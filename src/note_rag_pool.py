# note_rag_pool.py
#
# In-memory vector RAG pool for multi-chunk Zettelkasten note generation.
# Implements a Two-Tier retrieval architecture:
# - Tier 1: Global Concept Map (all accumulated notes with id, title, and 1-sentence core mechanism).
# - Tier 2: Focal Notes (selected via Microsoft Harrier 0.6B embeddings + Qwen3-Reranker 0.6B cross-encoder).
# Both the embedding model and reranker model are strictly mandatory.

import re
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from logger import log_debug, log_error
from semantic_memory_service import SemanticMemoryService, EmbeddingModelNotFoundError
from reranker_service import RerankerService, RerankerModelNotFoundError


class NoteRagPool:
    """
    Session-scoped in-memory vector RAG repository for atomic Zettelkasten notes.
    Collects newly generated notes chunk by chunk, computes embeddings, and performs
    two-stage retrieval (Harrier bi-encoder candidate selection + Qwen3 cross-encoder reranking)
    alongside a panoramic Tier 1 global concept map to eliminate duplicates across document chunks.
    """

    def __init__(
        self,
        semantic_memory_service: Optional[SemanticMemoryService] = None,
        reranker_service: Optional[RerankerService] = None,
        count_tokens_fn: Optional[Callable[[str], int]] = None,
        ai_provider: Optional[Any] = None
    ):
        """
        Initializes NoteRagPool with mandatory SemanticMemoryService and RerankerService.
        Raises EmbeddingModelNotFoundError or RerankerModelNotFoundError if models are not on disk.
        """
        self.ai_provider = ai_provider
        if semantic_memory_service is not None:
            self.semantic_memory_service = semantic_memory_service
        else:
            self.semantic_memory_service = SemanticMemoryService()

        if reranker_service is not None:
            self.reranker_service = reranker_service
        else:
            self.reranker_service = RerankerService()

        # Enforce that BOTH embedding and reranker models MUST exist on the system
        if not self.semantic_memory_service.is_model_available():
            raise EmbeddingModelNotFoundError(
                f"Embedding model is mandatory but not found at: '{self.semantic_memory_service.model_path}'. "
                "Please download Microsoft Harrier (0.6B) via Settings -> Model Manager before generating notes."
            )

        if not self.reranker_service.is_model_available():
            raise RerankerModelNotFoundError(
                f"Reranker model is mandatory but not found at: '{self.reranker_service.model_path}'. "
                "Please download Qwen3 Reranker (0.6B) via Settings -> Model Manager before generating notes."
            )

        self.count_tokens_fn = count_tokens_fn or self._default_count_tokens
        self._notes: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []
        self._seen_titles: set = set()
        self._title_redirects: Dict[str, str] = {}
        self._id_redirects: Dict[Any, Any] = {}
        self._current_max_id: int = 0
        self._is_reconciled: bool = False

    @staticmethod
    def _default_count_tokens(text: str) -> int:
        """Conservative token estimation for multilingual and CJK prose (~3.2 chars per token)."""
        if not text:
            return 0
        cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
        other_chars = len(text) - cjk_count
        return max(1, int(cjk_count * 1.5 + other_chars / 3.2) + 1)

    @staticmethod
    def _extract_core_mechanism(note: Dict[str, Any], content: str) -> str:
        """Extracts or derives a clean 1-sentence core mechanism summary (~25-35 tokens)."""
        if "core_mechanism" in note and note["core_mechanism"]:
            mechanism = str(note["core_mechanism"]).strip()
        elif "conceptual_analysis" in note and isinstance(note["conceptual_analysis"], dict):
            mechanism = str(note["conceptual_analysis"].get("core_thesis", "")).strip()
        else:
            mechanism = ""

        if not mechanism:
            clean_lines = [
                l.strip() for l in content.split("\n")
                if l.strip() and not l.strip().startswith("#")
            ]
            first_text = clean_lines[0] if clean_lines else content
            # Split at first sentence terminator
            parts = re.split(r'(?<=[.!?])\s+', first_text, maxsplit=1)
            mechanism = parts[0] if parts else first_text[:160]

        mechanism = mechanism.replace("\n", " ").strip()
        if len(mechanism) > 200:
            mechanism = mechanism[:197] + "..."
        return mechanism

    def add_notes(self, new_notes: List[Dict[str, Any]]) -> None:
        """
        Adds newly generated notes to the pool with semantic and lexical deduplication.
        Assigns persistent global 1-based IDs, vectorizes title + content,
        derives compact core mechanism, and merges duplicates into existing notes.
        """
        if not new_notes:
            return

        self._is_reconciled = False

        import difflib
        from ai_response_parser import AiResponseParser

        texts_to_embed = []
        candidates_to_embed = []

        for note in new_notes:
            if not isinstance(note, dict):
                continue
            title = str(note.get("title", "")).strip()
            content = str(note.get("content", "")).strip()

            if not title or not content:
                continue

            title_key = title.casefold()
            # Fast lexical check against exact previously seen titles
            if title_key in self._seen_titles:
                continue

            note_copy = dict(note)
            note_copy["title"] = title
            note_copy["content"] = content
            note_copy["core_mechanism"] = self._extract_core_mechanism(note, content)

            payload_text = f"{title}\n\n{content}"
            texts_to_embed.append(payload_text)
            candidates_to_embed.append(note_copy)

        if not candidates_to_embed:
            return

        # Vectorize using SemanticMemoryService
        try:
            log_debug(f"NoteRagPool: Embedding {len(candidates_to_embed)} candidate notes for pool ingestion...")
            vectors = self.semantic_memory_service.embed_texts(
                texts_to_embed,
                prefix=self.semantic_memory_service.DOCUMENT_PREFIX
            )
        except Exception as e:
            log_error(f"NoteRagPool: Failed to generate embeddings for new notes: {e}")
            raise

        added_count = 0
        merged_count = 0

        for idx, cand_note in enumerate(candidates_to_embed):
            cand_emb = vectors[idx]
            cand_note["_embedding"] = cand_emb
            title = cand_note["title"]
            content = cand_note["content"]

            # Semantic deduplication against existing notes in pool (Best-match selection across all existing notes)
            best_match = None
            best_match_score = -1.0
            best_sim = 0.0
            best_overlap = 0.0

            if self._embeddings:
                for exist_idx, candidate_existing in enumerate(self._notes):
                    c_emb = self._embeddings[exist_idx]
                    t1 = title
                    t2 = candidate_existing["title"]
                    c1 = content
                    c2 = candidate_existing["content"]

                    # 1. Never merge if content is too short (< 80 chars) to prevent false test/stub merges
                    if len(c1) < 80 or len(c2) < 80:
                        continue

                    # Normalized cosine similarity
                    norm1 = np.linalg.norm(cand_emb)
                    norm2 = np.linalg.norm(c_emb)
                    sim = float(np.dot(cand_emb, c_emb) / (norm1 * norm2)) if (norm1 >= 1e-12 and norm2 >= 1e-12) else 0.0

                    is_dup, match_score, content_overlap = AiResponseParser.is_duplicate_concept(
                        t1, t2, c1, c2, sim
                    )

                    if is_dup:
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match = candidate_existing
                            best_sim = sim
                            best_overlap = content_overlap

            if best_match is not None:
                matched_existing = best_match
                max_sim = best_sim
                content_overlap = best_overlap
                merged_count += 1
                log_debug(
                    f"NoteRagPool: Semantic duplicate concept detected: '{title}' matches '{matched_existing['title']}' "
                    f"(sim={max_sim:.4f}, overlap={content_overlap:.2f}). Merging into existing note."
                )
                old_matched_title = matched_existing["title"]
                self._title_redirects[title] = old_matched_title
                if "id" in cand_note:
                    cand_id = cand_note["id"]
                    if isinstance(cand_id, (list, tuple)):
                        cand_id = cand_id[0] if cand_id else None
                    if cand_id is not None:
                        try:
                            self._id_redirects[cand_id] = matched_existing["id"]
                        except TypeError:
                            pass

                # Merge unique connections
                raw_cand_conns = cand_note.get("connections", [])
                if isinstance(raw_cand_conns, list):
                    flat_cand_conns: List[str] = []
                    def _flat_c(val: Any) -> None:
                        if isinstance(val, (list, tuple)):
                            for sub in val:
                                _flat_c(sub)
                        elif isinstance(val, str) and val.strip():
                            flat_cand_conns.append(val.strip())
                    _flat_c(raw_cand_conns)
                    if "connections" not in matched_existing or not isinstance(matched_existing["connections"], list):
                        matched_existing["connections"] = []
                    for conn in flat_cand_conns:
                        if conn and conn not in matched_existing["connections"]:
                            matched_existing["connections"].append(conn)

                synthesized = False
                if self.ai_provider is not None and hasattr(self.ai_provider, "synthesize_note_cluster"):
                    try:
                        synth_res = self.ai_provider.synthesize_note_cluster([matched_existing, cand_note])
                        if isinstance(synth_res, dict) and synth_res.get("content"):
                            new_title = synth_res.get("title") or old_matched_title
                            matched_existing["title"] = new_title
                            matched_existing["content"] = synth_res["content"]
                            if new_title != old_matched_title:
                                self._title_redirects[old_matched_title] = new_title
                                self._title_redirects[title] = new_title
                                for k, v in list(self._title_redirects.items()):
                                    if v == old_matched_title:
                                        self._title_redirects[k] = new_title
                            if "connections" not in matched_existing or not isinstance(matched_existing["connections"], list):
                                matched_existing["connections"] = []
                            for c in synth_res.get("connections", []):
                                if c and c not in matched_existing["connections"]:
                                    matched_existing["connections"].append(c)
                            matched_existing["core_mechanism"] = self._extract_core_mechanism(matched_existing, synth_res["content"])
                            synthesized = True
                            log_debug(
                                f"NoteRagPool: LLM synthesized duplicate into '{new_title}' "
                                f"({len(matched_existing['content'])} chars)."
                            )
                    except Exception as synth_err:
                        log_debug(f"NoteRagPool: LLM cluster synthesis failed ({synth_err}). Falling back to algorithmic synthesis.")

                if not synthesized:
                    # Synthesize content non-redundantly to prevent any loss of novel details or examples
                    synthesized_content = AiResponseParser.synthesize_note_contents(
                        matched_existing["content"], content
                    )
                    if len(synthesized_content) > len(matched_existing["content"]):
                        matched_existing["content"] = synthesized_content
                        matched_existing["core_mechanism"] = self._extract_core_mechanism(matched_existing, synthesized_content)
                    elif len(content) > len(matched_existing["content"]) + 100:
                        matched_existing["content"] = content
                        matched_existing["core_mechanism"] = self._extract_core_mechanism(cand_note, content)

                try:
                    updated_emb = self.semantic_memory_service.embed_text(
                        f"{matched_existing['title']}\n\n{matched_existing['content']}",
                        prefix=self.semantic_memory_service.DOCUMENT_PREFIX
                    )
                    matched_existing["_embedding"] = updated_emb
                    self._embeddings[self._notes.index(matched_existing)] = updated_emb
                    log_debug(
                        f"NoteRagPool: Re-embedded enriched note '{matched_existing['title']}' "
                        f"(content size: {len(matched_existing['content'])} chars)."
                    )
                except Exception as e:
                    log_debug(f"NoteRagPool: Failed to re-embed enriched note: {e}")
            else:
                if "id" in cand_note:
                    raw_cand_id = cand_note["id"]
                    if isinstance(raw_cand_id, (list, tuple)):
                        raw_cand_id = raw_cand_id[0] if raw_cand_id else None
                    if raw_cand_id is not None:
                        try:
                            self._id_redirects[raw_cand_id] = self._current_max_id + 1
                        except TypeError:
                            pass
                self._current_max_id += 1
                cand_note["id"] = self._current_max_id
                self._seen_titles.add(title.casefold())
                self._notes.append(cand_note)
                self._embeddings.append(cand_emb)
                added_count += 1

        log_debug(
            f"NoteRagPool: Pool updated (+{added_count} distinct notes, {merged_count} duplicates merged). "
            f"Total notes in pool: {len(self._notes)}."
        )

    def get_compact_concept_map(self) -> List[Dict[str, Any]]:
        """
        Returns Tier 1 global concept map containing ALL notes accumulated in the pool
        with their integer ID, canonical title, and 1-sentence core mechanism.
        """
        return [
            {
                "id": n["id"],
                "title": n["title"],
                "core_mechanism": n.get("core_mechanism", "")
            }
            for n in self._notes
            if n.get("title")
        ]

    def retrieve_relevant_notes(
        self,
        query_chunk: str,
        top_k: int = 8,
        min_similarity: float = 0.15,
        max_tokens: int = 4000,
        candidate_pool_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval of focal notes:
        1. Stage 1 (Harrier Bi-Encoder): Computes cosine similarity and retrieves up to
           candidate_pool_size candidate notes (sim >= min_similarity).
        2. Stage 2 (Qwen3 Cross-Encoder): Reranks candidates using instruction-aware cross-attention.
        3. Packing: Greedily packs full note representations (id, title, content) up to top_k within max_tokens.
        """
        if not self._notes or not query_chunk or not str(query_chunk).strip():
            return []

        # Vectorize chunk query
        query_text = str(query_chunk).strip()[:4000]
        try:
            query_vec = self.semantic_memory_service.embed_text(
                query_text,
                prefix=self.semantic_memory_service.QUERY_PREFIX
            )
        except Exception as e:
            log_error(f"NoteRagPool: Failed to embed query chunk: {e}")
            raise

        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-12:
            return []
        q_vec = query_vec / q_norm

        # Matrix dot product (vectors are already L2-normalized)
        matrix = np.vstack(self._embeddings)
        sim_scores = np.dot(matrix, q_vec)

        # Stage 1: Gather top candidate notes via vector similarity
        scored_candidates = []
        for idx, sim in enumerate(sim_scores):
            score = float(sim)
            if score >= min_similarity:
                scored_candidates.append((score, self._notes[idx]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [note for _, note in scored_candidates[:candidate_pool_size]]

        if not top_candidates:
            # Fall back to top 3 notes even if below threshold
            top_candidates = [note for _, note in scored_candidates[:3]]

        if not top_candidates:
            return []

        # Stage 2: Cross-Encoder Reranking via Qwen3-Reranker
        try:
            log_debug(
                f"NoteRagPool: Reranking {len(top_candidates)} candidates using Qwen3-Reranker..."
            )
            reranked_pairs = self.reranker_service.score_candidates(
                query=query_text,
                candidates=top_candidates
            )
        except Exception as e:
            log_error(f"NoteRagPool: Reranking failed: {e}")
            raise

        # Stage 3: Greedily pack top notes within max_tokens budget
        selected_notes: List[Dict[str, Any]] = []
        accumulated_tokens = 0

        for score, note in reranked_pairs:
            if len(selected_notes) >= top_k:
                break

            note_repr = f"ID: {note['id']}\nTitle: {note['title']}\nContent: {note['content']}"
            note_tokens = self.count_tokens_fn(note_repr) + 30

            if selected_notes and (accumulated_tokens + note_tokens > max_tokens):
                continue
            elif not selected_notes and note_tokens > max_tokens:
                selected_notes.append({
                    "id": note["id"],
                    "title": note["title"],
                    "content": note["content"],
                    "rerank_score": score
                })
                accumulated_tokens += note_tokens
                break

            selected_notes.append({
                "id": note["id"],
                "title": note["title"],
                "content": note["content"],
                "rerank_score": score
            })
            accumulated_tokens += note_tokens

        top_score_str = f"{selected_notes[0]['rerank_score']:.2f}" if selected_notes and "rerank_score" in selected_notes[0] else "N/A"
        log_debug(
            f"NoteRagPool: Two-stage retrieval selected {len(selected_notes)}/{len(self._notes)} focal notes "
            f"(top score: {top_score_str}, tokens={accumulated_tokens}/{max_tokens})."
        )
        return selected_notes

    def retrieve_two_tier_context(
        self,
        query_chunk: str,
        total_budget: int = 5200,
        top_k_focal: int = 8
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieves two-tier context for a chunk:
        - Tier 1: Panoramic concept map (ALL notes with id, title, and 1-sentence core_mechanism).
        - Tier 2: Focal notes retrieved via Harrier embedding + Qwen3 reranker with full content.
        Total token footprint is dynamically budgeted to stay strictly within total_budget.
        """
        all_concepts = self.get_compact_concept_map()

        if not all_concepts:
            return [], []

        # Estimate Tier 1 tokens
        tier1_repr = "\n".join(
            f"[ID: {c['id']}] \"{c['title']}\" | {c.get('core_mechanism', '')}"
            for c in all_concepts
        )
        tier1_tokens = self.count_tokens_fn(tier1_repr) + 150

        # Allocate remainder to Tier 2 (minimum 3000 tokens for focal cards)
        tier2_budget = max(3000, total_budget - tier1_tokens)

        focal_notes = self.retrieve_relevant_notes(
            query_chunk=query_chunk,
            top_k=top_k_focal,
            max_tokens=tier2_budget
        )

        return all_concepts, focal_notes

    def get_all_titles(self) -> List[str]:
        """Returns the canonical titles of all notes currently accumulated in the pool."""
        return [n.get("title", "") for n in self._notes if n.get("title")]

    def _clean_and_remap_notes(self) -> List[Dict[str, Any]]:
        """Internal helper returning all current notes with connection redirects and body wikilinks resolved."""
        cleaned_notes = []
        for note in self._notes:
            note_copy = dict(note)
            orig_conns = note_copy.get("connections", [])
            title_clean = note_copy.get("title", "").strip().casefold()
            if isinstance(orig_conns, list):
                new_conns = []
                seen_conns = set()
                for c in orig_conns:
                    if not isinstance(c, str):
                        continue
                    # Remap if redirected
                    resolved = self._title_redirects.get(c, c)
                    resolved_lower = resolved.strip().casefold()
                    # Skip self-links and duplicates
                    if resolved_lower == title_clean or resolved_lower in seen_conns:
                        continue
                    seen_conns.add(resolved_lower)
                    new_conns.append(resolved)
                note_copy["connections"] = new_conns

            # Remap body wikilinks for any redirected titles
            for old_t, new_t in self._title_redirects.items():
                if old_t and new_t and old_t != new_t:
                    note_copy["content"] = re.sub(
                        r'\[\[' + re.escape(old_t) + r'(\]\]|\|)',
                        lambda m, nt=new_t: f"[[{nt}{m.group(1)}",
                        note_copy.get("content", "")
                    )

            cleaned_notes.append(note_copy)
        return cleaned_notes

    def reconcile_pool_globally(self) -> List[Dict[str, Any]]:
        """
        Global reconciliation pass across all accumulated notes in the pool using Union-Find.
        Runs once after all chunks have been processed.
        Discovers any multi-way or transitive duplicate clusters using pre-computed embeddings,
        character n-gram overlap, and title similarity.
        Consolidates each cluster into a single cohesive note via LLM synthesis (if ai_provider
        is available) or algorithmic non-redundant synthesis, remaps all graph connections
        and markdown wikilinks, and updates pool state in-place.
        """
        if self._is_reconciled or len(self._notes) <= 1:
            return self._clean_and_remap_notes()

        import difflib
        from collections import defaultdict
        from ai_response_parser import AiResponseParser

        num_notes = len(self._notes)
        parent = list(range(num_notes))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(num_notes):
            for j in range(i + 1, num_notes):
                n1 = self._notes[i]
                n2 = self._notes[j]
                t1 = n1.get("title", "")
                t2 = n2.get("title", "")
                c1 = n1.get("content", "")
                c2 = n2.get("content", "")

                emb1 = self._embeddings[i] if i < len(self._embeddings) else None
                emb2 = self._embeddings[j] if j < len(self._embeddings) else None

                if emb1 is not None and emb2 is not None:
                    norm1 = np.linalg.norm(emb1)
                    norm2 = np.linalg.norm(emb2)
                    sim = float(np.dot(emb1, emb2) / (norm1 * norm2)) if (norm1 >= 1e-12 and norm2 >= 1e-12) else 0.0
                else:
                    sim = 0.0

                is_dup, _, _ = AiResponseParser.is_duplicate_concept(t1, t2, c1, c2, sim)
                if is_dup:
                    union(i, j)

        clusters_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(num_notes):
            clusters_map[find(i)].append(i)

        has_duplicates = any(len(cl) > 1 for cl in clusters_map.values())
        if not has_duplicates:
            self._is_reconciled = True
            log_debug("NoteRagPool: Global reconciliation found 0 cross-chunk duplicates.")
            return self._clean_and_remap_notes()

        log_debug(
            f"NoteRagPool: Global reconciliation detected duplicate clusters across {num_notes} notes. "
            f"Consolidating..."
        )

        surviving_notes: List[Dict[str, Any]] = []
        surviving_embeddings: List[np.ndarray] = []

        for root, cluster in clusters_map.items():
            if len(cluster) == 1:
                idx = cluster[0]
                surviving_notes.append(self._notes[idx])
                if idx < len(self._embeddings):
                    surviving_embeddings.append(self._embeddings[idx])
                continue

            # Multi-note cluster: sort by length descending
            cluster.sort(key=lambda idx: len(str(self._notes[idx].get("content", ""))), reverse=True)
            primary_idx = cluster[0]
            primary_note = dict(self._notes[primary_idx])
            cluster_notes = [dict(self._notes[idx]) for idx in cluster]

            synthesized = False
            old_primary_title = primary_note.get("title", "")
            if self.ai_provider is not None and hasattr(self.ai_provider, "synthesize_note_cluster"):
                try:
                    synth_res = self.ai_provider.synthesize_note_cluster(cluster_notes)
                    if isinstance(synth_res, dict) and synth_res.get("content"):
                        new_title = synth_res.get("title") or old_primary_title
                        primary_note["title"] = new_title
                        primary_note["content"] = synth_res["content"]
                        if new_title != old_primary_title:
                            self._title_redirects[old_primary_title] = new_title
                            for k, v in list(self._title_redirects.items()):
                                if v == old_primary_title:
                                    self._title_redirects[k] = new_title
                        if "connections" not in primary_note or not isinstance(primary_note["connections"], list):
                            primary_note["connections"] = []
                        for c in synth_res.get("connections", []):
                            if c and c not in primary_note["connections"]:
                                primary_note["connections"].append(c)
                        primary_note["core_mechanism"] = self._extract_core_mechanism(primary_note, synth_res["content"])
                        synthesized = True
                        log_debug(
                            f"NoteRagPool: Global LLM synthesized cluster of {len(cluster)} notes into '{new_title}'"
                        )
                except Exception as e:
                    log_debug(f"NoteRagPool: Global LLM synthesis failed ({e}). Falling back to algorithmic synthesis.")

            if not synthesized:
                for dup_idx in cluster[1:]:
                    sec_content = str(self._notes[dup_idx].get("content", ""))
                    primary_note["content"] = AiResponseParser.synthesize_note_contents(
                        primary_note.get("content", ""), sec_content
                    )
                primary_note["core_mechanism"] = self._extract_core_mechanism(primary_note, primary_note["content"])

            # Merge connections from all duplicate notes in cluster
            if "connections" not in primary_note or not isinstance(primary_note["connections"], list):
                primary_note["connections"] = []
            for dup_idx in cluster[1:]:
                dup_note = self._notes[dup_idx]
                dup_title = dup_note.get("title", "")
                self._title_redirects[dup_title] = primary_note["title"]
                if "id" in dup_note and "id" in primary_note:
                    dup_id = dup_note["id"]
                    if isinstance(dup_id, (list, tuple)):
                        dup_id = dup_id[0] if dup_id else None
                    if dup_id is not None:
                        try:
                            self._id_redirects[dup_id] = primary_note["id"]
                        except TypeError:
                            pass
                raw_dup_conns = dup_note.get("connections", [])
                if isinstance(raw_dup_conns, list):
                    flat_dup_conns: List[str] = []
                    def _flat_dup_c(val: Any) -> None:
                        if isinstance(val, (list, tuple)):
                            for sub in val:
                                _flat_dup_c(sub)
                        elif isinstance(val, str) and val.strip():
                            flat_dup_conns.append(val.strip())
                    _flat_dup_c(raw_dup_conns)
                    for c in flat_dup_conns:
                        if c and c not in primary_note["connections"]:
                            primary_note["connections"].append(c)

            # Re-embed the synthesized primary note
            try:
                new_emb = self.semantic_memory_service.embed_text(
                    f"{primary_note['title']}\n\n{primary_note['content']}",
                    prefix=self.semantic_memory_service.DOCUMENT_PREFIX
                )
                primary_note["_embedding"] = new_emb
            except Exception as e:
                new_emb = self._embeddings[primary_idx] if primary_idx < len(self._embeddings) else np.zeros(1024, dtype=np.float32)

            surviving_notes.append(primary_note)
            surviving_embeddings.append(new_emb)

        self._notes = surviving_notes
        self._embeddings = surviving_embeddings
        self._is_reconciled = True

        log_debug(
            f"NoteRagPool: Global reconciliation completed: {num_notes} notes consolidated into "
            f"{len(self._notes)} unique notes."
        )

        return self._clean_and_remap_notes()

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Returns all notes collected across all chunks with global reconciliation and redirects resolved."""
        return self.reconcile_pool_globally()

    def __len__(self) -> int:
        return len(self._notes)

    def clear(self) -> None:
        """Resets the pool state."""
        self._notes.clear()
        self._embeddings.clear()
        self._seen_titles.clear()
        self._title_redirects.clear()
        self._id_redirects.clear()
        self._current_max_id = 0
        self._is_reconciled = False
