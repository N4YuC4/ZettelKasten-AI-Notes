# semantic_memory_service.py
#
# Core semantic memory engine for Zettelkasten AI Notes.
# Manages local embedding generation (Microsoft Harrier 0.6B), vector similarity search,
# intra-document candidate pair filtering, and cross-vault semantic retrieval.
# Runs strictly on CPU to isolate compute and eliminate VRAM conflicts with GPU generation LLMs.

import os
import re
import threading
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, deque
import numpy as np
from logger import log_debug, log_error
import local_models_catalog


class EmbeddingModelNotFoundError(FileNotFoundError):
    """Raised when the specified embedding GGUF model file does not exist on disk."""
    pass


class EmbeddingInferenceError(RuntimeError):
    """Raised when local embedding inference fails."""
    pass


class SemanticMemoryService:
    """
    Manages local embedding extraction, L2 vector normalization, and pairwise cosine similarity.
    Powered by Microsoft Harrier 0.6B (32K context, 1024 dimensions, Multilingual).
    """

    _cached_llm: Optional[Any] = None
    _cached_model_path: Optional[str] = None
    _lock = threading.Lock()

    # Recommended task-specific prefixes for Harrier and Nomic architectures
    DOCUMENT_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None
    ):
        if model_path:
            self.model_path = os.path.abspath(model_path)
        else:
            default_dir = local_models_catalog.get_default_models_dir()
            harrier_info = local_models_catalog.get_model_by_id(local_models_catalog.DEFAULT_EMBEDDING_MODEL_ID)
            filename = harrier_info.filename if harrier_info else "harrier-oss-v1-0.6b.Q4_K_M.gguf"
            self.model_path = os.path.abspath(os.path.join(default_dir, filename))

        self.n_threads = n_threads or 2

    def is_model_available(self) -> bool:
        """Checks if the local embedding model file exists on disk."""
        return os.path.exists(self.model_path)

    def _get_or_load_embedder(self) -> Any:
        """Retrieves cached Llama instance configured for embeddings or initializes a new one on CPU."""
        if not self.is_model_available():
            raise EmbeddingModelNotFoundError(
                f"Embedding model file not found at: '{self.model_path}'. "
                "Please download Microsoft Harrier (0.6B) via Settings -> Model Manager."
            )

        with SemanticMemoryService._lock:
            if (
                SemanticMemoryService._cached_llm is not None
                and SemanticMemoryService._cached_model_path == self.model_path
            ):
                return SemanticMemoryService._cached_llm

            try:
                import llama_cpp
                from llama_cpp import Llama
                # Respect model's native pooling architecture from GGUF metadata
                # Harrier 0.6B specifies LLAMA_POOLING_TYPE_LAST (3), decoder-based autoregressive models require LAST token pooling
                pooling_type = getattr(llama_cpp, "LLAMA_POOLING_TYPE_UNSPECIFIED", -1)
            except ImportError as e:
                raise EmbeddingInferenceError(
                    "The llama-cpp-python library is required for local embeddings. "
                    "Please run 'pip install llama-cpp-python'."
                ) from e

            log_debug(
                f"Loading local embedding model: {self.model_path} "
                f"(CPU-only, threads={self.n_threads}, native GGUF pooling)"
            )

            old_vk = os.environ.get("VK_DRIVER_FILES")
            try:
                os.environ["VK_DRIVER_FILES"] = ""
                embedder = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    pooling_type=pooling_type,
                    n_gpu_layers=0,  # Strictly CPU to prevent VRAM competition with generator LLM
                    n_threads=self.n_threads,
                    verbose=False
                )
                SemanticMemoryService._cached_llm = embedder
                SemanticMemoryService._cached_model_path = self.model_path
                log_debug("Local embedding model loaded successfully into CPU memory.")
                return embedder
            except Exception as e:
                log_error(f"Failed to initialize embedding model: {e}")
                raise EmbeddingInferenceError(f"Failed to load embedding model: {e}") from e
            finally:
                if old_vk is not None:
                    os.environ["VK_DRIVER_FILES"] = old_vk
                else:
                    os.environ.pop("VK_DRIVER_FILES", None)

    @classmethod
    def unload_cached_model(cls) -> None:
        """Releases cached embedding model instance from memory."""
        with cls._lock:
            if cls._cached_llm is not None:
                del cls._cached_llm
                cls._cached_llm = None
                cls._cached_model_path = None
                log_debug("Cached local embedding model unloaded from memory.")

    def embed_text(self, text: str, prefix: str = DOCUMENT_PREFIX) -> np.ndarray:
        """
        Generates an L2-normalized float32 embedding vector for a single text string.
        """
        if not text or not text.strip():
            return np.zeros(1024, dtype=np.float32)

        embedder = self._get_or_load_embedder()
        full_prompt = f"{prefix}{text.strip()}"

        try:
            output = embedder.create_embedding(full_prompt)
            if not output or "data" not in output or not output["data"]:
                raise EmbeddingInferenceError("Empty embedding response returned by llama-cpp.")

            raw_vec = output["data"][0]["embedding"]
            vec_arr = np.asarray(raw_vec, dtype=np.float32)

            # Apply L2 normalization
            norm = float(np.linalg.norm(vec_arr))
            if norm > 1e-12:
                vec_arr = vec_arr / norm

            return vec_arr
        except Exception as e:
            log_error(f"Error generating embedding for text: {e}")
            raise EmbeddingInferenceError(f"Embedding inference failed: {e}") from e

    def embed_texts(self, texts: List[str], prefix: str = DOCUMENT_PREFIX) -> np.ndarray:
        """
        Generates an (N, D) float32 matrix of L2-normalized vectors for a list of texts.
        """
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        vectors: List[np.ndarray] = []
        for text in texts:
            vectors.append(self.embed_text(text, prefix=prefix))

        return np.vstack(vectors)

    def compute_candidate_pairs(
        self,
        notes: List[Dict[str, Any]],
        similarity_threshold: float = 0.65,
        max_candidates_per_note: Optional[int] = None,
        cross_chunk_only: bool = True
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
        """
        Computes pairwise cosine similarity across all generated notes in memory.
        Filters out self-links, duplicates, already-connected notes, and (optionally) intra-chunk notes,
        returning genuine cross-section candidate pairs matching or exceeding similarity_threshold.
        
        Returns:
            (candidate_pairs, embeddings_matrix)
            where candidate_pairs is a sorted list of (source_id, target_id, similarity_score) tuples (descending score),
            and embeddings_matrix is the (N, D) float32 matrix of note embeddings.
        """
        if not notes or len(notes) <= 1:
            if notes:
                emb = self.embed_texts([f"{notes[0].get('title', '')}\n\n{notes[0].get('content', '')}"])
                return [], emb
            return [], np.empty((0, 1024), dtype=np.float32)

        # 1. Prepare text payload for each note
        note_texts = [
            f"{n.get('title', '').strip()}\n\n{n.get('content', '').strip()}"
            for n in notes
        ]

        log_debug(f"Vectorizing {len(note_texts)} notes for semantic candidate filtering...")
        embeddings = self.embed_texts(note_texts, prefix=self.DOCUMENT_PREFIX)

        # 2. Pairwise cosine similarity via dot product (since vectors are L2-normalized: dot(u, v) == cos(u, v))
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Map existing bidirectional connections established in Stage 1
        existing_connections: Dict[Any, Set[str]] = {}
        for idx, n in enumerate(notes):
            nid = n.get("id", idx + 1)
            raw_conns = n.get("connections", [])
            existing_connections[nid] = {
                str(c).strip().casefold() for c in raw_conns if c
            }

        # Check if notes span multiple chunks
        chunk_ids = {n.get("_chunk_id") for n in notes if n.get("_chunk_id") is not None}
        has_multiple_chunks = len(chunk_ids) > 1

        candidate_pair_map: Dict[Tuple[int, int], float] = {}
        num_notes = len(notes)

        # 3. Collect candidates filtering out already-connected and redundant intra-chunk pairs
        for i in range(num_notes):
            note_i = notes[i]
            source_id = note_i.get("id", i + 1)
            title_i_lower = note_i.get("title", "").strip().casefold()
            chunk_i = note_i.get("_chunk_id")
            neighbors: List[Tuple[int, float]] = []

            for j in range(num_notes):
                if i == j:
                    continue
                note_j = notes[j]
                target_id = note_j.get("id", j + 1)
                title_j_lower = note_j.get("title", "").strip().casefold()
                chunk_j = note_j.get("_chunk_id")

                # Skip if already linked in Stage 1
                if (title_j_lower and title_j_lower in existing_connections.get(source_id, set())) or \
                   (title_i_lower and title_i_lower in existing_connections.get(target_id, set())):
                    continue

                # Skip intra-chunk pairs if document spans multiple chunks and cross_chunk_only is requested
                # (since Stage 1 already analyzed notes within the same chunk together)
                if cross_chunk_only and has_multiple_chunks and chunk_i is not None and chunk_j is not None and chunk_i == chunk_j:
                    continue

                sim = float(similarity_matrix[i, j])
                if sim >= similarity_threshold:
                    neighbors.append((target_id, round(sim, 4)))

            # Sort candidate neighbors by similarity descending
            neighbors.sort(key=lambda item: item[1], reverse=True)

            if max_candidates_per_note is not None and max_candidates_per_note > 0:
                neighbors = neighbors[:max_candidates_per_note]

            for target_id, sim in neighbors:
                canonical = (min(source_id, target_id), max(source_id, target_id))
                if canonical not in candidate_pair_map or sim > candidate_pair_map[canonical]:
                    candidate_pair_map[canonical] = sim

        candidate_pairs = [
            (src, tgt, score)
            for (src, tgt), score in candidate_pair_map.items()
        ]
        # Sort descending by conceptual similarity
        candidate_pairs.sort(key=lambda item: item[2], reverse=True)

        total_possible = (num_notes * (num_notes - 1)) // 2
        log_debug(
            f"Semantic filtering completed: {len(candidate_pairs)} / {total_possible} candidate pairs "
            f"qualified (threshold >= {similarity_threshold}, cross_chunk_only={cross_chunk_only and has_multiple_chunks}, max_per_note={max_candidates_per_note})."
        )

        return candidate_pairs, embeddings

    def consolidate_and_deduplicate_notes(
        self,
        notes: List[Dict[str, Any]],
        similarity_threshold: float = 0.90,
        lexical_threshold: float = 0.60,
        min_combined_threshold: float = 0.85,
        ai_provider: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Global batch semantic deduplication across all notes in a document.
        Discovers duplicate/paraphrased concepts using embedding cosine similarity and
        title lexical similarity, merges connections and details into the primary note,
        and remaps all graph connections seamlessly.
        Supports N-way multi-note LLM synthesis when ai_provider is provided,
        falling back seamlessly to algorithmic paragraph synthesis.
        """
        if not notes or len(notes) <= 1:
            return notes

        import difflib
        from ai_response_parser import AiResponseParser

        # Ensure embeddings exist for all notes
        missing_indices = [i for i, n in enumerate(notes) if "_embedding" not in n or n["_embedding"] is None]
        if missing_indices:
            texts_to_embed = [
                f"{notes[i].get('title', '').strip()}\n\n{notes[i].get('content', '').strip()}"
                for i in missing_indices
            ]
            try:
                embs = self.embed_texts(texts_to_embed, prefix=self.DOCUMENT_PREFIX)
                for idx, orig_idx in enumerate(missing_indices):
                    notes[orig_idx]["_embedding"] = embs[idx]
            except Exception as e:
                log_debug(f"SemanticMemoryService: Could not embed notes for deduplication ({e}). Skipping.")
                return notes

        embeddings = np.array([n["_embedding"] for n in notes], dtype=np.float32)

        num_notes = len(notes)
        dup_graph: Dict[int, Set[int]] = defaultdict(set)

        for i in range(num_notes):
            for j in range(i + 1, num_notes):
                t1 = str(notes[i].get("title", "")).strip()
                t2 = str(notes[j].get("title", "")).strip()
                c1 = str(notes[i].get("content", "")).strip()
                c2 = str(notes[j].get("content", "")).strip()

                # Normalized cosine similarity
                norm1 = np.linalg.norm(embeddings[i])
                norm2 = np.linalg.norm(embeddings[j])
                sim = float(np.dot(embeddings[i], embeddings[j]) / (norm1 * norm2)) if (norm1 >= 1e-12 and norm2 >= 1e-12) else 0.0

                is_dup, _, _ = AiResponseParser.is_duplicate_concept(t1, t2, c1, c2, sim)
                if is_dup:
                    dup_graph[i].add(j)
                    dup_graph[j].add(i)

        if not dup_graph:
            return notes

        # Discover multi-way connected duplicate clusters using BFS
        visited_indices = set()
        clusters: List[List[int]] = []
        for i in range(num_notes):
            if i in dup_graph and i not in visited_indices:
                comp = []
                q = deque([i])
                visited_indices.add(i)
                while q:
                    curr = q.popleft()
                    comp.append(curr)
                    for neigh in dup_graph[curr]:
                        if neigh not in visited_indices:
                            visited_indices.add(neigh)
                            q.append(neigh)
                clusters.append(comp)

        dup_to_primary: Dict[int, int] = {}
        title_redirects: Dict[str, str] = {}
        id_redirects: Dict[Any, Any] = {}

        for cluster in clusters:
            # Sort cluster notes by content length descending (longest note is primary candidate)
            cluster.sort(key=lambda idx: len(str(notes[idx].get("content", ""))), reverse=True)
            primary_idx = cluster[0]
            cluster_notes = [dict(notes[idx]) for idx in cluster]

            synthesized = False
            # 1. Attempt N-way LLM Synthesis if ai_provider is available
            if ai_provider is not None and hasattr(ai_provider, "synthesize_note_cluster"):
                try:
                    synth_res = ai_provider.synthesize_note_cluster(cluster_notes)
                    if isinstance(synth_res, dict) and synth_res.get("content"):
                        old_title = notes[primary_idx].get("title", "")
                        new_title = synth_res.get("title") or old_title
                        notes[primary_idx]["title"] = new_title
                        notes[primary_idx]["content"] = synth_res["content"]
                        if new_title != old_title:
                            title_redirects[old_title] = new_title
                            for k, v in list(title_redirects.items()):
                                if v == old_title:
                                    title_redirects[k] = new_title
                        if "connections" not in notes[primary_idx] or not isinstance(notes[primary_idx]["connections"], list):
                            notes[primary_idx]["connections"] = []
                        for c in synth_res.get("connections", []):
                            if c and c not in notes[primary_idx]["connections"]:
                                notes[primary_idx]["connections"].append(c)
                        synthesized = True
                        log_debug(
                            f"SemanticMemoryService: N-way LLM synthesized {len(cluster)} notes into "
                            f"'{new_title}' ({len(notes[primary_idx]['content'])} chars)."
                        )
                except Exception as synth_err:
                    log_debug(f"SemanticMemoryService: LLM cluster synthesis failed ({synth_err}). Falling back to algorithmic synthesis.")

            # 2. Algorithmic Fallback (if LLM unavailable or failed)
            if not synthesized:
                for dup_idx in cluster[1:]:
                    pri_content = str(notes[primary_idx].get("content", ""))
                    sec_content = str(notes[dup_idx].get("content", ""))
                    notes[primary_idx]["content"] = AiResponseParser.synthesize_note_contents(pri_content, sec_content)

            # Re-embed the synthesized primary note
            try:
                new_emb = self.embed_text(
                    f"{notes[primary_idx].get('title', '')}\n\n{notes[primary_idx].get('content', '')}",
                    prefix=self.DOCUMENT_PREFIX
                )
                notes[primary_idx]["_embedding"] = new_emb
                embeddings[primary_idx] = new_emb
            except Exception as e:
                log_debug(f"SemanticMemoryService: Could not re-embed enriched note: {e}")

            # Merge all connections from duplicate notes and configure redirects
            if "connections" not in notes[primary_idx] or not isinstance(notes[primary_idx]["connections"], list):
                notes[primary_idx]["connections"] = []

            for dup_idx in cluster[1:]:
                dup_to_primary[dup_idx] = primary_idx
                title_redirects[notes[dup_idx].get("title", "")] = notes[primary_idx].get("title", "")
                if "id" in notes[dup_idx] and "id" in notes[primary_idx]:
                    dup_id = notes[dup_idx]["id"]
                    if isinstance(dup_id, (list, tuple)):
                        dup_id = dup_id[0] if dup_id else None
                    if dup_id is not None:
                        try:
                            id_redirects[dup_id] = notes[primary_idx]["id"]
                        except TypeError:
                            pass

                # Merge connections
                raw_conns = notes[dup_idx].get("connections", [])
                if isinstance(raw_conns, list):
                    flat_conns: List[str] = []
                    def _flat_c(val: Any) -> None:
                        if isinstance(val, (list, tuple)):
                            for sub in val:
                                _flat_c(sub)
                        elif isinstance(val, str) and val.strip():
                            flat_conns.append(val.strip())
                    _flat_c(raw_conns)
                    for c in flat_conns:
                        if c and c not in notes[primary_idx]["connections"]:
                            notes[primary_idx]["connections"].append(c)

                log_debug(
                    f"SemanticMemoryService: Consolidated duplicate note '{notes[dup_idx].get('title')}' "
                    f"into '{notes[primary_idx].get('title')}' (cluster of {len(cluster)} notes)."
                )

        # Build surviving notes list, remap connections, and remap markdown wikilinks
        surviving_notes: List[Dict[str, Any]] = []
        for idx, n in enumerate(notes):
            if idx in dup_to_primary:
                continue
            note_copy = dict(n)
            orig_conns = note_copy.get("connections", [])
            title_clean = note_copy.get("title", "").strip().casefold()
            if isinstance(orig_conns, list):
                new_conns = []
                seen_conns = set()
                for c in orig_conns:
                    if not isinstance(c, str):
                        continue
                    resolved = title_redirects.get(c, c)
                    resolved_lower = resolved.strip().casefold()
                    if resolved_lower == title_clean or resolved_lower in seen_conns:
                        continue
                    seen_conns.add(resolved_lower)
                    new_conns.append(resolved)
                note_copy["connections"] = new_conns

            # Remap body wikilinks for any redirected titles
            for old_t, new_t in title_redirects.items():
                if old_t and new_t and old_t != new_t:
                    note_copy["content"] = re.sub(
                        r'\[\[' + re.escape(old_t) + r'(\]\]|\|)',
                        lambda m, nt=new_t: f"[[{nt}{m.group(1)}",
                        note_copy.get("content", "")
                    )

            surviving_notes.append(note_copy)

        log_debug(
            f"SemanticMemoryService: Deduplication consolidated {len(notes)} notes into "
            f"{len(surviving_notes)} unique notes ({len(dup_to_primary)} duplicates removed)."
        )
        return surviving_notes

    def compute_semantic_links(
        self,
        notes: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
        sensitivity_k: float = 1.0,
        quality_floor: float = 0.65,
        cross_chunk_only: bool = True,
        max_links_per_note: int = 4,
        prevent_orphans: bool = True,
        min_orphan_similarity: float = 0.50
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        Computes pure semantic knowledge graph connections across notes in memory without an LLM.
        Uses statistical Z-score outlier detection (dynamic threshold = mean + k * std, default k=1.0, floor=0.65)
        with degree capping and orphan note protection to discover genuine conceptual resonances
        without hub bloat or isolated nodes.

        Returns:
            (link_pairs, embeddings_matrix, dynamic_threshold)
            where link_pairs is a list of (source_id, target_id) tuples,
            embeddings_matrix is the (N, D) float32 matrix of note embeddings,
            and dynamic_threshold is the calculated similarity cutoff.
        """
        if not notes or len(notes) <= 1:
            if notes:
                emb = self.embed_texts([f"{notes[0].get('title', '')}\n\n{notes[0].get('content', '')}"])
                return [], emb, quality_floor
            return [], np.empty((0, 1024), dtype=np.float32), quality_floor

        # 1. Prepare text payload for each note
        note_texts = [
            f"{n.get('title', '').strip()}\n\n{n.get('content', '').strip()}"
            for n in notes
        ]

        log_debug(f"Vectorizing {len(note_texts)} notes for pure semantic linking...")
        embeddings = self.embed_texts(note_texts, prefix=self.DOCUMENT_PREFIX)

        # 2. Pairwise cosine similarity matrix via dot product (L2-normalized)
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Map existing connections from Stage 1
        existing_connections: Dict[Any, Set[str]] = {}
        for idx, n in enumerate(notes):
            nid = n.get("id", idx + 1)
            raw_conns = n.get("connections", [])
            existing_connections[nid] = {
                str(c).strip().casefold() for c in raw_conns if c
            }

        # Check if notes span multiple chunks
        chunk_ids = {n.get("_chunk_id") for n in notes if n.get("_chunk_id") is not None}
        has_multiple_chunks = len(chunk_ids) > 1

        num_notes = len(notes)
        valid_scores: List[float] = []
        valid_pairs_with_scores: List[Tuple[int, int, float]] = []

        # 3. Collect candidate pairs (excluding self, already-linked, and intra-chunk if multi-chunk)
        for i in range(num_notes):
            note_i = notes[i]
            source_id = note_i.get("id", i + 1)
            title_i_lower = note_i.get("title", "").strip().casefold()
            chunk_i = note_i.get("_chunk_id")

            for j in range(i + 1, num_notes):
                note_j = notes[j]
                target_id = note_j.get("id", j + 1)
                title_j_lower = note_j.get("title", "").strip().casefold()
                chunk_j = note_j.get("_chunk_id")

                # Skip if already linked in Stage 1
                if (title_j_lower and title_j_lower in existing_connections.get(source_id, set())) or \
                   (title_i_lower and title_i_lower in existing_connections.get(target_id, set())):
                    continue

                # Skip intra-chunk pairs if multi-chunk and cross_chunk_only is enabled
                if cross_chunk_only and has_multiple_chunks and chunk_i is not None and chunk_j is not None and chunk_i == chunk_j:
                    continue

                sim = float(similarity_matrix[i, j])
                valid_scores.append(sim)
                valid_pairs_with_scores.append((source_id, target_id, sim))

        if not valid_scores:
            log_debug("No valid unlinked pairs found across document. Stage 2 linking complete.")
            return [], embeddings, quality_floor

        # 4. Determine dynamic threshold
        if similarity_threshold is not None and similarity_threshold > 0:
            effective_threshold = similarity_threshold
        else:
            arr_scores = np.asarray(valid_scores, dtype=np.float32)
            mean_sim = float(np.mean(arr_scores))
            std_sim = float(np.std(arr_scores))
            statistical_cutoff = mean_sim + (sensitivity_k * std_sim)
            # Cap at 0.90 to prevent small-sample Z-score blowup, and round to 4 decimals
            effective_threshold = round(min(0.90, max(quality_floor, statistical_cutoff)), 4)
            log_debug(
                f"Dynamic semantic threshold computed: mean={mean_sim:.4f}, std={std_sim:.4f}, "
                f"k={sensitivity_k}, cutoff={statistical_cutoff:.4f} -> effective_threshold={effective_threshold:.4f}"
            )

        # 5. Filter and sort pairs descending by score, applying degree capping
        valid_pairs_with_scores.sort(key=lambda item: item[2], reverse=True)
        link_pairs: List[Tuple[int, int]] = []
        new_degrees: Dict[int, int] = defaultdict(int)

        for src, tgt, score in valid_pairs_with_scores:
            if score >= effective_threshold:
                if max_links_per_note is None or (
                    new_degrees[src] < max_links_per_note and new_degrees[tgt] < max_links_per_note
                ):
                    link_pairs.append((src, tgt))
                    new_degrees[src] += 1
                    new_degrees[tgt] += 1

        # 6. Orphan Note Protection: Ensure no note is left with 0 connections
        if prevent_orphans and num_notes > 1:
            for i in range(num_notes):
                nid = notes[i].get("id", i + 1)
                total_conns = len(existing_connections.get(nid, set())) + new_degrees[nid]
                if total_conns == 0:
                    best_partner_id = None
                    best_score = 0.0
                    for j in range(num_notes):
                        if i == j:
                            continue
                        other_id = notes[j].get("id", j + 1)
                        sim = float(similarity_matrix[i, j])
                        if sim > best_score:
                            best_score = sim
                            best_partner_id = other_id

                    if best_partner_id is not None and best_score >= min_orphan_similarity:
                        link_pairs.append((nid, best_partner_id))
                        new_degrees[nid] += 1
                        new_degrees[best_partner_id] += 1
                        log_debug(
                            f"SemanticMemoryService: Connected orphan note [ID {nid}] '{notes[i].get('title')}' "
                            f"to closest conceptual neighbor [ID {best_partner_id}] (sim={best_score:.4f})."
                        )

        # 7. Disconnected Component Bridging: Ensure no isolated subgraphs/islands exist
        if prevent_orphans and num_notes > 2:
            title_to_nid = {
                str(n.get("title", "")).strip().casefold(): n.get("id", idx + 1)
                for idx, n in enumerate(notes)
            }
            nid_to_idx = {n.get("id", idx + 1): idx for idx, n in enumerate(notes)}
            adj_graph: Dict[Any, Set[Any]] = defaultdict(set)
            for idx, n in enumerate(notes):
                nid = n.get("id", idx + 1)
                for c_title in existing_connections.get(nid, set()):
                    target_nid = title_to_nid.get(c_title)
                    if target_nid is not None and target_nid != nid:
                        adj_graph[nid].add(target_nid)
                        adj_graph[target_nid].add(nid)
            for src, tgt in link_pairs:
                adj_graph[src].add(tgt)
                adj_graph[tgt].add(src)

            all_nids = [n.get("id", idx + 1) for idx, n in enumerate(notes)]
            visited_nids = set()
            components: List[List[Any]] = []
            for nid in all_nids:
                if nid not in visited_nids:
                    comp = []
                    q = deque([nid])
                    visited_nids.add(nid)
                    while q:
                        curr = q.popleft()
                        comp.append(curr)
                        for neigh in adj_graph[curr]:
                            if neigh not in visited_nids:
                                visited_nids.add(neigh)
                                q.append(neigh)
                    components.append(comp)

            if len(components) > 1:
                components.sort(key=len, reverse=True)
                main_component = set(components[0])
                bridge_threshold = min(quality_floor, max(min_orphan_similarity, 0.60))
                for comp in components[1:]:
                    best_pair = None
                    best_sim = -1.0
                    for u in comp:
                        u_idx = nid_to_idx.get(u)
                        if u_idx is None:
                            continue
                        for v in main_component:
                            v_idx = nid_to_idx.get(v)
                            if v_idx is None:
                                continue
                            sim = float(similarity_matrix[u_idx, v_idx])
                            if sim > best_sim:
                                best_sim = sim
                                best_pair = (u, v)

                    if best_pair and best_sim >= bridge_threshold:
                        canonical = (min(best_pair[0], best_pair[1]), max(best_pair[0], best_pair[1]))
                        existing_set = {(min(a, b), max(a, b)) for a, b in link_pairs}
                        if canonical not in existing_set:
                            link_pairs.append(canonical)
                            new_degrees[canonical[0]] += 1
                            new_degrees[canonical[1]] += 1
                            log_debug(
                                f"SemanticMemoryService: Bridged isolated knowledge cluster to main graph "
                                f"via link {canonical[0]} <--> {canonical[1]} (sim={best_sim:.4f} >= {bridge_threshold:.4f})."
                            )
                        main_component.update(comp)

        log_debug(
            f"Pure semantic linking completed: {len(link_pairs)} / {len(valid_scores)} pairs qualified "
            f"at threshold >= {effective_threshold:.4f}."
        )

        return link_pairs, embeddings, effective_threshold

    def find_similar_vault_notes(
        self,
        query_vector: np.ndarray,
        vault_embeddings: Dict[str, np.ndarray],
        similarity_threshold: float = 0.60,
        top_k: int = 5,
        exclude_note_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compares a single query vector against the entire vault of note embeddings.
        Returns top_k most similar notes matching or exceeding similarity_threshold.
        """
        if not vault_embeddings or query_vector is None or len(query_vector) == 0:
            return []

        exclude = exclude_note_ids or set()
        candidates: List[Tuple[str, float]] = []

        q_norm = np.linalg.norm(query_vector)
        if q_norm < 1e-12:
            return []
        q_vec = query_vector / q_norm

        for note_id, emb_vec in vault_embeddings.items():
            if note_id in exclude or emb_vec is None or len(emb_vec) == 0:
                continue

            v_norm = np.linalg.norm(emb_vec)
            if v_norm < 1e-12:
                continue

            cos_sim = float(np.dot(q_vec, emb_vec / v_norm))
            if cos_sim >= similarity_threshold:
                candidates.append((note_id, round(cos_sim, 4)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]
