# semantic_memory_service.py
#
# Core semantic memory engine for Zettelkasten AI Notes.
# Manages local embedding generation (Microsoft Harrier 0.6B), vector similarity search,
# intra-document candidate pair filtering, and cross-vault semantic retrieval.
# Runs strictly on CPU to isolate compute and eliminate VRAM conflicts with GPU generation LLMs.

import os
import threading
from typing import List, Dict, Any, Tuple, Optional, Set
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
                from llama_cpp import Llama, LLAMA_POOLING_TYPE_LAST
            except ImportError as e:
                raise EmbeddingInferenceError(
                    "The llama-cpp-python library is required for local embeddings. "
                    "Please run 'pip install llama-cpp-python'."
                ) from e

            log_debug(
                f"Loading local embedding model: {self.model_path} "
                f"(CPU-only, threads={self.n_threads}, pooling=LAST)"
            )

            try:
                embedder = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    pooling_type=LLAMA_POOLING_TYPE_LAST,
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

    def compute_semantic_links(
        self,
        notes: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
        sensitivity_k: float = 2.0,
        quality_floor: float = 0.0,
        cross_chunk_only: bool = True
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
        """
        Computes pure semantic knowledge graph connections across notes in memory without an LLM.
        Uses statistical Z-score outlier detection (dynamic threshold = mean + k * std, k=2.0)
        to discover genuine conceptual resonances above document background noise.

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

        # 5. Filter pairs exceeding the dynamic threshold
        link_pairs: List[Tuple[int, int]] = []
        for src, tgt, score in valid_pairs_with_scores:
            if score >= effective_threshold:
                link_pairs.append((src, tgt))

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
