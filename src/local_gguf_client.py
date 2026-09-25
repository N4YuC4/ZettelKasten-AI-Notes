# local_gguf_client.py
#
# Client for running embedded local GGUF models via llama-cpp-python.
# Manages in-process model lifecycle, thread allocation, context windows,
# intelligent semantic text chunking for large documents, and JSON schema enforcement.

import os
from env_config import configure_headless_environment

configure_headless_environment()

import math
import re
import gc
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from hardware_checker import HardwareChecker
from logger import log_debug, log_error
from ai_response_parser import AiResponseParser
from prompt_templates import (
    SYSTEM_INSTRUCTION_EXTRACTION,
    SYSTEM_INSTRUCTION_LINKING,
    build_note_extraction_prompt,
    build_graph_linking_prompt,
    build_candidate_pairs_verification_prompt,
    partition_candidate_pairs,
)
from ai_provider import BaseAiProvider
import semantic_chunker


class LocalModelNotFoundError(Exception):
    """Raised when the configured local GGUF model file does not exist on disk."""
    pass


class LocalModelOOMError(Exception):
    """Raised when system RAM is insufficient to safely load the GGUF model."""
    pass


class LocalLlmError(Exception):
    """Raised when llama-cpp-python encounters a fatal runtime or inference error."""
    pass


class LocalGgufClient(BaseAiProvider):
    """
    Manages in-process GGUF model lifecycle and generates structured Zettelkasten notes.
    Reuses model instances across calls to prevent expensive reload times.
    Features automatic context expansion and semantic chunking for large PDFs.
    """
    _cached_llm: Optional[Any] = None
    _cached_model_path: Optional[str] = None
    _cached_n_ctx: Optional[int] = None
    _cached_n_gpu_layers: Optional[int] = None

    DEFAULT_CONTEXT_WINDOW: int = 16384
    n_ctx: int = DEFAULT_CONTEXT_WINDOW
    n_gpu_layers: int = -1
    _explicit_n_ctx: bool = False

    def __init__(
        self,
        model_path: str,
        n_ctx: Optional[int] = None,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        text_content: Optional[str] = None,
        use_speculative: bool = False
    ):
        if not model_path or not os.path.exists(model_path):
            raise LocalModelNotFoundError(
                f"Local GGUF model file not found: '{model_path}'. "
                "Please download the model via Settings -> Model Manager."
            )

        self.model_path = os.path.abspath(model_path)
        self._explicit_n_ctx = (n_ctx is not None)
        if n_ctx is not None:
            self.n_ctx = n_ctx
        else:
            self.n_ctx = HardwareChecker.calculate_adaptive_context_window(
                text_content=text_content or "",
                model_path=self.model_path,
                model_max_ctx=self.DEFAULT_CONTEXT_WINDOW
            )
        self.n_gpu_layers = n_gpu_layers

        # Determine thread allocation
        cpu_info = HardwareChecker.get_cpu_info()
        self.n_threads = n_threads or cpu_info["optimal_threads"]

        # Validate hardware bounds before loading
        is_safe, msg, status = HardwareChecker.check_file_compatibility(self.model_path)
        if not is_safe:
            raise LocalModelOOMError(msg)

        self.llm = self._get_or_load_model()

    def _get_or_load_model(self) -> Any:
        """Retrieves cached Llama instance or initializes a new one with Q8_0 KV cache and 16K fallback."""
        if (
            LocalGgufClient._cached_llm is not None
            and LocalGgufClient._cached_model_path == self.model_path
            and LocalGgufClient._cached_n_ctx == self.n_ctx
            and LocalGgufClient._cached_n_gpu_layers == self.n_gpu_layers
        ):
            log_debug(
                f"Reusing cached GGUF model: {self.model_path} "
                f"(n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers})"
            )
            return LocalGgufClient._cached_llm

        # Configure headless Vulkan environment and route to discrete GPU
        HardwareChecker.configure_vulkan_environment(enable_gpu=(self.n_gpu_layers != 0))

        try:
            import llama_cpp
            from llama_cpp import Llama
            q8_type = getattr(llama_cpp, "GGML_TYPE_Q8_0", None)
        except ImportError as e:
            raise LocalLlmError(
                "The llama-cpp-python library is not installed. "
                "Please run 'pip install llama-cpp-python'."
            ) from e

        # Build 16K granular context fallback ladder
        target_ctx = self.n_ctx
        ctx_ladder: List[int] = [target_ctx]
        step = 16384
        curr = target_ctx - step
        while curr >= 8192:
            ctx_ladder.append(curr)
            curr -= step
        if 8192 not in ctx_ladder:
            ctx_ladder.append(8192)

        last_error = None
        for try_ctx in ctx_ladder:
            log_debug(
                f"Attempting to load GGUF model: {self.model_path} "
                f"(n_ctx={try_ctx}, n_threads={self.n_threads}, n_gpu_layers={self.n_gpu_layers}, kv_cache=Q8_0)"
            )
            t0 = time.time()

            # Attempt configurations:
            # 1. Q8_0 KV cache + Flash Attention (fastest & minimal VRAM)
            # 2. Q8_0 KV cache + Default Attention (fallback if flash-attn unsupported)
            # 3. Default KV + Default Attention (legacy CPU/driver fallback)
            attempts: List[Dict[str, Any]] = []
            if q8_type is not None:
                attempts.append({"type_k": q8_type, "type_v": q8_type, "flash_attn": True})
                attempts.append({"type_k": q8_type, "type_v": q8_type, "flash_attn": False})
            attempts.append({"flash_attn": False})

            for kwargs in attempts:
                try:
                    llm = Llama(
                        model_path=self.model_path,
                        n_ctx=try_ctx,
                        n_threads=self.n_threads,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=True,
                        **kwargs
                    )
                    load_sec = time.time() - t0
                    log_debug(
                        f"GGUF model loaded successfully into memory in {load_sec:.2f}s "
                        f"(n_ctx={try_ctx}, kwargs={kwargs}, path={self.model_path})"
                    )
                    self.n_ctx = try_ctx
                    LocalGgufClient._cached_llm = llm
                    LocalGgufClient._cached_model_path = self.model_path
                    LocalGgufClient._cached_n_ctx = try_ctx
                    LocalGgufClient._cached_n_gpu_layers = self.n_gpu_layers
                    return llm
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if any(m in err_str for m in ["failed to create context", "out of memory", "cannot allocate", "cuda", "vulkan memory"]):
                        log_debug(f"Memory allocation failed at n_ctx={try_ctx}: {e}. Stepping down context...")
                        break
                    else:
                        continue

        log_error(f"Failed to load GGUF model across all fallback context levels: {last_error}\n{traceback.format_exc()}")
        raise LocalModelOOMError(
            f"Failed to allocate memory for model (insufficient memory even at lowest context level): {last_error}"
        ) from last_error

    @classmethod
    def unload_cached_model(cls) -> None:
        """
        Explicitly frees the cached GGUF model and its context/KV cache buffers
        from RAM/VRAM and forces immediate garbage collection.
        """
        if cls._cached_llm is not None:
            path_unloaded = cls._cached_model_path
            log_debug(f"Unloading cached GGUF model from memory: {path_unloaded}")
            try:
                if hasattr(cls._cached_llm, "close"):
                    cls._cached_llm.close()
            except Exception as e:
                log_error(f"Error while closing GGUF model: {e}")
            finally:
                cls._cached_llm = None
                cls._cached_model_path = None
                cls._cached_n_ctx = None
                cls._cached_n_gpu_layers = None
                gc.collect()
                log_debug(f"Cached GGUF model memory freed successfully ({path_unloaded}).")

    @classmethod
    def unload_model(cls) -> None:
        """Alias for unload_cached_model."""
        cls.unload_cached_model()

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Returns True if a model is currently resident in memory."""
        return cls._cached_llm is not None

    def unload(self) -> None:
        """Instance helper to unload the active model."""
        LocalGgufClient.unload_cached_model()
        self.llm = None

    def _count_tokens(self, text: str) -> int:
        """Estimates or counts tokens accurately using the model's tokenizer or fallback."""
        if not text:
            return 0
        try:
            if hasattr(self, "llm") and self.llm is not None and hasattr(self.llm, "tokenize"):
                return len(self.llm.tokenize(text.encode("utf-8", errors="ignore")))
        except Exception:
            pass
        # Conservative estimate for Turkish and multilingual prose (~3.0 chars per token)
        return max(1, int(len(text) / 3.0) + 1)

    def _chunk_text(
        self,
        text: str,
        max_chunk_tokens: int = semantic_chunker.DEFAULT_EXTRACTION_CHUNK_TOKENS,
        overlap_tokens: int = semantic_chunker.DEFAULT_OVERLAP_TOKENS
    ) -> List[str]:
        """
        Universal dynamic chunker: splits text into balanced, coherent semantic chunks.
        Delegates to shared semantic_chunker.chunk_text using model tokenizer for accurate counts.
        """
        return semantic_chunker.chunk_text(
            text=text,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens_fn=self._count_tokens
        )

    @staticmethod
    def _clean_connections(raw_connections: Any, current_title: str = "") -> List[str]:
        """Cleans and sanitizes note connections via AiResponseParser."""
        return AiResponseParser.clean_connections(raw_connections, current_title)

    def _parse_notes_json(self, response_text: str) -> List[Dict[str, Any]]:
        """Parses, normalizes, and sanitizes JSON notes via AiResponseParser."""
        return AiResponseParser.parse_notes_json(response_text)

    def _execute_inference(
        self,
        chunk_text: str,
        previous_notes_json: Optional[str] = None,
        existing_titles: Optional[List[str]] = None,
        unified_general_title: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        rag_notes: Optional[List[Dict[str, Any]]] = None,
        global_concept_map: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes a single LLM chat completion on chunk_text with optional Two-Tier RAG context."""
        user_prompt = build_note_extraction_prompt(
            chunk_text=chunk_text,
            previous_notes_json=previous_notes_json,
            existing_titles=existing_titles,
            unified_general_title=unified_general_title,
            custom_system_prompt=custom_system_prompt,
            rag_notes=rag_notes,
            global_concept_map=global_concept_map,
        )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_EXTRACTION},
            {"role": "user", "content": user_prompt}
        ]

        # Estimate prompt tokens and dynamically calculate safe max_tokens so we do not exceed n_ctx
        prompt_tokens = self._count_tokens(user_prompt) + 300
        remaining_ctx = max(512, self.n_ctx - prompt_tokens - 100)
        output_cap = 2048 if self.n_ctx <= 8192 else 4096
        safe_max_tokens = min(output_cap, remaining_ctx)

        log_debug(f"Executing local GGUF inference (input length: {len(chunk_text)} chars, max_tokens: {safe_max_tokens})...")

        try:
            # Omit response_format={"type": "json_object"} to eliminate CPU BNF grammar parsing
            # which degrades GPU generation speed by ~85%. AiResponseParser handles JSON recovery.
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                repeat_penalty=1.1,
                top_p=0.95,
                max_tokens=safe_max_tokens,
            )
        except Exception as e:
            log_error(f"Local LLM inference failed: {e}\n{traceback.format_exc()}")
            raise LocalLlmError(f"Local model inference error: {e}") from e

        if not response or "choices" not in response or not response["choices"]:
            return []

        content = response["choices"][0]["message"].get("content", "")
        log_debug(f"Local LLM response received (len={len(content)} chars)")

        parsed_notes = self._parse_notes_json(content)
        if not parsed_notes and content.strip():
            if AiResponseParser.is_valid_empty_notes_response(content):
                log_debug("Local LLM explicitly returned valid empty notes array (no new concepts in chunk).")
                return []
            log_error(f"Failed to parse notes from local LLM response: {content[:300]}")
            raise LocalLlmError(f"Failed to parse local model output: {content[:200]}")

        return parsed_notes

    def generate_zettelkasten_notes(
        self,
        text_content: str,
        on_progress: Optional[Callable[[str], None]] = None,
        custom_system_prompt: Optional[str] = None,
        semantic_memory_service: Optional[Any] = None,
        reranker_service: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates Zettelkasten-style atomic notes from text content using the local GGUF model.
        Uses an in-memory vector RAG pool (NoteRagPool) with Two-Tier retrieval (Global Concept Map + Reranked Focal Notes).
        Strictly calibrated and compatible with 8k (8192) context windows.
        Returns:
            list of dicts containing 'general_title', 'title', 'content', 'connections'
        """
        if not text_content or not str(text_content).strip():
            return []

        sanitized_text = (
            str(text_content)
            .replace("<document_content>", "")
            .replace("</document_content>", "")
            .strip()
        )
        if not sanitized_text:
            return []

        # Enforce mandatory embedding service & reranker service & initialize RAG pool
        from note_rag_pool import NoteRagPool
        from semantic_memory_service import SemanticMemoryService
        from reranker_service import RerankerService

        memory_service = semantic_memory_service or SemanticMemoryService()
        rank_service = reranker_service or RerankerService()
        rag_pool = NoteRagPool(
            semantic_memory_service=memory_service,
            reranker_service=rank_service,
            count_tokens_fn=self._count_tokens,
            ai_provider=self
        )

        # Dynamic context and token budget: strictly fixed and calibrated for 16K (16,384) context windows
        # Exact 16K allocation:
        # 6000 (source chunk) + 5200 (Two-Tier RAG) + 4096 (output generation) + 688 (system instructions) + 400 (safety area) = 16,384 tokens
        is_small_ctx = (self.n_ctx <= 8192)
        max_output_tokens = 2048 if is_small_ctx else 4096
        system_overhead_tokens = 500 if is_small_ctx else 688
        safety_margin = 250 if is_small_ctx else 400
        rag_context_budget = 2400 if is_small_ctx else 5200

        safe_ctx_budget = max(
            1024,
            self.n_ctx - max_output_tokens - system_overhead_tokens - safety_margin - rag_context_budget
        )
        max_prompt_tokens = min(safe_ctx_budget, 2800 if is_small_ctx else semantic_chunker.DEFAULT_EXTRACTION_CHUNK_TOKENS)

        # Single pass budget (when no RAG reference context is required):
        single_pass_budget = max(
            max_prompt_tokens,
            self.n_ctx - max_output_tokens - system_overhead_tokens - safety_margin
        )

        total_tokens = self._count_tokens(sanitized_text)
        log_debug(
            f"Input text: {len(sanitized_text)} chars (~{total_tokens} tokens). "
            f"Model context: {self.n_ctx} tokens (single-pass budget: {single_pass_budget}, chunk budget: {max_prompt_tokens} tokens, RAG budget: {rag_context_budget})."
        )

        # 1. Single pass if within single-pass budget
        if total_tokens <= single_pass_budget:
            if on_progress:
                on_progress("AI is extracting notes...")
            return self._execute_inference(sanitized_text, custom_system_prompt=custom_system_prompt)

        # 2. Document exceeds budget -> semantic chunking with vector RAG pool
        log_debug(
            f"Document size (~{total_tokens} tokens) exceeds single-pass budget ({single_pass_budget} tokens). "
            f"Splitting document into semantic chunks ({max_prompt_tokens} tokens max) with vector RAG pool ({rag_context_budget} tokens)..."
        )
        chunks = self._chunk_text(
            sanitized_text,
            max_chunk_tokens=max_prompt_tokens,
            overlap_tokens=semantic_chunker.DEFAULT_OVERLAP_TOKENS
        )
        log_debug(f"Document split into {len(chunks)} chunks for sequential RAG processing.")

        unified_general_title: Optional[str] = None

        for idx, chunk in enumerate(chunks):
            progress_msg = f"AI is extracting notes (Part {idx + 1}/{len(chunks)})..."
            log_debug(f"{progress_msg} ({len(chunk)} chars)")
            if on_progress:
                on_progress(progress_msg)
            try:
                # Retrieve two-tier context: Tier 1 global concept map + Tier 2 reranked focal notes
                tier1_concepts, tier2_focal_notes = rag_pool.retrieve_two_tier_context(
                    query_chunk=chunk,
                    total_budget=rag_context_budget,
                    top_k_focal=4 if is_small_ctx else 8
                )
                chunk_notes = self._execute_inference(
                    chunk_text=chunk,
                    rag_notes=tier2_focal_notes if tier2_focal_notes else None,
                    global_concept_map=tier1_concepts if tier1_concepts else None,
                    existing_titles=rag_pool.get_all_titles() if len(rag_pool) > 0 and not tier1_concepts else None,
                    unified_general_title=unified_general_title,
                    custom_system_prompt=custom_system_prompt,
                )
                new_chunk_notes: List[Dict[str, Any]] = []
                for note in chunk_notes:
                    if not unified_general_title and note.get("general_title"):
                        unified_general_title = note["general_title"].strip()

                    note["_chunk_id"] = idx
                    new_chunk_notes.append(note)

                # Collect new notes into RAG pool (indexes embeddings immediately)
                if new_chunk_notes:
                    rag_pool.add_notes(new_chunk_notes)

            except Exception as ce:
                log_error(f"Error processing chunk {idx + 1}/{len(chunks)}: {ce}")
                if len(rag_pool) == 0 and idx == len(chunks) - 1:
                    raise

        # Retrieve all accumulated notes from RAG pool
        all_notes = rag_pool.get_all_notes()
        for idx, note in enumerate(all_notes):
            note["id"] = idx + 1
            if unified_general_title:
                note["general_title"] = unified_general_title

        return all_notes

    def _parse_links_json(self, response_text: str) -> List[Tuple[Any, Any]]:
        """Parses list of (source, target) link pairs via AiResponseParser."""
        return AiResponseParser.parse_links_json(response_text)

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None,
        similarity_threshold: Optional[float] = None,
        semantic_memory_service: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 of the Two-Stage Pipeline: Global Knowledge Graph Linking.
        When SemanticMemoryService (Microsoft Harrier 0.6B) is available:
        Performs Pure Semantic Vector Linking using statistical Z-score dynamic thresholding
        (threshold = max(quality_floor, mean + 1.8 * std)), establishing genuine knowledge graph
        connections in milliseconds with 0 LLM calls, 0 prompt tokens, and 0 context overflow risk.
        Falls back gracefully to legacy direct LLM prompt if the embedding model is not installed.
        """
        if not notes or len(notes) <= 1:
            return notes

        log_debug(f"Starting Stage 2: Global Knowledge Graph Linking for {len(notes)} notes...")
        if on_progress:
            on_progress("Analyzing conceptual links between notes...")

        # 1. Attempt Pure Semantic Linking via SemanticMemoryService
        memory_service = semantic_memory_service
        if memory_service is None:
            try:
                from semantic_memory_service import SemanticMemoryService
                memory_service = SemanticMemoryService()
            except Exception as e:
                log_debug(f"SemanticMemoryService init error (falling back to direct prompt): {e}")
                memory_service = None

        if memory_service and memory_service.is_model_available():
            try:
                # Ensure notes are consolidated & deduplicated before computing knowledge graph links
                if hasattr(memory_service, "consolidate_and_deduplicate_notes"):
                    deduped = memory_service.consolidate_and_deduplicate_notes(notes, ai_provider=self)
                    if isinstance(deduped, list) and (not deduped or isinstance(deduped[0], dict)):
                        notes = deduped
                if on_progress:
                    on_progress("Computing semantic connections with Harrier embedding...")
                link_pairs, embeddings, eff_threshold = memory_service.compute_semantic_links(
                    notes, similarity_threshold=similarity_threshold, cross_chunk_only=True
                )
                # Store embeddings on notes for subsequent persistence
                if embeddings is not None and len(embeddings) == len(notes):
                    for idx, note in enumerate(notes):
                        note["_embedding"] = embeddings[idx]

                log_debug(
                    f"Stage 2 pure semantic linking discovered {len(link_pairs)} connections "
                    f"(effective threshold: {eff_threshold:.4f}). Attaching links..."
                )
                return AiResponseParser.attach_links_to_notes(notes, link_pairs)
            except Exception as e:
                log_error(f"Pure semantic linking error (falling back to direct LLM prompt): {e}")

        # Fallback legacy prompt (if embedding model is not yet installed or failed)
        full_notes_payload = [
            {
                "id": idx + 1,
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for idx, n in enumerate(notes)
            if n.get("title")
        ]

        notes_json_str = json.dumps(full_notes_payload, ensure_ascii=False, indent=2)
        safe_prompt_budget = max(2048, self.n_ctx - 4096 - 500)
        if self._count_tokens(notes_json_str) > safe_prompt_budget:
            full_notes_payload = [
                {
                    "id": idx + 1,
                    "title": n.get("title", ""),
                    "content": (n.get("content", "")[:300] + "...") if len(n.get("content", "")) > 300 else n.get("content", "")
                }
                for idx, n in enumerate(notes)
                if n.get("title")
            ]

        user_prompt = build_graph_linking_prompt(full_notes_payload)
        prompt_tokens = self._count_tokens(user_prompt) + 200
        remaining_ctx = max(4096, self.n_ctx - prompt_tokens - 100)
        safe_max_tokens = min(8192, remaining_ctx)

        log_debug(
            f"Stage 2 linking budget (legacy): n_ctx={self.n_ctx}, prompt_tokens={prompt_tokens}, "
            f"remaining_ctx={remaining_ctx}, safe_max_tokens={safe_max_tokens}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_LINKING},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                repeat_penalty=1.05,
                max_tokens=safe_max_tokens,
            )
            raw_content = response["choices"][0]["message"].get("content", "") if response and "choices" in response else ""
            log_debug(f"Stage 2 legacy linking response received (len={len(raw_content)} chars): {raw_content[:400]}")
            pairs = self._parse_links_json(raw_content)
            log_debug(f"Discovered {len(pairs)} raw link pairs from global linking pass.")

            return AiResponseParser.attach_links_to_notes(notes, pairs)

        except Exception as e:
            log_error(f"Stage 2 legacy linking failed (non-fatal, continuing with notes without links): {e}\n{traceback.format_exc()}")
            return notes

        return notes

    def synthesize_note_cluster(
        self,
        cluster_notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """Synthesizes an N-way duplicate cluster into a single cohesive note via local GGUF model."""
        if not cluster_notes or len(cluster_notes) < 2:
            return cluster_notes[0] if cluster_notes else None

        from prompt_templates import SYSTEM_INSTRUCTION_SYNTHESIS, build_note_synthesis_prompt

        prompt = build_note_synthesis_prompt(cluster_notes)
        log_debug(f"LocalGgufClient: Executing N-way synthesis for {len(cluster_notes)} notes...")
        if on_progress:
            on_progress(f"Synthesizing {len(cluster_notes)} overlapping notes into unified concept...")

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_SYNTHESIS},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                repeat_penalty=1.1,
                top_p=0.95,
                max_tokens=4096,
            )
            if response and "choices" in response and response["choices"]:
                raw_content = response["choices"][0]["message"].get("content", "")
                if raw_content:
                    parsed = AiResponseParser.parse_synthesized_note(raw_content)
                    if parsed:
                        log_debug(f"LocalGgufClient: Successfully synthesized cluster into '{parsed['title']}'")
                        return parsed
        except Exception as e:
            log_error(f"LocalGgufClient: Failed to synthesize note cluster: {e}\n{traceback.format_exc()}")

        return None
