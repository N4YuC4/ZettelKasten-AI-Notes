# local_gguf_client.py
#
# Client for running embedded local GGUF models via llama-cpp-python.
# Manages in-process model lifecycle, thread allocation, context windows,
# intelligent semantic text chunking for large documents, and JSON schema enforcement.

import os
from env_config import configure_headless_environment

configure_headless_environment()

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
)
from ai_provider import BaseAiProvider


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

    DEFAULT_CONTEXT_WINDOW: int = 131072
    n_ctx: int = DEFAULT_CONTEXT_WINDOW
    n_gpu_layers: int = -1
    _explicit_n_ctx: bool = False

    def __init__(
        self,
        model_path: str,
        n_ctx: Optional[int] = None,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        text_content: Optional[str] = None
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

    def _chunk_text(self, text: str, max_chunk_tokens: int, overlap_tokens: int = 250) -> List[str]:
        """
        Splits text into coherent semantic chunks that each fit within max_chunk_tokens.
        Preserves paragraph and sentence boundaries.
        """
        if not text or self._count_tokens(text) <= max_chunk_tokens:
            return [text] if text else []

        raw_paragraphs = text.split("\n\n")
        paragraphs = []
        for p in raw_paragraphs:
            p = p.strip()
            if not p:
                continue
            p_tokens = self._count_tokens(p)
            if p_tokens > max_chunk_tokens:
                # Sub-split oversized paragraph by sentence endings
                sentences = re.split(r'(?<=[.!?])\s+', p)
                current_sub = ""
                for s in sentences:
                    if self._count_tokens((current_sub + " " + s).strip()) > max_chunk_tokens:
                        if current_sub:
                            paragraphs.append(current_sub.strip())
                        current_sub = s
                    else:
                        current_sub = (current_sub + " " + s).strip()
                if current_sub:
                    paragraphs.append(current_sub.strip())
            else:
                paragraphs.append(p)

        chunks = []
        current_chunk_paragraphs: List[str] = []
        current_chunk_tokens = 0

        for p in paragraphs:
            p_tokens = self._count_tokens(p)
            if current_chunk_paragraphs and (current_chunk_tokens + p_tokens > max_chunk_tokens):
                chunks.append("\n\n".join(current_chunk_paragraphs))

                # Retain overlap paragraphs
                overlap_paras: List[str] = []
                overlap_toks = 0
                for prev_p in reversed(current_chunk_paragraphs):
                    prev_toks = self._count_tokens(prev_p)
                    if overlap_toks + prev_toks <= overlap_tokens:
                        overlap_paras.insert(0, prev_p)
                        overlap_toks += prev_toks
                    else:
                        break

                current_chunk_paragraphs = overlap_paras + [p]
                current_chunk_tokens = overlap_toks + p_tokens
            else:
                current_chunk_paragraphs.append(p)
                current_chunk_tokens += p_tokens

        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        return chunks

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
    ) -> List[Dict[str, Any]]:
        """Executes a single LLM chat completion on chunk_text with optional chained context."""
        user_prompt = build_note_extraction_prompt(
            chunk_text=chunk_text,
            previous_notes_json=previous_notes_json,
            existing_titles=existing_titles,
            unified_general_title=unified_general_title,
            custom_system_prompt=custom_system_prompt,
        )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_EXTRACTION},
            {"role": "user", "content": user_prompt}
        ]

        log_debug(f"Executing local GGUF inference (input length: {len(chunk_text)} chars)...")

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=4096,
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
            log_error(f"Failed to parse notes from local LLM response: {content[:300]}")
            raise LocalLlmError(f"Failed to parse local model output: {content[:200]}")

        return parsed_notes

    def generate_zettelkasten_notes(
        self,
        text_content: str,
        on_progress: Optional[Callable[[str], None]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates Zettelkasten-style atomic notes from text content using the local GGUF model.
        Automatically chunks long documents if token count exceeds prompt token budget.
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

        # Dynamically adapt context window to document demand if not explicitly fixed
        if not getattr(self, "_explicit_n_ctx", False) and os.path.exists(getattr(self, "model_path", "")):
            recommended_ctx = HardwareChecker.calculate_adaptive_context_window(
                text_content=sanitized_text,
                model_path=self.model_path,
                model_max_ctx=self.DEFAULT_CONTEXT_WINDOW
            )
            if recommended_ctx > self.n_ctx:
                log_debug(f"Document demand requires expanding context window: {self.n_ctx} -> {recommended_ctx}")
                self.n_ctx = recommended_ctx
                self.llm = self._get_or_load_model()

        # Dynamic chunk budget: dynamically scales with model's actual context window (e.g. 32k or 128k)
        max_output_tokens = 4096
        system_overhead_tokens = 600
        safety_margin = 250
        previous_json_budget = 2500
        safe_ctx_budget = max(1024, self.n_ctx - max_output_tokens - system_overhead_tokens - safety_margin - previous_json_budget)
        max_prompt_tokens = safe_ctx_budget

        total_tokens = self._count_tokens(sanitized_text)
        log_debug(
            f"Input text: {len(sanitized_text)} chars (~{total_tokens} tokens). "
            f"Model context: {self.n_ctx} tokens (safe prompt budget: {max_prompt_tokens} tokens)."
        )

        # 1. Single pass if within context budget
        if total_tokens <= max_prompt_tokens:
            if on_progress:
                on_progress("AI is extracting notes...")
            return self._execute_inference(sanitized_text, custom_system_prompt=custom_system_prompt)

        # 2. Document exceeds budget -> semantic chunking with Chained JSON Context
        log_debug(
            f"Document size (~{total_tokens} tokens) exceeds single-pass budget ({max_prompt_tokens} tokens). "
            "Splitting document into semantic chunks with chained context..."
        )
        chunks = self._chunk_text(sanitized_text, max_chunk_tokens=max_prompt_tokens, overlap_tokens=250)
        log_debug(f"Document split into {len(chunks)} chunks for sequential chained processing.")

        all_notes: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        all_seen_titles_list: List[str] = []
        unified_general_title: Optional[str] = None
        previous_chunk_json: Optional[str] = None

        for idx, chunk in enumerate(chunks):
            progress_msg = f"AI is extracting notes (Part {idx + 1}/{len(chunks)})..."
            log_debug(f"{progress_msg} ({len(chunk)} chars)")
            if on_progress:
                on_progress(progress_msg)
            try:
                chunk_notes = self._execute_inference(
                    chunk_text=chunk,
                    previous_notes_json=previous_chunk_json,
                    existing_titles=all_seen_titles_list if all_seen_titles_list else None,
                    unified_general_title=unified_general_title,
                    custom_system_prompt=custom_system_prompt,
                )
                new_chunk_notes: List[Dict[str, Any]] = []
                for note in chunk_notes:
                    if not unified_general_title and note.get("general_title"):
                        unified_general_title = note["general_title"].strip()

                    title = note.get("title", "").strip()
                    title_key = title.lower()
                    if title_key and title_key not in seen_titles:
                        seen_titles.add(title_key)
                        all_seen_titles_list.append(title)
                        all_notes.append(note)
                        new_chunk_notes.append(note)
                    elif not title_key:
                        all_notes.append(note)
                        new_chunk_notes.append(note)

                # Prepare previous_chunk_json for the subsequent chunk
                if new_chunk_notes:
                    clean_export = {
                        "general_title": unified_general_title or "",
                        "notes": new_chunk_notes
                    }
                    json_str = json.dumps(clean_export, ensure_ascii=False, indent=2)
                    if self._count_tokens(json_str) > previous_json_budget:
                        # Compact note content if it exceeds token budget
                        compact_notes = [
                            {
                                "title": n.get("title", ""),
                                "content": (n.get("content", "")[:250] + "...") if len(n.get("content", "")) > 250 else n.get("content", ""),
                                "connections": n.get("connections", [])
                            }
                            for n in new_chunk_notes
                        ]
                        json_str = json.dumps({
                            "general_title": unified_general_title or "",
                            "notes": compact_notes
                        }, ensure_ascii=False, indent=2)
                    previous_chunk_json = json_str

            except Exception as ce:
                log_error(f"Error processing chunk {idx + 1}/{len(chunks)}: {ce}")
                if not all_notes and idx == len(chunks) - 1:
                    raise

        # Guarantee that all notes from the same document share the exact same overarching collection
        if unified_general_title:
            for note in all_notes:
                note["general_title"] = unified_general_title

        return all_notes

    def _parse_links_json(self, response_text: str) -> List[Tuple[Any, Any]]:
        """Parses list of (source, target) link pairs via AiResponseParser."""
        return AiResponseParser.parse_links_json(response_text)

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 of the Two-Stage Pipeline: Global Knowledge Graph Linking.
        Takes the complete set of notes generated in Stage 1 with full titles and contents.
        Queries the LLM with the numbered notes map to discover genuine conceptual connections.
        Attaches discovered connections directly to each note's 'connections' list.
        """
        if not notes or len(notes) <= 1:
            return notes

        log_debug(f"Starting Stage 2: Global Knowledge Graph Linking for {len(notes)} notes...")
        if on_progress:
            on_progress("Analyzing conceptual links between notes...")

        # Prepare numbered notes export for prompt input
        full_notes_payload = [
            {
                "id": idx + 1,
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for idx, n in enumerate(notes)
            if n.get("title")
        ]

        # Safety boundary: if an immense document exceeds 24k tokens, compact content slightly
        notes_json_str = json.dumps(full_notes_payload, ensure_ascii=False, indent=2)
        if self._count_tokens(notes_json_str) > 24000:
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

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION_LINKING},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                repeat_penalty=1.15,
                max_tokens=512,
            )
            raw_content = response["choices"][0]["message"].get("content", "") if response and "choices" in response else ""
            log_debug(f"Stage 2 linking response received (len={len(raw_content)} chars): {raw_content[:400]}")
            pairs = self._parse_links_json(raw_content)
            log_debug(f"Discovered {len(pairs)} raw link pairs from global linking pass.")

            return AiResponseParser.attach_links_to_notes(notes, pairs)

        except Exception as e:
            log_error(f"Stage 2 global linking failed (non-fatal, continuing with notes without links): {e}\n{traceback.format_exc()}")
            return notes

        return notes
