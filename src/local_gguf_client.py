# local_gguf_client.py
#
# Client for running embedded local GGUF models via llama-cpp-python.
# Manages in-process model lifecycle, thread allocation, context windows,
# intelligent semantic text chunking for large documents, and JSON schema enforcement.

import os

# Prevent Vulkan loader from injecting desktop presentation layers into headless compute
os.environ["VK_LOADER_LAYERS_DISABLE"] = "*"
os.environ["DISABLE_LAYER_NV_OPTIMUS_1"] = "1"
os.environ["DISABLE_LAYER_NV_PRESENT_1"] = "1"

import re
import gc
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from hardware_checker import HardwareChecker
from logger import log_debug, log_error


class LocalModelNotFoundError(Exception):
    """Raised when the configured local GGUF model file does not exist on disk."""
    pass


class LocalModelOOMError(Exception):
    """Raised when system RAM is insufficient to safely load the GGUF model."""
    pass


class LocalLlmError(Exception):
    """Raised when llama-cpp-python encounters a fatal runtime or inference error."""
    pass


class LocalGgufClient:
    """
    Manages in-process GGUF model lifecycle and generates structured Zettelkasten notes.
    Reuses model instances across calls to prevent expensive reload times.
    Features automatic context expansion and semantic chunking for large PDFs.
    """
    _cached_llm: Optional[Any] = None
    _cached_model_path: Optional[str] = None
    _cached_n_ctx: Optional[int] = None
    _cached_n_gpu_layers: Optional[int] = None

    DEFAULT_CONTEXT_WINDOW: int = 32768
    n_ctx: int = DEFAULT_CONTEXT_WINDOW

    def __init__(
        self,
        model_path: str,
        n_ctx: int = DEFAULT_CONTEXT_WINDOW,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None
    ):
        if not model_path or not os.path.exists(model_path):
            raise LocalModelNotFoundError(
                f"Lokal GGUF model dosyası bulunamadı: '{model_path}'. "
                "Lütfen Ayarlar -> Model Yöneticisi üzerinden modeli indiriniz."
            )

        self.model_path = os.path.abspath(model_path)
        self.n_ctx = n_ctx
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
        """Retrieves cached Llama instance or initializes a new one."""
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
            from llama_cpp import Llama
        except ImportError as e:
            raise LocalLlmError(
                "llama-cpp-python kütüphanesi kurulu değil. "
                "Lütfen 'pip install llama-cpp-python' komutunu çalıştırınız."
            ) from e

        log_debug(
            f"Loading GGUF model into memory: {self.model_path} "
            f"(n_ctx={self.n_ctx}, n_threads={self.n_threads}, n_gpu_layers={self.n_gpu_layers})"
        )

        t0 = time.time()
        try:
            try:
                llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    flash_attn=True,
                    verbose=True
                )
            except (TypeError, ValueError, Exception) as fe:
                log_debug(f"Flash attention not supported or failed ({fe}), falling back to default attention.")
                llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=True
                )
            load_sec = time.time() - t0
            log_debug(f"GGUF model loaded successfully into memory in {load_sec:.2f}s ({self.model_path})")
            LocalGgufClient._cached_llm = llm
            LocalGgufClient._cached_model_path = self.model_path
            LocalGgufClient._cached_n_ctx = self.n_ctx
            LocalGgufClient._cached_n_gpu_layers = self.n_gpu_layers
            return llm
        except Exception as e:
            log_error(f"Failed to load GGUF model: {e}\n{traceback.format_exc()}")
            raise LocalLlmError(f"Model yüklenirken hata oluştu: {e}") from e

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

    def _parse_notes_json(self, response_text: str) -> List[Dict[str, Any]]:
        """Parses and normalizes JSON notes from local LLM response."""
        if not response_text or not isinstance(response_text, str):
            return []

        def normalize_result(data):
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                gen_title = data.get("general_title", "")
                for key in ('notes', 'zettelkasten', 'data', 'items', 'result', 'generated_notes'):
                    if key in data and isinstance(data[key], list):
                        items = [item for item in data[key] if isinstance(item, dict)]
                        if gen_title:
                            for it in items:
                                if not it.get("general_title"):
                                    it["general_title"] = gen_title
                        return items
                if 'title' in data or 'content' in data:
                    return [data]
                dict_values = [v for v in data.values() if isinstance(v, dict)]
                if dict_values:
                    return dict_values
            return []

        cleaned_str = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', response_text)

        # 1. Direct JSON parse
        try:
            parsed = json.loads(cleaned_str, strict=False)
            norm = normalize_result(parsed)
            if norm:
                return norm
        except json.JSONDecodeError:
            pass

        # 2. Markdown code block extraction
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_str)
        for block in code_blocks:
            try:
                parsed = json.loads(block.strip(), strict=False)
                norm = normalize_result(parsed)
                if norm:
                    return norm
            except json.JSONDecodeError:
                pass

        # 3. Bracket extraction
        for start_char in ('[', '{'):
            idx = cleaned_str.find(start_char)
            if idx != -1:
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(cleaned_str[idx:])
                    norm = normalize_result(parsed)
                    if norm:
                        return norm
                except json.JSONDecodeError:
                    pass

        # 4. Salvage completed notes from truncated array if model hit max_tokens
        idx = cleaned_str.find('"notes"')
        if idx != -1:
            gen_match = re.search(r'"general_title"\s*:\s*"([^"]+)"', cleaned_str)
            gen_title = gen_match.group(1).strip() if gen_match else None
            array_start = cleaned_str.find('[', idx)
            if array_start != -1:
                cur = array_start + 1
                decoder = json.JSONDecoder()
                salvaged = []
                while cur < len(cleaned_str):
                    while cur < len(cleaned_str) and cleaned_str[cur] in ' \t\r\n,':
                        cur += 1
                    if cur >= len(cleaned_str) or cleaned_str[cur] == ']':
                        break
                    try:
                        obj, end = decoder.raw_decode(cleaned_str[cur:])
                        if isinstance(obj, dict) and ('title' in obj or 'content' in obj):
                            if gen_title and not obj.get("general_title"):
                                obj["general_title"] = gen_title
                            salvaged.append(obj)
                        cur += end
                    except json.JSONDecodeError:
                        break
                if salvaged:
                    log_debug(f"Successfully salvaged {len(salvaged)} notes from truncated JSON response.")
                    return salvaged

        return []

    def _execute_inference(
        self,
        chunk_text: str,
        previous_notes_json: Optional[str] = None,
        existing_titles: Optional[List[str]] = None,
        unified_general_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Executes a single LLM chat completion on chunk_text with optional chained context."""
        system_instruction = (
            "You are an expert AI specialized in Niklas Luhmann's Zettelkasten note-taking method. "
            "Extract key concepts, distinct arguments, definitions, and insights from the given text. "
            "Create multiple concise, self-contained, and atomic Zettelkasten notes. "
            "Do NOT produce a single summary note. Generate a rich list of individual atomic notes covering all core ideas. "
            "Notes must be in the same language as the input text. "
            "Return a valid JSON object containing a 'general_title' and a 'notes' list."
        )

        chained_context_block = ""
        if previous_notes_json or existing_titles or unified_general_title:
            context_sections = []
            if unified_general_title:
                context_sections.append(f"- ANA DOKÜMAN KONUSU (KATEGORİ): \"{unified_general_title}\"")
            if existing_titles:
                titles_list_str = ", ".join(f'"{t}"' for t in existing_titles)
                context_sections.append(
                    f"- DAHA ÖNCE OLUŞTURULMUŞ TÜM NOT BAŞLIKLARI (TEKRAR ETMEYİNİZ):\n[{titles_list_str}]"
                )
            if previous_notes_json:
                context_sections.append(
                    "ÖNCEKİ BÖLÜMDE ÜRETİLEN ZETTELKASTEN NOTLARI (REFERANS ÇIKTISI):\n"
                    "```json\n"
                    f"{previous_notes_json}\n"
                    "```"
                )

            chained_context_block = (
                "\n=== ÖNCEKİ BÖLÜMLERDEN AKTARILAN BAĞLAM ===\n"
                + "\n\n".join(context_sections)
                + "\n============================================\n\n"
            )

        category_rule = (
            f"1. 'general_title' alanını kesinlikle \"{unified_general_title}\" olarak belirleyin."
            if unified_general_title
            else "1. 'general_title': Belgenin genel ana konusunu belirleyin."
        )

        user_prompt = f"""Extract multiple atomic Zettelkasten notes from the text below and output a JSON object with a 'notes' array.
{chained_context_block}
CRITICAL RULES:
{category_rule}
2. DEDICATED NEW NOTES (NO DUPLICATES): Do NOT recreate, summarize again, or duplicate notes that already exist in previous sections. Only extract new, distinct concepts, definitions, and arguments introduced in this specific section.
3. CROSS-CONNECTIONS (NETWORK BUILDING): If a concept in this section relates to a note from previous sections (listed in the context above), include that exact previous note title in the 'connections' array!
4. CRITICAL RULE FOR CONNECTIONS: The 'connections' list of each note must ONLY contain the EXACT 'title' of other notes (either from previous sections or from this section). Do NOT use shortened or generalized topic names.

Example structure:
{{
  "general_title": "{unified_general_title or 'Document Main Topic'}",
  "notes": [
    {{
      "title": "Specific Concept Name",
      "content": "A self-contained, clear explanation of this specific atomic idea.",
      "connections": ["Related Concept Name"]
    }},
    {{
      "title": "Related Concept Name",
      "content": "A self-contained explanation of this related idea.",
      "connections": ["Specific Concept Name"]
    }}
  ]
}}

Text to process:
<document_content>
{chunk_text}
</document_content>
"""

        messages = [
            {"role": "system", "content": system_instruction},
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
            raise LocalLlmError(f"Yerel model çıkarım hatası: {e}") from e

        if not response or "choices" not in response or not response["choices"]:
            return []

        content = response["choices"][0]["message"].get("content", "")
        log_debug(f"Local LLM response received (len={len(content)} chars)")

        parsed_notes = self._parse_notes_json(content)
        if not parsed_notes and content.strip():
            log_error(f"Failed to parse notes from local LLM response: {content[:300]}")
            raise LocalLlmError(f"Yerel model çıktısı ayrıştırılamadı: {content[:200]}")

        return parsed_notes

    def generate_zettelkasten_notes(
        self,
        text_content: str,
        on_progress: Optional[Callable[[str], None]] = None
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

        # Dynamic chunk budget: for 32768 context, allow up to 24000 tokens (~75,000+ chars) in a single pass.
        # This prevents splitting medium-to-large PDFs (40-60 pages) into unnecessary separate chunks.
        TARGET_MAX_CHUNK_TOKENS = 24000
        max_output_tokens = 4096
        system_overhead_tokens = 600
        safety_margin = 250
        previous_json_budget = 2500
        safe_ctx_budget = max(1024, self.n_ctx - max_output_tokens - system_overhead_tokens - safety_margin - previous_json_budget)
        max_prompt_tokens = min(TARGET_MAX_CHUNK_TOKENS, safe_ctx_budget)

        total_tokens = self._count_tokens(sanitized_text)
        log_debug(
            f"Input text: {len(sanitized_text)} chars (~{total_tokens} tokens). "
            f"Model context: {self.n_ctx} tokens (safe prompt budget: {max_prompt_tokens} tokens)."
        )

        # 1. Single pass if within context budget
        if total_tokens <= max_prompt_tokens:
            if on_progress:
                on_progress("Yapay zeka notları çıkarıyor...")
            return self._execute_inference(sanitized_text)

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
            progress_msg = f"Yapay zeka notları çıkarıyor (Bölüm {idx + 1}/{len(chunks)})..."
            log_debug(f"{progress_msg} ({len(chunk)} chars)")
            if on_progress:
                on_progress(progress_msg)
            try:
                chunk_notes = self._execute_inference(
                    chunk_text=chunk,
                    previous_notes_json=previous_chunk_json,
                    existing_titles=all_seen_titles_list if all_seen_titles_list else None,
                    unified_general_title=unified_general_title
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

        # Guarantee that all notes from the same document share the exact same overarching category
        if unified_general_title:
            for note in all_notes:
                note["general_title"] = unified_general_title

        return all_notes

    def _parse_links_json(self, response_text: str) -> List[Tuple[str, str]]:
        """Parses list of (source, target) title pairs from LLM response."""
        if not response_text or not isinstance(response_text, str):
            return []

        cleaned_str = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', response_text)

        def extract_pairs(data: Any) -> List[Tuple[str, str]]:
            pairs: List[Tuple[str, str]] = []
            items: List[Any] = []
            if isinstance(data, dict):
                for k in ("links", "connections", "relations", "graph", "edges", "result"):
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break
                if not items:
                    for v in data.values():
                        if isinstance(v, list):
                            items = v
                            break
            elif isinstance(data, list):
                items = data

            for item in items:
                if isinstance(item, dict):
                    src = item.get("source") or item.get("from") or item.get("source_title") or item.get("note_a")
                    tgt = item.get("target") or item.get("to") or item.get("target_title") or item.get("note_b")
                    if src and tgt and isinstance(src, str) and isinstance(tgt, str):
                        src_clean = src.strip()
                        tgt_clean = tgt.strip()
                        if src_clean and tgt_clean and src_clean.lower() != tgt_clean.lower():
                            pairs.append((src_clean, tgt_clean))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    src, tgt = str(item[0]).strip(), str(item[1]).strip()
                    if src and tgt and src.lower() != tgt.lower():
                        pairs.append((src, tgt))
            return pairs

        # 1. Direct JSON parse
        try:
            parsed = json.loads(cleaned_str, strict=False)
            res = extract_pairs(parsed)
            if res:
                return res
        except Exception:
            pass

        # 2. Markdown code block
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_str)
        for block in code_blocks:
            try:
                parsed = json.loads(block.strip(), strict=False)
                res = extract_pairs(parsed)
                if res:
                    return res
            except Exception:
                pass

        # 3. Bracket extraction
        for start_char in ('{', '['):
            idx = cleaned_str.find(start_char)
            if idx != -1:
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(cleaned_str[idx:])
                    res = extract_pairs(parsed)
                    if res:
                        return res
                except Exception:
                    pass

        return []

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 of the Two-Stage Pipeline: Global Knowledge Graph Linking.
        Takes the complete set of notes generated in Stage 1 with full titles and contents.
        Queries the LLM with the entire notes map to discover genuine conceptual connections.
        Attaches discovered connections directly to each note's 'connections' list.
        """
        if not notes or len(notes) <= 1:
            return notes

        log_debug(f"Starting Stage 2: Global Knowledge Graph Linking for {len(notes)} notes...")
        if on_progress:
            on_progress("Notlar arası kavramsal bağlantılar çözümleniyor...")

        # Build title lookup maps for case-insensitive matching
        title_map: Dict[str, str] = {}
        note_by_title: Dict[str, Dict[str, Any]] = {}
        for n in notes:
            t = n.get("title", "").strip()
            if t:
                title_map[t.lower()] = t
                note_by_title[t] = n
                if "connections" not in n or not isinstance(n["connections"], list):
                    n["connections"] = []

        # Prepare full notes export for prompt input (all titles and complete note contents)
        full_notes_payload = [
            {
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for n in notes
            if n.get("title")
        ]

        notes_json_str = json.dumps(full_notes_payload, ensure_ascii=False, indent=2)

        # Safety boundary: if an immense document exceeds 24k tokens, compact content slightly
        if self._count_tokens(notes_json_str) > 24000:
            compact_payload = [
                {
                    "title": n.get("title", ""),
                    "content": (n.get("content", "")[:300] + "...") if len(n.get("content", "")) > 300 else n.get("content", "")
                }
                for n in notes
                if n.get("title")
            ]
            notes_json_str = json.dumps(compact_payload, ensure_ascii=False, indent=2)

        system_instruction = (
            "You are an expert AI specialized in Niklas Luhmann's Zettelkasten method and knowledge graphs. "
            "Your task is to analyze a completed set of atomic Zettelkasten notes from a document and discover "
            "genuine, meaningful conceptual links between them (such as prerequisite, cause-effect, contrast, or conceptual complement). "
            "Do NOT make forced or superficial connections. Return a valid JSON object with a 'links' array."
        )

        user_prompt = f"""Aşağıda bir belgeden yeni çıkarılmış tüm Zettelkasten notları başlıkları ve tam içerikleriyle yer almaktadır:

<all_notes>
{notes_json_str}
</all_notes>

GÖREV:
Yukarıdaki notların tamamını ve açıklamalarını bütüncül olarak analiz ediniz.
Aralarında doğrudan önkoşul, kavramsal tamamlayıcılık, sebep-sonuç veya mantıksal devamlılık bulunan kartları eşleştiriniz.

KURALLAR:
1. 'source' ve 'target' alanları KESİNLİKLE yukarıdaki listede yer alan 'title' değerleriyle BİREBİR AYNI olmalıdır.
2. ZORLAMA BAĞLANTI KURMAYINIZ: Sadece aralarında gerçek bir kavramsal bağ bulunan kartları eşleştiriniz. Bağımsız tanımlar veya öncüller bağlantısız kalabilir.
3. Kendi kendine bağlantı (source == target) kurmayınız.
4. Çift yönlü tekrardan kaçınınız (A -> B bağlandıysa ayrıca B -> A yazmayınız).

Çıktı JSON formatı:
{{
  "links": [
    {{"source": "Tam Not Başlığı A", "target": "Tam Not Başlığı B"}},
    {{"source": "Tam Not Başlığı C", "target": "Tam Not Başlığı D"}}
  ]
}}
"""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2048,
            )
            raw_content = response["choices"][0]["message"].get("content", "") if response and "choices" in response else ""
            log_debug(f"Stage 2 linking response received (len={len(raw_content)} chars)")
            pairs = self._parse_links_json(raw_content)
            log_debug(f"Discovered {len(pairs)} raw link pairs from global linking pass.")

            added_count = 0
            for src_raw, tgt_raw in pairs:
                src_key = src_raw.lower()
                tgt_key = tgt_raw.lower()
                if src_key in title_map and tgt_key in title_map and src_key != tgt_key:
                    canonical_src = title_map[src_key]
                    canonical_tgt = title_map[tgt_key]
                    src_note = note_by_title[canonical_src]
                    tgt_note = note_by_title[canonical_tgt]

                    if canonical_tgt not in src_note["connections"]:
                        src_note["connections"].append(canonical_tgt)
                        added_count += 1
                    if canonical_src not in tgt_note["connections"]:
                        tgt_note["connections"].append(canonical_src)

            log_debug(f"Successfully attached {added_count} graph connections to notes.")

        except Exception as e:
            log_error(f"Stage 2 global linking failed (non-fatal, continuing with notes without links): {e}\n{traceback.format_exc()}")

        return notes
