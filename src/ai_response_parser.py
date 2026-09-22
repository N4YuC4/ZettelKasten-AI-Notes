# ai_response_parser.py
#
# Unified response parser for LLM outputs (both Google Gemini and local GGUF models).
# Normalizes JSON schemas, cleans markdown code blocks, repairs malformed escapes,
# and sanitizes knowledge graph connection links across all document domains.

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from logger import log_debug, log_error


class AiResponseParser:
    """
    Robust JSON parser and sanitizer for LLM-generated Zettelkasten notes and graph links.
    Handles Markdown code blocks, malformed escapes, partial truncations, and schema normalization.
    """

    # Universal structural and navigational patterns to reject across all document domains
    FORBIDDEN_PATTERNS = [
        # Numeric & author-year citation brackets: [1], [38], [1, 2], [Smith et al., 2021]
        r"^\[\s*[\w\s\.,&]+(?:\s*,\s*\d{4})?\s*\]$",
        # Academic & Scientific structural markers
        r"^(?:figure|fig\.|table|tab\.|equation|eq\.|proposition|theorem|lemma|corollary|definition|appendix|footnote)\b",
        # Turkish academic / scientific markers
        r"^(?:şekil|sekil|tablo|denklem|önerme|onerme|teorem|aksiyom|tanım|tanim|dipnot|ek)\b",
        # Legal / Regulatory markers
        r"^(?:madde|fıkra|fikra|bent|paragraf|hüküm|hukum|article|clause|section|sec\.|paragraph|para\.|item)\b",
        # Book / Literature / Report structural markers
        r"^(?:chapter|ch\.|bölüm|bolum|kısım|kisim|part|page|p\.|sayfa|s\.)\b",
        # Transcript / Meeting markers & timestamps
        r"^(?:speaker|konuşmacı|konusmaci)\b",
        r"^\d{1,2}:\d{2}(?::\d{2})?$",
    ]
    _COMPILED_FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]

    @staticmethod
    def strip_qualifiers(title: str) -> str:
        """Strips parenthetical qualifiers, brackets, leading articles, and currency signs for canonical title comparison."""
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", title.strip(), flags=re.IGNORECASE)
        return re.sub(r"\s*[\(\[].*?[\)\]]", "", cleaned).replace("$", "").strip().casefold()

    @staticmethod
    def normalize_tokens(text: str) -> List[str]:
        """Extracts canonical word stems from text, stripping leading articles."""
        s = re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)
        words = re.findall(r"\b\w+\b", s.casefold())
        stemmed = []
        for w in words:
            if len(w) > 3 and w.endswith("ies"):
                stemmed.append(w[:-3] + "y")
            elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                stemmed.append(w[:-1])
            else:
                stemmed.append(w)
        return stemmed


    @classmethod
    def clean_connections(cls, raw_connections: Any, current_title: str = "") -> List[str]:
        """
        Cleans and sanitizes note connections.
        Filters out citations, structural coordinates, self-references, and duplicates.
        """
        if not raw_connections or not isinstance(raw_connections, list):
            return []

        cleaned: List[str] = []
        seen = set()
        title_lower = current_title.strip().lower()

        for item in raw_connections:
            if not isinstance(item, str):
                continue
            item_raw = item.strip()
            if not item_raw:
                continue

            # 1. Bracketed citations [1], [38], [Smith et al., 2021]
            if re.match(r"^\[\s*[\w\s\.,&]+(?:\s*,\s*\d{4})?\s*\]$", item_raw):
                continue

            text = item_raw.strip('"\'[]').strip()
            if not text or len(text) < 3:
                continue

            # 2. Bare numbers or coordinate digits (e.g. '1', '38', '4.1')
            if text.isdigit() or re.match(r"^\d+(?:[\.\-_]\d+)*$", text):
                continue

            # 3. Academic author citations (e.g. 'Smith et al., 2021')
            if re.search(r"\bet al\b", text, re.IGNORECASE):
                continue

            text_lower = text.lower()
            if text_lower == title_lower or text_lower in seen:
                continue

            # 4. Check against forbidden structural patterns
            is_forbidden = any(pat.search(text) for pat in cls._COMPILED_FORBIDDEN)
            if is_forbidden:
                continue

            seen.add(text_lower)
            cleaned.append(text)

        return cleaned

    @classmethod
    def attach_links_to_notes(
        cls,
        notes: List[Dict[str, Any]],
        pairs: List[Tuple[Any, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Resolves (source, target) link pairs and attaches bidirectional connections to notes.
        Supports integer IDs, string integer IDs, verbatim titles, and normalized titles.
        Sanitizes all connections against forbidden patterns and self-references.
        """
        if not notes or not pairs:
            return notes

        title_map: Dict[str, str] = {}
        stripped_title_map: Dict[str, str] = {}
        note_by_title: Dict[str, Dict[str, Any]] = {}
        id_to_note: Dict[Any, Dict[str, Any]] = {}

        for idx, n in enumerate(notes):
            t = n.get("title", "").strip()
            if t:
                title_map[t.casefold()] = t
                stripped = cls.strip_qualifiers(t)
                if stripped:
                    stripped_title_map[stripped] = t
                note_by_title[t] = n
                if "connections" not in n or not isinstance(n["connections"], list):
                    n["connections"] = []

            # 1-based index mapping (authoritative sequential index sent to LLM in full_notes_payload)
            num_id = idx + 1
            id_to_note[num_id] = n
            id_to_note[str(num_id)] = n

            # Map explicit non-numeric ID (e.g. UUID) without overriding sequential integer indices
            raw_id = n.get("id")
            if raw_id is not None and not isinstance(raw_id, int):
                raw_str = str(raw_id).strip()
                if not raw_str.isdigit():
                    id_to_note[raw_id] = n

        token_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for t, n in note_by_title.items():
            toks = tuple(cls.normalize_tokens(t))
            if len(toks) >= 2 and toks not in token_map:
                token_map[toks] = n

        def resolve_endpoint(val: Any) -> Optional[Dict[str, Any]]:
            if val is None:
                return None
            if val in id_to_note:
                return id_to_note[val]
            if isinstance(val, str):
                val_clean = val.strip()
                if val_clean.isdigit() and int(val_clean) in id_to_note:
                    return id_to_note[int(val_clean)]
                val_fold = val_clean.casefold()
                if val_fold in title_map:
                    return note_by_title[title_map[val_fold]]
                val_norm = cls.strip_qualifiers(val_clean)
                if val_norm in stripped_title_map:
                    return note_by_title[stripped_title_map[val_norm]]
                val_toks = tuple(cls.normalize_tokens(val_clean))
                if len(val_toks) >= 2 and val_toks in token_map:
                    return token_map[val_toks]
            return None

        for src_val, tgt_val in pairs:
            src_note = resolve_endpoint(src_val)
            tgt_note = resolve_endpoint(tgt_val)

            if src_note and tgt_note and src_note is not tgt_note:
                src_title = src_note.get("title", "")
                tgt_title = tgt_note.get("title", "")
                if src_title.lower() != tgt_title.lower():
                    if tgt_title not in src_note["connections"]:
                        src_note["connections"].append(tgt_title)
                    if src_title not in tgt_note["connections"]:
                        tgt_note["connections"].append(src_title)

        for n in notes:
            n["connections"] = cls.clean_connections(n.get("connections", []), n.get("title", ""))

        return notes

    @classmethod
    def normalize_notes_data(cls, data: Any) -> List[Dict[str, Any]]:
        """
        Normalizes parsed JSON data (either flat list or wrapped object with 'notes' and 'links')
        into a standardized list of note dictionaries with resolved 'connections'.
        """
        items: List[Dict[str, Any]] = []
        gen_title = ""
        raw_links: List[Any] = []

        found_notes_key = False
        if isinstance(data, list):
            items = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            gen_title = data.get("general_title", "")
            for key in ('notes', 'zettelkasten', 'data', 'items', 'result', 'generated_notes'):
                if key in data and isinstance(data[key], list):
                    items = [item for item in data[key] if isinstance(item, dict)]
                    found_notes_key = True
                    break
            if not found_notes_key:
                if 'title' in data and 'content' in data:
                    items = [data]
                else:
                    dict_values = [v for v in data.values() if isinstance(v, dict) and ('title' in v and 'content' in v)]
                    if dict_values:
                        items = dict_values

            # Extract top-level links if present
            for lkey in ('links', 'relations', 'edges', 'connections'):
                if lkey in data and isinstance(data[lkey], list):
                    raw_links = data[lkey]
                    break

        if not items:
            return []

        # Filter and sanitize valid notes: drop empty notes, meta-placeholders, and non-notes
        valid_items: List[Dict[str, Any]] = []
        for it in items:
            raw_title = str(it.get("title", "")).strip().strip('"\'')
            raw_content = str(it.get("content", "")).strip()
            # Require meaningful title and non-empty content
            if not raw_title or not raw_content:
                continue
            # Drop prompt meta-syntax placeholders (e.g. '<Primary Concept...>')
            if raw_title.startswith("<") and raw_title.endswith(">"):
                continue
            if raw_title.lower() in ("untitled note", "new note"):
                continue
            it["title"] = raw_title
            valid_items.append(it)

        items = valid_items
        if not items:
            return []

        # 1. Build ID and Title lookup tables
        id_to_note: Dict[Any, Dict[str, Any]] = {}
        clean_title_map: Dict[str, str] = {}

        for idx, it in enumerate(items):
            if gen_title and not it.get("general_title"):
                it["general_title"] = gen_title
            canon_title = it["title"]
            if "connections" not in it or not isinstance(it["connections"], list):
                it["connections"] = []

            # 1-based index mapping
            id_to_note[idx + 1] = it
            id_to_note[str(idx + 1)] = it

            # Explicit ID mapping if provided by model (non-numeric only to avoid clobbering)
            if "id" in it:
                raw_id = it["id"]
                if not isinstance(raw_id, int):
                    raw_str = str(raw_id).strip()
                    if not raw_str.isdigit():
                        id_to_note[raw_id] = it

            if canon_title:
                clean_title_map[canon_title.casefold()] = canon_title
                stripped = cls.strip_qualifiers(canon_title)
                if stripped:
                    clean_title_map[stripped] = canon_title

        # 2. Resolve per-note connections (supports numeric IDs, digit strings, or title strings)
        for it in items:
            raw_c = it.get("connections", [])
            if not isinstance(raw_c, list):
                raw_c = [raw_c] if raw_c else []
            resolved_c: List[str] = []
            for c in raw_c:
                if isinstance(c, int) and c in id_to_note:
                    target_title = id_to_note[c]["title"]
                    if target_title and target_title.lower() != it["title"].lower():
                        resolved_c.append(target_title)
                elif isinstance(c, str):
                    c_strip = c.strip()
                    if c_strip.isdigit() and int(c_strip) in id_to_note:
                        target_title = id_to_note[int(c_strip)]["title"]
                        if target_title and target_title.lower() != it["title"].lower():
                            resolved_c.append(target_title)
                    else:
                        c_fold = c_strip.casefold()
                        c_norm = cls.strip_qualifiers(c_strip)
                        if c_fold in clean_title_map:
                            resolved_c.append(clean_title_map[c_fold])
                        elif c_norm in clean_title_map:
                            resolved_c.append(clean_title_map[c_norm])
                        else:
                            resolved_c.append(c_strip)
            it["connections"] = cls.clean_connections(resolved_c, it["title"])

        # 3. Resolve top-level links if present
        if raw_links:
            for link in raw_links:
                src_val, tgt_val = None, None
                if isinstance(link, dict):
                    src_val = link.get("source") or link.get("from") or link.get("source_title") or link.get("note_a") or link.get("source_id")
                    tgt_val = link.get("target") or link.get("to") or link.get("target_title") or link.get("note_b") or link.get("target_id")
                elif isinstance(link, (list, tuple)) and len(link) >= 2:
                    src_val, tgt_val = link[0], link[1]

                if src_val is not None and tgt_val is not None:
                    src_note = None
                    tgt_note = None

                    # Resolve src
                    if src_val in id_to_note:
                        src_note = id_to_note[src_val]
                    elif isinstance(src_val, str) and src_val.strip().isdigit() and int(src_val.strip()) in id_to_note:
                        src_note = id_to_note[int(src_val.strip())]
                    elif isinstance(src_val, str):
                        s_clean = src_val.strip()
                        s_canon = clean_title_map.get(s_clean.casefold()) or clean_title_map.get(cls.strip_qualifiers(s_clean))
                        if s_canon:
                            for cand in items:
                                if cand["title"] == s_canon:
                                    src_note = cand
                                    break

                    # Resolve tgt
                    if tgt_val in id_to_note:
                        tgt_note = id_to_note[tgt_val]
                    elif isinstance(tgt_val, str) and tgt_val.strip().isdigit() and int(tgt_val.strip()) in id_to_note:
                        tgt_note = id_to_note[int(tgt_val.strip())]
                    elif isinstance(tgt_val, str):
                        t_clean = tgt_val.strip()
                        t_canon = clean_title_map.get(t_clean.casefold()) or clean_title_map.get(cls.strip_qualifiers(t_clean))
                        if t_canon:
                            for cand in items:
                                if cand["title"] == t_canon:
                                    tgt_note = cand
                                    break

                    if src_note and tgt_note and src_note is not tgt_note:
                        s_title = src_note["title"]
                        t_title = tgt_note["title"]
                        if s_title.lower() != t_title.lower():
                            if t_title not in src_note["connections"]:
                                src_note["connections"].append(t_title)
                            if s_title not in tgt_note["connections"]:
                                tgt_note["connections"].append(s_title)

        # Final sanitize of connections on all items
        for it in items:
            it["connections"] = cls.clean_connections(it.get("connections", []), it["title"])

        return items

    @staticmethod
    def sanitize_latex_escapes(raw_json: str) -> str:
        """
        Repairs unescaped backslashes in LaTeX formulas within raw JSON responses.
        Prevents JSON decoders from mangling LaTeX commands (e.g. converting \\times
        or \\text to TAB [0x09], or \\beta to BACKSPACE [0x08]).
        """
        if not raw_json or not isinstance(raw_json, str):
            return ""

        # 1. Protect all single backslashes inside math blocks ($...$ and $$...$$)
        def fix_math_block(match: re.Match) -> str:
            block = match.group(0)
            # Replace any single backslash that is not already double-escaped or preceding a quote
            return re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', block)

        fixed = re.sub(r'\$\$[\s\S]+?\$\$|\$[^\$\n]+?\$', fix_math_block, raw_json)

        # 2. Also repair known LaTeX commands that might appear without math dollar delimiters
        latex_keywords = (
            r'(?:times|beta|alpha|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|omega|'
            r'frac|sqrt|text|sum|prod|int|cdot|leq|geq|neq|approx|infty|pm|partial|nabla|'
            r'mathbf|mathrm|mathit|vec|hat|bar|tilde|left|right)'
        )
        fixed = re.sub(r'(?<!\\)\\(' + latex_keywords + r'\b)', r'\\\\\1', fixed)

        # 3. Sanitize math operators accidentally wrapped inside \text{...} (e.g. \text{∑} -> \sum)
        # Prevents KaTeX from failing with "Can't use function X in text mode"
        math_text_ops = {
            '∑': r'\\sum', r'\sum': r'\\sum',
            '∏': r'\\prod', r'\prod': r'\\prod',
            '∫': r'\\int', r'\int': r'\\int',
            '√': r'\\sqrt', r'\sqrt': r'\\sqrt',
            '±': r'\\pm', r'\pm': r'\\pm',
            '≤': r'\\le', r'\le': r'\\le',
            '≥': r'\\ge', r'\ge': r'\\ge',
            '≠': r'\\ne', r'\ne': r'\\ne',
            '≈': r'\\approx', r'\approx': r'\\approx',
            '×': r'\\times', r'\times': r'\\times',
            '÷': r'\\div', r'\div': r'\\div',
        }
        def fix_text_operators(m: re.Match) -> str:
            inner = m.group(1).strip()
            if inner in math_text_ops:
                return f" {math_text_ops[inner]} "
            res = inner
            for sym, repl in math_text_ops.items():
                if sym in res:
                    res = res.replace(sym, f" {repl} ")
            if res != inner:
                return res
            return m.group(0)

        fixed = re.sub(r'\\text\{([^}]*)\}', fix_text_operators, fixed)

        return fixed

    @staticmethod
    def repair_truncated_json(raw: str) -> str:
        """
        Repairs truncated JSON strings caused by LLMs hitting max_tokens limits.
        Closes dangling string escapes, removes trailing incomplete keys/colons/commas,
        and balances unclosed brackets ('[' -> ']' and '{' -> '}').
        """
        if not raw or not isinstance(raw, str):
            return ""

        s = raw.strip()

        # 1. Strip trailing dangling backslash if cut off mid-escape
        if s.endswith("\\"):
            s = s[:-1]

        # 2. Check if stopped inside an unclosed string
        in_str = False
        escape = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str

        # If cut off mid-string, the current element is incomplete.
        # Discard the incomplete element back to before its unclosed '{' brace.
        if in_str:
            last_open_brace = s.rfind("{")
            last_close_brace = s.rfind("}")
            if last_open_brace > last_close_brace:
                s = s[:last_open_brace].rstrip(" \t\r\n,")
            else:
                s += '"'

        # 3. Strip incomplete trailing key, colon, or comma
        # e.g., , "title": " or , "tit" or , {
        s = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', s)
        s = re.sub(r',\s*"[^"]*"\s*:\s*$', '', s)
        s = re.sub(r',\s*"[^"]*"\s*$', '', s)
        s = re.sub(r',\s*\{[^{}]*$', '', s)
        s = s.rstrip(' \t\r\n,:/')

        # 4. Recalculate bracket stack after stripping
        stack = []
        in_str = False
        escape = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and stack[-1] == ch:
                        stack.pop()

        # 5. Append missing closing brackets in reverse order
        while stack:
            s += stack.pop()

        return s

    @classmethod
    def parse_notes_json(cls, response_text: str) -> List[Dict[str, Any]]:
        """
        Parses notes JSON from raw model response text.
        Applies multi-stage recovery: clean parse, truncation repair, markdown code blocks,
        raw_decode (non-strict), backslash fixes, outermost bracket slicing, resilient array
        salvaging, and regex pattern fallback.
        """
        if not response_text or not isinstance(response_text, str):
            return []

        cleaned_str = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', response_text)

        # Safely strip outer thinking block ONLY if it occurs before the main JSON/code block.
        # Preserves any <think> keywords that exist inside actual note content/strings.
        outer_think_match = re.match(r'^\s*(?:<think>|<\|think\|>)[\s\S]*?(?:</think>|<think\|>|<\|channel\|>)\s*', cleaned_str, re.IGNORECASE)
        if outer_think_match:
            cleaned_str = cleaned_str[outer_think_match.end():].strip()

        # Sanitize unescaped LaTeX backslashes before JSON decoding
        cleaned_str = cls.sanitize_latex_escapes(cleaned_str)

        # 1. Direct JSON parse
        try:
            parsed = json.loads(cleaned_str, strict=False)
            norm = cls.normalize_notes_data(parsed)
            if norm:
                return norm
        except json.JSONDecodeError:
            pass

        # 1b. Truncated repair on raw string
        try:
            repaired_str = cls.repair_truncated_json(cleaned_str)
            if repaired_str and repaired_str != cleaned_str:
                parsed = json.loads(repaired_str, strict=False)
                norm = cls.normalize_notes_data(parsed)
                if norm:
                    log_debug(f"Successfully parsed {len(norm)} notes using repair_truncated_json.")
                    return norm
        except Exception:
            pass

        # 2. Markdown code block extraction
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_str)
        for block in code_blocks:
            block_cleaned = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', block)
            try:
                parsed = json.loads(block_cleaned, strict=False)
                norm = cls.normalize_notes_data(parsed)
                if norm:
                    return norm
            except json.JSONDecodeError:
                pass
            try:
                repaired_block = cls.repair_truncated_json(block_cleaned)
                if repaired_block and repaired_block != block_cleaned:
                    parsed = json.loads(repaired_block, strict=False)
                    norm = cls.normalize_notes_data(parsed)
                    if norm:
                        return norm
            except Exception:
                pass
            for start_char in ('[', '{'):
                start_idx = block_cleaned.find(start_char)
                if start_idx != -1:
                    try:
                        decoder = json.JSONDecoder(strict=False)
                        parsed, _ = decoder.raw_decode(block_cleaned[start_idx:])
                        norm = cls.normalize_notes_data(parsed)
                        if norm:
                            return norm
                    except json.JSONDecodeError:
                        pass

        # 3. Bracket extraction via raw_decode (strict=False)
        for start_char in ('[', '{'):
            idx = cleaned_str.find(start_char)
            if idx != -1:
                try:
                    decoder = json.JSONDecoder(strict=False)
                    parsed, _ = decoder.raw_decode(cleaned_str[idx:])
                    norm = cls.normalize_notes_data(parsed)
                    if norm:
                        return norm
                except json.JSONDecodeError:
                    pass

        # 4. Unescaped backslash repair
        try:
            fixed_str = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', cleaned_str)
            for start_char in ('[', '{'):
                start_idx = fixed_str.find(start_char)
                if start_idx != -1:
                    try:
                        decoder = json.JSONDecoder(strict=False)
                        parsed, _ = decoder.raw_decode(fixed_str[start_idx:])
                        norm = cls.normalize_notes_data(parsed)
                        if norm:
                            return norm
                    except json.JSONDecodeError:
                        pass
            parsed = json.loads(fixed_str, strict=False)
            norm = cls.normalize_notes_data(parsed)
            if norm:
                return norm
        except json.JSONDecodeError:
            pass

        # 5. Outermost regex bracket match fallback
        for pattern in (r"\[\s*\{[\s\S]*\}\s*\]", r"\{\s*\"[\s\S]*\}\s*"):
            match = re.search(pattern, cleaned_str)
            if match:
                try:
                    fixed_match = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', match.group(0))
                    parsed = json.loads(fixed_match, strict=False)
                    norm = cls.normalize_notes_data(parsed)
                    if norm:
                        return norm
                except json.JSONDecodeError:
                    pass

        # 6. Resilient salvage of completed notes from truncated array
        note_keys = ('"notes"', '"zettelkasten"', '"data"', '"items"', '"result"', '"generated_notes"')
        array_start = -1
        for key in note_keys:
            idx = cleaned_str.find(key)
            if idx != -1:
                array_start = cleaned_str.find('[', idx)
                if array_start != -1:
                    break

        if array_start == -1:
            array_start = cleaned_str.find('[')

        if array_start != -1:
            gen_match = re.search(r'"general_title"\s*:\s*"([^"]+)"', cleaned_str)
            gen_title = gen_match.group(1).strip() if gen_match else None
            cur = array_start + 1
            decoder = json.JSONDecoder(strict=False)
            salvaged = []
            while cur < len(cleaned_str):
                while cur < len(cleaned_str) and cleaned_str[cur] in ' \t\r\n,':
                    cur += 1
                if cur >= len(cleaned_str) or cleaned_str[cur] == ']':
                    break
                try:
                    obj, end = decoder.raw_decode(cleaned_str[cur:])
                    if isinstance(obj, dict) and (obj.get('title') or obj.get('content')):
                        salvaged.append(obj)
                    cur += end
                except json.JSONDecodeError:
                    # Do not abort immediately; advance to next note object '{'
                    next_brace = cleaned_str.find('{', cur + 1)
                    if next_brace != -1:
                        cur = next_brace
                    else:
                        break
            if salvaged:
                log_debug(f"Successfully salvaged {len(salvaged)} notes from truncated JSON response.")
                return cls.normalize_notes_data({"notes": salvaged, "general_title": gen_title})

        # 7. Regex pattern fallback for severely malformed note blocks
        raw_notes = []
        gen_match = re.search(r'"general_title"\s*:\s*"([^"]+)"', cleaned_str)
        gen_title = gen_match.group(1).strip() if gen_match else None

        for note_match in re.finditer(
            r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"title"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
            cleaned_str
        ):
            n_id, n_title, n_content = note_match.groups()
            n_content_clean = n_content.replace(r'\"', '"').replace(r'\n', '\n')
            raw_notes.append({
                "id": int(n_id),
                "title": n_title.strip(),
                "content": n_content_clean.strip(),
            })

        if raw_notes:
            log_debug(f"Successfully salvaged {len(raw_notes)} notes via regex pattern fallback.")
            return cls.normalize_notes_data({"notes": raw_notes, "general_title": gen_title})

        return []

    @staticmethod
    def parse_links_json(response_text: str) -> List[Tuple[Any, Any]]:
        """Parses list of (source, target) link pairs from LLM response for Stage 2 linking."""
        if not response_text or not isinstance(response_text, str):
            return []

        cleaned_str = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', response_text)

        # Safely strip outer thinking block ONLY if it occurs before the main JSON/code block.
        # Preserves any <think> keywords that exist inside actual note content/strings.
        outer_think_match = re.match(r'^\s*(?:<think>|<\|think\|>)[\s\S]*?(?:</think>|<think\|>|<\|channel\|>)\s*', cleaned_str, re.IGNORECASE)
        if outer_think_match:
            cleaned_str = cleaned_str[outer_think_match.end():].strip()

        # Sanitize unescaped LaTeX backslashes before JSON decoding
        cleaned_str = AiResponseParser.sanitize_latex_escapes(cleaned_str)

        def extract_pairs(data: Any) -> List[Tuple[Any, Any]]:
            pairs: List[Tuple[Any, Any]] = []
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
                    src = item.get("source") or item.get("from") or item.get("source_title") or item.get("note_a") or item.get("source_id")
                    tgt = item.get("target") or item.get("to") or item.get("target_title") or item.get("note_b") or item.get("target_id")
                    if src is not None and tgt is not None:
                        pairs.append((src, tgt))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((item[0], item[1]))
            return pairs

        # 1. Direct JSON parse
        try:
            parsed = json.loads(cleaned_str, strict=False)
            res = extract_pairs(parsed)
            if res:
                return res
        except Exception:
            pass

        # 1b. Truncated repair on raw string
        try:
            repaired_str = AiResponseParser.repair_truncated_json(cleaned_str)
            if repaired_str and repaired_str != cleaned_str:
                parsed = json.loads(repaired_str, strict=False)
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
            try:
                repaired_block = AiResponseParser.repair_truncated_json(block.strip())
                if repaired_block and repaired_block != block.strip():
                    parsed = json.loads(repaired_block, strict=False)
                    res = extract_pairs(parsed)
                    if res:
                        return res
            except Exception:
                pass

        # 3. Bracket extraction (strict=False)
        for start_char in ('{', '['):
            idx = cleaned_str.find(start_char)
            if idx != -1:
                try:
                    decoder = json.JSONDecoder(strict=False)
                    parsed, _ = decoder.raw_decode(cleaned_str[idx:])
                    res = extract_pairs(parsed)
                    if res:
                        return res
                except Exception:
                    pass

        # 4. Truncated array salvaging if model hit max_tokens
        array_start = -1
        for key in ('"links"', '"connections"', '"relations"', '"edges"', '"graph"'):
            k_idx = cleaned_str.find(key)
            if k_idx != -1:
                array_start = cleaned_str.find('[', k_idx)
                if array_start != -1:
                    break

        if array_start == -1:
            array_start = cleaned_str.find('[')

        if array_start != -1:
            cur = array_start + 1
            decoder = json.JSONDecoder(strict=False)
            salvaged_items: List[Any] = []
            while cur < len(cleaned_str):
                while cur < len(cleaned_str) and cleaned_str[cur] in ' \t\r\n,':
                    cur += 1
                if cur >= len(cleaned_str) or cleaned_str[cur] == ']':
                    break
                try:
                    obj, end = decoder.raw_decode(cleaned_str[cur:])
                    if isinstance(obj, (dict, list, tuple)):
                        salvaged_items.append(obj)
                    cur += end
                except Exception:
                    # Advance to next '{' or '[' to salvage subsequent links
                    next_brace = -1
                    for bc in ('{', '['):
                        found_idx = cleaned_str.find(bc, cur + 1)
                        if found_idx != -1 and (next_brace == -1 or found_idx < next_brace):
                            next_brace = found_idx
                    if next_brace != -1:
                        cur = next_brace
                    else:
                        break
            res = extract_pairs(salvaged_items)
            if res:
                return res

        # 5. Regex pattern fallback for individual link objects
        regex_pairs: List[Tuple[Any, Any]] = []

        def _clean_val(v: str) -> Any:
            val = v.strip().strip('"\'')
            if val.isdigit():
                return int(val)
            return val

        # Match individual link object blocks { ... } containing both source and target,
        # supporting arbitrary key orders and preceding reasoning fields (e.g. 'relationship_logic')
        for obj_match in re.finditer(r'\{([^{}]+)\}', cleaned_str):
            obj_str = obj_match.group(1)
            src_m = re.search(r'(?:"source"|"from"|"note_a"|"source_id")\s*:\s*(\"[^\"]+\"|[^,}\s]+)', obj_str, re.IGNORECASE)
            tgt_m = re.search(r'(?:"target"|"to"|"note_b"|"target_id")\s*:\s*(\"[^\"]+\"|[^,}\s]+)', obj_str, re.IGNORECASE)
            if src_m and tgt_m:
                s_val = _clean_val(src_m.group(1))
                t_val = _clean_val(tgt_m.group(1))
                regex_pairs.append((s_val, t_val))

        if regex_pairs:
            seen_pairs = set()
            unique_pairs = []
            for p in regex_pairs:
                if p not in seen_pairs:
                    seen_pairs.add(p)
                    unique_pairs.append(p)
            return unique_pairs

        return []

