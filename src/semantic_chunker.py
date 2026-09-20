# semantic_chunker.py
#
# Universal dynamic semantic text chunker shared between Cloud (Gemini) and Local (GGUF) providers.
# Divides long documents into coherent, balanced chunks along paragraph, sentence, and word boundaries.

import re
import math
from typing import List, Optional, Callable

DEFAULT_EXTRACTION_CHUNK_TOKENS = 4500
DEFAULT_OVERLAP_TOKENS = 100


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimation for Latin, Turkish, and multilingual prose (~3.2 chars/token).
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.2))


def chunk_text(
    text: str,
    max_chunk_tokens: int = DEFAULT_EXTRACTION_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens_fn: Optional[Callable[[str], int]] = None
) -> List[str]:
    """
    Universal dynamic chunker: splits text into balanced, coherent semantic chunks.
    Supports structured documents (Markdown headings, paragraphs) and unstructured
    text blobs (continuous OCR/transcripts without headings) by splitting along
    sentence and paragraph boundaries to reach an evenly balanced target size per chunk.
    """
    if not text or not str(text).strip():
        return []

    token_counter = count_tokens_fn or estimate_tokens
    total_tokens = token_counter(text)

    if total_tokens <= max_chunk_tokens:
        return [text]

    # Calculate balanced target chunk size to avoid oversized first chunks
    effective_cap = max(500, max_chunk_tokens - overlap_tokens)
    num_chunks = max(2, math.ceil(total_tokens / effective_cap))
    target_new_tokens = math.ceil(total_tokens / num_chunks)

    # Split into base units (paragraphs or sentences)
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paras:
        raw_paras = [text.strip()]

    units: List[str] = []
    for p in raw_paras:
        p_tok = token_counter(p)
        if p_tok > target_new_tokens:
            # Sub-split oversized paragraphs or unstructured text blobs by sentence endings
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
            for s in sentences:
                s_tok = token_counter(s)
                if s_tok > target_new_tokens:
                    # Fallback for text with no punctuation: split by words
                    words = s.split()
                    cur_w: List[str] = []
                    cur_w_tok = 0
                    for w in words:
                        wt = token_counter(w + " ")
                        if cur_w_tok + wt > target_new_tokens and cur_w:
                            units.append(" ".join(cur_w))
                            cur_w = [w]
                            cur_w_tok = wt
                        else:
                            cur_w.append(w)
                            cur_w_tok += wt
                    if cur_w:
                        units.append(" ".join(cur_w))
                else:
                    units.append(s)
        else:
            units.append(p)

    chunks: List[str] = []
    cur_chunk: List[str] = []
    cur_new_tokens = 0

    for u in units:
        u_tok = token_counter(u)
        if cur_chunk and (cur_new_tokens + u_tok > target_new_tokens):
            chunks.append("\n\n".join(cur_chunk))
            # Retain overlap from end of current chunk
            overlap_units: List[str] = []
            overlap_tok = 0
            for prev_u in reversed(cur_chunk):
                pt = token_counter(prev_u)
                if overlap_tok + pt <= overlap_tokens:
                    overlap_units.insert(0, prev_u)
                    overlap_tok += pt
                else:
                    break
            cur_chunk = overlap_units + [u]
            cur_new_tokens = u_tok
        else:
            cur_chunk.append(u)
            cur_new_tokens += u_tok

    if cur_chunk:
        # If the last chunk is very small (< 300 tokens) and previous chunk exists, merge if within budget
        if len(chunks) >= 1 and cur_new_tokens < 300 and (token_counter(chunks[-1]) + cur_new_tokens <= max_chunk_tokens + 200):
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join([u for u in cur_chunk if u not in chunks[-1]])
        else:
            chunks.append("\n\n".join(cur_chunk))

    return chunks

