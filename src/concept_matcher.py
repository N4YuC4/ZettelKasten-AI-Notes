"""
Concept matching and lexical deduplication engine for Zettelkasten AI Notes.

Provides domain-agnostic, multilingual tools for:
- Canonical title qualifier stripping
- Token normalization & stemming
- Language-agnostic character n-gram overlap
- Multi-tiered concept duplication detection
"""

import re
import difflib
from typing import List, Set, Tuple


def strip_qualifiers(title: str) -> str:
    """Strips parenthetical qualifiers, brackets, leading articles, and currency signs for canonical title comparison."""
    cleaned = re.sub(
        r"^(?:the|a|an|der|die|das|ein|eine|le|la|les|un|une|el|los|las|il|lo|gli|um|uma)\s+",
        "",
        title.strip(),
        flags=re.IGNORECASE
    )
    return re.sub(r"\s*[\(\[（【].*?[\)\]）】]", "", cleaned).replace("$", "").strip().casefold()


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


def char_ngram_overlap(text1: str, text2: str, n: int = 4) -> float:
    """
    Computes language-agnostic character n-gram containment overlap.
    Immune to agglutinative suffix differences (Turkish -ler/-in/-e),
    compound words (German), verb conjugations, and CJK tokenization quirks.
    Returns the containment ratio of the smaller text within the larger text.
    """
    if not text1 or not text2:
        return 0.0
    t1 = re.sub(r'\s+', ' ', text1.casefold()).strip()
    t2 = re.sub(r'\s+', ' ', text2.casefold()).strip()
    if len(t1) < n or len(t2) < n:
        return 0.0
    ngrams1 = {t1[i:i + n] for i in range(len(t1) - n + 1)}
    ngrams2 = {t2[i:i + n] for i in range(len(t2) - n + 1)}
    intersection = len(ngrams1 & ngrams2)
    return intersection / max(1, min(len(ngrams1), len(ngrams2)))


# Multilingual generic domain qualifiers that must not trigger false title entity matching on their own
GENERIC_TITLE_WORDS: Set[str] = {
    "formula", "formülü", "formulu", "formule", "formel", "fórmula",
    "model", "modeli", "modèle", "modelo",
    "theory", "teorisi", "théorie", "teoría",
    "method", "yöntemi", "yontemi", "yöntem", "yontem", "méthode", "metodo", "methode",
    "approach", "yaklaşımı", "yaklasimi", "approche", "enfoque", "ansatz",
    "analysis", "analizi", "analyse", "análisis",
    "concept", "kavramı", "kavrami", "konzept", "concepto",
    "effect", "etkisi", "effet", "efecto", "wirkung",
    "system", "sistemi", "système", "sistema",
    "structure", "yapısı", "yapisi", "struktur", "estructura",
    "overview", "genel", "bakış", "bakis", "aperçu", "resumen", "überblick",
    # Generic descriptive modifiers across domains and languages
    "economic", "ekonomik", "economique", "ökonomisch", "económico",
    "basic", "temel", "fundamental", "fondamental",
    "dynamic", "dynamics", "dinamik", "dinamikleri", "dinamiği",
    "component", "components", "bileşen", "bileşenleri", "unsur", "unsurları",
    "ve", "and", "et", "und", "y", "e",
    "nin", "nın", "nun", "nün", "in", "ın", "un", "ün"
}


def is_duplicate_concept(
    t1: str,
    t2: str,
    c1: str,
    c2: str,
    sim: float
) -> Tuple[bool, float, float]:
    """
    Determines if two notes represent the same underlying concept using a domain-agnostic,
    multilingual, multi-tiered verification combining canonical title identity, title similarity,
    lexical overlap, distinctive entity matching, and embedding cosine similarity.

    Returns:
        (is_dup, match_score, content_overlap) where match_score is a composite ranking score
        for best-match selection.
    """
    c1_str = str(c1).strip()
    c2_str = str(c2).strip()

    # 1. Never merge if content is too short (< 80 chars) to prevent false test/stub merges
    if len(c1_str) < 80 or len(c2_str) < 80:
        return False, 0.0, 0.0

    t1_clean = strip_qualifiers(t1)
    t2_clean = strip_qualifiers(t2)

    # 2. Never merge if canonical titles contain distinct numbers/indices (e.g. 'CIE94' vs 'CIEDE2000', 'Part 1' vs 'Part 2')
    digits1 = set(re.findall(r'\d+', t1_clean))
    digits2 = set(re.findall(r'\d+', t2_clean))
    if digits1 and digits2 and digits1 != digits2:
        return False, 0.0, 0.0

    t_sim = difflib.SequenceMatcher(None, t1_clean, t2_clean).ratio()

    # Language-agnostic char 4-gram overlap + fallback token overlap
    char_overlap = char_ngram_overlap(c1_str, c2_str, n=4)
    words1 = set(normalize_tokens(c1_str))
    words2 = set(normalize_tokens(c2_str))
    tok_overlap = len(words1 & words2) / max(1, min(len(words1), len(words2)))
    content_overlap = max(char_overlap, tok_overlap)

    min_len = min(len(t1_clean), len(t2_clean))
    max_len = max(len(t1_clean), len(t2_clean))
    is_title_substr = (
        (t1_clean in t2_clean or t2_clean in t1_clean)
        if (min_len >= 6 and max_len > 0 and (min_len / max_len >= 0.35))
        else False
    )

    t_toks1 = {w for w in re.findall(r'\b\w+\b', t1_clean) if len(w) >= 3 and w not in GENERIC_TITLE_WORDS}
    t_toks2 = {w for w in re.findall(r'\b\w+\b', t2_clean) if len(w) >= 3 and w not in GENERIC_TITLE_WORDS}
    shared_distinctive = bool(t_toks1 & t_toks2)

    # Core non-generic token sets across the complete title string (capturing permutations and author references)
    raw_toks1 = set(normalize_tokens(t1))
    raw_toks2 = set(normalize_tokens(t2))
    core_toks1 = {w for w in raw_toks1 if len(w) >= 3 and w not in GENERIC_TITLE_WORDS}
    core_toks2 = {w for w in raw_toks2 if len(w) >= 3 and w not in GENERIC_TITLE_WORDS}
    core_identity = (core_toks1 == core_toks2 and len(core_toks1) >= 2)

    is_dup = (
        # Tier 1: Exact canonical title match (after stripping qualifiers/articles) with modest corroboration
        (t1_clean == t2_clean and (sim >= 0.75 or content_overlap >= 0.20)) or
        # Tier 1b: Core concept identity (same distinctive entities/words across permutations/modifiers) with corroboration
        (core_identity and (content_overlap >= 0.20 or sim >= 0.50)) or
        # Tier 2: Substring or strong title similarity with high semantic similarity
        ((is_title_substr or t_sim >= 0.65) and sim >= 0.88 and (content_overlap >= 0.25 or t_sim >= 0.75)) or
        # Tier 3: Substantial title similarity with significant content overlap and solid semantic similarity
        (t_sim >= 0.55 and content_overlap >= 0.40 and sim >= 0.80) or
        # Tier 4: Near-identical text content (safety fallback for verbatim/near-verbatim re-extraction)
        (content_overlap >= 0.75) or
        # Tier 5: Shared distinctive title entity with extremely high semantic similarity
        (shared_distinctive and sim >= 0.95 and (t_sim >= 0.40 or content_overlap >= 0.25))
    )

    # Composite match score: prioritize core concept identity or balanced title/embedding alignment
    if core_identity and (content_overlap >= 0.20 or sim >= 0.50):
        match_score = max(sim * 0.50 + t_sim * 0.30 + content_overlap * 0.20, 0.85 + content_overlap * 0.15)
    else:
        match_score = sim * 0.50 + t_sim * 0.30 + content_overlap * 0.20
    return is_dup, match_score, content_overlap

