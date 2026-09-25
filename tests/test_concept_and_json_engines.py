import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from concept_matcher import (
    strip_qualifiers,
    normalize_tokens,
    char_ngram_overlap,
    is_duplicate_concept,
)
from json_repair_engine import (
    sanitize_latex_escapes,
    repair_truncated_json,
)


def test_strip_qualifiers():
    assert strip_qualifiers("The Theory of Relativity (Special)") == "theory of relativity"
    assert strip_qualifiers("Die Quantenmechanik [Teil 1]") == "quantenmechanik"
    assert strip_qualifiers("Le Capital (Marx)") == "capital"
    assert strip_qualifiers("Birinci Madde (Giriş)") == "birinci madde"


def test_normalize_tokens():
    tokens = normalize_tokens("The batteries and categories are active")
    assert "battery" in tokens
    assert "category" in tokens


def test_char_ngram_overlap():
    # Turkish inflectional differences
    sim = char_ngram_overlap("kuantum dolanıklığı", "kuantum dolanıklığının")
    assert sim > 0.8
    assert char_ngram_overlap("", "anything") == 0.0


def test_is_duplicate_concept_distinct_numbers():
    t1 = "Model Part 1"
    t2 = "Model Part 2"
    c1 = "A" * 100
    c2 = "A" * 100
    is_dup, _, _ = is_duplicate_concept(t1, t2, c1, c2, sim=0.99)
    assert is_dup is False


def test_is_duplicate_concept_short_stub():
    is_dup, _, _ = is_duplicate_concept("Title A", "Title A", "Short", "Short", sim=0.99)
    assert is_dup is False


def test_sanitize_latex_escapes():
    raw = '{"formula": "$E = \\alpha + \\beta \\times 10$"}'
    fixed = sanitize_latex_escapes(raw)
    assert "\\\\alpha" in fixed
    assert "\\\\beta" in fixed
    assert "\\\\times" in fixed


def test_repair_truncated_json():
    import json
    # Truncated in middle of string discards unclosed object and balances brackets
    raw = '{"notes": [{"id": 1, "title": "Quantum", "content": "Partially written co'
    repaired = repair_truncated_json(raw)
    data = json.loads(repaired)
    assert "notes" in data
