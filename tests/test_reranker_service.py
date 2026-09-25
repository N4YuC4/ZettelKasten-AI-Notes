# test_reranker_service.py
#
# Unit tests for RerankerService (Qwen3-Reranker-0.6B-GGUF cross-encoder).
# Tests mandatory model verification, prompt formatting, CPU parameter initialization,
# sigmoid score normalization, and candidate ranking.

import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from reranker_service import (
    RerankerService,
    RerankerModelNotFoundError,
    RerankerInferenceError
)


def test_reranker_model_mandatory_raises_when_missing():
    service = RerankerService(model_path="/nonexistent/reranker_path.gguf")
    assert not service.is_model_available()

    with pytest.raises(RerankerModelNotFoundError) as exc_info:
        service.score_candidates("Query", [{"title": "Note", "content": "Content"}])

    assert "Reranker model is mandatory but not found" in str(exc_info.value)


def test_format_rerank_input():
    query = "Solow-Swan neoclassical growth model convergence"
    title = "Beta Convergence"
    content = "Beta convergence implies poorer economies grow faster."

    formatted = RerankerService.format_rerank_input(query, title, content)
    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<Query>:" in formatted
    assert "<Document>: Title: Beta Convergence" in formatted
    assert content in formatted
    assert "<|im_start|>assistant" in formatted
    assert "<think>" in formatted


def test_sigmoid_normalization():
    assert RerankerService._sigmoid(0.0) == pytest.approx(0.5, abs=1e-5)
    assert RerankerService._sigmoid(100.0) == pytest.approx(1.0, abs=1e-5)
    assert RerankerService._sigmoid(-100.0) == pytest.approx(0.0, abs=1e-5)
    assert 0.0 <= RerankerService._sigmoid(2.5) <= 1.0


def test_score_candidates_sorts_descending_by_relevance():
    service = RerankerService(model_path="/fake/reranker.gguf")

    candidates = [
        {"id": 1, "title": "Low Relevance", "content": "Unrelated topic."},
        {"id": 2, "title": "High Relevance", "content": "Direct conceptual match."},
        {"id": 3, "title": "Medium Relevance", "content": "Partially related."}
    ]

    mock_llm = MagicMock()
    mock_llm.tokenize.side_effect = lambda b: [9693] if b == b"yes" else ([2152] if b == b"no" else [1, 2, 3])

    # Candidate 1: diff = 8.5 - 10.0 = -1.5
    # Candidate 2: diff = 13.2 - 10.0 = 3.2
    # Candidate 3: diff = 10.8 - 10.0 = 0.8
    logits_seq = [
        {9693: 8.5, 2152: 10.0},
        {9693: 13.2, 2152: 10.0},
        {9693: 10.8, 2152: 10.0},
    ]
    mock_llm._ctx.get_logits_ith.side_effect = logits_seq

    with patch.object(service, "is_model_available", return_value=True), \
         patch.object(service, "_get_or_load_reranker", return_value=mock_llm):

        scored = service.score_candidates("Target query text", candidates)

        assert len(scored) == 3
        # Candidate 2 (High Relevance, diff=3.2) must be first
        assert scored[0][1]["id"] == 2
        assert scored[0][0] > scored[1][0]
        # Candidate 3 (Medium Relevance, diff=0.8) must be second
        assert scored[1][1]["id"] == 3
        assert scored[1][0] > scored[2][0]
        # Candidate 1 (Low Relevance, diff=-1.5) must be third
        assert scored[2][1]["id"] == 1


def test_rerank_top_k_and_min_score():
    service = RerankerService(model_path="/fake/reranker.gguf")

    candidates = [
        {"id": 1, "title": "Note 1", "content": "Text 1"},
        {"id": 2, "title": "Note 2", "content": "Text 2"},
        {"id": 3, "title": "Note 3", "content": "Text 3"},
        {"id": 4, "title": "Note 4", "content": "Text 4"},
    ]

    # Scores: candidate 1 -> 0.9 (diff=2.2), candidate 2 -> 0.8 (diff=1.386),
    #         candidate 3 -> 0.4 (diff=-0.4), candidate 4 -> 0.1 (diff=-2.2)
    mock_llm = MagicMock()
    mock_llm.tokenize.side_effect = lambda b: [9693] if b == b"yes" else ([2152] if b == b"no" else [1, 2, 3])

    def make_logits(diff):
        return {9693: 10.0 + diff, 2152: 10.0}

    with patch.object(service, "is_model_available", return_value=True), \
         patch.object(service, "_get_or_load_reranker", return_value=mock_llm):

        mock_llm._ctx.get_logits_ith.side_effect = [
            make_logits(2.2), make_logits(1.386), make_logits(-0.4), make_logits(-2.2)
        ]
        top_2 = service.rerank("Query", candidates, top_k=2)
        assert len(top_2) == 2
        assert top_2[0][1]["id"] == 1
        assert top_2[1][1]["id"] == 2

        # min_score=0.5
        mock_llm._ctx.get_logits_ith.side_effect = [
            make_logits(2.2), make_logits(1.386), make_logits(-0.4), make_logits(-2.2)
        ]
        filtered = service.rerank("Query", candidates, top_k=10, min_score=0.5)
        assert len(filtered) == 2
        assert all(score >= 0.5 for score, _ in filtered)


def test_unload_cached_model():
    mock_llm = MagicMock()
    RerankerService._cached_llm = mock_llm
    RerankerService._cached_model_path = "/fake/model.gguf"

    RerankerService.unload_cached_model()

    assert RerankerService._cached_llm is None
    assert RerankerService._cached_model_path is None
    mock_llm.close.assert_called_once()

