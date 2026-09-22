# tests/test_semantic_memory_service.py

import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from semantic_memory_service import (
    SemanticMemoryService,
    EmbeddingModelNotFoundError,
    EmbeddingInferenceError
)


def test_is_model_available_false(tmp_path):
    fake_path = tmp_path / "non_existent_harrier.gguf"
    service = SemanticMemoryService(model_path=str(fake_path))
    assert service.is_model_available() is False

    with pytest.raises(EmbeddingModelNotFoundError):
        service._get_or_load_embedder()


def test_embed_text_mocked(tmp_path):
    model_file = tmp_path / "harrier-oss-v1-0.6b.Q4_K_M.gguf"
    model_file.write_text("dummy gguf")

    service = SemanticMemoryService(model_path=str(model_file))
    assert service.is_model_available() is True

    # Mock embedder output
    mock_vec = [0.6, 0.8] + [0.0] * 1022  # norm = 1.0
    mock_embedder = MagicMock()
    mock_embedder.create_embedding.return_value = {
        "data": [{"embedding": mock_vec}]
    }

    with patch.object(service, "_get_or_load_embedder", return_value=mock_embedder):
        vec = service.embed_text("Test Note Content")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 1024
        assert np.isclose(np.linalg.norm(vec), 1.0)
        # Check prompt formatting with prefix
        mock_embedder.create_embedding.assert_called_with("search_document: Test Note Content")


def test_compute_candidate_pairs():
    service = SemanticMemoryService(model_path="/fake/path")

    # Construct 3 notes where:
    # Note 1 and Note 2 are very similar (cos ~ 0.96)
    # Note 3 is orthogonal (cos ~ 0.0)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.96, 0.28, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    fake_matrix = np.vstack([v1, v2, v3])

    notes = [
        {"id": 1, "title": "Beta Convergence", "content": "Economic growth models"},
        {"id": 2, "title": "Conditional Convergence", "content": "Empirical testing of Beta"},
        {"id": 3, "title": "Quantum Mechanics", "content": "Wave particle duality"}
    ]

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        candidate_pairs, matrix = service.compute_candidate_pairs(notes, similarity_threshold=0.65)

        # Only Note 1 and Note 2 should qualify
        assert len(candidate_pairs) == 1
        src, tgt, score = candidate_pairs[0]
        assert src == 1
        assert tgt == 2
        assert score >= 0.65
        assert np.array_equal(matrix, fake_matrix)


def test_find_similar_vault_notes():
    service = SemanticMemoryService(model_path="/fake/path")

    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    vault = {
        "note-1": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "note-2": np.array([0.7, 0.5, 0.0], dtype=np.float32),
        "note-3": np.array([0.1, 0.9, 0.0], dtype=np.float32),
    }

    # With threshold 0.65, note-1 (~0.99) and note-2 (~0.81) qualify
    results = service.find_similar_vault_notes(q, vault, similarity_threshold=0.65, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "note-1"
    assert results[1][0] == "note-2"

    # Test exclusion
    results_excluded = service.find_similar_vault_notes(
        q, vault, similarity_threshold=0.65, top_k=2, exclude_note_ids={"note-1"}
    )
    assert len(results_excluded) == 1
    assert results_excluded[0][0] == "note-2"


def test_compute_candidate_pairs_bounded_by_max_per_note():
    service = SemanticMemoryService(model_path="/fake/path")

    # Create 10 notes with high mutual similarity
    notes = [
        {"id": i, "title": f"Note {i}", "content": f"Content {i}"}
        for i in range(1, 11)
    ]
    # Matrix of all ones (normalized vectors) -> all pairs have similarity 1.0
    fake_matrix = np.ones((10, 1024), dtype=np.float32)
    fake_matrix = fake_matrix / np.linalg.norm(fake_matrix, axis=1, keepdims=True)

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        # With unbounded candidates, 10 notes would have 10*9/2 = 45 pairs
        all_pairs, _ = service.compute_candidate_pairs(notes, similarity_threshold=0.65, max_candidates_per_note=None)
        assert len(all_pairs) == 45

        # With max_candidates_per_note=2, total pairs should be strictly bounded (at most 10*2 = 20 undirected pairs)
        bounded_pairs, _ = service.compute_candidate_pairs(notes, similarity_threshold=0.65, max_candidates_per_note=2)
        assert len(bounded_pairs) <= 10 * 2
        assert len(bounded_pairs) < 45
        assert len(bounded_pairs) > 0


def test_partition_candidate_pairs():
    from prompt_templates import partition_candidate_pairs

    candidate_pairs = [
        (i, i + 1, 0.85)
        for i in range(1, 101)  # 100 pairs
    ]
    notes_map = {
        i: {"id": i, "title": f"Note {i}", "content": f"Full body content for note {i}"}
        for i in range(1, 105)
    }

    # Partition with max 25 pairs per batch
    batches = partition_candidate_pairs(
        candidate_pairs=candidate_pairs,
        notes_map=notes_map,
        max_prompt_tokens=50000,
        max_pairs_per_batch=25
    )

    assert len(batches) == 4
    for b in batches:
        assert len(b) <= 25

    # Flattened pairs should match original
    flattened = [pair for b in batches for pair in b]
    assert len(flattened) == 100
    assert flattened == candidate_pairs


def test_compute_candidate_pairs_skips_already_connected():
    service = SemanticMemoryService(model_path="/fake/path")

    notes = [
        {"id": 1, "title": "Note Alpha", "content": "Content Alpha", "connections": ["Note Beta"]},
        {"id": 2, "title": "Note Beta", "content": "Content Beta", "connections": ["Note Alpha"]},
        {"id": 3, "title": "Note Gamma", "content": "Content Gamma", "connections": []},
    ]
    fake_matrix = np.ones((3, 1024), dtype=np.float32)
    fake_matrix = fake_matrix / np.linalg.norm(fake_matrix, axis=1, keepdims=True)

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        # Note Alpha and Note Beta are already linked in Stage 1 -> must NOT be in candidate pairs!
        pairs, _ = service.compute_candidate_pairs(notes, similarity_threshold=0.65)
        # Pairs should only connect (Alpha, Gamma) and (Beta, Gamma)
        pair_tuples = [(p[0], p[1]) for p in pairs]
        assert (1, 2) not in pair_tuples
        assert (1, 3) in pair_tuples
        assert (2, 3) in pair_tuples


def test_compute_candidate_pairs_cross_chunk_filtering():
    service = SemanticMemoryService(model_path="/fake/path")

    # 4 notes across 2 chunks:
    # Chunk 0: Notes 1 and 2
    # Chunk 1: Notes 3 and 4
    notes = [
        {"id": 1, "title": "Chunk 0 Note A", "content": "Content", "connections": [], "_chunk_id": 0},
        {"id": 2, "title": "Chunk 0 Note B", "content": "Content", "connections": [], "_chunk_id": 0},
        {"id": 3, "title": "Chunk 1 Note C", "content": "Content", "connections": [], "_chunk_id": 1},
        {"id": 4, "title": "Chunk 1 Note D", "content": "Content", "connections": [], "_chunk_id": 1},
    ]
    fake_matrix = np.ones((4, 1024), dtype=np.float32)
    fake_matrix = fake_matrix / np.linalg.norm(fake_matrix, axis=1, keepdims=True)

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        # With cross_chunk_only=True (default when multiple chunks exist), intra-chunk pairs (1,2) and (3,4) must be skipped!
        pairs, _ = service.compute_candidate_pairs(notes, similarity_threshold=0.65, cross_chunk_only=True)
        pair_tuples = [(p[0], p[1]) for p in pairs]

        assert (1, 2) not in pair_tuples  # Same chunk (0)
        assert (3, 4) not in pair_tuples  # Same chunk (1)

        # Cross-chunk pairs must be present:
        assert (1, 3) in pair_tuples
        assert (1, 4) in pair_tuples
        assert (2, 3) in pair_tuples
        assert (2, 4) in pair_tuples


def test_compute_semantic_links_dynamic_threshold():
    service = SemanticMemoryService(model_path="/fake/path")

    # 4 notes:
    # Pair (1, 2) has high similarity 0.95
    # Pairs (1, 3), (1, 4), (2, 3), (2, 4), (3, 4) have low similarity ~0.2 - 0.4
    notes = [
        {"id": 1, "title": "Solow Growth Model", "content": "Capital accumulation"},
        {"id": 2, "title": "Beta Convergence", "content": "Catching up process in economic growth"},
        {"id": 3, "title": "Cooking Recipes", "content": "Italian pasta and pizza"},
        {"id": 4, "title": "Gardening Tips", "content": "Pruning roses in spring"}
    ]

    # Create synthetic normalized embeddings:
    # v1 and v2 close to [1, 0, 0, 0]
    # v3 close to [0, 1, 0, 0]
    # v4 close to [0, 0, 1, 0]
    v1 = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 1020, dtype=np.float32)
    v2 = np.array([0.95, 0.31, 0.0, 0.0] + [0.0] * 1020, dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.array([0.1, 0.99, 0.0, 0.0] + [0.0] * 1020, dtype=np.float32)
    v3 = v3 / np.linalg.norm(v3)
    v4 = np.array([0.05, 0.1, 0.99, 0.0] + [0.0] * 1020, dtype=np.float32)
    v4 = v4 / np.linalg.norm(v4)

    fake_matrix = np.vstack([v1, v2, v3, v4])

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        # Without explicit threshold: uses dynamic mean + 1.8*std floored by 0.70
        link_pairs, embeddings, eff_threshold = service.compute_semantic_links(notes, sensitivity_k=1.8, quality_floor=0.70)

        # Only (1, 2) should resonate above the statistical dynamic threshold
        assert len(link_pairs) == 1
        assert (1, 2) in link_pairs or (2, 1) in link_pairs
        assert eff_threshold >= 0.70
        assert np.array_equal(embeddings, fake_matrix)


def test_compute_semantic_links_empty_or_single():
    service = SemanticMemoryService(model_path="/fake/path")

    # Empty
    pairs, emb, thresh = service.compute_semantic_links([])
    assert pairs == []
    assert len(emb) == 0

    # Single note
    with patch.object(service, "embed_texts", return_value=np.ones((1, 1024), dtype=np.float32)):
        pairs, emb, thresh = service.compute_semantic_links([{"id": 1, "title": "Single", "content": "Body"}])
        assert pairs == []
        assert len(emb) == 1


def test_compute_semantic_links_explicit_threshold_and_filters():
    service = SemanticMemoryService(model_path="/fake/path")

    # 3 notes:
    # 1 and 2 already linked in Stage 1
    # 2 and 3 unlinked, similarity 0.85
    notes = [
        {"id": 1, "title": "Note A", "content": "Content A", "connections": ["Note B"]},
        {"id": 2, "title": "Note B", "content": "Content B", "connections": ["Note A"]},
        {"id": 3, "title": "Note C", "content": "Content C", "connections": []},
    ]

    v1 = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
    v2 = np.array([1.0] + [0.0] * 1023, dtype=np.float32)  # Sim(1, 2) = 1.0 but already linked!
    v3 = np.array([0.85, 0.5267] + [0.0] * 1022, dtype=np.float32)
    v3 = v3 / np.linalg.norm(v3)  # Sim(2, 3) ~ 0.85

    fake_matrix = np.vstack([v1, v2, v3])

    with patch.object(service, "embed_texts", return_value=fake_matrix):
        # Explicit threshold 0.80
        link_pairs, _, eff_thresh = service.compute_semantic_links(notes, similarity_threshold=0.80)
        assert eff_thresh == 0.80
        # (1, 2) is skipped because already linked; (1, 3) and (2, 3) qualify
        assert (1, 2) not in link_pairs
        assert (1, 3) in link_pairs
        assert (2, 3) in link_pairs


