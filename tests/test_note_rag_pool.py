# test_note_rag_pool.py
#
# Unit tests for NoteRagPool:
# Tests mandatory embedding and reranker model enforcement, persistent ID tracking,
# two-stage retrieval (Harrier bi-encoder + Qwen3 reranker), Tier 1 global concept map,
# and token budget caps.

import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from note_rag_pool import NoteRagPool
from semantic_memory_service import SemanticMemoryService, EmbeddingModelNotFoundError
from reranker_service import RerankerService, RerankerModelNotFoundError


def create_mock_services():
    """Helper creating mock SemanticMemoryService and RerankerService."""
    mock_memory = MagicMock(spec=SemanticMemoryService)
    mock_memory.is_model_available.return_value = True
    mock_memory.DOCUMENT_PREFIX = "search_document: "
    mock_memory.QUERY_PREFIX = "search_query: "

    mock_reranker = MagicMock(spec=RerankerService)
    mock_reranker.is_model_available.return_value = True
    # Default reranker score_candidates mock: pairs each candidate with a score
    mock_reranker.score_candidates.side_effect = lambda query, candidates: [
        (0.95 - 0.05 * i, c) for i, c in enumerate(candidates)
    ]

    return mock_memory, mock_reranker


def test_embedding_model_mandatory_raises_when_missing():
    mock_memory = MagicMock(spec=SemanticMemoryService)
    mock_memory.is_model_available.return_value = False
    mock_memory.model_path = "/nonexistent/harrier.gguf"

    mock_reranker = MagicMock(spec=RerankerService)
    mock_reranker.is_model_available.return_value = True

    with pytest.raises(EmbeddingModelNotFoundError) as exc_info:
        NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    assert "Embedding model is mandatory but not found" in str(exc_info.value)


def test_reranker_model_mandatory_raises_when_missing():
    mock_memory = MagicMock(spec=SemanticMemoryService)
    mock_memory.is_model_available.return_value = True

    mock_reranker = MagicMock(spec=RerankerService)
    mock_reranker.is_model_available.return_value = False
    mock_reranker.model_path = "/nonexistent/qwen3_reranker.gguf"

    with pytest.raises(RerankerModelNotFoundError) as exc_info:
        NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    assert "Reranker model is mandatory but not found" in str(exc_info.value)


def test_add_notes_assigns_persistent_sequential_ids_and_extracts_mechanism():
    mock_memory, mock_reranker = create_mock_services()

    # 1024-dim dummy embeddings
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[1] = 1.0
    mock_memory.embed_texts.return_value = np.vstack([v1, v2])

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    assert len(pool) == 0

    chunk1_notes = [
        {"title": "Concept Alpha", "content": "Alpha establishes the core axiom. It functions as the foundation."},
        {"title": "Concept Beta", "content": "Beta operationalizes the model. It estimates parameters."}
    ]
    pool.add_notes(chunk1_notes)

    assert len(pool) == 2
    all_notes = pool.get_all_notes()
    assert all_notes[0]["id"] == 1
    assert all_notes[0]["title"] == "Concept Alpha"
    assert "Alpha establishes the core axiom" in all_notes[0]["core_mechanism"]

    assert all_notes[1]["id"] == 2
    assert all_notes[1]["title"] == "Concept Beta"
    assert pool.get_all_titles() == ["Concept Alpha", "Concept Beta"]

    # Tier 1 Global Concept Map check
    concept_map = pool.get_compact_concept_map()
    assert len(concept_map) == 2
    assert concept_map[0]["id"] == 1
    assert concept_map[0]["title"] == "Concept Alpha"
    assert "core_mechanism" in concept_map[0]


def test_retrieve_relevant_notes_empty_pool():
    mock_memory, mock_reranker = create_mock_services()
    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    assert pool.retrieve_relevant_notes("Any query text") == []
    assert pool.retrieve_relevant_notes("") == []


def test_retrieve_relevant_notes_two_stage_reranking():
    mock_memory, mock_reranker = create_mock_services()

    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[1] = 1.0
    mock_memory.embed_texts.return_value = np.vstack([v1, v2])

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool.add_notes([
        {
            "title": "Quantum Mechanics",
            "content": "Quantum mechanics describes nature at atomic scales."
        },
        {
            "title": "Classical Mechanics",
            "content": "Classical mechanics deals with macroscopic projectiles."
        }
    ])

    q_vec = np.zeros(1024, dtype=np.float32)
    q_vec[0] = 0.8
    q_vec[1] = 0.8
    mock_memory.embed_text.return_value = q_vec

    # Configure mock reranker to rank Classical Mechanics FIRST with higher cross-attention score
    def custom_rerank(query, candidates):
        results = []
        for c in candidates:
            score = 0.98 if c["title"] == "Classical Mechanics" else 0.45
            results.append((score, c))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    mock_reranker.score_candidates.side_effect = custom_rerank

    retrieved = pool.retrieve_relevant_notes("Macroscopic motion analysis", top_k=2)

    assert len(retrieved) == 2
    # Reranker should put Classical Mechanics at index 0 because score 0.98 > 0.45
    assert retrieved[0]["title"] == "Classical Mechanics"
    assert retrieved[0]["rerank_score"] == 0.98
    assert retrieved[1]["title"] == "Quantum Mechanics"


def test_retrieve_two_tier_context():
    mock_memory, mock_reranker = create_mock_services()

    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    mock_memory.embed_texts.return_value = np.vstack([v1])

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool.add_notes([
        {"title": "Note 1", "content": "Comprehensive text for Note 1."}
    ])

    mock_memory.embed_text.return_value = v1

    tier1_map, tier2_focal = pool.retrieve_two_tier_context("Query", total_budget=3500)

    assert len(tier1_map) == 1
    assert tier1_map[0]["title"] == "Note 1"
    assert len(tier2_focal) == 1
    assert tier2_focal[0]["title"] == "Note 1"


def test_retrieve_relevant_notes_respects_token_budget():
    mock_memory, mock_reranker = create_mock_services()

    notes = [
        {"title": f"Theory Part {i}", "content": f"Elaborated exposition of Theory {i}. " * 30}
        for i in range(5)
    ]
    vectors = [np.ones(1024, dtype=np.float32) / np.sqrt(1024) for _ in range(5)]
    mock_memory.embed_texts.return_value = np.vstack(vectors)

    def exact_token_counter(text: str) -> int:
        return len(text.split())

    pool = NoteRagPool(
        semantic_memory_service=mock_memory,
        reranker_service=mock_reranker,
        count_tokens_fn=exact_token_counter
    )
    pool.add_notes(notes)

    query_vec = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
    mock_memory.embed_text.return_value = query_vec

    retrieved = pool.retrieve_relevant_notes("Query text", top_k=5, max_tokens=500)

    assert len(retrieved) < 5
    total_tokens = sum(exact_token_counter(n["content"]) for n in retrieved)
    assert total_tokens <= 600


def test_seen_titles_deduplication():
    mock_memory, mock_reranker = create_mock_services()
    v1 = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
    mock_memory.embed_texts.return_value = np.vstack([v1])

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool.add_notes([
        {"title": "Entropy in Thermodynamics", "content": "First description of entropy."}
    ])
    assert len(pool) == 1

    # Duplicate title should be skipped
    pool.add_notes([
        {"title": "entropy in thermodynamics", "content": "Duplicate note should be skipped."}
    ])
    assert len(pool) == 1
    assert pool.get_all_notes()[0]["content"] == "First description of entropy."


def test_semantic_deduplication_and_connection_redirect():
    mock_memory, mock_reranker = create_mock_services()

    # Create two nearly identical normalized vectors (sim ~0.98) and one orthogonal vector
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.98
    v2[1] = 0.20
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.zeros(1024, dtype=np.float32)
    v3[2] = 1.0

    mock_memory.embed_texts.side_effect = [
        np.vstack([v1]),      # Chunk 1: Note A
        np.vstack([v2, v3])   # Chunk 2: Note A duplicate, Note B
    ]

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    # Chunk 1
    pool.add_notes([
        {
            "title": "Chromatic Adaptation Transform",
            "content": "Chromatic adaptation transform modifies color space coordinates to match human visual contours under varying light."
        }
    ])
    assert len(pool) == 1

    # Chunk 2: proposes a duplicate of Note A with altered title, and Note B pointing to the duplicate
    pool.add_notes([
        {
            "title": "Chromatic Adaptation Mechanism",
            "content": "Chromatic adaptation mechanism modifies color space coordinates to match human visual contours under varying illuminants.",
            "connections": ["Surface Roughness"]
        },
        {
            "title": "Surface Roughness",
            "content": "Microscopic texture irregularities create optical shadowing across physical materials as seen in [[Chromatic Adaptation Mechanism]].",
            "connections": ["Chromatic Adaptation Mechanism"]
        }
    ])

    # The duplicate should be merged -> only 2 notes in pool!
    assert len(pool) == 2
    titles = [n["title"] for n in pool.get_all_notes()]
    assert "Chromatic Adaptation Transform" in titles
    assert "Surface Roughness" in titles
    assert "Chromatic Adaptation Mechanism" not in titles

    # Connection and markdown wikilink to the duplicate should be redirected to the surviving canonical title!
    all_notes = pool.get_all_notes()
    surface_note = next(n for n in all_notes if n["title"] == "Surface Roughness")
    assert "Chromatic Adaptation Transform" in surface_note["connections"]
    assert "[[Chromatic Adaptation Transform]]" in surface_note["content"]


def test_lexical_fallback_deduplication_low_embedding_similarity():
    """Verifies that high textual overlap or title match triggers deduplication even when embedding similarity is low (e.g. 0.55)."""
    mock_memory, mock_reranker = create_mock_services()

    # Vectors with low cosine similarity (~0.55)
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.55
    v2[1] = np.sqrt(1 - 0.55**2)

    mock_memory.embed_texts.side_effect = [
        np.vstack([v1]),
        np.vstack([v2])
    ]

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    # Note 1
    pool.add_notes([
        {
            "title": "Ayrım Boşluğu ve Hassasiyeti",
            "content": "Renk örnekleri arasında fiziksel ayrım boşluğu bulunmadığında insan gözünün parlaklık ve ton algı hassasiyeti belirgin şekilde artış gösterir. " * 3
        }
    ])
    assert len(pool) == 1

    # Note 2: Parenthetical qualifier in title, near identical content
    pool.add_notes([
        {
            "title": "Ayrım Boşluğu (Gap Effect No-Separation) ve Hassasiyeti",
            "content": "Renk örnekleri arasında fiziksel ayrım boşluğu bulunmadığında insan gözünün parlaklık ve ton algı hassasiyeti belirgin şekilde artış gösterir ve toleransları daraltır. " * 3
        }
    ])

    # Should merge despite embedding similarity being only 0.55
    assert len(pool) == 1
    surviving = pool.get_all_notes()[0]
    assert "Ayrım Boşluğu" in surviving["title"]


def test_note_rag_pool_llm_synthesis_integration():
    """Verifies that NoteRagPool invokes ai_provider.synthesize_note_cluster when merging duplicates."""
    mock_memory, mock_reranker = create_mock_services()

    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0

    mock_memory.embed_texts.side_effect = [
        np.vstack([v1]),
        np.vstack([v1])
    ]

    mock_ai_provider = MagicMock()
    mock_ai_provider.synthesize_note_cluster.return_value = {
        "title": "Unified CIEDE2000 Formulation",
        "content": "Synthesized unified explanation containing all formulas and boundary conditions.",
        "connections": ["Related Concept"]
    }

    pool = NoteRagPool(
        semantic_memory_service=mock_memory,
        reranker_service=mock_reranker,
        ai_provider=mock_ai_provider
    )

    pool.add_notes([
        {"title": "CIEDE2000 Formula", "content": "Basic lightness weighting and formula structure. " * 5}
    ])

    pool.add_notes([
        {"title": "CIEDE2000 Formula Structure", "content": "Basic lightness weighting and formula structure with parameters. " * 5}
    ])

    assert len(pool) == 1
    assert mock_ai_provider.synthesize_note_cluster.call_count == 1
    surviving = pool.get_all_notes()[0]
    assert surviving["title"] == "Unified CIEDE2000 Formulation"
    assert surviving["content"] == "Synthesized unified explanation containing all formulas and boundary conditions."
    assert "Related Concept" in surviving["connections"]


def test_reconcile_pool_globally_transitive_clusters():
    """Verifies that reconcile_pool_globally connects transitive multi-way duplicate clusters across the pool."""
    mock_memory, mock_reranker = create_mock_services()

    # Note 1 and Note 2 are duplicates; Note 2 and Note 3 are duplicates
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.98
    v2[1] = 0.20
    v2 /= np.linalg.norm(v2)
    v3 = np.zeros(1024, dtype=np.float32)
    v3[0] = 0.96
    v3[1] = 0.28
    v3 /= np.linalg.norm(v3)
    v_distinct = np.zeros(1024, dtype=np.float32)
    v_distinct[5] = 1.0

    mock_memory.embed_texts.side_effect = [
        np.vstack([v1, v_distinct]),
        np.vstack([v2]),
        np.vstack([v3])
    ]
    mock_memory.embed_text.return_value = v1

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    # Chunk 1
    pool.add_notes([
        {"title": "Kromatik Uyum Mekanizması", "content": "İnsan gözü değişen ışık spektrumuna göre koni hücrelerinin duyarlılığını ayarlar. " * 3},
        {"title": "Yüzey Dokusu ve Pürüzlülük", "content": "Fiziksel yüzeylerin mikro yapısı ışığı saçarak ölçüm hatalarına yol açar. " * 3, "connections": ["Kromatik Uyum Mekanizması"]}
    ])

    # Chunk 2
    pool.add_notes([
        {"title": "Kromatik Uyum Modeli", "content": "İnsan gözü değişen aydınlatıcı spektrumuna göre koni hücrelerinin duyarlılığını dinamik ayarlar. " * 3}
    ])

    # Chunk 3
    pool.add_notes([
        {"title": "Kromatik Adaptasyon Mekanizması", "content": "İnsan gözünün koni hücre duyarlılığını aydınlatma koşullarına göre dengeleme sürecidir. " * 3}
    ])

    # Reconcile pool globally (called automatically by get_all_notes)
    all_notes = pool.get_all_notes()

    # The 3 chromatic adaptation variants should be consolidated into 1! Total notes in pool = 2
    assert len(all_notes) == 2
    titles = [n["title"] for n in all_notes]
    assert any("Kromatik" in t for t in titles)
    assert "Yüzey Dokusu ve Pürüzlülük" in titles

    # Connection on Yüzey Dokusu should point to the surviving chromatic note
    texture_note = next(n for n in all_notes if n["title"] == "Yüzey Dokusu ve Pürüzlülük")
    surviving_chroma = next(t for t in titles if "Kromatik" in t)
    assert surviving_chroma in texture_note["connections"]


def test_reconcile_pool_globally_direct_disjoint_sets():
    """Verifies that reconcile_pool_globally correctly clusters separate notes already inside the pool."""
    mock_memory, mock_reranker = create_mock_services()

    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.98
    v2[1] = 0.20
    v2 /= np.linalg.norm(v2)
    v_distinct = np.zeros(1024, dtype=np.float32)
    v_distinct[8] = 1.0

    mock_memory.embed_text.return_value = v1

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool._notes = [
        {"id": 1, "title": "Ayrım Boşluğu ve Hassasiyeti", "content": "Ayrım boşluğu olmadan algı hassasiyeti artar. " * 5, "connections": []},
        {"id": 2, "title": "Ayrım Boşluğu (No Gap Effect) ve Hassasiyeti", "content": "Ayrım boşluğu olmadan algı hassasiyeti artar. " * 5, "connections": ["Fotoğrafçılık"]},
        {"id": 3, "title": "Fotoğrafçılık", "content": "Fotoğrafik pozlama ve sensör kalibrasyonu süreçleridir. " * 5, "connections": ["Ayrım Boşluğu (No Gap Effect) ve Hassasiyeti"]}
    ]
    pool._embeddings = [v1, v2, v_distinct]

    reconciled = pool.reconcile_pool_globally()

    assert len(reconciled) == 2
    titles = [n["title"] for n in reconciled]
    assert "Fotoğrafçılık" in titles
    photo_note = next(n for n in reconciled if n["title"] == "Fotoğrafçılık")
    # Connection should be remapped to the surviving canonical Ayrım Boşluğu title
    assert photo_note["connections"][0] == "Ayrım Boşluğu ve Hassasiyeti"


def test_note_rag_pool_handles_list_ids_and_nested_connections():
    mock_memory, mock_reranker = create_mock_services()
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[1] = 1.0
    # Duplicate of note 2 (very high similarity)
    v3 = np.zeros(1024, dtype=np.float32)
    v3[1] = 0.99
    v3[2] = 0.10
    v3 /= np.linalg.norm(v3)

    mock_memory.embed_texts.side_effect = [
        np.vstack([v1, v2]),
        np.vstack([v3])
    ]
    mock_memory.embed_text.return_value = v2

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)

    # Ingestion 1: candidate notes with list IDs and nested connections
    pool.add_notes([
        {
            "id": [1],
            "title": "Solow Growth Model",
            "content": "Solow growth model examines capital accumulation and steady state economics in detail.",
            "connections": [2]
        },
        {
            "id": [2],
            "title": "Technological Convergence",
            "content": "Technological convergence across developing and developed economies enables sustained economic growth and productivity.",
            "connections": [1, [3, "Solow Growth Model"]]
        }
    ])
    assert len(pool) == 2

    # Ingestion 2: duplicate of Technological Convergence with list ID and list connections
    pool.add_notes([
        {
            "id": [5],
            "title": "Technological Convergence Dynamics",
            "content": "Technological convergence across developing and developed economies enables sustained economic growth and productivity.",
            "connections": [[1, 2], "Solow Growth Model"]
        }
    ])
    # The duplicate should merge smoothly without TypeError: unhashable type: 'list'
    assert len(pool) == 2
    all_notes = pool.get_all_notes()
    assert len(all_notes) == 2
    for n in all_notes:
        assert isinstance(n["id"], int)
        assert isinstance(n["connections"], list)
        for c in n["connections"]:
            assert isinstance(c, str)


def test_retrieve_relevant_notes_fallback_when_below_min_similarity():
    """Verifies that retrieve_relevant_notes falls back to top notes when no candidates meet min_similarity."""
    mock_memory, mock_reranker = create_mock_services()

    # Create dummy embeddings
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[1] = 1.0

    mock_memory.embed_texts.return_value = np.vstack([v1, v2])
    # Query vector has 0.1 similarity with v1 and 0.05 with v2
    q_vec = np.zeros(1024, dtype=np.float32)
    q_vec[0] = 0.1
    q_vec[1] = 0.05
    q_vec[2] = 0.99
    q_vec = q_vec / np.linalg.norm(q_vec)
    mock_memory.embed_text.return_value = q_vec

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool.add_notes([
        {"id": 1, "title": "Note Alpha", "content": "Alpha content", "connections": []},
        {"id": 2, "title": "Note Beta", "content": "Beta content", "connections": []}
    ])

    # With min_similarity=0.99, neither note passes threshold (dot product is ~0.1)
    retrieved = pool.retrieve_relevant_notes("Some search query", min_similarity=0.99)
    assert len(retrieved) > 0
    assert any(n["title"] == "Note Alpha" for n in retrieved)


def test_clean_and_remap_notes_transitive_redirects():
    """Verifies that transitive redirects (C -> B -> A) are properly resolved to A in connections and wikilinks."""
    mock_memory, mock_reranker = create_mock_services()
    v1 = np.ones(1024, dtype=np.float32)
    v1 = v1 / np.linalg.norm(v1)
    mock_memory.embed_texts.return_value = np.vstack([v1])

    pool = NoteRagPool(semantic_memory_service=mock_memory, reranker_service=mock_reranker)
    pool.add_notes([
        {
            "id": 1,
            "title": "Alpha Note",
            "content": "This note references [[Note Charlie]] and [[Note Bravo|alias]].",
            "connections": ["Note Charlie", "Note Bravo"]
        }
    ])

    # Simulate multi-hop transitive redirects: Charlie -> Bravo -> Final Target
    pool._title_redirects["Note Charlie"] = "Note Bravo"
    pool._title_redirects["Note Bravo"] = "Final Target"

    cleaned = pool._clean_and_remap_notes()
    assert len(cleaned) == 1
    assert "Final Target" in cleaned[0]["connections"]
    assert "Note Charlie" not in cleaned[0]["connections"]
    assert "Note Bravo" not in cleaned[0]["connections"]
    assert "[[Final Target]]" in cleaned[0]["content"]
    assert "[[Final Target|alias]]" in cleaned[0]["content"]


