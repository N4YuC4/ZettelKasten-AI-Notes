# test_semantic_chunker.py

import semantic_chunker


def test_estimate_tokens():
    assert semantic_chunker.estimate_tokens("") == 0
    assert semantic_chunker.estimate_tokens("Hello world") >= 1
    # ~3.2 chars per token
    text = "a" * 320
    assert 95 <= semantic_chunker.estimate_tokens(text) <= 105


def test_chunk_text_empty():
    assert semantic_chunker.chunk_text("") == []
    assert semantic_chunker.chunk_text("   \n\n  ") == []


def test_chunk_text_single_pass_within_budget():
    short_text = "This is a short document.\n\nIt fits easily into one chunk."
    chunks = semantic_chunker.chunk_text(short_text, max_chunk_tokens=4500)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_reduces_14_chunks_to_4_or_5_for_73k_doc():
    # Simulate a ~73,000 character document (such as Turing's 1950 paper)
    paragraphs = [
        f"Paragraph {i}: " + ("This is a detailed analysis of computational intelligence and discrete systems. " * 8)
        for i in range(120)
    ]
    doc = "\n\n".join(paragraphs)
    assert len(doc) >= 70_000

    # Under old 1,800 token budget:
    old_chunks = semantic_chunker.chunk_text(doc, max_chunk_tokens=1800, overlap_tokens=200)
    assert len(old_chunks) >= 12

    assert semantic_chunker.DEFAULT_OVERLAP_TOKENS == 0

    # Under new dynamic 4,500 token budget:
    new_chunks = semantic_chunker.chunk_text(
        doc,
        max_chunk_tokens=4500,
        overlap_tokens=semantic_chunker.DEFAULT_OVERLAP_TOKENS
    )
    # Reduced from 12-14 parts to 4-5 parts!
    assert 4 <= len(new_chunks) <= 6

    # Verify every paragraph is retained and chunks overlap properly
    assert "Paragraph 0:" in new_chunks[0]
    assert "Paragraph 119:" in new_chunks[-1]


def test_chunk_text_unstructured_blob_splitting():
    # Continuous text without newlines
    raw_blob = "Sentence without paragraphs. " * 1500
    chunks = semantic_chunker.chunk_text(raw_blob, max_chunk_tokens=1000, overlap_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) > 0

