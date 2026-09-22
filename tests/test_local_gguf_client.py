# tests/test_local_gguf_client.py

import os
import sys
import json
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from local_gguf_client import (
    LocalGgufClient,
    LocalLlmError,
    LocalModelNotFoundError,
    LocalModelOOMError
)


def test_missing_model_raises_not_found():
    with pytest.raises(LocalModelNotFoundError):
        LocalGgufClient("/path/to/nonexistent/model.gguf")


def test_parse_notes_json_direct():
    client = LocalGgufClient.__new__(LocalGgufClient)
    sample_json = json.dumps([
        {
            "general_title": "Zettelkasten",
            "title": "Atomic Notes",
            "content": "One idea per note.",
            "connections": []
        }
    ])
    res = client._parse_notes_json(sample_json)
    assert len(res) == 1
    assert res[0]["title"] == "Atomic Notes"


def test_parse_notes_json_root_object_with_notes_list():
    client = LocalGgufClient.__new__(LocalGgufClient)
    sample_json = json.dumps({
        "general_title": "Quantum Physics",
        "notes": [
            {"title": "Note A", "content": "Desc A", "connections": ["Note B"]},
            {"title": "Note B", "content": "Desc B", "connections": ["Note A"]}
        ]
    })
    res = client._parse_notes_json(sample_json)
    assert len(res) == 2
    assert res[0]["title"] == "Note A"
    assert res[0]["general_title"] == "Quantum Physics"
    assert res[1]["title"] == "Note B"
    assert res[1]["general_title"] == "Quantum Physics"



def test_parse_notes_json_markdown_wrapped():
    client = LocalGgufClient.__new__(LocalGgufClient)
    sample_text = """Here is the result:
```json
[
  {
    "general_title": "AI",
    "title": "LLM Inference",
    "content": "Runs locally on CPU/GPU.",
    "connections": []
  }
]
```
"""
    res = client._parse_notes_json(sample_text)
    assert len(res) == 1
    assert res[0]["title"] == "LLM Inference"


def test_generate_zettelkasten_notes_empty():
    client = LocalGgufClient.__new__(LocalGgufClient)
    assert client.generate_zettelkasten_notes("") == []
    assert client.generate_zettelkasten_notes("   ") == []


def test_generate_zettelkasten_notes_mocked(tmp_path):
    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_text("dummy gguf header")

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps([
                        {
                            "general_title": "Topic",
                            "title": "Note 1",
                            "content": "Content 1",
                            "connections": []
                        }
                    ])
                }
            }
        ]
    }

    with patch("local_gguf_client.HardwareChecker.check_file_compatibility", return_value=(True, "OK", "OK")):
        with patch.object(LocalGgufClient, "_get_or_load_model", return_value=mock_llm):
            client = LocalGgufClient(str(dummy_model))
            notes = client.generate_zettelkasten_notes("Some input text")
            assert len(notes) == 1
            assert notes[0]["title"] == "Note 1"


def test_chunk_text():
    client = LocalGgufClient.__new__(LocalGgufClient)
    paragraphs = [
        "Paragraph 1: " + "alpha " * 40,
        "Paragraph 2: " + "beta " * 40,
        "Paragraph 3: " + "gamma " * 40,
    ]
    full_text = "\n\n".join(paragraphs)
    # Estimate ~40 words is ~55 tokens per paragraph. Set max_chunk_tokens to 70
    chunks = client._chunk_text(full_text, max_chunk_tokens=70, overlap_tokens=10)
    assert len(chunks) >= 2


def test_generate_zettelkasten_notes_large_doc_chunks(tmp_path):
    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_text("dummy gguf header")

    mock_llm = MagicMock()
    mock_llm.tokenize.side_effect = lambda b: list(range(max(1, len(b) // 2)))
    # Return different notes for successive calls
    mock_llm.create_chat_completion.side_effect = [
        {
            "choices": [{
                "message": {"content": json.dumps([{"general_title": "T", "title": "Part 1 Note", "content": "C1", "connections": []}])}
            }]
        },
        {
            "choices": [{
                "message": {"content": json.dumps([{"general_title": "T", "title": "Part 2 Note", "content": "C2", "connections": []}])}
            }]
        },
    ]

    with patch("local_gguf_client.HardwareChecker.check_file_compatibility", return_value=(True, "OK", "OK")):
        with patch.object(LocalGgufClient, "_get_or_load_model", return_value=mock_llm):
            # Set n_ctx small so that text exceeds budget
            client = LocalGgufClient(str(dummy_model), n_ctx=2800)
            large_text = ("This is paragraph one with insightful concepts.\n\n" * 40) + ("This is paragraph two with further arguments.\n\n" * 40)
            notes = client.generate_zettelkasten_notes(large_text)
            assert len(notes) >= 2
            titles = [n["title"] for n in notes]
            assert "Part 1 Note" in titles
            assert "Part 2 Note" in titles


def test_unload_cached_model():
    mock_llm = MagicMock()
    LocalGgufClient._cached_llm = mock_llm
    LocalGgufClient._cached_model_path = "/dummy/model.gguf"
    LocalGgufClient._cached_n_ctx = 4096
    LocalGgufClient._cached_n_gpu_layers = -1

    assert LocalGgufClient.is_model_loaded() is True
    LocalGgufClient.unload_cached_model()

    mock_llm.close.assert_called_once()
    assert LocalGgufClient._cached_llm is None
    assert LocalGgufClient._cached_model_path is None
    assert LocalGgufClient._cached_n_ctx is None
    assert LocalGgufClient._cached_n_gpu_layers is None
    assert LocalGgufClient.is_model_loaded() is False


def test_unload_cached_model_handles_none_gracefully():
    LocalGgufClient._cached_llm = None
    LocalGgufClient._cached_n_gpu_layers = None
    LocalGgufClient.unload_cached_model()
    assert LocalGgufClient.is_model_loaded() is False


def test_cache_invalidation_on_gpu_layers_change(tmp_path):
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"dummy gguf content")

    mock_llm_gpu = MagicMock()
    mock_llm_cpu = MagicMock()

    LocalGgufClient.unload_cached_model()

    with patch("hardware_checker.HardwareChecker.check_file_compatibility", return_value=(True, "OK", "OK")):
        with patch("llama_cpp.Llama", side_effect=[mock_llm_gpu, mock_llm_cpu]) as mock_llama_cls:
            # 1. Initialize with GPU (-1)
            client1 = LocalGgufClient(str(dummy_model), n_gpu_layers=-1)
            assert LocalGgufClient._cached_n_gpu_layers == -1
            assert mock_llama_cls.call_count == 1

            # 2. Same settings -> reused
            client2 = LocalGgufClient(str(dummy_model), n_gpu_layers=-1)
            assert mock_llama_cls.call_count == 1

            # 3. Changed n_gpu_layers to 0 (CPU) -> cache miss, re-instantiated
            client3 = LocalGgufClient(str(dummy_model), n_gpu_layers=0)
            assert LocalGgufClient._cached_n_gpu_layers == 0
            assert mock_llama_cls.call_count == 2

    LocalGgufClient.unload_cached_model()


def test_parse_notes_json_truncated_salvage():
    client = LocalGgufClient.__new__(LocalGgufClient)
    # Simulate truncated JSON where output was cut off mid-way
    truncated_json = '''
    {
      "general_title": "Quantum Foundations",
      "notes": [
        {
          "title": "Superposition",
          "content": "States exist simultaneously.",
          "connections": ["Entanglement"]
        },
        {
          "title": "Entanglement",
          "content": "Correlated particles regardless of distance.",
          "connections": ["Superposition"]
        },
        {
          "title": "Incomplete Note",
          "content": "This got cut off by token limit
    '''
    notes = client._parse_notes_json(truncated_json)
    assert len(notes) == 2
    assert notes[0]["title"] == "Superposition"
    assert notes[0]["general_title"] == "Quantum Foundations"
    assert notes[1]["title"] == "Entanglement"
    assert notes[1]["general_title"] == "Quantum Foundations"


def test_generate_zettelkasten_notes_calls_on_progress():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.tokenize.return_value = list(range(100))
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps([{
                    "title": "Progress Note",
                    "content": "Progress content",
                    "connections": []
                }])
            }
        }]
    }
    client.llm = mock_llm
    client.model_path = "/dummy/model.gguf"
    client.n_ctx = 4096

    progress_messages = []
    def on_progress(msg: str):
        progress_messages.append(msg)

    notes = client.generate_zettelkasten_notes("Test text content", on_progress=on_progress)
    assert len(notes) == 1
    assert notes[0]["title"] == "Progress Note"
    assert len(progress_messages) > 0
    assert any("notes" in m.lower() for m in progress_messages)


def test_default_context_window_is_128k():
    assert LocalGgufClient.DEFAULT_CONTEXT_WINDOW == 131072


def test_execute_inference_with_chained_context():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "general_title": "AI Ethics",
                    "notes": [{
                        "title": "Responsibility Gap",
                        "content": "Who is responsible for autonomous actions?",
                        "connections": ["Turing Test"]
                    }]
                })
            }
        }]
    }
    client.llm = mock_llm

    prev_json = json.dumps({"general_title": "AI Ethics", "notes": [{"title": "Turing Test"}]})
    notes = client._execute_inference(
        chunk_text="Autonomous systems create novel accountability challenges.",
        previous_notes_json=prev_json,
        existing_titles=["Turing Test"],
        unified_general_title="AI Ethics"
    )

    assert len(notes) == 1
    assert notes[0]["title"] == "Responsibility Gap"

    call_args = mock_llm.create_chat_completion.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]

    assert "=== PREVIOUS SECTION CONTEXT ===" in user_prompt
    assert "Turing Test" in user_prompt
    assert "AI Ethics" in user_prompt
    assert "DEDICATED NEW NOTES (NO DUPLICATES)" in user_prompt


def test_generate_zettelkasten_notes_chained_multi_chunk(tmp_path):
    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_text("dummy gguf header")

    mock_llm = MagicMock()
    # Simulate token count: return list of ints based on length
    mock_llm.tokenize.side_effect = lambda b: list(range(max(1, len(b) // 2)))

    call_prompts = []

    def mock_chat_completion(**kwargs):
        messages = kwargs.get("messages", [])
        prompt_content = messages[1]["content"] if len(messages) > 1 else ""
        call_prompts.append(prompt_content)
        if len(call_prompts) == 1:
            return {
                "choices": [{
                    "message": {"content": json.dumps({
                        "general_title": "Unified Cognitive Architecture",
                        "notes": [{"title": "Sensory Perception", "content": "Raw inputs.", "connections": []}]
                    })}
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {"content": json.dumps({
                        "general_title": "Unified Cognitive Architecture",
                        "notes": [{"title": "Working Memory", "content": "Short-term buffer.", "connections": ["Sensory Perception"]}]
                    })}
                }]
            }

    mock_llm.create_chat_completion.side_effect = mock_chat_completion

    with patch("local_gguf_client.HardwareChecker.check_file_compatibility", return_value=(True, "OK", "OK")):
        with patch.object(LocalGgufClient, "_get_or_load_model", return_value=mock_llm):
            # Use small n_ctx so text is chunked into 2 parts
            client = LocalGgufClient(str(dummy_model), n_ctx=2800)
            large_text = ("This is section one discussing sensory apparatus in cognition.\n\n" * 20) + \
                         ("This is section two discussing working memory storage.\n\n" * 20)

            notes = client.generate_zettelkasten_notes(large_text)

            assert len(notes) == 2
            assert notes[0]["title"] == "Sensory Perception"
            assert notes[1]["title"] == "Working Memory"
            assert notes[0]["general_title"] == "Unified Cognitive Architecture"
            assert notes[1]["general_title"] == "Unified Cognitive Architecture"

            # Check that the second call received the first call's notes in its chained context!
            assert len(call_prompts) == 2
            assert "Sensory Perception" in call_prompts[1]
            assert "=== PREVIOUS SECTION CONTEXT ===" in call_prompts[1]


def test_parse_links_json_formats():
    client = LocalGgufClient.__new__(LocalGgufClient)

    # 1. Direct JSON
    sample_json = json.dumps({
        "links": [
            {"source": "Note 1", "target": "Note 2"},
            {"source": "Note 2", "target": "Note 3"}
        ]
    })
    pairs = client._parse_links_json(sample_json)
    assert len(pairs) == 2
    assert pairs[0] == ("Note 1", "Note 2")

    # 2. Markdown wrapped
    sample_md = """Here are the relations:
```json
{
  "links": [
    {"source": "Alpha", "target": "Beta"}
  ]
}
```"""
    pairs_md = client._parse_links_json(sample_md)
    assert len(pairs_md) == 1
    assert pairs_md[0] == ("Alpha", "Beta")

    # 3. Empty or malformed
    assert client._parse_links_json("") == []
    assert client._parse_links_json("not valid json at all") == []


def test_generate_note_links_full_content_and_attachment():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "links": [
                        {"source": "PHP Variables", "target": "PHP Data Types"}
                    ]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = [
        {"title": "PHP Variables", "content": "Variables are defined with $ and store values.", "connections": []},
        {"title": "PHP Data Types", "content": "Types such as string, integer, and boolean are supported.", "connections": []}
    ]

    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    result = client.generate_note_links(notes, semantic_memory_service=mock_no_mem)

    # Check that LLM received full content
    call_args = mock_llm.create_chat_completion.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]

    assert "PHP Variables" in user_prompt
    assert "Variables are defined with $" in user_prompt
    assert "PHP Data Types" in user_prompt
    assert "string, integer, and boolean" in user_prompt

    # Check connection was attached
    assert "PHP Data Types" in result[0]["connections"]
    assert "PHP Variables" in result[1]["connections"]
    assert call_args["max_tokens"] >= 4096


def test_generate_note_links_with_semantic_memory_service():
    client = LocalGgufClient.__new__(LocalGgufClient)
    client.n_ctx = 16384
    mock_llm = MagicMock()
    client.llm = mock_llm

    notes = [
        {"id": 1, "title": "Solow Model", "content": "Capital accumulation", "connections": []},
        {"id": 2, "title": "Beta Convergence", "content": "Catching up process", "connections": []},
        {"id": 3, "title": "Quantum Spin", "content": "Unrelated physics concept", "connections": []}
    ]

    mock_memory = MagicMock()
    mock_memory.is_model_available.return_value = True
    import numpy as np
    dummy_emb = np.ones((3, 1024), dtype=np.float32)
    # Pure vector linking returns (1, 2) directly above dynamic threshold
    mock_memory.compute_semantic_links.return_value = ([(1, 2)], dummy_emb, 0.78)

    linked = client.generate_note_links(notes, semantic_memory_service=mock_memory)

    # ZERO LLM calls made!
    mock_llm.create_chat_completion.assert_not_called()

    # Check embedding attached to notes for DB saving
    assert "_embedding" in linked[0]
    assert len(linked[0]["_embedding"]) == 1024

    # Bidirectional connections established via pure vector linking
    assert "Beta Convergence" in linked[0]["connections"]
    assert "Solow Model" in linked[1]["connections"]
    assert "Quantum Spin" not in linked[0]["connections"]




def test_generate_note_links_max_tokens_floor_4k():
    client = LocalGgufClient.__new__(LocalGgufClient)
    client.n_ctx = 8192
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({"links": []})}}]
    }
    client.llm = mock_llm

    notes = [
        {"title": f"Note {i}", "content": "Detailed discourse on concepts."}
        for i in range(20)
    ]
    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    client.generate_note_links(notes, semantic_memory_service=mock_no_mem)

    call_args = mock_llm.create_chat_completion.call_args[1]
    assert call_args["max_tokens"] >= 4096


def test_generate_note_links_single_note_noop():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    client.llm = mock_llm

    single_note = [{"title": "Only Note", "content": "Content", "connections": []}]
    res = client.generate_note_links(single_note)
    assert len(res) == 1
    mock_llm.create_chat_completion.assert_not_called()


def test_clean_connections_filters_structural_and_citations_across_domains():
    raw_connections = [
        # Academic citations and propositions
        "[1]",
        "[38]",
        "[Smith et al., 2021]",
        "Figure 3",
        "Proposition 4.1",
        "Theorem 2",
        "Table 1",
        "Equation (4)",
        "Footnote 5",
        # Legal articles and clauses
        "Madde 5",
        "Article 12",
        "Fıkra 2",
        "Clause 3.1",
        "Bent (a)",
        "Ek-1",
        # Books and structure
        "Chapter 4",
        "Bölüm 2",
        "Page 45",
        "Sayfa 12",
        # Transcripts and media
        "00:14:22",
        "Speaker 1",
        # Self title
        "Target Concept Title",
        # Genuine conceptual note titles (MUST BE PRESERVED)
        "Hyperbolic Embedding Space",
        "Sözleşmeden Dönme Hakkı",
        "Asynchronous Event Loop",
        "Hyperbolic Embedding Space",  # duplicate, must be deduplicated
    ]

    cleaned = LocalGgufClient._clean_connections(raw_connections, current_title="Target Concept Title")
    assert cleaned == [
        "Hyperbolic Embedding Space",
        "Sözleşmeden Dönme Hakkı",
        "Asynchronous Event Loop"
    ]


def test_execute_inference_prompt_contains_universal_rule():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "general_title": "Legal & Tech Analysis",
                    "notes": [{
                        "title": "Autonomous Liability",
                        "content": "Liability principles in autonomous systems.",
                        "connections": ["[12]", "Madde 5", "Strict Liability Doctrine"]
                    }]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = client._execute_inference("Sample text discussing legal liability in AI systems.")
    assert len(notes) == 1
    # Check that structural connections were automatically filtered out
    assert notes[0]["connections"] == ["Strict Liability Doctrine"]

    # Verify prompt contains universal instruction across all document types
    call_args = mock_llm.create_chat_completion.call_args[1]
    prompt_text = call_args["messages"][1]["content"]
    assert "Never use structural labels, numbers, or citations as titles" in prompt_text
    assert "Article 5" in prompt_text
    assert "[12]" in prompt_text
    assert "Figure 3" in prompt_text


def test_granular_16k_fallback_ladder_on_memory_error(tmp_path):
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"dummy gguf content")

    LocalGgufClient.unload_cached_model()

    # Simulate: 131072 fails with OOM, but 114688 succeeds!
    mock_successful_llm = MagicMock()

    call_contexts = []
    def mock_llama_init(*args, **kwargs):
        ctx = kwargs.get("n_ctx")
        call_contexts.append(ctx)
        if ctx == 131072:
            raise MemoryError("failed to create context: out of memory")
        return mock_successful_llm

    with patch("hardware_checker.HardwareChecker.check_file_compatibility", return_value=(True, "OK", "OK")):
        with patch("llama_cpp.Llama", side_effect=mock_llama_init):
            client = LocalGgufClient(str(dummy_model), n_ctx=131072)
            assert client.n_ctx == 114688
            assert 131072 in call_contexts
            assert 114688 in call_contexts

    LocalGgufClient.unload_cached_model()


def test_parse_notes_json_with_numeric_ids_and_top_level_links():
    client = LocalGgufClient.__new__(LocalGgufClient)
    sample_response = json.dumps({
        "general_title": "Consciousness Theory",
        "notes": [
            {
                "id": 1,
                "title": "Collapse Mechanism",
                "content": "Deterministic reduction of quantum state.",
                "connections": []
            },
            {
                "id": 2,
                "title": "Candidate Generation Process",
                "content": "Derivation of probable states.",
                "connections": []
            }
        ],
        "links": [
            {"source": 1, "target": 2}
        ]
    })
    res = client._parse_notes_json(sample_response)
    assert len(res) == 2
    assert res[0]["title"] == "Collapse Mechanism"
    assert res[0]["connections"] == ["Candidate Generation Process"]
    assert res[1]["title"] == "Candidate Generation Process"
    assert res[1]["connections"] == ["Collapse Mechanism"]


def test_parse_notes_json_with_in_note_numeric_connections():
    client = LocalGgufClient.__new__(LocalGgufClient)
    sample_response = json.dumps({
        "general_title": "Philosophy",
        "notes": [
            {
                "id": 1,
                "title": "Conscious State",
                "content": "Description 1",
                "connections": [2]
            },
            {
                "id": 2,
                "title": "Observer Effect",
                "content": "Description 2",
                "connections": ["1"]
            }
        ]
    })
    res = client._parse_notes_json(sample_response)
    assert len(res) == 2
    assert res[0]["connections"] == ["Observer Effect"]
    assert res[1]["connections"] == ["Conscious State"]


def test_generate_note_links_with_integer_ids():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "links": [
                        {"source": 1, "target": 2},
                        {"source": 2, "target": 3}
                    ]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = [
        {"title": "Concept A", "content": "Description A", "connections": []},
        {"title": "Concept B", "content": "Description B", "connections": []},
        {"title": "Concept C", "content": "Description C", "connections": []}
    ]

    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    linked_notes = client.generate_note_links(notes, semantic_memory_service=mock_no_mem)
    assert len(linked_notes) == 3
    assert "Concept B" in linked_notes[0]["connections"]
    assert "Concept A" in linked_notes[1]["connections"]
    assert "Concept C" in linked_notes[1]["connections"]
    assert "Concept B" in linked_notes[2]["connections"]


def test_generate_note_links_with_string_digits():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "links": [
                        {"source": "1", "target": "2"}
                    ]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = [
        {"title": "Alpha", "content": "Desc Alpha", "connections": []},
        {"title": "Beta", "content": "Desc Beta", "connections": []}
    ]

    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    linked_notes = client.generate_note_links(notes, semantic_memory_service=mock_no_mem)
    assert "Beta" in linked_notes[0]["connections"]
    assert "Alpha" in linked_notes[1]["connections"]


def test_generate_note_links_with_parenthetical_stripped_fallback():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "links": [
                        {"source": "Collapse Mechanism", "target": "Candidate Generation Process"}
                    ]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = [
        {"title": "Collapse Mechanism (Boltzmann-Softmax)", "content": "Desc 1", "connections": []},
        {"title": "Candidate Generation Process ($G$)", "content": "Desc 2", "connections": []}
    ]

    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    linked_notes = client.generate_note_links(notes, semantic_memory_service=mock_no_mem)
    assert "Candidate Generation Process ($G$)" in linked_notes[0]["connections"]
    assert "Collapse Mechanism (Boltzmann-Softmax)" in linked_notes[1]["connections"]


def test_execute_inference_prompt_contains_id_and_few_shot():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "general_title": "Test Topic",
                    "notes": [{"id": 1, "title": "Test Note", "content": "Desc", "connections": []}],
                    "links": []
                })
            }
        }]
    }
    client.llm = mock_llm

    client._execute_inference("Sample document text")

    call_args = mock_llm.create_chat_completion.call_args[1]
    sys_prompt = call_args["messages"][0]["content"]
    prompt_text = call_args["messages"][1]["content"]
    assert "STRICT LANGUAGE MATCHING:" in prompt_text
    assert "NEVER MIX LANGUAGES" in prompt_text
    assert "ATOMIC ZETTELKASTEN NOTES:" in prompt_text
    assert "CONCEPTUAL CONNECTIONS" not in prompt_text
    assert "JSON FORMAT:" in prompt_text
    assert "Always generate all titles, contents, and collections in the exact same language" in sys_prompt
    assert "Document Topic" in prompt_text
    assert "Concept Name" in prompt_text


def test_execute_inference_language_agnostic_prompt():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "general_title": "Hukuk",
                    "notes": [{"id": 1, "title": "Sözleşme Feshi", "content": "Açıklama", "connections": []}],
                    "links": []
                })
            }
        }]
    }
    client.llm = mock_llm

    tr_text = "Borçlar Kanunu uyarınca sözleşmeden dönme hakkı borçlunun temerrüdü halinde alacaklıya tanınan seçimlik bir haktır."
    client._execute_inference(tr_text)

    call_args = mock_llm.create_chat_completion.call_args[1]
    sys_prompt = call_args["messages"][0]["content"]
    user_prompt = call_args["messages"][1]["content"]

    # System instruction enforces universal fidelity to whatever language is input
    assert "Always generate all titles, contents, and collections in the exact same language as the source text." in sys_prompt
    assert "STRICT LANGUAGE MATCHING:" in user_prompt
    assert "NEVER MIX LANGUAGES:" in user_prompt


def test_generate_note_links_universal_language_agnostic_prompt():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({"links": [{"source": 1, "target": 2}]})}}]
    }
    client.llm = mock_llm

    # Notes in any language (English, Turkish, etc.) use the same universal ID linking prompt
    notes = [
        {"title": "Çökme Mekanizması", "content": "Kuantum durumunun deterministik indirgenmesi.", "connections": []},
        {"title": "Gözlemci Etkisi", "content": "Gözlemci ile gözlenen sistem arasındaki etkileşim.", "connections": []}
    ]
    mock_no_mem = MagicMock()
    mock_no_mem.is_model_available.return_value = False
    linked_notes = client.generate_note_links(notes, semantic_memory_service=mock_no_mem)
    call_prompt = mock_llm.create_chat_completion.call_args[1]["messages"][1]["content"]

    assert ("Below are all Zettelkasten notes" in call_prompt or "Below are the notes and semantically matched candidate pairs" in call_prompt)
    assert "RULES:" in call_prompt
    assert '"target":' in call_prompt
    assert "Gözlemci Etkisi" in linked_notes[0]["connections"]
    assert "Çökme Mekanizması" in linked_notes[1]["connections"]


def test_universal_dynamic_chunking_unstructured_blob():
    client = LocalGgufClient.__new__(LocalGgufClient)
    # Raw unstructured blob with NO newlines, only sentences
    blob_sentences = [
        f"Sentence number {i} explains an important scientific mechanism in color analysis."
        for i in range(1, 300)
    ]
    raw_blob = " ".join(blob_sentences)

    total_tok = client._count_tokens(raw_blob)
    assert total_tok > 2000

    chunks = client._chunk_text(raw_blob, max_chunk_tokens=1800, overlap_tokens=200)
    assert len(chunks) >= 2

    # Verify every chunk is within safe budget
    for ch in chunks:
        ch_tokens = client._count_tokens(ch)
        assert ch_tokens <= 2100  # allowing for overlap
        # Verify no mid-sentence truncation
        assert ch.endswith(".") or ch.endswith("!") or ch.endswith("?")


def test_universal_dynamic_chunking_short_single_pass():
    client = LocalGgufClient.__new__(LocalGgufClient)
    short_text = "This is a concise short document with two paragraphs.\n\nIt covers an atomic concept."
    chunks = client._chunk_text(short_text, max_chunk_tokens=1800)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_universal_dynamic_chunking_balanced_distribution():
    client = LocalGgufClient.__new__(LocalGgufClient)
    # 5,000+ tokens of paragraphs
    paras = [f"Paragraph {i}: " + ("data science and colorimetry concepts " * 15) for i in range(1, 50)]
    full_text = "\n\n".join(paras)
    total_tokens = client._count_tokens(full_text)
    assert total_tokens > 4000

    chunks = client._chunk_text(full_text, max_chunk_tokens=1800, overlap_tokens=200)
    assert len(chunks) >= 3

    # Ensure chunks are fairly balanced (none are extreme outliers like 100 tokens vs 4000 tokens)
    chunk_tokens = [client._count_tokens(c) for c in chunks]
    for ct in chunk_tokens:
        assert ct >= 500
        assert ct <= 2000


def test_execute_inference_sampling_parameters_forwarded():
    client = LocalGgufClient.__new__(LocalGgufClient)
    client.n_ctx = 8192
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps([{"title": "T", "content": "C"}])}}]
    }
    client.llm = mock_llm

    res = client._execute_inference("Sample text")
    assert len(res) == 1

    call_kwargs = mock_llm.create_chat_completion.call_args[1]
    assert call_kwargs.get("temperature") == 0.2
    assert call_kwargs.get("repeat_penalty") == 1.1
    assert call_kwargs.get("top_p") == 0.95
    assert call_kwargs.get("max_tokens") <= 4096
    # response_format is omitted to avoid CPU BNF grammar slowdown
    assert "response_format" not in call_kwargs






