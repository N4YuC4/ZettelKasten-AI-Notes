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
    assert any("notlar" in m.lower() for m in progress_messages)


def test_default_context_window_is_32k():
    assert LocalGgufClient.DEFAULT_CONTEXT_WINDOW == 32768


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

    assert "=== ÖNCEKİ BÖLÜMLERDEN AKTARILAN BAĞLAM ===" in user_prompt
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
            assert "=== ÖNCEKİ BÖLÜMLERDEN AKTARILAN BAĞLAM ===" in call_prompts[1]


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
                        {"source": "PHP Değişkenleri", "target": "PHP Veri Tipleri"}
                    ]
                })
            }
        }]
    }
    client.llm = mock_llm

    notes = [
        {"title": "PHP Değişkenleri", "content": "Değişkenler $ ile tanımlanır ve değer saklar.", "connections": []},
        {"title": "PHP Veri Tipleri", "content": "String, integer, boolean gibi türler mevcuttur.", "connections": []}
    ]

    result = client.generate_note_links(notes)

    # Check that LLM received full content
    call_args = mock_llm.create_chat_completion.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]

    assert "PHP Değişkenleri" in user_prompt
    assert "Değişkenler $ ile tanımlanır" in user_prompt
    assert "PHP Veri Tipleri" in user_prompt
    assert "String, integer, boolean" in user_prompt

    # Check connection was attached
    assert "PHP Veri Tipleri" in result[0]["connections"]
    assert "PHP Değişkenleri" in result[1]["connections"]


def test_generate_note_links_single_note_noop():
    client = LocalGgufClient.__new__(LocalGgufClient)
    mock_llm = MagicMock()
    client.llm = mock_llm

    single_note = [{"title": "Only Note", "content": "Content", "connections": []}]
    res = client.generate_note_links(single_note)
    assert len(res) == 1
    mock_llm.create_chat_completion.assert_not_called()



