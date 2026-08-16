import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gemini_api_client import (
    GeminiApiClient,
    GeminiApiError,
    GeminiAuthError,
    GeminiRateLimitError,
)

def test_exception_hierarchy():
    assert issubclass(GeminiAuthError, GeminiApiError)
    assert issubclass(GeminiRateLimitError, GeminiApiError)
    assert issubclass(GeminiApiError, Exception)

def test_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiAuthError):
        GeminiApiClient()

def test_model_name_preserved(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    assert client.model_name == 'gemma-4-31b-it'

def test_parse_notes_json_clean_json():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = '[{"general_title": "Topic", "title": "Note 1", "content": "Text", "connections": []}]'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Note 1"

def test_parse_notes_json_markdown_block():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = 'Here is the JSON:\n```json\n[{"general_title": "Topic", "title": "Note 2", "content": "Text", "connections": []}]\n```'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Note 2"

def test_parse_notes_json_dict_wrapper():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = '{"general_title": "Topic", "notes": [{"title": "Note 3", "content": "Text 3", "connections": []}]}'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Note 3"

def test_parse_notes_json_single_dict():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = '{"general_title": "Topic", "title": "Single Note", "content": "Single Content", "connections": []}'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Single Note"

def test_parse_notes_json_invalid():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = 'This is invalid text with no json'
    notes = client._parse_notes_json(raw)
    assert notes == []

def test_generate_zettelkasten_notes_empty_input(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    assert client.generate_zettelkasten_notes("") == []
    assert client.generate_zettelkasten_notes("   ") == []

def test_generate_zettelkasten_notes_auth_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    client.client = MagicMock()
    client.client.models.generate_content.side_effect = Exception("403 API_KEY_INVALID: User not authorized")
    
    with pytest.raises(GeminiAuthError):
        client.generate_zettelkasten_notes("Sample text")

def test_generate_zettelkasten_notes_rate_limit_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    client.client = MagicMock()
    client.client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
    
    with pytest.raises(GeminiRateLimitError):
        client.generate_zettelkasten_notes("Sample text")

def test_generate_zettelkasten_notes_generic_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    client.client = MagicMock()
    client.client.models.generate_content.side_effect = Exception("500 Internal Server Error")
    
    with pytest.raises(GeminiApiError):
        client.generate_zettelkasten_notes("Sample text")

def test_parse_notes_json_trailing_garbage():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = '[{"general_title": "Topic", "title": "Note with trailing", "content": "Text", "connections": []}]\nHope this helps!'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Note with trailing"

def test_parse_notes_json_unescaped_backslashes():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = r'[{"general_title": "Math", "title": "Formula", "content": "Formula: \frac{a}{b} and \alpha", "connections": []}]'
    notes = client._parse_notes_json(raw)
    assert len(notes) == 1
    assert notes[0]['title'] == "Formula"

def test_generate_zettelkasten_notes_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    client = GeminiApiClient()
    client.client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = ""
    client.client.models.generate_content.return_value = mock_resp
    
    res = client.generate_zettelkasten_notes("Valid text")
    assert res == []

