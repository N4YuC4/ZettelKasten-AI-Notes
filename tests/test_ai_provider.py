import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_provider import BaseAiProvider, create_ai_provider
from gemini_api_client import GeminiApiClient
from local_gguf_client import LocalGgufClient


class DummyProvider(BaseAiProvider):
    def generate_zettelkasten_notes(self, text_content, on_progress=None):
        return [{"title": "Test", "content": text_content, "connections": []}]


def test_base_ai_provider_interface():
    provider = DummyProvider()
    notes = provider.generate_zettelkasten_notes("Sample text")
    assert len(notes) == 1
    assert notes[0]["title"] == "Test"

    # Default generate_note_links should return notes unchanged
    linked = provider.generate_note_links(notes)
    assert linked == notes


def test_create_ai_provider_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    provider = create_ai_provider("gemini")
    assert isinstance(provider, BaseAiProvider)
    assert isinstance(provider, GeminiApiClient)


def test_create_ai_provider_local(monkeypatch, tmp_path):
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"dummy")

    mock_llm = MagicMock()
    monkeypatch.setattr("local_gguf_client.LocalGgufClient._get_or_load_model", lambda self: mock_llm)
    monkeypatch.setattr("hardware_checker.HardwareChecker.check_file_compatibility", lambda path: (True, "OK", "success"))

    provider = create_ai_provider("local", model_path=str(dummy_model), n_gpu_layers=0)
    assert isinstance(provider, BaseAiProvider)
    assert isinstance(provider, LocalGgufClient)
