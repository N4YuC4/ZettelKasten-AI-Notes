import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gemini_api_client import GeminiApiClient

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

def test_parse_notes_json_invalid():
    client = GeminiApiClient.__new__(GeminiApiClient)
    raw = 'This is invalid text with no json'
    notes = client._parse_notes_json(raw)
    assert notes == []
