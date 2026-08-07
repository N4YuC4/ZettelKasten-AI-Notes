import pytest
import os
import sys

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import note_manager
from database_manager import DatabaseManager

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_notes.db"
    import database_manager
    database_manager.DATABASE_FILE = str(db_path)
    db = DatabaseManager()
    yield db
    db.close_connection()

def test_generate_unique_id():
    uid1 = note_manager.generate_unique_id()
    uid2 = note_manager.generate_unique_id()
    assert isinstance(uid1, str)
    assert len(uid1) == 36
    assert uid1 != uid2

def test_get_sanitized_title():
    raw_markdown = "# **My Title** with [link](http://example.com) & `code`"
    cleaned = note_manager.get_sanitized_title(raw_markdown)
    assert "My Title" in cleaned

    empty = note_manager.get_sanitized_title("")
    assert empty == "Untitled Note"

def test_save_and_rename_note(temp_db):
    content = "# Test Note\n\nThis is a test content."
    note_id, title = note_manager.save_note(temp_db, None, content, "General")
    assert title == "Test Note"
    assert note_id is not None

    retrieved_content = note_manager.get_note_content(temp_db, note_id)
    assert retrieved_content == content

    success, new_title = note_manager.rename_note(temp_db, note_id, "Renamed Title", "General")
    assert success is True
    assert new_title == "Renamed Title"

    note_manager.delete_note(temp_db, note_id)
    assert note_manager.get_note_content(temp_db, note_id) is None
