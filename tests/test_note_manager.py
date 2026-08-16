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

    # Leading blank lines and deeper heading levels
    multiline = "\n\n  \n### Deep Header Note\nSome content here"
    assert note_manager.get_sanitized_title(multiline) == "Deep Header Note"

    # Empty content
    empty = note_manager.get_sanitized_title("")
    assert empty == "Untitled Note"

    # Whitespace only content
    whitespace_only = note_manager.get_sanitized_title("   \n\n  \t ")
    assert whitespace_only == "Untitled Note"

    # Content with markdown symbols stripped down to nothing
    symbols_only = note_manager.get_sanitized_title("# **`~~`**")
    assert symbols_only == "Untitled Note"

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

def test_save_note_duplicate_titles_creates_distinct_notes(temp_db):
    content1 = "# Duplicate Title\nContent 1"
    content2 = "# Duplicate Title\nContent 2"

    note_id_1, title1 = note_manager.save_note(temp_db, None, content1, "General")
    note_id_2, title2 = note_manager.save_note(temp_db, None, content2, "General")

    assert note_id_1 != note_id_2
    assert title1 == "Duplicate Title"
    assert title2 == "Duplicate Title"

    # Verify both notes exist in the DB independently
    assert note_manager.get_note_content(temp_db, note_id_1) == content1
    assert note_manager.get_note_content(temp_db, note_id_2) == content2

def test_save_note_update_with_explicit_id(temp_db):
    content = "# Original Title\nInitial body"
    note_id, title = note_manager.save_note(temp_db, None, content, "Cat1")

    updated_content = "# Updated Title\nUpdated body"
    updated_id, updated_title = note_manager.save_note(temp_db, note_id, updated_content, "Cat2")

    assert updated_id == note_id
    assert updated_title == "Updated Title"
    assert note_manager.get_note_content(temp_db, note_id) == updated_content

def test_create_category():
    assert note_manager.create_category("Science") is True
    assert note_manager.create_category("   ") is False
    assert note_manager.create_category("") is False

def test_load_all_notes_metadata(temp_db):
    note_manager.save_note(temp_db, None, "# Note A\nContent", "Category 1")
    note_manager.save_note(temp_db, None, "# Note B\nContent", "Category 2")
    note_manager.save_note(temp_db, None, "# Note C\nContent", "")

    notes_meta, categories = note_manager.load_all_notes_metadata(temp_db)
    assert len(notes_meta) == 3
    assert "Category 1" in categories
    assert "Category 2" in categories
    assert categories == sorted(categories)


