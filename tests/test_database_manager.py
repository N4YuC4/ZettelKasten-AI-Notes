import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from database_manager import DatabaseManager
import database_manager

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_db.db"
    database_manager.DATABASE_FILE = str(db_path)
    db = DatabaseManager()
    yield db
    db.close_connection()

def test_settings_crud(temp_db):
    assert temp_db.get_setting("UI_THEME") is None
    temp_db.set_setting("UI_THEME", "Dark")
    assert temp_db.get_setting("UI_THEME") == "Dark"

def test_notes_crud_and_links(temp_db):
    temp_db.insert_note("id-1", "Note 1", "Content 1", "Category A")
    temp_db.insert_note("id-2", "Note 2", "Content 2", "Category A")

    assert temp_db.note_count("Category A") == 2
    assert temp_db.get_note_id_by_title("Note 1") == "id-1"

    # Links
    link_success = temp_db.insert_note_link("id-1", "id-2")
    assert link_success is True

    links = temp_db.get_note_links("id-1")
    assert "id-2" in links

    all_links = temp_db.get_all_note_links()
    assert ("id-1", "id-2") in all_links

    # Deleting note cascaded link
    temp_db.delete_note("id-1")
    assert temp_db.get_note("id-1") is None
    assert temp_db.get_note_links("id-2") == []
