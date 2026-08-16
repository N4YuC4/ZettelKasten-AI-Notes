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

def test_pragmas_and_indexes(temp_db):
    cursor = temp_db.conn.cursor()

    # Verify pragmas
    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.upper() == "WAL"

    cursor.execute("PRAGMA foreign_keys;")
    foreign_keys = cursor.fetchone()[0]
    assert foreign_keys == 1

    cursor.execute("PRAGMA busy_timeout;")
    busy_timeout = cursor.fetchone()[0]
    assert busy_timeout == 5000

    # Verify secondary indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_notes_category" in indexes
    assert "idx_links_source" in indexes
    assert "idx_links_target" in indexes

def test_delete_note_link_bidirectional(temp_db):
    temp_db.insert_note("n1", "Note 1", "Body 1", "Cat")
    temp_db.insert_note("n2", "Note 2", "Body 2", "Cat")

    # Link from n1 to n2
    temp_db.insert_note_link("n1", "n2")
    assert "n2" in temp_db.get_note_links("n1")
    assert "n1" in temp_db.get_note_links("n2")

    # Delete specifying reversed order (n2, n1)
    deleted = temp_db.delete_note_link("n2", "n1")
    assert deleted is True
    assert temp_db.get_note_links("n1") == []
    assert temp_db.get_note_links("n2") == []

def test_delete_category_cascades_notes_and_links(temp_db):
    # Setup notes across categories
    temp_db.insert_note("cat-note-1", "Cat Note 1", "Body 1", "ProjectX")
    temp_db.insert_note("cat-note-2", "Cat Note 2", "Body 2", "ProjectX")
    temp_db.insert_note("other-note", "Other Note", "Body 3", "General")

    # Links:
    # 1. Inside ProjectX (cat-note-1 -> cat-note-2)
    # 2. Between categories (cat-note-1 -> other-note)
    # 3. Between categories reversed (other-note -> cat-note-2)
    temp_db.insert_note_link("cat-note-1", "cat-note-2")
    temp_db.insert_note_link("cat-note-1", "other-note")
    temp_db.insert_note_link("other-note", "cat-note-2")

    # Delete category ProjectX
    success = temp_db.delete_category("ProjectX")
    assert success is True

    # Notes under ProjectX are deleted
    assert temp_db.get_note("cat-note-1") is None
    assert temp_db.get_note("cat-note-2") is None
    assert temp_db.note_count("ProjectX") == 0

    # Note in General remains
    assert temp_db.get_note("other-note") is not None
    assert temp_db.note_count("General") == 1

    # All links associated with ProjectX notes are deleted
    assert temp_db.get_note_links("other-note") == []
    assert temp_db.get_all_note_links() == []

def test_bulk_operations(temp_db):
    now = "2026-08-17T00:00:00"
    notes_data = [
        ("bulk-1", "Bulk Note 1", "Content 1", "BulkCat", now, now),
        ("bulk-2", "Bulk Note 2", "Content 2", "BulkCat", now, now),
        ("bulk-3", "Bulk Note 3", "Content 3", "BulkCat", now, now),
    ]
    temp_db.bulk_insert_notes(notes_data)
    assert temp_db.note_count("BulkCat") == 3

    links_data = [
        ("bulk-1", "bulk-2"),
        ("bulk-2", "bulk-3"),
        ("bulk-1", "bulk-2") # Duplicate should be ignored
    ]
    temp_db.bulk_insert_links(links_data)
    all_links = temp_db.get_all_note_links()
    assert ("bulk-1", "bulk-2") in all_links
    assert ("bulk-2", "bulk-3") in all_links
    assert len(all_links) == 2

def test_get_all_notes_metadata(temp_db):
    temp_db.insert_note("m1", "Metadata Note 1", "Content 1", "CatAlpha")
    temp_db.insert_note("m2", "Metadata Note 2", "Content 2", "CatBeta")
    temp_db.insert_note("m3", "Metadata Note 3", "Content 3", "")

    metadata, categories = temp_db.get_all_notes_metadata()
    assert len(metadata) == 3
    assert "CatAlpha" in categories
    assert "CatBeta" in categories
    assert "" not in categories


