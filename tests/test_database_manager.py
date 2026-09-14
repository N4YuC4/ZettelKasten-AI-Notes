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
    assert "idx_notes_collection" in indexes
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


def test_delete_category_subquery_scalability(temp_db):
    now = "2026-08-31T00:00:00"
    # Insert 50 notes under "BigCat"
    notes = [(f"big-{i}", f"Big Note {i}", f"Content {i}", "BigCat", now, now) for i in range(50)]
    links = [(f"big-{i}", f"big-{i+1}") for i in range(49)]
    temp_db.bulk_insert_notes_and_links(notes, links)

    assert temp_db.note_count("BigCat") == 50
    assert len(temp_db.get_all_note_links()) == 49

    success = temp_db.delete_category("BigCat")
    assert success is True
    assert temp_db.note_count("BigCat") == 0
    assert len(temp_db.get_all_note_links()) == 0


def test_database_manager_recovers_when_db_file_deleted_from_disk(temp_db):
    temp_db.insert_note("rec-1", "Note 1", "Content", "Cat")
    assert temp_db.note_count() == 1
    assert os.path.exists(temp_db.db_path)

    # Delete database file from disk while connection is open
    os.remove(temp_db.db_path)
    assert not os.path.exists(temp_db.db_path)

    # Next call should detect file deletion, close stale connection, reconnect and ensure schema
    assert temp_db.note_count() == 0
    assert os.path.exists(temp_db.db_path)

    # Can insert and query immediately without 'no such table' error
    temp_db.insert_note("rec-2", "Note 2", "Content 2", "Cat")
    assert temp_db.note_count() == 1
    assert temp_db.get_note("rec-2") is not None


def test_sqlite_category_to_collection_migration(tmp_path):
    import sqlite3
    legacy_db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy_db_file))
    cur = conn.cursor()
    # Create legacy schema with category column and idx_notes_category index
    cur.execute("""
        CREATE TABLE notes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            category TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_notes_category ON notes(category);")
    cur.execute("""
        CREATE TABLE note_links (
            source_note_id TEXT NOT NULL,
            target_note_id TEXT NOT NULL,
            PRIMARY KEY (source_note_id, target_note_id)
        )
    """)
    cur.execute("""
        INSERT INTO notes (id, title, content, category, created_at, updated_at)
        VALUES ('legacy-1', 'Legacy Title', 'Legacy Content', 'OldCategory', '2026-01-01', '2026-01-01')
    """)
    conn.commit()
    conn.close()

    # Now open with DatabaseManager
    database_manager.DATABASE_FILE = str(legacy_db_file)
    db = DatabaseManager()
    try:
        cur = db.conn.cursor()
        cur.execute("PRAGMA table_info(notes);")
        columns = [row[1] for row in cur.fetchall()]
        assert "collection" in columns
        assert "category" not in columns

        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_notes_collection" in indexes
        assert "idx_notes_category" not in indexes

        # Verify data preserved
        note = db.get_note_model("legacy-1")
        assert note is not None
        assert note.collection == "OldCategory"
        assert note.category == "OldCategory"

        # Verify collection methods work
        assert db.delete_collection("OldCategory") is True
        assert db.get_note("legacy-1") is None
    finally:
        db.close_connection()
