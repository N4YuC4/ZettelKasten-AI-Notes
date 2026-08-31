# test_models_and_service.py
#
# Unit tests for Domain Models, NoteService, and AppState.

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models import Note, NoteMetadata, NoteLink, GeneratedAiNote, DocumentStats
import note_service
from note_service import NoteService
from app_state import AppState
import database_manager


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_service_db.db"
    database_manager.DATABASE_FILE = str(db_path)
    db = database_manager.DatabaseManager()
    yield db
    db.close_connection()


def test_models_creation_and_properties():
    # Note and NoteMetadata
    note = Note(
        id="test-1",
        title="Test Title",
        content="# Test Title\nBody",
        category="General"
    )
    assert note.id == "test-1"
    assert note.metadata.id == "test-1"
    assert note.metadata.title == "Test Title"
    assert note.metadata.category == "General"
    assert note.metadata.to_tuple() == ("test-1", "Test Title", "General")

    db_tuple = note.to_db_tuple()
    assert len(db_tuple) == 6
    assert db_tuple[0] == "test-1"
    assert db_tuple[1] == "Test Title"

    # NoteLink
    link = NoteLink(source_note_id="n1", target_note_id="n2")
    assert link.to_tuple() == ("n1", "n2")

    # GeneratedAiNote
    ai_note = GeneratedAiNote(
        title="AI Title",
        content="AI Content",
        general_title="Topic",
        connections=["Link 1"]
    )
    assert ai_note.title == "AI Title"
    assert ai_note.connections == ["Link 1"]

    # DocumentStats
    stats = DocumentStats(words=10, chars=50, lines=2, reading_time_min=1)
    assert stats.words == 10
    assert stats["words"] == 10
    assert stats["chars"] == 50
    assert stats.to_dict()["words"] == 10


def test_disambiguate_title():
    existing = {"Summary", "Summary (2)", "Introduction"}
    assert note_service.disambiguate_title("New Topic", existing) == "New Topic"
    assert note_service.disambiguate_title("Summary", existing) == "Summary (3)"


def test_note_service_crud_and_links(temp_db):
    service = NoteService(temp_db)

    # Save new note
    nid, title = service.save_note(None, "# Atomic Note\nNote details", "Science")
    assert title == "Atomic Note"
    assert nid is not None

    # Retrieve content
    content = service.get_note_content(nid)
    assert content == "# Atomic Note\nNote details"

    # Rename note
    success, new_title = service.rename_note(nid, "Updated Atomic Note", "Science")
    assert success is True
    assert new_title == "Updated Atomic Note"
    assert service.get_note_content(nid).startswith("# Updated Atomic Note")

    # Rename note preserving category when category is None
    success, new_title = service.rename_note(nid, "Renamed Preserving Category")
    assert success is True
    note_model = service.get_note(nid)
    assert note_model.category == "Science"
    assert service.get_note_content(nid).startswith("# Renamed Preserving Category")

    # Rename non-existent note
    fail_success, fail_msg = service.rename_note("fake-id", "New Title")
    assert fail_success is False
    assert "not found" in fail_msg.lower()

    # Rename to empty
    fail_empty, fail_empty_msg = service.rename_note(nid, "   ")
    assert fail_empty is False

    # Second note for linking
    nid2, _ = service.save_note(None, "# Connected Note\nBody 2", "Science")

    # Create Link
    assert service.create_link(nid, nid2) is True
    assert service.create_link(nid, nid) is False  # Self link rejected
    assert nid2 in service.get_linked_note_ids(nid)

    # Delete Link
    assert service.delete_link(nid, nid2) is True
    assert nid2 not in service.get_linked_note_ids(nid)

    # Load all metadata
    meta, cats = service.load_all_notes_metadata()
    assert len(meta) == 2
    assert "Science" in cats

    # Delete category
    assert service.delete_category("Science") is True
    assert service.delete_category("") is False
    meta_after, _ = service.load_all_notes_metadata()
    assert len(meta_after) == 0


def test_sanitize_title_preserves_link_text():
    assert note_service.sanitize_title("# [Python 3.13 Guide](https://python.org)") == "Python 3.13 Guide"
    assert note_service.sanitize_title("# [[Quantum Computing]]") == "Quantum Computing"
    assert note_service.sanitize_title("# [[Target Note|Custom Alias]]") == "Custom Alias"
    assert note_service.sanitize_title("# ![Image](pic.png) Actual Heading") == "Actual Heading"
    assert note_service.sanitize_title("") == "Untitled Note"
    assert note_service.sanitize_title("   ") == "Untitled Note"


def test_app_state_listeners():
    state = AppState(theme_mode="Dark", auto_save=True)
    events = []

    def on_change(s: AppState):
        events.append((s.current_note_id, s.is_dirty, s.selected_category_filter))

    state.add_listener(on_change)

    state.select_note("n1", "Title 1", "Cat 1")
    assert len(events) == 1
    assert events[-1] == ("n1", False, "")

    state.set_dirty(True)
    assert len(events) == 2
    assert events[-1] == ("n1", True, "")

    state.set_category_filter("Cat 1")
    assert len(events) == 3
    assert events[-1] == ("n1", True, "Cat 1")

    state.set_search_query("query")
    assert state.search_query == "query"

    state.set_auto_save(False)
    assert state.auto_save is False

    state.set_theme_mode("Light")
    assert state.theme_mode == "Light"

    state.remove_listener(on_change)
    state.set_dirty(False)
    assert len(events) == 6  # No additional events after removal


def test_process_markdown_wikilinks_ignores_code_blocks_and_inline_code():
    text = (
        "Here is a note with [[Normal WikiLink]].\n\n"
        "```python\n"
        "# This is code with [[Code Block WikiLink]]\n"
        "matrix[[0]]\n"
        "```\n\n"
        "And inline code `[[Inline WikiLink]]` should not change.\n"
        "Finally [[Another WikiLink|Alias]]."
    )
    processed = note_service.process_markdown_wikilinks(text)
    assert "[🔗 Normal WikiLink](zettel://note/Normal%20WikiLink)" in processed
    assert "[🔗 Alias](zettel://note/Another%20WikiLink)" in processed
    # Verify code block was not mutated
    assert "[[Code Block WikiLink]]" in processed
    assert "matrix[[0]]" in processed
    assert "`[[Inline WikiLink]]`" in processed


def test_preserve_single_linebreaks_preserves_tables_and_headings():
    table_text = (
        "# Main Heading\n"
        "| Header 1 | Header 2 |\n"
        "| :--- | :--- |\n"
        "| Cell A | Cell B |\n\n"
        "Normal line 1\n"
        "Normal line 2"
    )
    res = note_service.preserve_single_linebreaks(table_text)
    lines = res.split("\n")
    assert lines[0] == "# Main Heading"  # No trailing double spaces on headings
    assert lines[1] == "| Header 1 | Header 2 |"  # No trailing double spaces on tables
    assert lines[2] == "| :--- | :--- |"
    assert lines[3] == "| Cell A | Cell B |"
    assert lines[5] == "Normal line 1  "


def test_rename_note_with_none_or_leading_blank_lines(temp_db):
    service = NoteService(temp_db)
    # 1. Test None content in DB
    cursor = temp_db.conn.cursor()
    cursor.execute(
        "INSERT INTO notes (id, title, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("none-id", "None Title", None, "General", "2026-08-31", "2026-08-31")
    )
    temp_db.conn.commit()

    success, new_title = service.rename_note("none-id", "Fixed None Title")
    assert success is True
    assert new_title == "Fixed None Title"
    assert service.get_note_content("none-id") == "# Fixed None Title"

    # 2. Test leading blank lines
    nid, _ = service.save_note(None, "\n\n  \n# Original Header\nSome details here", "General")
    success2, new_title2 = service.rename_note(nid, "New Clean Header")
    assert success2 is True
    content2 = service.get_note_content(nid)
    lines2 = content2.splitlines()
    assert lines2[3] == "# New Clean Header"
    assert "Original Header" not in content2


