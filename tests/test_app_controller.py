import pytest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import flet as ft
from app_state import AppState
from app_controller import AppController
from note_service import NoteService
import database_manager


@pytest.fixture
def test_setup(tmp_path):
    db_path = tmp_path / "test_controller.db"
    database_manager.DATABASE_FILE = str(db_path)
    db = database_manager.DatabaseManager(init_tables=True)
    note_service = NoteService(db)
    state = AppState(theme_mode="Dark", auto_save=True)

    page = MagicMock(spec=ft.Page)
    page.theme_mode = ft.ThemeMode.DARK
    page.overlay = []

    dialog_manager = MagicMock()

    controller = AppController(
        page=page,
        db_manager=db,
        note_service=note_service,
        dialog_manager=dialog_manager,
        state=state
    )

    sidebar = MagicMock()
    right_panel = MagicMock()
    editor_workspace = MagicMock()
    note_title_text = MagicMock()
    collection_chip = MagicMock()
    theme_btn = MagicMock()
    auto_save_switch = MagicMock()
    pdf_file_picker = MagicMock()

    controller.attach_views(
        sidebar=sidebar,
        right_panel=right_panel,
        editor_workspace=editor_workspace,
        note_title_text=note_title_text,
        collection_chip=collection_chip,
        theme_btn=theme_btn,
        auto_save_switch=auto_save_switch,
        pdf_file_picker=pdf_file_picker
    )
    assert controller.collection_chip is collection_chip
    assert controller.category_chip is collection_chip

    yield controller, db, note_service, state, dialog_manager
    db.close_connection()


def test_controller_new_note_and_save(test_setup):
    controller, db, note_service, state, _ = test_setup

    controller.new_note_internal(initial_title="Alpha Note", initial_content="# Alpha Note\n\nContent here")
    assert state.current_note_id is not None
    assert state.current_note_title == "Alpha Note"

    # Modify content and save
    controller.editor_workspace.get_content.return_value = "# Alpha Note (Updated)\n\nNew body"
    controller.save_current_note()
    assert state.current_note_title == "Alpha Note (Updated)"
    assert state.is_dirty is False


def test_controller_collection_create_and_delete(test_setup):
    controller, db, note_service, state, dialog_manager = test_setup

    controller.handle_create_collection("Science")
    assert state.selected_collection_filter == "Science"
    assert state.selected_category_filter == "Science"
    assert "Science" in state.all_collections
    assert "Science" in state.all_categories

    # Prevent duplicate collection creation
    controller.handle_create_collection("Science")
    # Prevent reserved collection name
    controller.handle_create_collection("All Notes")

    # Confirmed delete
    controller.handle_delete_collection_confirmed("Science")
    assert state.selected_collection_filter == ""
    assert state.selected_category_filter == ""

    # Test backward compatibility aliases
    controller.handle_create_category("History")
    assert state.selected_collection_filter == "History"
    controller.handle_delete_category_confirmed("History")
    assert state.selected_collection_filter == ""


def test_controller_rename_and_delete_note(test_setup):
    controller, db, note_service, state, _ = test_setup

    controller.new_note_internal("Original Title", "# Original Title\n\nBody")
    note_id = state.current_note_id

    controller.handle_rename_note(note_id, "Renamed Title")
    assert state.current_note_title == "Renamed Title"

    # Delete note
    controller.handle_delete_note(note_id, "Renamed Title")
    all_notes, _ = note_service.load_all_notes_metadata()
    assert not any(nid == note_id for nid, _, _ in all_notes)


def test_controller_toggle_theme(test_setup):
    controller, db, _, _, _ = test_setup

    assert controller.page.theme_mode == ft.ThemeMode.DARK
    controller.toggle_theme()
    assert controller.page.theme_mode == ft.ThemeMode.LIGHT
    assert db.get_setting("UI_THEME") == "Light"

    controller.toggle_theme()
    assert controller.page.theme_mode == ft.ThemeMode.DARK
    assert db.get_setting("UI_THEME") == "Dark"


def test_controller_guard_unsaved_changes_auto_save_true(test_setup):
    controller, db, _, state, dialog_manager = test_setup

    controller.new_note_internal("Note A", "# Note A\n\nBody")
    controller.editor_workspace.get_content.return_value = "# Note A\n\nModified"
    state.set_dirty(True)
    state.set_auto_save(True)

    action_called = False
    def target_action():
        nonlocal action_called
        action_called = True

    controller.guard_unsaved_changes(target_action)
    assert action_called is True
    assert state.is_dirty is False
    dialog_manager.show_unsaved_changes_prompt.assert_not_called()


def test_controller_guard_unsaved_changes_auto_save_false(test_setup):
    controller, db, _, state, dialog_manager = test_setup

    controller.new_note_internal("Note B", "# Note B\n\nBody")
    state.set_dirty(True)
    state.set_auto_save(False)

    action_called = False
    def target_action():
        nonlocal action_called
        action_called = True

    controller.guard_unsaved_changes(target_action)
    assert action_called is False
    dialog_manager.show_unsaved_changes_prompt.assert_called_once()


def test_controller_handle_ai_finished_flow(test_setup):
    controller, db, note_service, state, dialog_manager = test_setup

    with patch("time.sleep") as mock_sleep, patch.object(controller, "show_snack_bar") as mock_snack:
        controller.handle_ai_finished([{"title": "Note 1"}])
        dialog_manager.hide_loading.assert_called_once()
        mock_sleep.assert_called_with(0.35)
        mock_snack.assert_called_once_with("1 notes successfully generated and saved!")


def test_controller_cancel_worker_flow(test_setup):
    controller, db, note_service, state, dialog_manager = test_setup

    mock_worker = MagicMock()
    controller.active_worker = mock_worker

    with patch("time.sleep") as mock_sleep, patch.object(controller, "show_snack_bar") as mock_snack:
        controller.cancel_worker()
        mock_worker.cancel.assert_called_once()
        assert controller.active_worker is None
        dialog_manager.hide_loading.assert_called_once()
        mock_sleep.assert_called_with(0.35)
        mock_snack.assert_called_once()


def test_controller_handle_rename_note_preserves_content_structure(test_setup):
    """Verifies that handle_rename_note accurately updates editor workspace content without duplicate headers."""
    controller, db, note_service, state, dialog_manager = test_setup

    initial_content = "\n\n# Original Title\n\nSome body text."
    controller.new_note_internal("Original Title", initial_content)
    note_id = state.current_note_id
    assert note_id is not None

    with patch.object(controller, "show_snack_bar"):
        controller.handle_rename_note(note_id, "Renamed Title")

    assert state.current_note_title == "Renamed Title"
    assert controller.editor_workspace.set_content.call_count == 2
    saved_arg = controller.editor_workspace.set_content.call_args[0][0]
    assert "# Renamed Title" in saved_arg
    assert "# Original Title" not in saved_arg
    # Ensure there is no duplicated title heading
    headings = [l for l in saved_arg.split('\n') if l.strip().startswith('# ')]
    assert len(headings) == 1
    assert headings[0] == "# Renamed Title"


