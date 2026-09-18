import pytest
import os
import sys
import threading
from unittest.mock import MagicMock
import flet as ft

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from settings_manager import SettingsManager
from database_manager import DatabaseManager
import prompt_templates
from app_controller import AppController
from app_state import AppState
from note_service import NoteService


@pytest.fixture
def temp_settings_manager(tmp_path):
    db_path = str(tmp_path / "test_settings.db")
    mgr = SettingsManager(db_path=db_path)
    yield mgr
    mgr.close_connection()


def test_settings_manager_crud(temp_settings_manager):
    assert temp_settings_manager.get_setting("CUSTOM_KEY") is None
    assert temp_settings_manager.get_setting("CUSTOM_KEY", default="fallback") == "fallback"

    temp_settings_manager.set_setting("CUSTOM_KEY", "custom_val")
    assert temp_settings_manager.get_setting("CUSTOM_KEY") == "custom_val"

    all_settings = temp_settings_manager.get_all_settings()
    assert all_settings.get("CUSTOM_KEY") == "custom_val"

    temp_settings_manager.remove_setting("CUSTOM_KEY")
    assert temp_settings_manager.get_setting("CUSTOM_KEY") is None


def test_settings_manager_defaults_and_reset(temp_settings_manager):
    defaults = temp_settings_manager.get_defaults()
    assert defaults["CONFIRM_DELETE_NOTE"] == "True"
    assert defaults["CONFIRM_DELETE_COLLECTION"] == "True"
    assert defaults["AI_PROVIDER"] == "gemini"
    assert defaults["UI_THEME"] == "Dark"

    # Mutate settings
    temp_settings_manager.set_setting("CONFIRM_DELETE_NOTE", "False")
    temp_settings_manager.set_setting("AI_PROVIDER", "local")
    temp_settings_manager.set_setting("CUSTOM_PARAM", "123")
    assert temp_settings_manager.get_setting("CONFIRM_DELETE_NOTE") == "False"
    assert temp_settings_manager.get_setting("CUSTOM_PARAM") == "123"

    # Reset to defaults
    temp_settings_manager.reset_to_defaults()
    assert temp_settings_manager.get_setting("CONFIRM_DELETE_NOTE") == "True"
    assert temp_settings_manager.get_setting("AI_PROVIDER") == "gemini"
    assert temp_settings_manager.get_setting("CUSTOM_PARAM") is None


def test_settings_manager_thread_safety(temp_settings_manager):
    errors = []

    def worker_thread(thread_id: int):
        try:
            for i in range(10):
                temp_settings_manager.set_setting(f"thread_{thread_id}_key_{i}", f"val_{i}")
                val = temp_settings_manager.get_setting(f"thread_{thread_id}_key_{i}")
                if val != f"val_{i}":
                    errors.append(f"Mismatch in thread {thread_id}: {val} != val_{i}")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker_thread, args=(tid,)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    all_settings = temp_settings_manager.get_all_settings()
    assert len(all_settings) == 50


def test_database_manager_delegation_to_settings(tmp_path):
    notes_db = str(tmp_path / "custom_notes.db")
    settings_db = str(tmp_path / "custom_settings.db")
    sm = SettingsManager(db_path=settings_db)
    db = DatabaseManager(db_path=notes_db, settings_manager=sm)

    assert db.get_setting("UI_THEME") is None
    db.set_setting("UI_THEME", "Light")
    assert db.get_setting("UI_THEME") == "Light"
    assert sm.get_setting("UI_THEME") == "Light"

    db.close_connection()


def test_custom_system_prompt_precedence_in_template():
    custom_instruction = "Always format notes with bullet points and ignore markdown headers."
    prompt = prompt_templates.build_note_extraction_prompt("Some source text", custom_system_prompt=custom_instruction)

    assert custom_instruction in prompt
    assert "PRECEDENCE & CONFLICT RESOLUTION RULE" in prompt
    assert "strictly prevail" in prompt
    assert "disregarded" in prompt


def test_delete_note_confirmation_setting_bypass(tmp_path):
    notes_db = str(tmp_path / "test_notes.db")
    settings_db = str(tmp_path / "test_settings.db")
    sm = SettingsManager(db_path=settings_db)
    db = DatabaseManager(db_path=notes_db, settings_manager=sm)

    state = AppState(theme_mode="Dark", auto_save=True)
    page = MagicMock()
    note_svc = NoteService(db)
    dialog_mgr = MagicMock()
    controller = AppController(
        page=page,
        db_manager=db,
        note_service=note_svc,
        dialog_manager=dialog_mgr,
        state=state
    )
    controller.attach_views(
        sidebar=MagicMock(),
        right_panel=MagicMock(),
        editor_workspace=MagicMock(),
        note_title_text=MagicMock(),
        collection_chip=MagicMock(),
        theme_btn=MagicMock(),
        auto_save_switch=MagicMock(),
        pdf_file_picker=MagicMock()
    )

    db.insert_note("n-1", "Test Note", "Content", "General")
    state.notes = [note_svc.get_note("n-1")]
    state.active_note_id = "n-1"

    # By default, CONFIRM_DELETE_NOTE is True -> opens dialog, does not delete immediately
    controller.confirm_or_delete_note("n-1", "Test Note")
    dialog_mgr.show_delete_note_confirm.assert_called_once()
    assert note_svc.get_note("n-1") is not None

    # Set CONFIRM_DELETE_NOTE to False -> bypasses dialog, deletes immediately
    dialog_mgr.show_delete_note_confirm.reset_mock()
    sm.set_setting("CONFIRM_DELETE_NOTE", "False")
    controller.confirm_or_delete_note("n-1", "Test Note")
    dialog_mgr.show_delete_note_confirm.assert_not_called()
    assert note_svc.get_note("n-1") is None

    db.close_connection()


def test_delete_collection_confirmation_setting_bypass(tmp_path):
    notes_db = str(tmp_path / "test_notes2.db")
    settings_db = str(tmp_path / "test_settings2.db")
    sm = SettingsManager(db_path=settings_db)
    db = DatabaseManager(db_path=notes_db, settings_manager=sm)

    state = AppState(theme_mode="Dark", auto_save=True)
    page = MagicMock()
    note_svc = NoteService(db)
    dialog_mgr = MagicMock()
    controller = AppController(
        page=page,
        db_manager=db,
        note_service=note_svc,
        dialog_manager=dialog_mgr,
        state=state
    )
    controller.attach_views(
        sidebar=MagicMock(),
        right_panel=MagicMock(),
        editor_workspace=MagicMock(),
        note_title_text=MagicMock(),
        collection_chip=MagicMock(),
        theme_btn=MagicMock(),
        auto_save_switch=MagicMock(),
        pdf_file_picker=MagicMock()
    )

    db.insert_note("n-2", "Project Note", "Content", "ProjectX")
    state.notes = [note_svc.get_note("n-2")]
    state.collections = ["ProjectX"]
    state.selected_collection_filter = "ProjectX"

    # When CONFIRM_DELETE_COLLECTION is True (default) -> opens confirmation dialog
    controller.handle_delete_collection_click()
    dialog_mgr.show_delete_collection_confirm.assert_called_once()

    # When CONFIRM_DELETE_COLLECTION is False -> bypasses dialog
    dialog_mgr.show_delete_collection_confirm.reset_mock()
    sm.set_setting("CONFIRM_DELETE_COLLECTION", "False")
    controller.handle_delete_collection_click()
    dialog_mgr.show_delete_collection_confirm.assert_not_called()

    db.close_connection()


def test_settings_dialog_auto_save_and_close(tmp_path):
    settings_db = str(tmp_path / "test_autosave_settings.db")
    sm = SettingsManager(db_path=settings_db)
    notes_db = str(tmp_path / "test_notes.db")
    db = DatabaseManager(db_path=notes_db, settings_manager=sm)

    mock_page = MagicMock()
    mock_page.overlay = []
    from ui.dialog_manager import DialogManager

    dm = DialogManager(mock_page)

    # Controller wiring
    state = AppState(theme_mode="Dark", auto_save=True)
    note_svc = NoteService(db)
    controller = AppController(
        page=mock_page,
        db_manager=db,
        note_service=note_svc,
        dialog_manager=dm,
        state=state
    )

    dm.show_settings_dialog(
        theme_btn=ft.IconButton(icon=ft.Icons.DARK_MODE),
        auto_save_switch=ft.Switch(),
        current_ai_provider="gemini",
        current_active_model_id="gemma-4-e2b",
        on_setting_changed=controller.handle_setting_change,
        on_open_model_manager=lambda: None,
        current_gpu_acceleration=True,
    )
    assert dm.settings_dialog.open is True

    # 1. Verify single "Close" button in actions
    assert len(dm.settings_dialog.actions) == 1
    close_btn = dm.settings_dialog.actions[0]

    # 2. Simulate user editing setting (auto-save without Save button)
    controller.handle_setting_change("CONFIRM_DELETE_NOTE", "False")
    assert sm.get_setting("CONFIRM_DELETE_NOTE") == "False"
    assert db.get_setting("CONFIRM_DELETE_NOTE") == "False"

    controller.handle_setting_change("GEMINI_API_KEY", "new-api-key-12345")
    assert sm.get_setting("GEMINI_API_KEY") == "new-api-key-12345"

    # 3. Simulate clicking Close button -> closes dialog
    close_btn.on_click(None)
    assert dm.settings_dialog.open is False

    db.close_connection()

