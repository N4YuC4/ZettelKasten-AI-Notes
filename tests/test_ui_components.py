# test_ui_components.py
#
# Unit tests for UI components (SidebarView, RightPanelView, EditorWorkspaceView, DialogManager, Splitters).

import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ui import SidebarView, RightPanelView, EditorWorkspaceView, DialogManager, create_vertical_splitter
import flet as ft


def test_sidebar_view_initialization_and_methods():
    clicked_note = []
    renamed_note = []
    deleted_note = []
    changed_cat = []
    new_cat = []
    del_cat = []
    searched = []
    settings = []

    sidebar = SidebarView(
        on_category_changed=lambda c: changed_cat.append(c),
        on_new_category_clicked=lambda: new_cat.append(True),
        on_delete_category_clicked=lambda: del_cat.append(True),
        on_search_changed=lambda q: searched.append(q),
        on_note_clicked=lambda nid, t, c: clicked_note.append((nid, t, c)),
        on_rename_note_clicked=lambda nid, t: renamed_note.append((nid, t)),
        on_delete_note_clicked=lambda nid, t: deleted_note.append((nid, t)),
        on_settings_clicked=lambda: settings.append(True),
    )

    assert sidebar.width == 300
    assert len(sidebar.collection_dropdown.options) == 1
    assert sidebar.collection_dropdown is sidebar.category_dropdown

    # Update collections (modern method)
    sidebar.update_collections(["Col 1", "Col 2"], selected_collection="Col 1")
    assert len(sidebar.collection_dropdown.options) == 3
    assert sidebar.collection_dropdown.value == "Col 1"

    # Backward compatibility: Update categories
    sidebar.update_categories(["Cat 1", "Cat 2"], selected_category="Cat 1")
    assert len(sidebar.category_dropdown.options) == 3
    assert sidebar.category_dropdown.value == "Cat 1"

    # Render notes with collection_name
    notes = [
        ("id1", "Note 1", "Cat 1"),
        ("id2", "Note 2", "Cat 1"),
    ]
    sidebar.render_notes(notes, current_note_id="id1", collection_name="Cat 1", total_count=2)
    assert len(sidebar.notes_listview.controls) == 2
    assert "Cat 1 count: 2" in sidebar.note_count_label.value

    # Test modern on_collection_changed instantiation
    col_changed = []
    sidebar_modern = SidebarView(
        on_collection_changed=lambda c: col_changed.append(c),
        on_search_changed=lambda q: None,
        on_note_clicked=lambda nid, t, c: None,
        on_rename_note_clicked=lambda nid, t: None,
        on_delete_note_clicked=lambda nid, t: None,
        on_settings_clicked=lambda: None,
    )
    assert sidebar_modern.collection_dropdown is not None


def test_right_panel_view_initialization_and_methods():
    selected_map = []
    clicked_linked = []
    unlinked = []

    mock_db = MagicMock()
    mock_db.get_all_note_links.return_value = []
    mock_db.get_all_notes_metadata.return_value = ([], set())

    panel = RightPanelView(
        db_manager=mock_db,
        on_map_note_selected=lambda nid: selected_map.append(nid),
        on_linked_note_clicked=lambda nid: clicked_linked.append(nid),
        on_unlink_note_clicked=lambda nid, t: unlinked.append((nid, t)),
    )

    assert panel.width == 350
    assert panel.mind_map_container is not None

    # Render linked notes when none
    panel.render_linked_notes("id1", [])
    assert len(panel.linked_notes_listview.controls) == 1
    assert "No linked notes" in panel.linked_notes_listview.controls[0].value

    # Render linked notes with items
    panel.render_linked_notes("id1", [("id2", "Linked Note 2")])
    assert len(panel.linked_notes_listview.controls) == 1


def test_editor_workspace_view():
    content_changes = []
    wikilinks = []
    new_note = []
    save_note = []
    del_note = []
    link_note = []
    gen_ai = []

    workspace = EditorWorkspaceView(
        on_content_change=lambda t: content_changes.append(t),
        on_wikilink_clicked=lambda t: wikilinks.append(t),
        get_all_notes_callback=lambda: [("1", "Title", "Cat")],
        on_new_note_clicked=lambda: new_note.append(True),
        on_save_note_clicked=lambda: save_note.append(True),
        on_delete_note_clicked=lambda: del_note.append(True),
        on_link_note_clicked=lambda: link_note.append(True),
        on_generate_ai_clicked=lambda: gen_ai.append(True),
    )

    assert workspace.expand is True
    assert len(workspace.action_buttons.controls) == 5

    # Set content
    workspace.set_content("# Test Content\nDetails", mark_dirty=True)
    assert workspace.get_content() == "# Test Content\nDetails"

    workspace.set_dirty(False)
    assert workspace.live_editor.dirty_indicator.value == ""


def test_dialog_manager_initialization():
    mock_page = MagicMock()
    mock_page.overlay = []

    dm = DialogManager(mock_page)
    assert len(mock_page.overlay) >= 10

    # Test error dialog trigger
    dm.show_error("Test Error", "An error occurred.")
    assert dm.error_dialog.open is True
    assert mock_page.update.called

    dm.close(dm.error_dialog)
    assert dm.error_dialog.open is False


def test_create_vertical_splitter():
    dragged = []
    splitter = create_vertical_splitter(on_drag=lambda e: dragged.append(e))
    assert splitter.mouse_cursor == ft.MouseCursor.RESIZE_LEFT_RIGHT


def test_sidebar_view_collapse_callback():
    collapsed = []
    sidebar = SidebarView(
        on_category_changed=lambda c: None,
        on_new_category_clicked=lambda: None,
        on_delete_category_clicked=lambda: None,
        on_search_changed=lambda q: None,
        on_note_clicked=lambda nid, t, c: None,
        on_rename_note_clicked=lambda nid, t: None,
        on_delete_note_clicked=lambda nid, t: None,
        on_settings_clicked=lambda: None,
        on_collapse_clicked=lambda: collapsed.append(True),
    )
    assert sidebar.on_collapse_clicked is not None
    # Trigger collapse callback
    sidebar.on_collapse_clicked()
    assert collapsed == [True]


def test_right_panel_view_collapse_callback():
    collapsed = []
    mock_db = MagicMock()
    mock_db.get_all_note_links.return_value = []
    mock_db.get_all_notes_metadata.return_value = ([], set())

    panel = RightPanelView(
        db_manager=mock_db,
        on_map_note_selected=lambda nid: None,
        on_linked_note_clicked=lambda nid: None,
        on_unlink_note_clicked=lambda nid, t: None,
        on_collapse_clicked=lambda: collapsed.append(True),
    )
    assert panel.on_collapse_clicked is not None
    # Trigger collapse callback
    panel.on_collapse_clicked()
    assert collapsed == [True]


def test_sidebar_view_collapsed_rail():
    sidebar = SidebarView(
        on_category_changed=lambda c: None,
        on_new_category_clicked=lambda: None,
        on_delete_category_clicked=lambda: None,
        on_search_changed=lambda q: None,
        on_note_clicked=lambda nid, t, c: None,
        on_rename_note_clicked=lambda nid, t: None,
        on_delete_note_clicked=lambda nid, t: None,
        on_settings_clicked=lambda: None,
    )
    assert sidebar.is_collapsed is False
    assert sidebar.width == 300

    # Collapse to narrow rail
    sidebar.set_collapsed(True)
    assert sidebar.is_collapsed is True
    assert sidebar.width == 50

    # Toggle back to expanded
    sidebar.toggle_collapsed()
    assert sidebar.is_collapsed is False
    assert sidebar.width == 300


def test_right_panel_view_collapsed_rail():
    mock_db = MagicMock()
    mock_db.get_all_note_links.return_value = []
    mock_db.get_all_notes_metadata.return_value = ([], set())

    panel = RightPanelView(
        db_manager=mock_db,
        on_map_note_selected=lambda nid: None,
        on_linked_note_clicked=lambda nid: None,
        on_unlink_note_clicked=lambda nid, t: None,
    )
    assert panel.is_collapsed is False
    assert panel.width == 350

    # Collapse to narrow rail
    panel.set_collapsed(True)
    assert panel.is_collapsed is True
    assert panel.width == 50

    # Toggle back to expanded
    panel.toggle_collapsed()
    assert panel.is_collapsed is False
    assert panel.width == 350


def test_editor_workspace_clean_layout():
    workspace = EditorWorkspaceView(
        on_content_change=lambda t: None,
        on_wikilink_clicked=lambda t: None,
        get_all_notes_callback=lambda: [],
        on_new_note_clicked=lambda: None,
        on_save_note_clicked=lambda: None,
        on_delete_note_clicked=lambda: None,
        on_link_note_clicked=lambda: None,
        on_generate_ai_clicked=lambda: None,
    )
    assert workspace.expand is True
    assert workspace.live_editor is not None
    # Live editor is full container content
    assert len(workspace.content.controls) == 1
    assert workspace.content.controls[0] == workspace.live_editor


def test_dialog_manager_settings_and_model_manager():
    mock_page = MagicMock()
    mock_page.overlay = []

    dm = DialogManager(mock_page)

    saved_settings = []
    dm.show_settings_dialog(
        theme_btn=ft.IconButton(icon=ft.Icons.DARK_MODE),
        auto_save_switch=ft.Switch(),
        current_ai_provider="local",
        current_active_model_id="gemma-4-e2b",
        on_save_settings=lambda key, prov, mid, gpu: saved_settings.append((key, prov, mid, gpu)),
        on_open_model_manager=lambda: None,
        current_gpu_acceleration=True
    )
    assert dm.settings_dialog.open is True

    # Trigger save action
    save_btn = dm.settings_dialog.actions[0]
    save_btn.on_click(None)
    assert len(saved_settings) == 1
    assert saved_settings[0][1:] == ("local", "gemma-4-e2b", True)

    selected_model = []
    dm.show_model_manager_dialog(
        models_dir="/tmp/test_models",
        active_model_id="gemma-4-e2b",
        on_select_model=lambda mid: selected_model.append(mid)
    )
    assert dm.model_manager_dialog.open is True


def test_dialog_manager_loading_dialog_lifecycle():
    mock_page = MagicMock()
    mock_page.overlay = []

    dm = DialogManager(mock_page)

    # Initial state
    assert dm.loading_dialog.open is False

    # Show loading
    cancelled = []
    dm.show_loading(
        title="PDF Analiz Ediliyor",
        message="Metin çıkarılıyor...",
        on_cancel=lambda: cancelled.append(True)
    )
    assert dm.loading_dialog.open is True
    assert dm.loading_msg.value == "Metin çıkarılıyor..."
    assert len(dm.loading_dialog.actions) == 1

    # Trigger cancel button
    cancel_btn = dm.loading_dialog.actions[0]
    cancel_btn.on_click(None)
    assert cancelled == [True]

    # Update loading message
    dm.update_loading_message("Notlar veritabanına kaydediliyor...")
    assert dm.loading_msg.value == "Notlar veritabanına kaydediliyor..."

    # Hide loading cleanly
    dm.hide_loading()
    assert dm.loading_dialog.open is False


def test_dialog_manager_persistent_controls_identity():
    """Verify that DialogManager maintains persistent control references across show/hide cycles."""
    mock_page = MagicMock()
    mock_page.overlay = []

    dm = DialogManager(mock_page)

    initial_title = dm.loading_dialog.title
    initial_content = dm.loading_dialog.content
    initial_cancel_btn = dm.loading_dialog.actions[0]

    dm.show_loading(title="Pass 1", message="Message 1", on_cancel=lambda: None)
    assert dm.loading_dialog.title is initial_title
    assert dm.loading_dialog.content is initial_content
    assert dm.loading_dialog.actions[0] is initial_cancel_btn
    assert dm.loading_title.value == "Pass 1"
    assert dm.loading_msg.value == "Message 1"
    assert dm.loading_cancel_btn.visible is True

    dm.update_loading_message("Message 1 updated")
    assert dm.loading_dialog.content is initial_content
    assert dm.loading_msg.value == "Message 1 updated"

    dm.hide_loading()
    assert dm.loading_dialog.open is False

    # Second show cycle (e.g. subsequent AI run) must NOT replace control instances
    dm.show_loading(title="Pass 2", message="Message 2", on_cancel=None)
    assert dm.loading_dialog.title is initial_title
    assert dm.loading_dialog.content is initial_content
    assert dm.loading_dialog.actions[0] is initial_cancel_btn
    assert dm.loading_title.value == "Pass 2"
    assert dm.loading_msg.value == "Message 2"
    assert dm.loading_cancel_btn.visible is False
    assert dm.loading_dialog.open is True





