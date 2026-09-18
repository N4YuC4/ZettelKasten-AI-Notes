# main.py
#
# Main application entry point for Zettelkasten AI Notes.
# Wires together AppState, Services, AppController, and responsive modular UI components.

import os
from env_config import configure_headless_environment

configure_headless_environment()

import flet as ft
from hardware_checker import HardwareChecker

# Configure headless Vulkan environment and route to discrete GPU
HardwareChecker.configure_vulkan_environment()

import database_manager
from settings_manager import SettingsManager
from note_service import NoteService
from app_state import AppState
from app_controller import AppController
from ui import DialogManager, SidebarView, RightPanelView, EditorWorkspaceView, create_vertical_splitter


def main(page: ft.Page):
    # 1. Page Configuration
    page.title = "Zettelkasten AI Notes"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1400
    page.window_height = 900
    page.padding = ft.Padding.all(6)

    # 2. Service & State Initialization
    settings_mgr = SettingsManager.get_instance()
    custom_db = settings_mgr.get_setting("CUSTOM_DB_PATH")
    db_path = custom_db if custom_db and os.path.exists(os.path.dirname(os.path.abspath(custom_db))) else None
    db_manager = database_manager.DatabaseManager(db_path=db_path, settings_manager=settings_mgr)
    note_service = NoteService(db_manager)

    saved_theme = db_manager.get_setting("UI_THEME") or "Dark"
    auto_save_val = db_manager.get_setting("AUTO_SAVE") != "False"
    page.theme_mode = ft.ThemeMode.DARK if saved_theme == "Dark" else ft.ThemeMode.LIGHT

    state = AppState(theme_mode=saved_theme, auto_save=auto_save_val)
    dialog_manager = DialogManager(page)

    # 3. Dedicated Controller Initialization
    controller = AppController(page, db_manager, note_service, dialog_manager, state)

    # 4. Global Widgets & Controls
    theme_btn = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE if page.theme_mode == ft.ThemeMode.DARK else ft.Icons.DARK_MODE,
        tooltip="Switch to Light Mode" if page.theme_mode == ft.ThemeMode.DARK else "Switch to Dark Mode",
        on_click=controller.toggle_theme
    )

    auto_save_switch = ft.Switch(
        label="Auto Save",
        value=state.auto_save,
        on_change=controller.toggle_auto_save
    )

    pdf_file_picker = ft.FilePicker()
    if hasattr(page, 'services'):
        page.services.append(pdf_file_picker)
    else:
        page.overlay.append(pdf_file_picker)

    # Top AppBar Title & Status
    note_title_text = ft.Text(
        "New Note",
        size=15,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.ON_SURFACE,
        overflow=ft.TextOverflow.ELLIPSIS,
    )
    collection_chip = ft.Container(
        content=ft.Text("All Notes", size=11, color=ft.Colors.PRIMARY, weight=ft.FontWeight.W_600),
        padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        bgcolor=ft.Colors.PRIMARY_CONTAINER,
        border_radius=12,
        visible=False,
    )
    title_row = ft.Row([
        note_title_text,
        collection_chip,
    ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # 5. UI Views Construction
    sidebar = SidebarView(
        on_collection_changed=lambda col: (state.set_collection_filter(col), controller.refresh_notes()),
        on_new_collection_clicked=lambda: dialog_manager.show_new_collection_dialog(
            on_submit=controller.handle_create_collection
        ),
        on_delete_collection_clicked=controller.handle_delete_collection_click,
        on_search_changed=lambda query: (state.set_search_query(query), controller.refresh_notes()),
        on_note_clicked=lambda nid, title, col: controller.guard_unsaved_changes(controller.open_note_internal, nid, title, col),
        on_rename_note_clicked=lambda nid, title: dialog_manager.show_rename_dialog(
            current_title=title,
            on_submit=lambda new_title: controller.handle_rename_note(nid, new_title)
        ),
        on_delete_note_clicked=controller.confirm_or_delete_note,
        on_settings_clicked=controller.handle_open_settings,
        on_collapse_clicked=controller.toggle_sidebar
    )

    right_panel = RightPanelView(
        db_manager=db_manager,
        on_map_note_selected=controller.open_note_by_id,
        on_linked_note_clicked=controller.open_note_by_id,
        on_unlink_note_clicked=lambda nid, title: dialog_manager.show_unlink_confirm(
            target_title=title,
            on_confirm=lambda: controller.handle_unlink_note(nid, title)
        ),
        on_collapse_clicked=controller.toggle_right_panel
    )

    editor_workspace = EditorWorkspaceView(
        on_content_change=controller.handle_editor_content_change,
        on_wikilink_clicked=controller.handle_wikilink_tap,
        get_all_notes_callback=lambda: note_service.load_all_notes_metadata()[0],
        on_new_note_clicked=lambda: controller.guard_unsaved_changes(controller.new_note_internal),
        on_save_note_clicked=controller.save_current_note,
        on_delete_note_clicked=controller.handle_delete_current_note,
        on_link_note_clicked=controller.handle_link_picker_open,
        on_generate_ai_clicked=lambda: page.run_task(controller.trigger_pdf_generation),
        on_blur=controller.handle_editor_blur,
    )

    # Attach instantiated UI controls to controller
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

    # 6. Responsive Splitters & Layout Assembly
    left_splitter = create_vertical_splitter(controller.on_left_drag)
    right_splitter = create_vertical_splitter(controller.on_right_drag)

    page.appbar = ft.AppBar(
        automatically_imply_leading=False,
        leading=None,
        leading_width=16,
        title=title_row,
        actions=[
            ft.IconButton(
                icon=ft.Icons.NOTE_ADD_OUTLINED,
                icon_color=ft.Colors.PRIMARY,
                tooltip="New Note (Ctrl+N)",
                on_click=lambda e: controller.guard_unsaved_changes(controller.new_note_internal)
            ),
            ft.IconButton(
                icon=ft.Icons.SAVE_OUTLINED,
                icon_color=ft.Colors.SECONDARY,
                tooltip="Save Note (Ctrl+S)",
                on_click=lambda e: controller.save_current_note()
            ),
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                icon_color=ft.Colors.ERROR,
                tooltip="Delete Note",
                on_click=lambda e: controller.handle_delete_current_note()
            ),
            ft.IconButton(
                icon=ft.Icons.LINK,
                icon_color=ft.Colors.TERTIARY,
                tooltip="Link Note (Ctrl+K)",
                on_click=lambda e: controller.handle_link_picker_open()
            ),
            ft.IconButton(
                icon=ft.Icons.AUTO_AWESOME,
                icon_color=ft.Colors.AMBER_400,
                tooltip="Generate AI Notes from PDF",
                on_click=lambda e: page.run_task(controller.trigger_pdf_generation)
            ),
            ft.Container(width=8),
        ],
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        center_title=False,
        elevation=1,
    )

    main_row = ft.Row([
        sidebar,
        left_splitter,
        editor_workspace,
        right_splitter,
        right_panel
    ], expand=True, spacing=4)

    page.on_resize = controller.update_layout
    page.on_keyboard_event = controller.handle_keyboard_event

    page.add(main_row)

    # 7. Initial data load
    controller.initial_load()


if __name__ == '__main__':
    ft.run(main)
