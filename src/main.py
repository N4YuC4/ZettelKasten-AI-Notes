# main.py
#
# Main application entry point and coordinator for Zettelkasten AI Notes.
# Wires together AppState, NoteService, DialogManager, and modular UI components.

import os
import threading
import flet as ft
from dotenv import set_key
from typing import Optional

from logger import log_debug, log_error
import database_manager
from note_service import NoteService
import pdf_processor
from ai_note_generator_worker import AiNoteGeneratorWorker
from app_state import AppState
from ui import DialogManager, SidebarView, RightPanelView, EditorWorkspaceView, create_vertical_splitter


def main(page: ft.Page):
    # 1. Page Configuration
    page.title = "Zettelkasten AI Notes"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1400
    page.window_height = 900
    page.padding = 10

    # 2. Service & State Initialization
    db_manager = database_manager.DatabaseManager()
    note_service = NoteService(db_manager)

    saved_theme = db_manager.get_setting("UI_THEME") or "Dark"
    auto_save_val = db_manager.get_setting("AUTO_SAVE") != "False"
    page.theme_mode = ft.ThemeMode.DARK if saved_theme == "Dark" else ft.ThemeMode.LIGHT

    state = AppState(theme_mode=saved_theme, auto_save=auto_save_val)

    # 3. Notification & Dialog Helpers
    def show_snack_bar(message: str, color=ft.Colors.PRIMARY):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.ON_PRIMARY_CONTAINER),
            bgcolor=color,
            duration=3000
        )
        page.snack_bar.open = True
        page.update()

    dialog_manager = DialogManager(page)

    # 4. Action / Controller Methods
    def refresh_categories(select_category: Optional[str] = None):
        notes_metadata, all_categories = note_service.load_all_notes_metadata()
        state.all_categories = all_categories
        selected = select_category if select_category is not None else state.selected_category_filter
        sidebar.update_categories(all_categories, selected)

    def refresh_notes(category_to_select: Optional[str] = None):
        if category_to_select is not None:
            state.selected_category_filter = category_to_select

        all_notes_metadata, _ = note_service.load_all_notes_metadata()
        state.displayed_notes = all_notes_metadata

        # Filter notes by active category and search text
        selected_cat = state.selected_category_filter
        query = state.search_query.lower()

        filtered = []
        for nid, title, cat in all_notes_metadata:
            if not selected_cat or cat == selected_cat:
                if not query or query in title.lower():
                    filtered.append((nid, title, cat))

        total_count = db_manager.note_count(selected_cat)
        sidebar.render_notes(filtered, state.current_note_id, selected_cat, total_count)
        refresh_linked_notes()
        refresh_mind_map()

    def refresh_linked_notes():
        linked_data = []
        if state.current_note_id:
            linked_ids = note_service.get_linked_note_ids(state.current_note_id)
            for lid in linked_ids:
                note_row = db_manager.get_note(lid)
                if note_row:
                    linked_data.append((lid, note_row[1]))
        right_panel.render_linked_notes(state.current_note_id, linked_data)

    def refresh_mind_map():
        all_notes, _ = note_service.load_all_notes_metadata()
        all_links = db_manager.get_all_note_links()

        selected_cat = state.selected_category_filter
        filtered_notes = [n for n in all_notes if not selected_cat or n[2] == selected_cat]
        filtered_ids = {n[0] for n in filtered_notes}
        filtered_links = [(s, t) for s, t in all_links if s in filtered_ids and t in filtered_ids]

        right_panel.update_mind_map(filtered_notes, filtered_links, state.current_note_id)

    def guard_unsaved_changes(action_fn, *args):
        if state.is_dirty:
            if state.auto_save:
                save_current_note()
                action_fn(*args)
            else:
                dialog_manager.show_unsaved_changes_prompt(
                    on_save=lambda: (save_current_note(), action_fn(*args)),
                    on_discard=lambda: (state.set_dirty(False), action_fn(*args))
                )
        else:
            action_fn(*args)

    def open_note_internal(note_id: str, display_title: str, category_path: str):
        content = note_service.get_note_content(note_id)
        if content is not None:
            state.select_note(note_id, display_title, category_path or "")
            editor_workspace.set_content(content, mark_dirty=False)
            page.title = f"Zettelkasten AI Notes - {display_title}"
            refresh_notes()
        else:
            show_snack_bar(f"Could not read content for note: {display_title}", color=ft.Colors.ERROR)
            new_note_internal()

    def open_note_by_id(note_id: str):
        all_notes, _ = note_service.load_all_notes_metadata()
        for nid, title, cat in all_notes:
            if nid == note_id:
                guard_unsaved_changes(open_note_internal, nid, title, cat)
                break

    def new_note_internal():
        state.select_note(None, "New Note", state.selected_category_filter)
        editor_workspace.set_content("", mark_dirty=False)
        page.title = "Zettelkasten AI Notes - New Note"
        page.update()

    def save_current_note():
        content = editor_workspace.get_content()
        if not content.strip():
            show_snack_bar("Cannot save empty note.", color=ft.Colors.ERROR)
            return

        cat_to_save = state.current_note_category
        note_id, title = note_service.save_note(state.current_note_id, content, cat_to_save)

        if note_id and title:
            state.select_note(note_id, title, cat_to_save)
            editor_workspace.set_dirty(False)
            page.title = f"Zettelkasten AI Notes - {title}"
            show_snack_bar(f"Note '{title}' saved successfully.")
            refresh_categories(cat_to_save)
            refresh_notes(cat_to_save)
        else:
            show_snack_bar("Failed to save note.", color=ft.Colors.ERROR)

    # 5. UI Views Construction
    def handle_wikilink_tap(target_title: str):
        all_notes, _ = note_service.load_all_notes_metadata()
        for nid, title, cat in all_notes:
            if title.strip().lower() == target_title.strip().lower():
                guard_unsaved_changes(open_note_internal, nid, title, cat)
                show_snack_bar(f"'{title}' notuna geçildi.")
                return

        # Not not found, prompt creation
        def create_linked_note():
            new_note_internal()
            initial_content = f"# {target_title}\n\n"
            editor_workspace.set_content(initial_content, mark_dirty=True)
            save_current_note()
            show_snack_bar(f"'{target_title}' notu oluşturuldu.")

        dialog_manager.show_create_linked_note_prompt(target_title, on_create=create_linked_note)

    sidebar = SidebarView(
        on_category_changed=lambda cat: (state.set_category_filter(cat), refresh_notes()),
        on_new_category_clicked=lambda: dialog_manager.show_new_category_dialog(
            on_submit=lambda cat_name: handle_create_category(cat_name)
        ),
        on_delete_category_clicked=lambda: handle_delete_category_click(),
        on_search_changed=lambda query: (state.set_search_query(query), refresh_notes()),
        on_note_clicked=lambda nid, title, cat: guard_unsaved_changes(open_note_internal, nid, title, cat),
        on_rename_note_clicked=lambda nid, title: dialog_manager.show_rename_dialog(
            current_title=title,
            on_submit=lambda new_title: handle_rename_note(nid, new_title)
        ),
        on_delete_note_clicked=lambda nid, title: dialog_manager.show_delete_note_confirm(
            note_title=title,
            on_confirm=lambda: handle_delete_note(nid, title)
        ),
        on_settings_clicked=lambda: dialog_manager.show_settings_dialog(
            theme_btn=theme_btn,
            auto_save_switch=auto_save_switch,
            on_save_api_key=handle_save_api_key
        )
    )

    right_panel = RightPanelView(
        db_manager=db_manager,
        on_map_note_selected=lambda nid: open_note_by_id(nid),
        on_linked_note_clicked=lambda nid: open_note_by_id(nid),
        on_unlink_note_clicked=lambda nid, title: dialog_manager.show_unlink_confirm(
            target_title=title,
            on_confirm=lambda: handle_unlink_note(nid, title)
        )
    )

    editor_workspace = EditorWorkspaceView(
        on_content_change=lambda text: state.set_dirty(True),
        on_wikilink_clicked=handle_wikilink_tap,
        get_all_notes_callback=lambda: note_service.load_all_notes_metadata()[0],
        on_new_note_clicked=lambda: guard_unsaved_changes(new_note_internal),
        on_save_note_clicked=save_current_note,
        on_delete_note_clicked=lambda: handle_delete_current_note(),
        on_link_note_clicked=lambda: handle_link_picker_open(),
        on_generate_ai_clicked=lambda: trigger_pdf_generation()
    )

    # 6. Specific Action Handlers
    def handle_create_category(category_name: str):
        cleaned = category_name.strip()
        if not cleaned:
            show_snack_bar("Category name cannot be empty.", color=ft.Colors.ERROR)
            return

        if cleaned in state.all_categories:
            show_snack_bar(f"Category '{cleaned}' already exists.", color=ft.Colors.ERROR)
            return

        dialog_manager.close(dialog_manager.new_category_dialog)
        state.set_category_filter(cleaned)
        refresh_categories(cleaned)
        refresh_notes(cleaned)
        show_snack_bar(f"Category '{cleaned}' created successfully.")

    def handle_delete_category_click():
        selected_cat = state.selected_category_filter
        if not selected_cat or selected_cat == "All Notes":
            show_snack_bar("Please select a valid category to delete.", color=ft.Colors.ERROR)
            return

        dialog_manager.show_delete_category_confirm(
            category_name=selected_cat,
            on_confirm=lambda: handle_delete_category_confirmed(selected_cat)
        )

    def handle_delete_category_confirmed(cat_to_delete: str):
        success = note_service.delete_category(cat_to_delete)
        if success:
            state.select_note(None, "New Note", "")
            state.set_category_filter("")
            editor_workspace.set_content("", mark_dirty=False)
            page.title = "Zettelkasten AI Notes - New Note"
            right_panel.mind_map_widget.invalidate_cache()
            refresh_categories("")
            refresh_notes("")
            show_snack_bar(f"Kategori '{cat_to_delete}' ve içerdiği tüm notlar silindi.")
        else:
            show_snack_bar("Failed to delete category.", color=ft.Colors.ERROR)

    def handle_rename_note(note_id: str, new_title: str):
        if not new_title.strip():
            show_snack_bar("Note title cannot be empty.", color=ft.Colors.ERROR)
            return

        success, msg_or_title = note_service.rename_note(note_id, new_title, state.current_note_category)
        if success:
            dialog_manager.close(dialog_manager.rename_dialog)
            if state.current_note_id == note_id:
                state.current_note_title = msg_or_title
                page.title = f"Zettelkasten AI Notes - {msg_or_title}"
                curr_val = editor_workspace.get_content()
                lines = curr_val.split('\n') if curr_val else []
                if lines:
                    lines[0] = f"# {msg_or_title}"
                else:
                    lines = [f"# {msg_or_title}"]
                editor_workspace.set_content('\n'.join(lines), mark_dirty=False)

            right_panel.mind_map_widget.invalidate_cache()
            refresh_notes()
            show_snack_bar("Note renamed successfully.")
        else:
            show_snack_bar(f"Failed to rename note: {msg_or_title}", color=ft.Colors.ERROR)

    def handle_delete_note(note_id: str, note_title: str):
        success = note_service.delete_note(note_id)
        if success:
            if state.current_note_id == note_id:
                new_note_internal()
            right_panel.mind_map_widget.invalidate_cache()
            refresh_notes()
            show_snack_bar(f"Note '{note_title}' deleted successfully.")
        else:
            show_snack_bar("Failed to delete note.", color=ft.Colors.ERROR)

    def handle_delete_current_note():
        if not state.current_note_id:
            show_snack_bar("Please select or save a note first to delete.", color=ft.Colors.ERROR)
            return
        dialog_manager.show_delete_note_confirm(
            note_title=state.current_note_title,
            on_confirm=lambda: handle_delete_note(state.current_note_id, state.current_note_title)
        )

    def handle_link_picker_open():
        if not state.current_note_id:
            show_snack_bar("Please select or save a source note first.", color=ft.Colors.ERROR)
            return
        all_notes, _ = note_service.load_all_notes_metadata()
        dialog_manager.show_link_picker(
            current_note_id=state.current_note_id,
            all_notes=all_notes,
            on_link_selected=lambda target_id, target_title: handle_create_link(target_id, target_title)
        )

    def handle_create_link(target_id: str, target_title: str):
        success = note_service.create_link(state.current_note_id, target_id)
        if success:
            show_snack_bar(f"Successfully linked to '{target_title}'.")
            refresh_linked_notes()
            refresh_mind_map()
        else:
            show_snack_bar(f"Link to '{target_title}' already exists or failed.", color=ft.Colors.ERROR)

    def handle_unlink_note(target_id: str, target_title: str):
        success = note_service.delete_link(state.current_note_id, target_id)
        if success:
            right_panel.mind_map_widget.invalidate_cache()
            refresh_linked_notes()
            refresh_mind_map()
            show_snack_bar(f"Successfully unlinked '{target_title}'.")
        else:
            show_snack_bar(f"Failed to unlink note: {target_title}.", color=ft.Colors.ERROR)

    def handle_save_api_key(api_key: str):
        if api_key:
            dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
            set_key(dotenv_path, "GEMINI_API_KEY", api_key)
            show_snack_bar("Gemini API Key saved successfully.")

    # 7. PDF AI Generation Worker Dispatch
    pdf_file_picker = ft.FilePicker()
    if hasattr(page, 'services'):
        page.services.append(pdf_file_picker)
    else:
        page.overlay.append(pdf_file_picker)

    def handle_ai_finished(generated_notes):
        dialog_manager.hide_loading()
        if generated_notes:
            state.set_category_filter("")
            right_panel.mind_map_widget.invalidate_cache()
            refresh_categories("")
            refresh_notes("")
            show_snack_bar(f"{len(generated_notes)} notes were successfully generated and saved!")
        else:
            show_snack_bar("No notes were generated by the AI.", color=ft.Colors.TERTIARY)

    def handle_ai_error(err_msg: str):
        dialog_manager.hide_loading()
        dialog_manager.show_error("AI Note Generation Error", f"An error occurred during AI note generation:\n{err_msg}")
        show_snack_bar(f"Error: {err_msg}", color=ft.Colors.ERROR)

    async def trigger_pdf_generation():
        files = await pdf_file_picker.pick_files(
            dialog_title="Select PDF File",
            allowed_extensions=["pdf"]
        )
        if files:
            pdf_path = files[0].path
            dialog_manager.show_loading(
                "Generating AI Notes From PDF",
                "Extracting text from PDF... This may take a moment."
            )
            try:
                extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)
            except Exception as ex:
                dialog_manager.hide_loading()
                show_snack_bar(f"Failed to read PDF: {ex}", color=ft.Colors.ERROR)
                return

            if extracted_text and extracted_text.strip():
                dialog_manager.update_loading_message("Generating notes with AI... This may take longer.")
                worker = AiNoteGeneratorWorker(
                    extracted_text,
                    on_finished=handle_ai_finished,
                    on_error=handle_ai_error
                )
                threading.Thread(target=worker.run, daemon=True).start()
            else:
                dialog_manager.hide_loading()
                show_snack_bar("Selected PDF file is empty or contains no readable text.", color=ft.Colors.ERROR)
        else:
            show_snack_bar("No PDF file selected.", color=ft.Colors.TERTIARY)

    # 8. Theme & Auto-Save Settings Toggles
    def toggle_theme(e):
        if page.theme_mode == ft.ThemeMode.DARK:
            page.theme_mode = ft.ThemeMode.LIGHT
            db_manager.set_setting("UI_THEME", "Light")
            theme_btn.icon = ft.Icons.DARK_MODE
            theme_btn.tooltip = "Switch to Dark Mode"
        else:
            page.theme_mode = ft.ThemeMode.DARK
            db_manager.set_setting("UI_THEME", "Dark")
            theme_btn.icon = ft.Icons.LIGHT_MODE
            theme_btn.tooltip = "Switch to Light Mode"
        page.update()

    theme_btn = ft.IconButton(
        icon=ft.Icons.LIGHT_MODE if page.theme_mode == ft.ThemeMode.DARK else ft.Icons.DARK_MODE,
        tooltip="Switch to Light Mode" if page.theme_mode == ft.ThemeMode.DARK else "Switch to Dark Mode",
        on_click=toggle_theme
    )

    def toggle_auto_save(e):
        val = "True" if auto_save_switch.value else "False"
        db_manager.set_setting("AUTO_SAVE", val)
        state.set_auto_save(auto_save_switch.value)

    auto_save_switch = ft.Switch(
        label="Auto Save",
        value=state.auto_save,
        on_change=toggle_auto_save
    )

    # 9. Layout Assembly with Resizable Splitters
    def on_left_drag(e: ft.DragUpdateEvent):
        delta = e.local_delta.x if e.local_delta else 0
        new_width = sidebar.width + delta
        if 180 <= new_width <= 600:
            sidebar.width = new_width
            page.update()

    def on_right_drag(e: ft.DragUpdateEvent):
        delta = e.local_delta.x if e.local_delta else 0
        new_width = right_panel.width - delta
        if 200 <= new_width <= 700:
            right_panel.width = new_width
            page.update()

    left_splitter = create_vertical_splitter(on_left_drag)
    right_splitter = create_vertical_splitter(on_right_drag)

    # Global Keyboard Shortcuts
    def handle_keyboard_event(e: ft.KeyboardEvent):
        if e.ctrl:
            k = e.key.lower() if e.key else ""
            if k == "s":
                save_current_note()
            elif k == "e":
                editor_workspace.toggle_editor_mode()
            elif k == "n":
                guard_unsaved_changes(new_note_internal)
            elif k == "b":
                editor_workspace.wrap_selection("**", "**", "kalın metin")
            elif k == "i":
                editor_workspace.wrap_selection("*", "*", "italik metin")
            elif k == "k":
                editor_workspace.open_wikilink_picker()

    page.on_keyboard_event = handle_keyboard_event

    page.add(
        ft.Row([
            sidebar,
            left_splitter,
            editor_workspace,
            right_splitter,
            right_panel
        ], expand=True, spacing=5)
    )

    # Initial data load
    refresh_categories()
    refresh_notes()


if __name__ == '__main__':
    ft.run(main)