# app_controller.py
#
# Dedicated application controller coordinating domain logic, UI views,
# dialogs, asynchronous AI background tasks, and application state.
# Extracted from monolithic main.py for clean architecture and SOLID compliance.

import os
import asyncio
import time
import traceback
from typing import Optional, Any
from dotenv import set_key

import flet as ft
from logger import log_debug, log_error
from note_service import NoteService, sanitize_title, disambiguate_title
import pdf_processor
from ai_note_generator_worker import AiNoteGeneratorWorker
from local_gguf_client import LocalGgufClient
import local_models_catalog
from app_state import AppState
from ui import DialogManager, SidebarView, RightPanelView, EditorWorkspaceView


class AppController:
    """
    Coordinates application actions, UI updates, and background workers.
    """

    def __init__(
        self,
        page: ft.Page,
        db_manager: Any,
        note_service: NoteService,
        dialog_manager: DialogManager,
        state: AppState
    ):
        self.page = page
        self.db_manager = db_manager
        self.note_service = note_service
        self.dialog_manager = dialog_manager
        self.state = state

        self.sidebar: Optional[SidebarView] = None
        self.right_panel: Optional[RightPanelView] = None
        self.editor_workspace: Optional[EditorWorkspaceView] = None
        self.note_title_text: Optional[ft.Text] = None
        self.collection_chip: Optional[ft.Container] = None
        self.theme_btn: Optional[ft.IconButton] = None
        self.auto_save_switch: Optional[ft.Switch] = None
        self.pdf_file_picker: Optional[ft.FilePicker] = None

        self._auto_save_seq: int = 0
        self.active_worker: Optional[AiNoteGeneratorWorker] = None

    @property
    def category_chip(self) -> Optional[ft.Container]:
        """Backward compatibility alias for collection_chip."""
        return self.collection_chip

    @category_chip.setter
    def category_chip(self, val: Optional[ft.Container]) -> None:
        self.collection_chip = val

    def attach_views(
        self,
        sidebar: SidebarView,
        right_panel: RightPanelView,
        editor_workspace: EditorWorkspaceView,
        note_title_text: ft.Text,
        collection_chip: Optional[ft.Container] = None,
        theme_btn: Optional[ft.IconButton] = None,
        auto_save_switch: Optional[ft.Switch] = None,
        pdf_file_picker: Optional[ft.FilePicker] = None,
        category_chip: Optional[ft.Container] = None
    ):
        """Attaches instantiated UI views and controls to the controller."""
        self.sidebar = sidebar
        self.right_panel = right_panel
        self.editor_workspace = editor_workspace
        self.note_title_text = note_title_text
        self.collection_chip = collection_chip or category_chip
        self.theme_btn = theme_btn
        self.auto_save_switch = auto_save_switch
        self.pdf_file_picker = pdf_file_picker

    def show_snack_bar(self, message: str, color=ft.Colors.PRIMARY):
        """Displays a non-blocking toast/snack-bar notification."""
        sb = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.ON_PRIMARY_CONTAINER),
            bgcolor=color,
            duration=3000,
        )
        try:
            self.page.show_dialog(sb)
        except Exception:
            try:
                self.page.overlay.append(sb)
                sb.open = True
                sb.update()
            except Exception:
                pass

    def refresh_collections(self, select_collection: Optional[str] = None, select_category: Optional[str] = None):
        """Reloads distinct collections from database and refreshes sidebar dropdown."""
        target_sel = select_collection if select_category is None else select_category
        notes_metadata, all_collections = self.note_service.load_all_notes_metadata()
        selected = target_sel if target_sel is not None else self.state.selected_collection_filter
        if selected and selected != "All Notes" and selected not in all_collections:
            all_collections = sorted(list(set(all_collections + [selected])))
        self.state.all_collections = all_collections
        if self.sidebar:
            self.sidebar.update_collections(all_collections, selected)

    # Backward compatibility alias
    refresh_categories = refresh_collections

    def refresh_notes(self, collection_to_select: Optional[str] = None, category_to_select: Optional[str] = None):
        """Filters and refreshes notes in the sidebar, linked panel, and mind map."""
        target_col = collection_to_select if category_to_select is None else category_to_select
        if target_col is not None:
            self.state.selected_collection_filter = target_col

        all_notes_metadata, _ = self.note_service.load_all_notes_metadata()
        self.state.displayed_notes = all_notes_metadata

        selected_col = self.state.selected_collection_filter
        query = self.state.search_query.lower()

        filtered = []
        for nid, title, col in all_notes_metadata:
            if not selected_col or col == selected_col:
                if not query or query in title.lower():
                    filtered.append((nid, title, col))

        total_count = self.db_manager.note_count(selected_col)
        if self.sidebar:
            self.sidebar.render_notes(filtered, self.state.current_note_id, selected_col, total_count)
        self.refresh_linked_notes()
        self.refresh_mind_map()
        try:
            self.page.update()
        except Exception:
            pass

    def refresh_linked_notes(self):
        """Refreshes notes linked to currently active note in the right panel."""
        linked_data = []
        if self.state.current_note_id:
            linked_ids = self.note_service.get_linked_note_ids(self.state.current_note_id)
            for lid in linked_ids:
                note_row = self.db_manager.get_note(lid)
                if note_row:
                    linked_data.append((lid, note_row[1]))
        if self.right_panel:
            self.right_panel.render_linked_notes(self.state.current_note_id, linked_data)

    def refresh_mind_map(self):
        """Refreshes knowledge graph visualization in the right panel."""
        all_notes, _ = self.note_service.load_all_notes_metadata()
        all_links = self.db_manager.get_all_note_links()

        selected_col = self.state.selected_collection_filter
        filtered_notes = [n for n in all_notes if not selected_col or n[2] == selected_col]
        filtered_ids = {n[0] for n in filtered_notes}
        filtered_links = [(s, t) for s, t in all_links if s in filtered_ids and t in filtered_ids]

        if self.right_panel:
            self.right_panel.update_mind_map(filtered_notes, filtered_links, self.state.current_note_id)

    def update_header_status(self, title: str, collection: str = "", category: Optional[str] = None):
        """Updates note title and collection badge in top appbar."""
        col_val = collection if category is None else category
        if self.note_title_text:
            self.note_title_text.value = title or "New Note"
        col_clean = (col_val or "").strip()
        if self.collection_chip:
            if col_clean and col_clean != "All Notes":
                self.collection_chip.content.value = col_clean
                self.collection_chip.visible = True
            else:
                self.collection_chip.visible = False
        try:
            if self.note_title_text:
                self.note_title_text.update()
            if self.collection_chip:
                self.collection_chip.update()
        except Exception:
            pass

    def cancel_auto_save_timer(self):
        """Invalidates pending debounced auto-save requests."""
        self._auto_save_seq += 1

    def guard_unsaved_changes(self, action_fn, *args):
        """Prompts user if current note has unsaved changes before performing an action."""
        self.cancel_auto_save_timer()
        if self.state.is_dirty:
            if self.state.auto_save:
                self.save_current_note()
                action_fn(*args)
            else:
                self.dialog_manager.show_unsaved_changes_prompt(
                    on_save=lambda: (self.save_current_note(), action_fn(*args)),
                    on_discard=lambda: (self.state.set_dirty(False), action_fn(*args))
                )
        else:
            action_fn(*args)

    def open_note_internal(self, note_id: str, display_title: str, collection_path: str = "", category_path: Optional[str] = None):
        """Loads note content into the editor workspace."""
        col = collection_path if category_path is None else category_path
        self.cancel_auto_save_timer()
        content = self.note_service.get_note_content(note_id)
        if content is not None:
            self.state.select_note(note_id, display_title, col or "")
            if self.editor_workspace:
                self.editor_workspace.set_content(content, mark_dirty=False)
            self.page.title = f"Zettelkasten AI Notes - {display_title}"
            self.update_header_status(display_title, col)
            self.refresh_notes()
        else:
            self.show_snack_bar(f"Could not read content for note: {display_title}", color=ft.Colors.ERROR)
            self.new_note_internal()

    def open_note_by_id(self, note_id: str):
        """Finds note by ID and opens it with dirty guard."""
        all_notes, _ = self.note_service.load_all_notes_metadata()
        for nid, title, col in all_notes:
            if nid == note_id:
                self.guard_unsaved_changes(self.open_note_internal, nid, title, col)
                break

    def new_note_internal(self, initial_title: Optional[str] = None, initial_content: Optional[str] = None):
        """Creates and selects a new blank or seeded note."""
        self.cancel_auto_save_timer()
        all_notes, _ = self.note_service.load_all_notes_metadata()
        existing_titles = {t for _, t, _ in all_notes}

        base_title = initial_title or "New Note"
        unique_title = disambiguate_title(base_title, existing_titles)

        collection = self.state.selected_collection_filter if self.state.selected_collection_filter and self.state.selected_collection_filter != "All Notes" else ""
        content = initial_content if initial_content is not None else f"# {unique_title}\n\n"

        note_id, title = self.note_service.save_note(None, content, collection)
        if note_id and title:
            self.state.select_note(note_id, title, collection)
            if self.editor_workspace:
                self.editor_workspace.set_content(content, mark_dirty=False)
            self.page.title = f"Zettelkasten AI Notes - {title}"
            self.update_header_status(title, collection)
            self.refresh_collections(collection)
            self.refresh_notes(collection)
            try:
                self.page.update()
            except Exception:
                pass

    def save_current_note(self):
        """Saves current editor content to SQLite database."""
        self.cancel_auto_save_timer()
        if not self.editor_workspace:
            return
        content = self.editor_workspace.get_content()
        col_to_save = self.state.current_note_collection
        note_id, title = self.note_service.save_note(self.state.current_note_id, content, col_to_save)

        if note_id and title:
            self.state.select_note(note_id, title, col_to_save)
            self.editor_workspace.set_dirty(False)
            self.page.title = f"Zettelkasten AI Notes - {title}"
            self.update_header_status(title, col_to_save)
            self.show_snack_bar(f"Note '{title}' saved successfully.")
            self.refresh_collections(col_to_save)
            self.refresh_notes(col_to_save)
        else:
            self.show_snack_bar("Failed to save note.", color=ft.Colors.ERROR)

    def handle_wikilink_tap(self, target_title: str):
        """Handles click on [[WikiLink]] tag, navigating or prompting to create."""
        all_notes, _ = self.note_service.load_all_notes_metadata()
        for nid, title, cat in all_notes:
            if title.strip().lower() == target_title.strip().lower():
                self.guard_unsaved_changes(self.open_note_internal, nid, title, cat)
                self.show_snack_bar(f"'{title}' notuna geçildi.")
                return

        source_note_id = self.state.current_note_id

        def create_linked_note():
            self.new_note_internal(initial_title=target_title, initial_content=f"# {target_title}\n\n")
            if source_note_id and self.state.current_note_id and source_note_id != self.state.current_note_id:
                self.note_service.create_link(source_note_id, self.state.current_note_id)
                self.refresh_linked_notes()
                self.refresh_mind_map()
            self.show_snack_bar(f"'{target_title}' notu oluşturuldu ve bağlandı.")

        self.dialog_manager.show_create_linked_note_prompt(target_title, on_create=create_linked_note)

    def toggle_theme(self, e=None):
        """Toggles application between Dark and Light themes."""
        if self.page.theme_mode == ft.ThemeMode.DARK:
            self.page.theme_mode = ft.ThemeMode.LIGHT
            self.db_manager.set_setting("UI_THEME", "Light")
            if self.theme_btn:
                self.theme_btn.icon = ft.Icons.DARK_MODE
                self.theme_btn.tooltip = "Switch to Dark Mode"
        else:
            self.page.theme_mode = ft.ThemeMode.DARK
            self.db_manager.set_setting("UI_THEME", "Dark")
            if self.theme_btn:
                self.theme_btn.icon = ft.Icons.LIGHT_MODE
                self.theme_btn.tooltip = "Switch to Light Mode"
        self.refresh_mind_map()
        try:
            self.page.update()
        except Exception:
            pass

    def toggle_auto_save(self, e=None):
        """Updates auto-save setting in state and persistent storage."""
        if self.auto_save_switch:
            val = "True" if self.auto_save_switch.value else "False"
            self.db_manager.set_setting("AUTO_SAVE", val)
            self.state.set_auto_save(self.auto_save_switch.value)

    def toggle_sidebar(self):
        """Collapses or expands sidebar."""
        if self.sidebar:
            self.sidebar.toggle_collapsed()
            self.update_layout()

    def toggle_right_panel(self):
        """Collapses or expands right graph panel."""
        if self.right_panel:
            self.right_panel.toggle_collapsed()
            self.update_layout()

    async def async_auto_save(self, seq: int):
        """Debounced asynchronous auto-save task."""
        await asyncio.sleep(0.3)
        if seq == self._auto_save_seq and self.state.auto_save and self.state.is_dirty and self.state.current_note_id:
            try:
                if not self.editor_workspace:
                    return
                cnt = self.editor_workspace.get_content()
                col = self.state.current_note_collection
                nid, saved_title = self.note_service.save_note(self.state.current_note_id, cnt, col)
                if nid and saved_title:
                    self.state.current_note_title = saved_title
                    self.state.set_dirty(False)
                    self.editor_workspace.set_dirty(False)
                    self.refresh_notes()
                    try:
                        self.page.update()
                    except Exception:
                        pass
            except Exception as ex:
                log_error(f"Async auto-save error: {ex}")

    def handle_editor_blur(self):
        """Auto-saves dirty note when editor loses focus."""
        self._auto_save_seq += 1
        if self.state.auto_save and self.state.is_dirty and self.state.current_note_id:
            try:
                if not self.editor_workspace:
                    return
                cnt = self.editor_workspace.get_content()
                col = self.state.current_note_collection
                nid, saved_title = self.note_service.save_note(self.state.current_note_id, cnt, col)
                if nid and saved_title:
                    self.state.current_note_title = saved_title
                    self.state.set_dirty(False)
                    self.editor_workspace.set_dirty(False)
                    self.refresh_notes()
                    try:
                        self.page.update()
                    except Exception:
                        pass
            except Exception as ex:
                log_error(f"Blur auto-save error: {ex}")

    def handle_editor_content_change(self, text: str):
        """Updates dirty state, header preview, and schedules debounced auto-save."""
        self._auto_save_seq += 1
        self.state.set_dirty(True)
        live_title = sanitize_title(text)
        self.update_header_status(live_title, self.state.current_note_collection)
        self.page.title = f"Zettelkasten AI Notes - {live_title}"

        if self.state.auto_save and self.state.current_note_id:
            self.page.run_task(self.async_auto_save, self._auto_save_seq)

    def handle_create_collection(self, collection_name: str):
        """Creates a new collection and selects it."""
        cleaned = collection_name.strip()
        if not cleaned:
            self.show_snack_bar("Collection name cannot be empty.", color=ft.Colors.ERROR)
            return

        if cleaned.lower() == "all notes":
            self.show_snack_bar("Collection name 'All Notes' is reserved.", color=ft.Colors.ERROR)
            return

        if cleaned in self.state.all_collections:
            self.show_snack_bar(f"Collection '{cleaned}' already exists.", color=ft.Colors.ERROR)
            return

        self.dialog_manager.close(self.dialog_manager.new_collection_dialog)
        self.state.set_collection_filter(cleaned)
        self.refresh_collections(cleaned)
        self.refresh_notes(cleaned)
        self.show_snack_bar(f"Collection '{cleaned}' created successfully.")

    # Backward compatibility alias
    handle_create_category = handle_create_collection

    def handle_delete_collection_click(self):
        """Prompts confirmation for deleting active collection."""
        selected_col = self.state.selected_collection_filter
        if not selected_col or selected_col == "All Notes":
            self.show_snack_bar("Please select a valid collection to delete.", color=ft.Colors.ERROR)
            return

        self.dialog_manager.show_delete_collection_confirm(
            collection_name=selected_col,
            on_confirm=lambda: self.handle_delete_collection_confirmed(selected_col)
        )

    # Backward compatibility alias
    handle_delete_category_click = handle_delete_collection_click

    def handle_delete_collection_confirmed(self, col_to_delete: str):
        """Deletes specified collection and its contained notes."""
        self.cancel_auto_save_timer()
        success = self.note_service.delete_collection(col_to_delete)
        if success:
            self.state.set_collection_filter("")
            if self.right_panel:
                self.right_panel.mind_map_widget.invalidate_cache()
            self.refresh_collections("")
            all_notes, _ = self.note_service.load_all_notes_metadata()
            if all_notes:
                self.open_note_internal(all_notes[0][0], all_notes[0][1], all_notes[0][2])
            else:
                self.new_note_internal()
            self.show_snack_bar(f"Koleksiyon '{col_to_delete}' ve içerdiği tüm notlar silindi.")
        else:
            self.show_snack_bar("Failed to delete collection.", color=ft.Colors.ERROR)

    # Backward compatibility alias
    handle_delete_category_confirmed = handle_delete_collection_confirmed

    def handle_rename_note(self, note_id: str, new_title: str):
        """Renames a note and updates editor title line."""
        if not new_title.strip():
            self.show_snack_bar("Note title cannot be empty.", color=ft.Colors.ERROR)
            return

        success, msg_or_title = self.note_service.rename_note(note_id, new_title)
        if success:
            self.dialog_manager.close(self.dialog_manager.rename_dialog)
            if self.state.current_note_id == note_id:
                self.state.current_note_title = msg_or_title
                self.page.title = f"Zettelkasten AI Notes - {msg_or_title}"
                self.update_header_status(msg_or_title, self.state.current_note_collection)
                if self.editor_workspace:
                    curr_val = self.editor_workspace.get_content()
                    lines = curr_val.split('\n') if curr_val else []
                    if lines:
                        lines[0] = f"# {msg_or_title}"
                    else:
                        lines = [f"# {msg_or_title}"]
                    self.editor_workspace.set_content('\n'.join(lines), mark_dirty=False)

            if self.right_panel:
                self.right_panel.mind_map_widget.invalidate_cache()
            self.refresh_notes()
            self.show_snack_bar("Note renamed successfully.")
        else:
            self.show_snack_bar(f"Failed to rename note: {msg_or_title}", color=ft.Colors.ERROR)

    def handle_delete_note(self, note_id: str, note_title: str):
        """Deletes a note from SQLite database."""
        success = self.note_service.delete_note(note_id)
        if success:
            if self.state.current_note_id == note_id:
                self.new_note_internal()
            if self.right_panel:
                self.right_panel.mind_map_widget.invalidate_cache()
            self.refresh_notes()
            self.show_snack_bar(f"Note '{note_title}' deleted successfully.")
        else:
            self.show_snack_bar("Failed to delete note.", color=ft.Colors.ERROR)

    def handle_delete_current_note(self):
        """Prompts confirmation to delete currently active note."""
        if not self.state.current_note_id:
            self.show_snack_bar("Please select or save a note first to delete.", color=ft.Colors.ERROR)
            return
        self.dialog_manager.show_delete_note_confirm(
            note_title=self.state.current_note_title,
            on_confirm=lambda: self.handle_delete_note(self.state.current_note_id, self.state.current_note_title)
        )

    def handle_link_picker_open(self):
        """Opens link picker dialog for current note."""
        if not self.state.current_note_id:
            self.show_snack_bar("Please select or save a source note first.", color=ft.Colors.ERROR)
            return
        all_notes, _ = self.note_service.load_all_notes_metadata()
        self.dialog_manager.show_link_picker(
            current_note_id=self.state.current_note_id,
            all_notes=all_notes,
            on_link_selected=lambda target_id, target_title: self.handle_create_link(target_id, target_title)
        )

    def handle_create_link(self, target_id: str, target_title: str):
        """Creates bidirectional or directional link between notes."""
        success = self.note_service.create_link(self.state.current_note_id, target_id)
        if success:
            self.show_snack_bar(f"Successfully linked to '{target_title}'.")
            self.refresh_linked_notes()
            self.refresh_mind_map()
        else:
            self.show_snack_bar(f"Link to '{target_title}' already exists or failed.", color=ft.Colors.ERROR)

    def handle_unlink_note(self, target_id: str, target_title: str):
        """Removes link between current note and target note."""
        success = self.note_service.delete_link(self.state.current_note_id, target_id)
        if success:
            if self.right_panel:
                self.right_panel.mind_map_widget.invalidate_cache()
            self.refresh_linked_notes()
            self.refresh_mind_map()
            self.show_snack_bar(f"Successfully unlinked '{target_title}'.")
        else:
            self.show_snack_bar(f"Failed to unlink note: {target_title}.", color=ft.Colors.ERROR)

    def handle_save_settings(self, api_key: str, ai_provider: str, active_model_id: str, gpu_acceleration: bool = True):
        """Persists user configuration for AI models, keys, and GPU acceleration."""
        cleaned_key = (api_key or "").strip()
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
        if cleaned_key:
            os.environ["GEMINI_API_KEY"] = cleaned_key
            set_key(dotenv_path, "GEMINI_API_KEY", cleaned_key)
        else:
            os.environ.pop("GEMINI_API_KEY", None)
            if os.path.exists(dotenv_path):
                from dotenv import unset_key
                try:
                    unset_key(dotenv_path, "GEMINI_API_KEY")
                except Exception:
                    pass

        self.db_manager.set_setting("AI_PROVIDER", ai_provider)
        self.db_manager.set_setting("ACTIVE_LOCAL_MODEL", active_model_id)
        self.db_manager.set_setting("GPU_ACCELERATION", "True" if gpu_acceleration else "False")

        prov_title = "Google Gemini" if ai_provider == "gemini" else "Lokal GGUF"
        gpu_status = "Açık" if gpu_acceleration else "Kapalı"
        self.show_snack_bar(f"Ayarlar kaydedildi. (Sağlayıcı: {prov_title}, GPU: {gpu_status})")

    def handle_open_model_manager(self):
        """Opens modal model manager dialog."""
        models_dir = self.db_manager.get_setting("MODELS_DIR") or local_models_catalog.get_default_models_dir()
        active_model_id = self.db_manager.get_setting("ACTIVE_LOCAL_MODEL") or local_models_catalog.DEFAULT_MODEL_ID
        self.dialog_manager.show_model_manager_dialog(
            models_dir=models_dir,
            active_model_id=active_model_id,
            on_select_model=lambda mid: (
                self.db_manager.set_setting("ACTIVE_LOCAL_MODEL", mid),
                self.show_snack_bar(f"Aktif model seçildi: {local_models_catalog.get_model_by_id(mid).display_name if local_models_catalog.get_model_by_id(mid) else mid}")
            ),
            on_model_deleted=lambda mid: self.show_snack_bar("Model dosyası silindi.")
        )

    def handle_open_settings(self):
        """Opens main application settings dialog."""
        try:
            log_debug("Opening settings dialog...")
            self.dialog_manager.show_settings_dialog(
                theme_btn=self.theme_btn,
                auto_save_switch=self.auto_save_switch,
                current_ai_provider=self.db_manager.get_setting("AI_PROVIDER") or "gemini",
                current_active_model_id=self.db_manager.get_setting("ACTIVE_LOCAL_MODEL") or local_models_catalog.DEFAULT_MODEL_ID,
                current_gpu_acceleration=(self.db_manager.get_setting("GPU_ACCELERATION") != "False"),
                on_save_settings=self.handle_save_settings,
                on_open_model_manager=self.handle_open_model_manager
            )
        except Exception as ex:
            log_error(f"Error opening settings dialog: {ex}")
            self.show_snack_bar(f"Ayarlar açılırken hata oluştu: {ex}", color=ft.Colors.ERROR)

    def cancel_worker(self):
        """Cancels active background AI worker and frees memory."""
        if self.active_worker:
            log_debug("User requested cancellation of AI note generation.")
            self.active_worker.cancel()
            self.active_worker = None
        try:
            LocalGgufClient.unload_cached_model()
        except Exception as e:
            log_error(f"Error unloading cached model on cancel: {e}")
        self.dialog_manager.hide_loading()
        self.show_snack_bar("Not çıkarma işlemi iptal edildi.", color=ft.Colors.TERTIARY)

    def handle_ai_finished(self, generated_notes):
        """Invoked when AI note generation successfully completes."""
        self.active_worker = None
        log_debug(f"handle_ai_finished invoked with {len(generated_notes) if generated_notes else 0} notes.")
        self.dialog_manager.hide_loading()
        time.sleep(0.05)
        if generated_notes:
            try:
                self.state.set_collection_filter("")
                if self.right_panel:
                    self.right_panel.mind_map_widget.invalidate_cache()
                self.refresh_collections("")
                self.refresh_notes("")
                self.show_snack_bar(f"{len(generated_notes)} not başarıyla üretildi ve kaydedildi!")
                log_debug("handle_ai_finished UI refresh completed successfully.")
            except Exception as e:
                log_error(f"Error during UI refresh in handle_ai_finished: {e}\n{traceback.format_exc()}")
        else:
            self.show_snack_bar("Yapay zeka tarafından not üretilemedi.", color=ft.Colors.TERTIARY)

    def handle_ai_error(self, err_msg: str):
        """Invoked when AI note generation fails."""
        self.active_worker = None
        log_error(f"handle_ai_error invoked: {err_msg}")
        self.dialog_manager.hide_loading()
        self.dialog_manager.show_error("AI Not Çıkarma Hatası", f"Not çıkarma sırasında bir hata oluştu:\n{err_msg}")
        self.show_snack_bar(f"Hata: {err_msg}", color=ft.Colors.ERROR)

    async def trigger_pdf_generation(self, e=None):
        """Prompts user to select a PDF and launches background AI worker."""
        if not self.pdf_file_picker:
            return
        files = await self.pdf_file_picker.pick_files(
            dialog_title="PDF Dosyası Seç",
            allowed_extensions=["pdf"]
        )
        if files:
            pdf_path = files[0].path
            self.dialog_manager.show_loading(
                title="PDF'ten Not Çıkarılıyor",
                message="PDF'ten metin çıkarılıyor... Lütfen bekleyiniz.",
                on_cancel=self.cancel_worker
            )
            try:
                extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)
            except Exception as ex:
                self.dialog_manager.hide_loading()
                self.show_snack_bar(f"PDF okunamadı: {ex}", color=ft.Colors.ERROR)
                return

            if extracted_text and extracted_text.strip():
                self.dialog_manager.update_loading_message("Notlar üretiliyor... Lütfen bekleyiniz.")
                worker = AiNoteGeneratorWorker(
                    extracted_text,
                    on_finished=self.handle_ai_finished,
                    on_error=self.handle_ai_error,
                    on_progress=lambda msg: self.dialog_manager.update_loading_message(msg)
                )
                self.active_worker = worker
                self.page.run_thread(worker.run)
            else:
                self.dialog_manager.hide_loading()
                self.show_snack_bar("Seçilen PDF dosyası boş veya okunabilir metin içermiyor.", color=ft.Colors.ERROR)
        else:
            self.show_snack_bar("PDF dosyası seçilmedi.", color=ft.Colors.TERTIARY)

    def on_left_drag(self, e: ft.DragUpdateEvent):
        """Resizes or collapses left sidebar."""
        if not self.sidebar:
            return
        delta = e.local_delta.x if e.local_delta else 0
        new_width = (self.sidebar.width or 300) + delta
        if new_width < 120:
            self.sidebar.set_collapsed(True)
        else:
            if self.sidebar.is_collapsed:
                self.sidebar.set_collapsed(False)
            if 120 <= new_width <= 550:
                self.sidebar.width = new_width
                self.sidebar.expanded_width = new_width
        try:
            self.page.update()
        except Exception:
            pass

    def on_right_drag(self, e: ft.DragUpdateEvent):
        """Resizes or collapses right panel."""
        if not self.right_panel:
            return
        delta = e.local_delta.x if e.local_delta else 0
        new_width = (self.right_panel.width or 350) - delta
        if new_width < 120:
            self.right_panel.set_collapsed(True)
        else:
            if self.right_panel.is_collapsed:
                self.right_panel.set_collapsed(False)
            if 120 <= new_width <= 600:
                self.right_panel.width = new_width
                self.right_panel.expanded_width = new_width
        try:
            self.page.update()
        except Exception:
            pass

    def update_layout(self, e=None):
        """Triggers responsive page update."""
        try:
            self.page.update()
        except Exception:
            pass

    def handle_keyboard_event(self, e: ft.KeyboardEvent):
        """Global keyboard shortcut dispatcher."""
        if e.ctrl:
            k = e.key.lower() if e.key else ""
            if k == "s":
                self.save_current_note()
            elif k == "e":
                if self.editor_workspace:
                    self.editor_workspace.toggle_editor_mode()
            elif k == "n":
                self.guard_unsaved_changes(self.new_note_internal)
            elif k == "b":
                if self.editor_workspace:
                    self.editor_workspace.wrap_selection("**", "**", "kalın metin")
            elif k == "i":
                if self.editor_workspace:
                    self.editor_workspace.wrap_selection("*", "*", "italik metin")
            elif k == "k":
                if self.editor_workspace:
                    self.editor_workspace.open_wikilink_picker()
            elif k in ("[", "{"):
                self.toggle_sidebar()
            elif k in ("]", "}"):
                self.toggle_right_panel()

    def initial_load(self):
        """Performs initial data loading and renders default notes."""
        self.refresh_collections()
        self.refresh_notes()
        self.update_layout()
