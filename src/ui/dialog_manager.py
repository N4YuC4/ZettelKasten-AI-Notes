# ui/dialog_manager.py
#
# Encapsulates all modal dialog creations and display workflows.

import flet as ft
from typing import Callable, Optional, List, Tuple, Any, Dict
from .dialogs import build_settings_dialog, build_model_manager_dialog


class DialogManager:
    """
    Manages all modal dialog lifecycle and event dispatching for Zettelkasten AI Notes.
    """
    def __init__(self, page: ft.Page):
        self.page = page
        self._init_dialogs()

    def _init_dialogs(self):
        # 1. Simple Action Confirm Dialogs
        self.delete_note_msg = ft.Text("")
        self.delete_note_dialog = ft.AlertDialog(
            title=ft.Text("Delete Note"),
            content=self.delete_note_msg,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.delete_col_dialog = ft.AlertDialog(
            title=ft.Text("Delete Collection"),
            content=ft.Text("Are you sure you want to delete this collection and all its notes? This action cannot be undone."),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.delete_cat_dialog = self.delete_col_dialog
        self.unlink_msg = ft.Text("")
        self.unlink_dialog = ft.AlertDialog(
            title=ft.Text("Unlink Note"),
            content=self.unlink_msg,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.error_title = ft.Text("Error")
        self.error_msg = ft.Text("")
        self.error_dialog = ft.AlertDialog(
            title=self.error_title,
            content=self.error_msg,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.unsaved_dialog = ft.AlertDialog(
            title=ft.Text("Unsaved Changes"),
            content=ft.Text("You have unsaved changes. Do you want to save them?"),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.create_linked_msg = ft.Text("")
        self.create_linked_dialog = ft.AlertDialog(
            title=ft.Text("New Note Link"),
            content=self.create_linked_msg,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 2. Input Dialogs
        self.new_collection_field = ft.TextField(label="Collection Name", border_radius=8)
        self.new_collection_dialog = ft.AlertDialog(
            title=ft.Text("New Collection"),
            content=self.new_collection_field,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.new_category_field = self.new_collection_field
        self.new_category_dialog = self.new_collection_dialog

        self.rename_note_field = ft.TextField(label="New Note Title", border_radius=8)
        self.rename_dialog = ft.AlertDialog(
            title=ft.Text("Rename Note"),
            content=self.rename_note_field,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 3. Complex Selection Dialogs (Link Note)
        self.link_search_field = ft.TextField(
            hint_text="Search notes to link...",
            prefix_icon=ft.Icons.SEARCH,
            border_radius=8
        )
        self.linkable_notes_listview = ft.ListView(spacing=5, height=300)
        self.link_note_dialog = ft.AlertDialog(
            title=ft.Text("Select Note to Link"),
            content=ft.Column([
                self.link_search_field,
                self.linkable_notes_listview
            ], tight=True, spacing=15, width=400),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 4. Progress / Loading Dialog (Persistent controls to prevent duplicate DialogRoute stacking)
        self.loading_title = ft.Text("Generating AI Notes", weight=ft.FontWeight.BOLD)
        self.loading_ring = ft.ProgressRing(width=28, height=28, stroke_width=3)
        self.loading_msg = ft.Text("Extracting text... Please wait.", size=13)
        self.loading_cancel_btn = ft.TextButton("Cancel", visible=False)
        self.loading_dialog = ft.AlertDialog(
            modal=True,
            title=self.loading_title,
            content=ft.Row([
                self.loading_ring,
                self.loading_msg
            ], spacing=15, alignment=ft.MainAxisAlignment.CENTER),
            actions=[self.loading_cancel_btn],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 5. Settings Dialog
        self.api_key_field = ft.TextField(
            label="Gemini API Key",
            password=True,
            can_reveal_password=True,
            border_radius=8
        )
        self.settings_dialog = ft.AlertDialog(
            title=ft.Row([
                ft.Icon(ft.Icons.SETTINGS, color=ft.Colors.PRIMARY),
                ft.Text("Settings", weight=ft.FontWeight.BOLD),
            ], spacing=10),
            content=ft.Container(width=580),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 6. Model Manager Dialog
        self.model_manager_dialog = ft.AlertDialog(
            title=ft.Row([
                ft.Icon(ft.Icons.MEMORY, color=ft.Colors.PRIMARY),
                ft.Text("Local Model Manager", weight=ft.FontWeight.BOLD),
            ], spacing=10),
            content=ft.Container(width=540),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 7. Reset Settings Confirmation Dialog
        self.reset_settings_dialog = ft.AlertDialog(
            title=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.ERROR),
                ft.Text("Reset Settings to Default", weight=ft.FontWeight.BOLD),
            ], spacing=10),
            content=ft.Text(
                "Are you sure you want to reset all settings to factory defaults?\n"
                "This will not delete your notes, but all custom preferences and API keys will be restored to defaults."
            ),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # Register all dialogs in page overlay with on_dismiss sync
        dialogs = [
            self.delete_note_dialog,
            self.delete_col_dialog,
            self.unlink_dialog,
            self.error_dialog,
            self.unsaved_dialog,
            self.create_linked_dialog,
            self.new_collection_dialog,
            self.rename_dialog,
            self.link_note_dialog,
            self.loading_dialog,
            self.settings_dialog,
            self.model_manager_dialog,
            self.reset_settings_dialog,
        ]
        for dlg in dialogs:
            dlg.on_dismiss = lambda e, d=dlg: self._handle_dialog_dismiss(d)
            self.page.overlay.append(dlg)

    def _handle_dialog_dismiss(self, dlg: ft.AlertDialog) -> None:
        """Keeps dialog open state strictly synchronized when closed via backdrop or ESC."""
        dlg.open = False

    def _safe_open(self, dlg: ft.AlertDialog) -> None:
        """Opens a modal dialog safely, pushing state updates to the dialog and page."""
        dlg.open = True
        try:
            dlg.update()
        except Exception:
            pass
        try:
            self.page.update()
        except Exception:
            pass

    def _safe_close(self, dlg: ft.AlertDialog) -> None:
        """Closes a modal dialog safely, pushing state updates to the dialog and page."""
        dlg.open = False
        try:
            dlg.update()
        except Exception:
            pass
        try:
            self.page.update()
        except Exception:
            pass

    def close(self, dlg: ft.AlertDialog) -> None:
        """Closes any given modal dialog safely."""
        self._safe_close(dlg)

    def show_error(self, title: str, message: str) -> None:
        """Displays an error alert dialog."""
        self.error_title.value = title
        self.error_msg.value = message
        self.error_dialog.actions = [
            ft.TextButton("OK", on_click=lambda e: self.close(self.error_dialog))
        ]
        self._safe_open(self.error_dialog)

    def show_delete_note_confirm(self, note_title: str, on_confirm: Callable) -> None:
        """Shows note deletion confirmation dialog."""
        self.delete_note_msg.value = (
            f"Are you sure you want to delete '{note_title}'?\nThis action cannot be undone."
        )
        self.delete_note_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.delete_note_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.delete_note_dialog))
        ]
        self._safe_open(self.delete_note_dialog)

    def show_delete_collection_confirm(self, collection_name: str, on_confirm: Callable) -> None:
        """Shows collection deletion confirmation dialog."""
        self.delete_col_dialog.content = ft.Text(
            f"Are you sure you want to delete this collection ('{collection_name}') and all its notes? This action cannot be undone."
        )
        self.delete_col_dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.delete_col_dialog)),
            ft.TextButton("Delete", on_click=lambda e: (self.close(self.delete_col_dialog), on_confirm()))
        ]
        self._safe_open(self.delete_col_dialog)

    # Backward compatibility alias
    show_delete_category_confirm = show_delete_collection_confirm

    def show_reset_settings_confirm(self, on_confirm: Callable) -> None:
        """Shows confirmation prompt before resetting settings to defaults."""
        self.reset_settings_dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.reset_settings_dialog)),
            ft.TextButton("Reset", style=ft.ButtonStyle(color=ft.Colors.ERROR), on_click=lambda e: (self.close(self.reset_settings_dialog), on_confirm()))
        ]
        self._safe_open(self.reset_settings_dialog)

    def show_unlink_confirm(self, target_title: str, on_confirm: Callable) -> None:
        """Shows unlinking confirmation dialog."""
        self.unlink_msg.value = (
            f"Are you sure you want to unlink '{target_title}' from the current note?"
        )
        self.unlink_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unlink_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.unlink_dialog))
        ]
        self._safe_open(self.unlink_dialog)

    def show_unsaved_changes_prompt(self, on_save: Callable, on_discard: Callable) -> None:
        """Shows prompt when navigating away with unsaved changes."""
        self.unsaved_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unsaved_dialog), on_save())),
            ft.TextButton("No", on_click=lambda e: (self.close(self.unsaved_dialog), on_discard())),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.unsaved_dialog))
        ]
        self._safe_open(self.unsaved_dialog)

    def show_create_linked_note_prompt(self, target_title: str, on_create: Callable) -> None:
        """Shows prompt when clicking a WikiLink to a non-existent note."""
        self.create_linked_msg.value = (
            f"No note titled '{target_title}' was found.\nWould you like to create it as a new note?"
        )
        self.create_linked_dialog.actions = [
            ft.TextButton("Create", on_click=lambda e: (self.close(self.create_linked_dialog), on_create())),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.create_linked_dialog))
        ]
        self._safe_open(self.create_linked_dialog)

    def show_new_collection_dialog(self, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to create a new collection."""
        self.new_collection_field.value = ""
        self.new_collection_field.on_submit = lambda e: on_submit(self.new_collection_field.value)
        self.new_collection_dialog.actions = [
            ft.TextButton("Create", on_click=lambda e: on_submit(self.new_collection_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.new_collection_dialog))
        ]
        self._safe_open(self.new_collection_dialog)

    # Backward compatibility alias
    show_new_category_dialog = show_new_collection_dialog

    def show_rename_dialog(self, current_title: str, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to rename a note."""
        self.rename_note_field.value = current_title
        self.rename_note_field.on_submit = lambda e: on_submit(self.rename_note_field.value)
        self.rename_dialog.actions = [
            ft.TextButton("Rename", on_click=lambda e: on_submit(self.rename_note_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.rename_dialog))
        ]
        self._safe_open(self.rename_dialog)

    def show_link_picker(
        self,
        current_note_id: Optional[str],
        all_notes: List[Tuple[str, str, str]],
        on_link_selected: Callable[[str, str], None]
    ) -> None:
        """Opens search list dialog to link to another note."""
        def update_link_list(query=""):
            self.linkable_notes_listview.controls.clear()
            q = query.lower().strip()
            for nid, title, _ in all_notes:
                if nid == current_note_id:
                    continue
                if not q or q in title.lower():
                    item = ft.ListTile(
                        leading=ft.Icon(ft.Icons.ARTICLE_OUTLINED, color=ft.Colors.PRIMARY),
                        title=ft.Text(title, weight=ft.FontWeight.W_500),
                        on_click=lambda e, target_id=nid, target_title=title: (
                            self.close(self.link_note_dialog),
                            on_link_selected(target_id, target_title)
                        ),
                        hover_color=ft.Colors.ON_INVERSE_SURFACE
                    )
                    self.linkable_notes_listview.controls.append(item)
            self.linkable_notes_listview.update()

        self.link_search_field.value = ""
        self.link_search_field.on_change = lambda e: update_link_list(self.link_search_field.value)
        self.link_note_dialog.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.link_note_dialog))
        ]
        self._safe_open(self.link_note_dialog)
        update_link_list("")

    def show_loading(
        self,
        title: str = "Generating AI Notes",
        message: str = "Preparing...",
        on_cancel: Optional[Callable[[], None]] = None
    ) -> None:
        """Shows loading progress dialog with custom text and optional cancellation."""
        self.loading_title.value = title
        self.loading_msg.value = message
        if on_cancel:
            self.loading_cancel_btn.visible = True
            self.loading_cancel_btn.on_click = lambda e: on_cancel()
        else:
            self.loading_cancel_btn.visible = False
            self.loading_cancel_btn.on_click = None
        self._safe_open(self.loading_dialog)

    def update_loading_message(self, message: str) -> None:
        """Updates text of an already open loading dialog."""
        self.loading_msg.value = message
        try:
            self.loading_msg.update()
        except Exception:
            try:
                self.loading_dialog.update()
            except Exception:
                try:
                    self.page.update()
                except Exception:
                    pass

    def hide_loading(self) -> None:
        """Hides the loading dialog."""
        self._safe_close(self.loading_dialog)

    def show_settings_dialog(
        self,
        theme_btn: ft.IconButton,
        auto_save_switch: ft.Switch,
        current_ai_provider: str,
        current_active_model_id: str,
        on_save_settings: Optional[Callable[..., None]] = None,
        on_open_model_manager: Optional[Callable[[], None]] = None,
        current_gpu_acceleration: bool = True,
        current_api_key: str = "",
        current_db_path: str = "",
        current_models_dir: str = "",
        current_custom_prompt: str = "",
        current_confirm_delete_note: bool = True,
        current_confirm_delete_col: bool = True,
        current_custom_db_path: str = "",
        on_reset_defaults: Optional[Callable[[], None]] = None,
        on_setting_changed: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        """Opens comprehensive 4-tab settings dialog with live hardware status and instant auto-save."""
        build_settings_dialog(
            dm=self,
            theme_btn=theme_btn,
            auto_save_switch=auto_save_switch,
            current_ai_provider=current_ai_provider,
            current_active_model_id=current_active_model_id,
            on_save_settings=on_save_settings,
            on_open_model_manager=on_open_model_manager,
            current_gpu_acceleration=current_gpu_acceleration,
            current_api_key=current_api_key,
            current_db_path=current_db_path,
            current_models_dir=current_models_dir,
            current_custom_prompt=current_custom_prompt,
            current_confirm_delete_note=current_confirm_delete_note,
            current_confirm_delete_col=current_confirm_delete_col,
            current_custom_db_path=current_custom_db_path,
            on_reset_defaults=on_reset_defaults,
            on_setting_changed=on_setting_changed,
        )

    def show_model_manager_dialog(
        self,
        models_dir: str,
        active_model_id: str,
        on_select_model: Callable[[str], None],
        on_model_deleted: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Opens Model Manager dialog displaying curated models, hardware status, and download controls."""
        build_model_manager_dialog(
            dm=self,
            models_dir=models_dir,
            active_model_id=active_model_id,
            on_select_model=on_select_model,
            on_model_deleted=on_model_deleted,
        )
