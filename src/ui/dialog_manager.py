# ui/dialog_manager.py
#
# Encapsulates all modal dialog creations and display workflows.

import flet as ft
from typing import Callable, Optional, List, Tuple


class DialogManager:
    """
    Manages all modal dialog lifecycle and event dispatching for Zettelkasten AI Notes.
    """
    def __init__(self, page: ft.Page):
        self.page = page
        self._init_dialogs()

    def _init_dialogs(self):
        # 1. Simple Action Confirm Dialogs
        self.delete_note_dialog = ft.AlertDialog(
            title=ft.Text("Delete Note"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.delete_cat_dialog = ft.AlertDialog(
            title=ft.Text("Kategoriyi Sil"),
            content=ft.Text("Bu kategoriyi ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz."),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.unlink_dialog = ft.AlertDialog(
            title=ft.Text("Unlink Note"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.error_dialog = ft.AlertDialog(
            title=ft.Text("Error"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.unsaved_dialog = ft.AlertDialog(
            title=ft.Text("Unsaved Changes"),
            content=ft.Text("You have unsaved changes. Do you want to save them?"),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )
        self.create_linked_dialog = ft.AlertDialog(
            title=ft.Text("Yeni Not Bağlantısı"),
            content=ft.Text(""),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # 2. Input Dialogs
        self.new_category_field = ft.TextField(label="Category Name", border_radius=8)
        self.new_category_dialog = ft.AlertDialog(
            title=ft.Text("New Category"),
            content=self.new_category_field,
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

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

        # 4. Progress / Loading Dialog
        self.loading_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Generating AI Notes From PDF"),
            content=ft.Row([
                ft.ProgressRing(),
                ft.Text("Extracting text from PDF... This may take a moment.")
            ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
        )

        # 5. Settings Dialog
        self.api_key_field = ft.TextField(
            label="Gemini API Key",
            password=True,
            can_reveal_password=True,
            border_radius=8
        )
        self.settings_dialog = ft.AlertDialog(
            title=ft.Text("Settings"),
            content=ft.Container(width=400),
            actions=[],
            actions_alignment=ft.MainAxisAlignment.END
        )

        # Register all dialogs in page overlay
        dialogs = [
            self.delete_note_dialog,
            self.delete_cat_dialog,
            self.unlink_dialog,
            self.error_dialog,
            self.unsaved_dialog,
            self.create_linked_dialog,
            self.new_category_dialog,
            self.rename_dialog,
            self.link_note_dialog,
            self.loading_dialog,
            self.settings_dialog,
        ]
        for dlg in dialogs:
            self.page.overlay.append(dlg)

    def close(self, dlg: ft.AlertDialog) -> None:
        """Closes any given modal dialog safely."""
        dlg.open = False
        self.page.update()

    def show_error(self, title: str, message: str) -> None:
        """Displays an error alert dialog."""
        self.error_dialog.title = ft.Text(title)
        self.error_dialog.content = ft.Text(message)
        self.error_dialog.actions = [
            ft.TextButton("OK", on_click=lambda e: self.close(self.error_dialog))
        ]
        self.error_dialog.open = True
        self.page.update()

    def show_delete_note_confirm(self, note_title: str, on_confirm: Callable) -> None:
        """Shows note deletion confirmation dialog."""
        self.delete_note_dialog.content = ft.Text(
            f"Are you sure you want to delete '{note_title}'?\nThis action cannot be undone."
        )
        self.delete_note_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.delete_note_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.delete_note_dialog))
        ]
        self.delete_note_dialog.open = True
        self.page.update()

    def show_delete_category_confirm(self, category_name: str, on_confirm: Callable) -> None:
        """Shows category deletion confirmation dialog."""
        self.delete_cat_dialog.content = ft.Text(
            f"Bu kategoriyi ('{category_name}') ve içerdiği tüm notları silmek istediğinizden emin misiniz? Bu işlem geri alınamaz."
        )
        self.delete_cat_dialog.actions = [
            ft.TextButton("İptal", on_click=lambda e: self.close(self.delete_cat_dialog)),
            ft.TextButton("Sil", on_click=lambda e: (self.close(self.delete_cat_dialog), on_confirm()))
        ]
        self.delete_cat_dialog.open = True
        self.page.update()

    def show_unlink_confirm(self, target_title: str, on_confirm: Callable) -> None:
        """Shows unlinking confirmation dialog."""
        self.unlink_dialog.content = ft.Text(
            f"Are you sure you want to unlink '{target_title}' from the current note?"
        )
        self.unlink_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unlink_dialog), on_confirm())),
            ft.TextButton("No", on_click=lambda e: self.close(self.unlink_dialog))
        ]
        self.unlink_dialog.open = True
        self.page.update()

    def show_unsaved_changes_prompt(self, on_save: Callable, on_discard: Callable) -> None:
        """Shows prompt when navigating away with unsaved changes."""
        self.unsaved_dialog.actions = [
            ft.TextButton("Yes", on_click=lambda e: (self.close(self.unsaved_dialog), on_save())),
            ft.TextButton("No", on_click=lambda e: (self.close(self.unsaved_dialog), on_discard())),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.unsaved_dialog))
        ]
        self.unsaved_dialog.open = True
        self.page.update()

    def show_create_linked_note_prompt(self, target_title: str, on_create: Callable) -> None:
        """Shows prompt when clicking a WikiLink to a non-existent note."""
        self.create_linked_dialog.content = ft.Text(
            f"'{target_title}' başlıklı bir not bulunamadı.\nYeni bir not olarak oluşturmak ister misiniz?"
        )
        self.create_linked_dialog.actions = [
            ft.TextButton("Oluştur", on_click=lambda e: (self.close(self.create_linked_dialog), on_create())),
            ft.TextButton("İptal", on_click=lambda e: self.close(self.create_linked_dialog))
        ]
        self.create_linked_dialog.open = True
        self.page.update()

    def show_new_category_dialog(self, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to create a new category."""
        self.new_category_field.value = ""
        self.new_category_field.on_submit = lambda e: on_submit(self.new_category_field.value)
        self.new_category_dialog.actions = [
            ft.TextButton("Create", on_click=lambda e: on_submit(self.new_category_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.new_category_dialog))
        ]
        self.new_category_dialog.open = True
        self.page.update()

    def show_rename_dialog(self, current_title: str, on_submit: Callable[[str], None]) -> None:
        """Opens dialog to rename a note."""
        self.rename_note_field.value = current_title
        self.rename_note_field.on_submit = lambda e: on_submit(self.rename_note_field.value)
        self.rename_dialog.actions = [
            ft.TextButton("Rename", on_click=lambda e: on_submit(self.rename_note_field.value)),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.rename_dialog))
        ]
        self.rename_dialog.open = True
        self.page.update()

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
        self.link_note_dialog.open = True
        self.page.update()
        update_link_list("")

    def show_loading(self, title: str, message: str) -> None:
        """Shows loading progress dialog with custom text."""
        self.loading_dialog.title = ft.Text(title)
        self.loading_dialog.content = ft.Row([
            ft.ProgressRing(),
            ft.Text(message)
        ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
        self.loading_dialog.open = True
        self.page.update()

    def update_loading_message(self, message: str) -> None:
        """Updates text of an already open loading dialog."""
        self.loading_dialog.content = ft.Row([
            ft.ProgressRing(),
            ft.Text(message)
        ], spacing=20, alignment=ft.MainAxisAlignment.CENTER)
        self.loading_dialog.update()

    def hide_loading(self) -> None:
        """Hides the loading dialog."""
        self.loading_dialog.open = False
        self.page.update()

    def show_settings_dialog(
        self,
        theme_btn: ft.IconButton,
        auto_save_switch: ft.Switch,
        on_save_api_key: Callable[[str], None]
    ) -> None:
        """Opens settings dialog for theme, auto-save, and API key."""
        self.settings_dialog.content = ft.Column([
            ft.Row([ft.Text("Theme Mode", weight=ft.FontWeight.BOLD), theme_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Auto Save", weight=ft.FontWeight.BOLD), auto_save_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Divider(),
            ft.Text("Gemini API Configuration", weight=ft.FontWeight.BOLD),
            ft.Text("Your API Key will be securely saved to your local .env file.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            self.api_key_field
        ], tight=True, spacing=15, width=400)

        self.settings_dialog.actions = [
            ft.TextButton("Save", on_click=lambda e: (
                on_save_api_key(self.api_key_field.value.strip()),
                self.close(self.settings_dialog)
            )),
            ft.TextButton("Cancel", on_click=lambda e: self.close(self.settings_dialog))
        ]
        self.settings_dialog.open = True
        self.page.update()

