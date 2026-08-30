# ui/editor_workspace.py
#
# Center editor workspace view component containing MarkdownEditorWidget and action toolbar.

import flet as ft
from typing import Callable, Optional, List, Tuple
from markdown_editor_widget import MarkdownEditorWidget


class EditorWorkspaceView(ft.Container):
    """
    Encapsulates the middle workspace: Markdown editor with source/reading modes and action buttons.
    """
    def __init__(
        self,
        on_content_change: Callable[[str], None],
        on_wikilink_clicked: Callable[[str], None],
        get_all_notes_callback: Callable[[], List[Tuple[str, str, str]]],
        on_new_note_clicked: Callable[[], None],
        on_save_note_clicked: Callable[[], None],
        on_delete_note_clicked: Callable[[], None],
        on_link_note_clicked: Callable[[], None],
        on_generate_ai_clicked: Callable[[], None],
        on_blur: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.on_content_change = on_content_change
        self.on_wikilink_clicked = on_wikilink_clicked
        self.get_all_notes_callback = get_all_notes_callback
        self.on_new_note_clicked = on_new_note_clicked
        self.on_save_note_clicked = on_save_note_clicked
        self.on_delete_note_clicked = on_delete_note_clicked
        self.on_link_note_clicked = on_link_note_clicked
        self.on_generate_ai_clicked = on_generate_ai_clicked
        self.on_blur = on_blur

        self.expand = True
        self.padding = 15
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW
        self.border_radius = 12

        self._init_controls()
        self._build_layout()

    def _init_controls(self):
        # Live Markdown Editor (Source and Reading modes)
        self.live_editor = MarkdownEditorWidget(
            on_content_change=self.on_content_change,
            on_wikilink_clicked=self.on_wikilink_clicked,
            get_all_notes_callback=self.get_all_notes_callback,
            on_save_shortcut=self.on_save_note_clicked,
            on_blur=self.on_blur,
        )

        # Retain action_buttons reference for unit tests / programmatic access
        self.action_buttons = ft.Row([
            ft.Button(content="New Note", on_click=lambda e: self.on_new_note_clicked()),
            ft.Button(content="Save Note", on_click=lambda e: self.on_save_note_clicked()),
            ft.Button(content="Delete Note", on_click=lambda e: self.on_delete_note_clicked()),
            ft.Button(content="Link Note", on_click=lambda e: self.on_link_note_clicked()),
            ft.Button(content="Generate AI Notes", on_click=lambda e: self.on_generate_ai_clicked()),
        ], visible=False)

    def _build_layout(self):
        self.content = ft.Column([
            self.live_editor,
        ], expand=True, spacing=0)

    def get_content(self) -> str:
        """Returns the current markdown content in editor."""
        return self.live_editor.get_value()

    def set_content(self, text: str, mark_dirty: bool = False):
        """Sets the editor content."""
        self.live_editor.set_value(text, mark_dirty=mark_dirty)

    def set_dirty(self, is_dirty: bool):
        """Updates dirty state indicator in the editor."""
        self.live_editor.set_dirty(is_dirty)

    def toggle_editor_mode(self):
        """Toggles between Source and Reading modes."""
        self.live_editor.toggle_mode()

    def wrap_selection(self, prefix: str, suffix: str, default_text: str = ""):
        """Applies markdown wrapping to selected text."""
        self.live_editor.wrap_selection(prefix, suffix, default_text)

    def open_wikilink_picker(self):
        """Opens WikiLink insertion picker dialog."""
        self.live_editor.open_wikilink_picker()

