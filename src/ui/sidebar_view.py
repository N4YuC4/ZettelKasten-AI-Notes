# ui/sidebar_view.py
#
# Left sidebar view component: category navigation, searching, and note list management.

import flet as ft
from typing import Callable, Optional, List, Tuple


class SidebarView(ft.Container):
    """
    Encapsulates the left sidebar UI including categories, search, notes list, and footer counters.
    """
    def __init__(
        self,
        on_category_changed: Callable[[str], None],
        on_new_category_clicked: Callable[[], None],
        on_delete_category_clicked: Callable[[], None],
        on_search_changed: Callable[[str], None],
        on_note_clicked: Callable[[str, str, str], None],
        on_rename_note_clicked: Callable[[str, str], None],
        on_delete_note_clicked: Callable[[str, str], None],
        on_settings_clicked: Callable[[], None],
        on_collapse_clicked: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.on_category_changed = on_category_changed
        self.on_new_category_clicked = on_new_category_clicked
        self.on_delete_category_clicked = on_delete_category_clicked
        self.on_search_changed = on_search_changed
        self.on_note_clicked = on_note_clicked
        self.on_rename_note_clicked = on_rename_note_clicked
        self.on_delete_note_clicked = on_delete_note_clicked
        self.on_settings_clicked = on_settings_clicked
        self.on_collapse_clicked = on_collapse_clicked

        self.is_collapsed = False
        self.expanded_width = 300
        self.width = 300
        self.padding = 15
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST
        self.border_radius = 12

        self._init_controls()
        self._build_layout()

    def _init_controls(self):
        self.category_dropdown = ft.Dropdown(
            label="Category",
            options=[ft.dropdown.Option(key="", text="All Notes")],
            value="",
            expand=True,
            on_select=lambda e: self.on_category_changed(self.category_dropdown.value or "")
        )

        self.search_textfield = ft.TextField(
            hint_text="Search notes...",
            prefix_icon=ft.Icons.SEARCH,
            border_radius=8,
            on_change=lambda e: self.on_search_changed(self.search_textfield.value or "")
        )

        self.notes_listview = ft.ListView(
            expand=True,
            spacing=8,
            padding=5
        )

        self.note_count_label = ft.Text(
            "All Notes count: 0",
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
            weight=ft.FontWeight.BOLD
        )

    def _build_layout(self):
        if self.is_collapsed:
            # Narrow Rail Layout (50px wide column)
            self.content = ft.Column([
                ft.IconButton(
                    icon=ft.Icons.KEYBOARD_DOUBLE_ARROW_RIGHT,
                    icon_color=ft.Colors.PRIMARY,
                    icon_size=20,
                    tooltip="Genişlet (Ctrl+[)",
                    on_click=lambda e: self.on_collapse_clicked() if self.on_collapse_clicked else self.toggle_collapsed()
                ),
                ft.Divider(height=1, thickness=1),
                ft.IconButton(
                    icon=ft.Icons.FOLDER_OPEN,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Kategoriler",
                    on_click=lambda e: self.set_collapsed(False)
                ),
                ft.IconButton(
                    icon=ft.Icons.SEARCH,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Not Ara",
                    on_click=lambda e: self.set_collapsed(False)
                ),
                ft.IconButton(
                    icon=ft.Icons.ARTICLE,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Notlar Listesi",
                    on_click=lambda e: self.set_collapsed(False)
                ),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.SETTINGS,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Ayarlar",
                    on_click=lambda e: self.on_settings_clicked()
                )
            ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6)
        else:
            # Full Sidebar Layout
            header_controls = [
                ft.Icon(ft.Icons.AUTO_STORIES, color=ft.Colors.PRIMARY, size=22),
                ft.Text("Zettelkasten", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY, expand=True),
            ]
            if self.on_collapse_clicked:
                header_controls.append(
                    ft.IconButton(
                        icon=ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT,
                        icon_color=ft.Colors.ON_SURFACE_VARIANT,
                        icon_size=18,
                        tooltip="Daralt (Ctrl+[)",
                        on_click=lambda e: self.on_collapse_clicked() if self.on_collapse_clicked else self.toggle_collapsed()
                    )
                )

            self.content = ft.Column([
                ft.Row(header_controls, alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1, thickness=1),
                ft.Row([
                    self.category_dropdown,
                    ft.IconButton(
                        ft.Icons.ADD_BOX,
                        on_click=lambda e: self.on_new_category_clicked(),
                        tooltip="New Category"
                    ),
                    ft.IconButton(
                        ft.Icons.DELETE_FOREVER,
                        on_click=lambda e: self.on_delete_category_clicked(),
                        tooltip="Delete Category"
                    )
                ], spacing=5),
                self.search_textfield,
                ft.Divider(height=1, thickness=1),
                self.notes_listview,
                ft.Divider(height=1, thickness=1),
                ft.Row([
                    self.note_count_label,
                    ft.Row([
                        ft.IconButton(
                            ft.Icons.SETTINGS,
                            on_click=lambda e: self.on_settings_clicked(),
                            tooltip="Settings"
                        )
                    ])
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
            ], expand=True, spacing=8)

    def set_collapsed(self, collapsed: bool):
        """Switches between narrow rail and expanded sidebar."""
        if self.is_collapsed == collapsed:
            return
        self.is_collapsed = collapsed
        if collapsed:
            if self.width and self.width > 120:
                self.expanded_width = self.width
            self.width = 50
            self.padding = ft.Padding.symmetric(horizontal=4, vertical=10)
        else:
            self.width = self.expanded_width or 300
            self.padding = 15
        self._build_layout()
        try:
            if self.page:
                self.update()
        except Exception:
            pass

    def toggle_collapsed(self):
        """Toggles between expanded and narrow rail."""
        self.set_collapsed(not self.is_collapsed)

    def update_categories(self, categories: List[str], selected_category: str = ""):
        """Updates category dropdown options and selected value."""
        self.category_dropdown.options.clear()
        self.category_dropdown.options.append(ft.dropdown.Option(key="", text="All Notes"))
        added_keys = {""}
        for cat in categories:
            cat_clean = cat.strip()
            if cat_clean and cat_clean not in added_keys:
                self.category_dropdown.options.append(ft.dropdown.Option(key=cat_clean, text=cat_clean))
                added_keys.add(cat_clean)
        
        # Ensure newly created empty category is selectable before notes exist
        selected_clean = (selected_category or "").strip()
        if selected_clean and selected_clean != "All Notes" and selected_clean not in added_keys:
            self.category_dropdown.options.append(ft.dropdown.Option(key=selected_clean, text=selected_clean))
            added_keys.add(selected_clean)

        self.category_dropdown.value = selected_clean if selected_clean != "All Notes" else ""
        try:
            self.category_dropdown.update()
        except Exception:
            pass

    def render_notes(
        self,
        notes: List[Tuple[str, str, str]],
        current_note_id: Optional[str],
        category_name: str,
        total_count: int
    ):
        """Renders list of note cards."""
        self.notes_listview.controls.clear()

        for note_id, display_title, category_path in notes:
            is_selected = (note_id == current_note_id)
            bg_color = ft.Colors.PRIMARY_CONTAINER if is_selected else ft.Colors.TRANSPARENT
            text_color = ft.Colors.ON_PRIMARY_CONTAINER if is_selected else ft.Colors.ON_SURFACE

            item = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.ARTICLE, color=ft.Colors.PRIMARY),
                    ft.Text(display_title, color=text_color, weight=ft.FontWeight.BOLD, expand=True),
                    ft.IconButton(
                        icon=ft.Icons.EDIT,
                        icon_color=ft.Colors.ON_SURFACE_VARIANT,
                        on_click=lambda e, nid=note_id, title=display_title: self.on_rename_note_clicked(nid, title),
                        icon_size=16,
                        tooltip="Rename Note"
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE,
                        icon_color=ft.Colors.ERROR,
                        on_click=lambda e, nid=note_id, title=display_title: self.on_delete_note_clicked(nid, title),
                        icon_size=16,
                        tooltip="Delete Note"
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.Padding.all(8),
                border_radius=8,
                bgcolor=bg_color,
                on_click=lambda e, nid=note_id, title=display_title, cat=category_path: self.on_note_clicked(nid, title, cat),
                on_hover=lambda e, bg=bg_color: setattr(e.control, 'bgcolor', ft.Colors.SURFACE_CONTAINER_HIGH if e.data == "true" else bg) or e.control.update()
            )
            self.notes_listview.controls.append(item)

        label_prefix = category_name if category_name and category_name != "All Notes" else "All Notes"
        self.note_count_label.value = f"{label_prefix} count: {total_count}"

        try:
            self.notes_listview.update()
            self.note_count_label.update()
        except Exception:
            pass

