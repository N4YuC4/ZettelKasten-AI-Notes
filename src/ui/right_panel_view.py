# ui/right_panel_view.py
#
# Right sidebar view component: Interactive Mind Map visualization and Linked Notes list.

import flet as ft
from typing import Callable, Optional, List, Tuple
from mind_map_widget import MindMapWidget


class RightPanelView(ft.Container):
    """
    Encapsulates the right sidebar containing the MindMapWidget and Linked Connections list.
    """
    def __init__(
        self,
        db_manager,
        on_map_note_selected: Callable[[str], None],
        on_linked_note_clicked: Callable[[str], None],
        on_unlink_note_clicked: Callable[[str, str], None],
        on_collapse_clicked: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.db_manager = db_manager
        self.on_map_note_selected = on_map_note_selected
        self.on_linked_note_clicked = on_linked_note_clicked
        self.on_unlink_note_clicked = on_unlink_note_clicked
        self.on_collapse_clicked = on_collapse_clicked

        self.is_collapsed = False
        self.expanded_width = 350
        self.width = 350
        self.padding = 15
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST
        self.border_radius = 12

        self._init_controls()
        self._build_layout()

    def _init_controls(self):
        # Mind Map Widget instance
        self.mind_map_widget = MindMapWidget(self.db_manager, on_note_selected=self.on_map_note_selected)

        # Mind Map Container
        self.mind_map_container = ft.Container(
            content=self.mind_map_widget,
            height=350,
            border_radius=8,
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(1.5, ft.Colors.ON_INVERSE_SURFACE)
        )

        # Splitter between Mind Map and Linked Notes
        self.mind_map_splitter = ft.GestureDetector(
            content=ft.Container(
                height=6,
                bgcolor="transparent",
                alignment=ft.Alignment(0, 0),
                content=ft.Container(
                    height=2,
                    bgcolor=ft.Colors.OUTLINE_VARIANT,
                    border_radius=1,
                ),
                on_hover=self._horizontal_splitter_hover
            ),
            mouse_cursor=ft.MouseCursor.RESIZE_UP_DOWN,
            on_pan_update=self._mind_map_drag
        )

        # Linked Notes list view
        self.linked_notes_listview = ft.ListView(
            expand=True,
            spacing=5,
            padding=5
        )

    def _horizontal_splitter_hover(self, e):
        if e.data == "true":
            e.control.content.height = 3
            e.control.content.bgcolor = ft.Colors.PRIMARY
        else:
            e.control.content.height = 2
            e.control.content.bgcolor = ft.Colors.OUTLINE_VARIANT
        try:
            e.control.update()
        except Exception:
            pass

    def _mind_map_drag(self, e: ft.DragUpdateEvent):
        delta = e.local_delta.y if e.local_delta else 0
        new_height = self.mind_map_container.height + delta
        if 150 <= new_height <= 600:
            self.mind_map_container.height = new_height
            if self.page:
                self.page.update()

    def _build_layout(self):
        if self.is_collapsed:
            # Narrow Rail Layout (50px wide column)
            self.content = ft.Column([
                ft.IconButton(
                    icon=ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT,
                    icon_color=ft.Colors.PRIMARY,
                    icon_size=20,
                    tooltip="Genişlet (Ctrl+])",
                    on_click=lambda e: self.on_collapse_clicked() if self.on_collapse_clicked else self.toggle_collapsed()
                ),
                ft.Divider(height=1, thickness=1),
                ft.IconButton(
                    icon=ft.Icons.HUB,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Zihin Haritası",
                    on_click=lambda e: self.set_collapsed(False)
                ),
                ft.IconButton(
                    icon=ft.Icons.LINK,
                    icon_color=ft.Colors.ON_SURFACE_VARIANT,
                    icon_size=18,
                    tooltip="Bağlantılar",
                    on_click=lambda e: self.set_collapsed(False)
                ),
            ], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6)
        else:
            # Full Right Panel Layout
            header_controls = [
                ft.Icon(ft.Icons.HUB, color=ft.Colors.PRIMARY, size=20),
                ft.Text("Mind Map", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY, expand=True),
            ]
            if self.on_collapse_clicked:
                header_controls.append(
                    ft.IconButton(
                        icon=ft.Icons.KEYBOARD_DOUBLE_ARROW_RIGHT,
                        icon_color=ft.Colors.ON_SURFACE_VARIANT,
                        icon_size=18,
                        tooltip="Daralt (Ctrl+])",
                        on_click=lambda e: self.on_collapse_clicked() if self.on_collapse_clicked else self.toggle_collapsed()
                    )
                )

            self.content = ft.Column([
                ft.Row(header_controls, alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=1, thickness=1),
                self.mind_map_container,
                self.mind_map_splitter,
                ft.Divider(height=1, thickness=1),
                ft.Row([
                    ft.Icon(ft.Icons.LINK, color=ft.Colors.PRIMARY, size=18),
                    ft.Text("Linked Connections", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                self.linked_notes_listview
            ], expand=True, spacing=8)

    def set_collapsed(self, collapsed: bool):
        """Switches between narrow rail and expanded right panel."""
        if self.is_collapsed == collapsed:
            return
        self.is_collapsed = collapsed
        if collapsed:
            if self.width and self.width > 120:
                self.expanded_width = self.width
            self.width = 50
            self.padding = ft.Padding.symmetric(horizontal=4, vertical=10)
        else:
            self.width = self.expanded_width or 350
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

    def update_mind_map(
        self,
        notes_metadata: List[Tuple[str, str, str]],
        links: List[Tuple[str, str]],
        current_note_id: Optional[str]
    ):
        """Refreshes the mind map layout."""
        self.mind_map_widget.update_map(notes_metadata, links, current_note_id)

    def render_linked_notes(
        self,
        current_note_id: Optional[str],
        linked_notes: List[Tuple[str, str]]
    ):
        """
        Renders the list of linked notes for the currently active note.
        linked_notes: List of (note_id, note_title)
        """
        self.linked_notes_listview.controls.clear()

        if current_note_id:
            if linked_notes:
                for linked_id, linked_title in linked_notes:
                    item = ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.LINK, color=ft.Colors.PRIMARY),
                            ft.Text(linked_title, color=ft.Colors.ON_SURFACE, weight=ft.FontWeight.W_500, expand=True),
                            ft.IconButton(
                                icon=ft.Icons.LINK_OFF,
                                icon_color=ft.Colors.ERROR,
                                on_click=lambda e, nid=linked_id, title=linked_title: self.on_unlink_note_clicked(nid, title),
                                icon_size=18,
                                tooltip="Unlink Note"
                            )
                        ]),
                        padding=5,
                        border_radius=5,
                        on_click=lambda e, nid=linked_id: self.on_linked_note_clicked(nid),
                        on_hover=lambda e: setattr(e.control, 'bgcolor', ft.Colors.SURFACE_CONTAINER_HIGH if e.data == "true" else ft.Colors.TRANSPARENT) or e.control.update()
                    )
                    self.linked_notes_listview.controls.append(item)
            else:
                self.linked_notes_listview.controls.append(
                    ft.Text("No linked notes.", color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
                )
        else:
            self.linked_notes_listview.controls.append(
                ft.Text("Select a note to see its links.", color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
            )

        try:
            self.linked_notes_listview.update()
        except Exception:
            pass

