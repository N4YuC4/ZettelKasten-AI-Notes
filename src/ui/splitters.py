# ui/splitters.py
#
# Reusable draggable splitter bars for responsive panel resizing.

import flet as ft
from typing import Callable


def create_vertical_splitter(on_drag: Callable[[ft.DragUpdateEvent], None]) -> ft.GestureDetector:
    """Creates a vertical resize splitter handle."""
    def on_hover(e):
        if e.data == "true":
            e.control.content.width = 3
            e.control.content.bgcolor = ft.Colors.PRIMARY
        else:
            e.control.content.width = 2
            e.control.content.bgcolor = ft.Colors.OUTLINE_VARIANT
        try:
            e.control.update()
        except Exception:
            pass

    return ft.GestureDetector(
        content=ft.Container(
            width=6,
            bgcolor="transparent",
            alignment=ft.Alignment(0, 0),
            content=ft.Container(
                width=2,
                bgcolor=ft.Colors.OUTLINE_VARIANT,
                border_radius=1,
            ),
            on_hover=on_hover
        ),
        mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
        on_pan_update=on_drag
    )

