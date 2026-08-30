# ui/__init__.py
# UI components package for Zettelkasten AI Notes

from ui.dialog_manager import DialogManager
from ui.sidebar_view import SidebarView
from ui.right_panel_view import RightPanelView
from ui.editor_workspace import EditorWorkspaceView
from ui.splitters import create_vertical_splitter

__all__ = [
    "DialogManager",
    "SidebarView",
    "RightPanelView",
    "EditorWorkspaceView",
    "create_vertical_splitter",
]

