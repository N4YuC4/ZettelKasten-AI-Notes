# app_state.py
#
# Centralized application state management for Zettelkasten AI Notes.
# Replaces scattered closure variables with a testable, observable state container.

from typing import Optional, List, Tuple, Callable, Any
from models import NoteMetadata


class AppState:
    """
    Holds and manages mutable application runtime state.
    Provides observer notification callbacks on state changes.
    """
    def __init__(
        self,
        theme_mode: str = "Dark",
        auto_save: bool = True,
    ):
        self.current_note_id: Optional[str] = None
        self.current_note_title: str = ""
        self.current_note_collection: str = ""
        self.selected_collection_filter: str = ""
        self.search_query: str = ""
        
        self.is_dirty: bool = False
        self.auto_save: bool = auto_save
        self.theme_mode: str = theme_mode
        
        self.displayed_notes: List[Tuple[str, str, str]] = []
        self.all_collections: List[str] = []
        
        # Observers for reactive UI updates
        self._listeners: List[Callable[['AppState'], None]] = []

    @property
    def current_note_category(self) -> str:
        """Backward compatibility alias for current_note_collection."""
        return self.current_note_collection

    @current_note_category.setter
    def current_note_category(self, val: str) -> None:
        self.current_note_collection = val

    @property
    def selected_category_filter(self) -> str:
        """Backward compatibility alias for selected_collection_filter."""
        return self.selected_collection_filter

    @selected_category_filter.setter
    def selected_category_filter(self, val: str) -> None:
        self.selected_collection_filter = val

    @property
    def all_categories(self) -> List[str]:
        """Backward compatibility alias for all_collections."""
        return self.all_collections

    @all_categories.setter
    def all_categories(self, val: List[str]) -> None:
        self.all_collections = val

    def add_listener(self, callback: Callable[['AppState'], None]) -> None:
        """Subscribes a listener callback to state updates."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[['AppState'], None]) -> None:
        """Unsubscribes a listener callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def notify(self) -> None:
        """Notifies all subscribed listeners."""
        for callback in self._listeners:
            try:
                callback(self)
            except Exception:
                pass

    def select_note(self, note_id: Optional[str], title: str = "", collection: str = "", category: Optional[str] = None) -> None:
        """Sets the actively selected note in state."""
        col = collection if category is None else category
        self.current_note_id = note_id
        self.current_note_title = title
        self.current_note_collection = col
        self.is_dirty = False
        self.notify()

    def set_dirty(self, dirty: bool) -> None:
        """Updates unsaved changes status."""
        if self.is_dirty != dirty:
            self.is_dirty = dirty
            self.notify()

    def set_collection_filter(self, collection: str) -> None:
        """Updates the active collection filter."""
        self.selected_collection_filter = collection
        self.notify()

    # Backward compatibility alias
    set_category_filter = set_collection_filter

    def set_search_query(self, query: str) -> None:
        """Updates the active search query."""
        self.search_query = query
        self.notify()

    def set_auto_save(self, enabled: bool) -> None:
        """Toggles auto save setting."""
        self.auto_save = enabled
        self.notify()

    def set_theme_mode(self, mode: str) -> None:
        """Updates active UI theme mode ('Dark' or 'Light')."""
        self.theme_mode = mode
        self.notify()

