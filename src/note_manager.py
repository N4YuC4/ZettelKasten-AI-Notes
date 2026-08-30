# note_manager.py
#
# Backward-compatibility adapter for Zettelkasten note management.
# Delegates core operations to note_service for clean separation of concerns.

from typing import Optional, List, Tuple, Set
import note_service
from note_service import NoteService, generate_unique_id, sanitize_title as get_sanitized_title


def save_note(db_manager, note_id: Optional[str], note_content: str, category_path: str = "") -> Tuple[str, str]:
    """Saves or updates a note in the database via NoteService."""
    service = NoteService(db_manager)
    return service.save_note(note_id, note_content, category_path)


def delete_note(db_manager, note_id: str) -> bool:
    """Deletes a note from the database via NoteService."""
    service = NoteService(db_manager)
    return service.delete_note(note_id)


def rename_note(db_manager, note_id: str, new_title: str, category_path: str = "") -> Tuple[bool, str]:
    """Renames a note via NoteService."""
    service = NoteService(db_manager)
    return service.rename_note(note_id, new_title, category_path)


def load_all_notes_metadata(db_manager) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Loads metadata and categories via NoteService."""
    service = NoteService(db_manager)
    return service.load_all_notes_metadata()


def get_note_content(db_manager, note_id: str) -> Optional[str]:
    """Retrieves content of a note via NoteService."""
    service = NoteService(db_manager)
    return service.get_note_content(note_id)


def create_category(category_name: str) -> bool:
    """Validates category name format."""
    return bool(category_name and category_name.strip())
