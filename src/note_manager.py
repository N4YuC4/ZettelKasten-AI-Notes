# note_manager.py
#
# Backward-compatibility adapter for Zettelkasten note management.
# Delegates core operations to note_service for clean separation of concerns.

from typing import Optional, List, Tuple, Set
import note_service
from note_service import NoteService, generate_unique_id, sanitize_title as get_sanitized_title


def save_note(db_manager, note_id: Optional[str], note_content: str, collection_path: str = "", category_path: Optional[str] = None) -> Tuple[str, str]:
    """Saves or updates a note in the database via NoteService."""
    service = NoteService(db_manager)
    col = collection_path if category_path is None else category_path
    return service.save_note(note_id, note_content, col)


def delete_note(db_manager, note_id: str) -> bool:
    """Deletes a note from the database via NoteService."""
    service = NoteService(db_manager)
    return service.delete_note(note_id)


def rename_note(db_manager, note_id: str, new_title: str, collection_path: Optional[str] = None, category_path: Optional[str] = None) -> Tuple[bool, str]:
    """Renames a note via NoteService."""
    service = NoteService(db_manager)
    col = collection_path if category_path is None else category_path
    return service.rename_note(note_id, new_title, col)


def load_all_notes_metadata(db_manager) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Loads metadata and collections via NoteService."""
    service = NoteService(db_manager)
    return service.load_all_notes_metadata()


def get_note_content(db_manager, note_id: str) -> Optional[str]:
    """Retrieves content of a note via NoteService."""
    service = NoteService(db_manager)
    return service.get_note_content(note_id)


def create_collection(collection_name: str) -> bool:
    """Validates collection name format."""
    return bool(collection_name and collection_name.strip())


# Backward compatibility alias
create_category = create_collection
