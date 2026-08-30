# note_service.py
#
# Core business logic and service layer for Zettelkasten notes management.
# Decoupled from presentation and raw UI handlers.

import re
import uuid
import math
import urllib.parse
from typing import Optional, List, Tuple, Set
from models import Note, NoteMetadata, NoteLink, DocumentStats
from logger import log_debug, log_error


def generate_unique_id() -> str:
    """Creates a unique UUID4 string for new notes."""
    return str(uuid.uuid4())


def sanitize_title(content: str) -> str:
    """
    Extracts and sanitizes a title from the first non-empty line of the note content.
    Removes Markdown heading symbols, inline formatting, links, and filesystem special characters.
    """
    if not content:
        return "Untitled Note"

    first_non_empty_line = ""
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped:
            first_non_empty_line = stripped
            break

    if not first_non_empty_line:
        return "Untitled Note"

    # Remove Markdown heading syntax (e.g., #, ##, etc.) from the beginning of the line
    cleaned = re.sub(r'^#+\s*', '', first_non_empty_line)
    # Remove bold/italic markers
    cleaned = re.sub(r'(\*\*|__|\*|_)', '', cleaned)
    # Remove inline code markers
    cleaned = re.sub(r'`', '', cleaned)
    # Remove strikethrough markers
    cleaned = re.sub(r'~~', '', cleaned)
    # Remove image syntax
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
    # Extract text from standard markdown links: [text](url) -> text
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned)
    # Extract text from wikilinks: [[target|alias]] -> alias, [[target]] -> target
    def _extract_wikilink_text(m):
        inner = m.group(1).strip()
        if "|" in inner:
            return inner.split("|", 1)[1].strip()
        return inner
    cleaned = re.sub(r'\[\[(.*?)\]\]', _extract_wikilink_text, cleaned)
    # Remove remaining special characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '', cleaned)
    
    cleaned = cleaned.strip()
    return cleaned if cleaned else "Untitled Note"


def disambiguate_title(base_title: str, existing_titles: Set[str]) -> str:
    """
    Ensures a title is unique among a set of existing titles by appending (2), (3), etc.
    """
    if base_title not in existing_titles:
        return base_title

    counter = 2
    candidate = f"{base_title} ({counter})"
    while candidate in existing_titles:
        counter += 1
        candidate = f"{base_title} ({counter})"
    return candidate


def preserve_single_linebreaks(text: str) -> str:
    """
    Adds two spaces (Markdown hard break) at line ends outside code blocks
    so that single newlines render properly in Markdown views.
    """
    if not text:
        return ""
    lines = text.split("\n")
    processed = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            processed.append(line)
        elif in_code:
            processed.append(line)
        else:
            if line.strip() and not line.endswith("  "):
                processed.append(line + "  ")
            else:
                processed.append(line)
    return "\n".join(processed)


def process_markdown_wikilinks(text: str) -> str:
    """
    Converts [[Note Title]] or [[Target Note|Custom Alias]] WikiLinks
    into standard Markdown links using the zettel://note/ protocol.
    """
    if not text:
        return ""

    def replace_wikilink(match):
        inner = match.group(1).strip()
        if "|" in inner:
            target, alias = inner.split("|", 1)
            target = target.strip()
            alias = alias.strip()
        else:
            target = inner
            alias = inner
        
        encoded_target = urllib.parse.quote(target)
        return f"[🔗 {alias}](zettel://note/{encoded_target})"

    text = re.sub(r'\[\[(.*?)\]\]', replace_wikilink, text)
    return preserve_single_linebreaks(text)


def calculate_document_stats(text: str) -> DocumentStats:
    """Calculates word count, character count, line count, and estimated reading time."""
    if not text:
        return DocumentStats()

    words = len(text.split())
    chars = len(text)
    lines = len(text.splitlines()) if text else 0
    reading_time = max(1, math.ceil(words / 200)) if words > 0 else 0

    return DocumentStats(
        words=words,
        chars=chars,
        lines=lines,
        reading_time_min=reading_time
    )


def toggle_task_in_text(text: str, task_index: int) -> str:
    """Toggles the N-th task checkbox (- [ ] <-> - [x]) in the markdown text."""
    if not text:
        return text

    pattern = re.compile(r'^(\s*[-*+]\s+\[)( |x|X)(\]\s+.*)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if 0 <= task_index < len(matches):
        match = matches[task_index]
        current_state = match.group(2)
        new_state = " " if current_state.lower() == "x" else "x"
        start, end = match.span()
        new_line = match.group(1) + new_state + match.group(3)
        return text[:start] + new_line + text[end:]
    return text


class NoteService:
    """
    Orchestrates domain business logic for note authoring, linking, and taxonomy.
    """
    sanitize_title = staticmethod(sanitize_title)
    disambiguate_title = staticmethod(disambiguate_title)

    def __init__(self, db_manager):
        self.db = db_manager

    def save_note(self, note_id: Optional[str], content: str, category: str = "") -> Tuple[str, str]:
        """
        Saves or updates a note with sanitized title extraction.
        Returns (note_id, sanitized_title).
        """
        title = sanitize_title(content)
        category_clean = category.strip() if category else ""

        if note_id:
            self.db.update_note(note_id, title, content, category_clean)
            return note_id, title
        else:
            new_id = generate_unique_id()
            self.db.insert_note(new_id, title, content, category_clean)
            return new_id, title

    def rename_note(self, note_id: str, new_title: str, category: Optional[str] = None) -> Tuple[bool, str]:
        """
        Renames a note and updates its first line heading in content.
        Preserves existing category if category is None or empty string.
        Returns (success, new_title_or_error_message).
        """
        if not new_title or not new_title.strip():
            return False, "New title cannot be empty or result in an empty sanitized title."

        sanitized = sanitize_title(new_title)
        if not sanitized or (sanitized == "Untitled Note" and not new_title.strip()):
            return False, "New title cannot be empty or result in an empty sanitized title."

        note_data = self.db.get_note(note_id)
        if not note_data:
            return False, "Note not found."

        existing_category = note_data[3] if isinstance(note_data, (tuple, list)) else getattr(note_data, "category", "")
        cat_to_save = category if (category is not None and category != "") else (existing_category or "")

        # Update the first line of the content with new heading
        current_content = note_data[2] if isinstance(note_data, (tuple, list)) else note_data.content
        lines = current_content.split('\n')
        if lines:
            lines[0] = f"# {sanitized}"
        else:
            lines = [f"# {sanitized}"]
        updated_content = '\n'.join(lines)

        self.db.update_note(note_id, sanitized, updated_content, cat_to_save)
        log_debug(f"Note with ID {note_id} renamed to {sanitized}.")
        return True, sanitized

    def delete_note(self, note_id: str) -> bool:
        """Deletes a note and its associated links."""
        try:
            return self.db.delete_note(note_id)
        except Exception as e:
            log_error(f"Error deleting note {note_id}: {e}")
            return False

    def get_note(self, note_id: str) -> Optional[Note]:
        """Retrieves typed Note model for a note ID."""
        return self.db.get_note_model(note_id)

    def get_note_content(self, note_id: str) -> Optional[str]:
        """Retrieves raw markdown content of a note."""
        note = self.db.get_note(note_id)
        if note:
            return note[2] if isinstance(note, (tuple, list)) else note.content
        return None

    def load_all_notes_metadata(self) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Returns all notes metadata tuples (id, title, category) and sorted unique categories."""
        notes_metadata_from_db, all_categories_from_db = self.db.get_all_notes_metadata()
        metadata_list = [(nid, title, cat) for nid, title, cat in notes_metadata_from_db]
        categories_list = sorted([cat for cat in all_categories_from_db if cat])
        return metadata_list, categories_list

    def create_link(self, source_id: str, target_id: str) -> bool:
        """Creates a link between two notes."""
        if not source_id or not target_id or source_id == target_id:
            return False
        return self.db.insert_note_link(source_id, target_id)

    def delete_link(self, source_id: str, target_id: str) -> bool:
        """Deletes a link between two notes."""
        return self.db.delete_note_link(source_id, target_id)

    def get_linked_note_ids(self, note_id: str) -> List[str]:
        """Returns IDs of notes linked to the given note."""
        return self.db.get_note_links(note_id)

    def delete_category(self, category_name: str) -> bool:
        """Deletes a category and all notes belonging to it."""
        if not category_name or not category_name.strip():
            return False
        return self.db.delete_category(category_name)

