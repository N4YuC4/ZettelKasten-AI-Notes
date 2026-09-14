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
    for line in content.splitlines():
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
    Adds two spaces (Markdown hard break) at line ends outside code blocks and tables
    so that single newlines render properly in Markdown views without breaking table rows.
    """
    if not text:
        return ""
    lines = text.split("\n")
    processed = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            processed.append(line)
        elif stripped in ("$$", r"\[", r"\]"):
            in_code = not in_code
            processed.append(line)
        elif in_code:
            processed.append(line)
        else:
            # Do not append trailing spaces to table rows (|...|), headings (#), or blank lines
            if stripped and not stripped.startswith("|") and not stripped.startswith("#") and not line.endswith("  "):
                processed.append(line + "  ")
            else:
                processed.append(line)
    return "\n".join(processed)


MATH_IN_TEXT_REPLACEMENTS = {
    '∑': r'\sum',
    r'\sum': r'\sum',
    '∏': r'\prod',
    r'\prod': r'\prod',
    '∫': r'\int',
    r'\int': r'\int',
    '√': r'\sqrt',
    r'\sqrt': r'\sqrt',
    '±': r'\pm',
    r'\pm': r'\pm',
    '≤': r'\le',
    r'\le': r'\le',
    '≥': r'\ge',
    r'\ge': r'\ge',
    '≠': r'\ne',
    r'\ne': r'\ne',
    '≈': r'\approx',
    r'\approx': r'\approx',
    '×': r'\times',
    r'\times': r'\times',
    '÷': r'\div',
    r'\div': r'\div',
    '∂': r'\partial',
    r'\partial': r'\partial',
    '∇': r'\nabla',
    r'\nabla': r'\nabla',
    '∞': r'\infty',
    r'\infty': r'\infty',
}


def sanitize_math_mode_syntax(text: str) -> str:
    """
    Repairs LaTeX math expressions by converting math symbols mistakenly placed inside \\text{...}
    back into valid math mode (e.g. \\text{∑} -> \\sum). KaTeX fails with "Can't use function X in text mode"
    if math operators are placed inside text mode.
    """
    def replace_text_content(match: re.Match) -> str:
        inner = match.group(1).strip()
        if inner in MATH_IN_TEXT_REPLACEMENTS:
            return f" {MATH_IN_TEXT_REPLACEMENTS[inner]} "
        res = inner
        for sym, repl in MATH_IN_TEXT_REPLACEMENTS.items():
            if sym in res:
                res = res.replace(sym, f" {repl} ")
        if res != inner:
            return res
        return match.group(0)

    return re.sub(r'\\text\{([^}]*)\}', replace_text_content, text)


def normalize_markdown_latex(text: str) -> str:
    """
    Normalizes LaTeX formulas so they render correctly in Flutter's Markdown renderer.
    flutter_markdown_plus_latex requires the closing delimiter ($ or $$) to be followed
    by whitespace, punctuation in [?!.,:？！。，：], or end of line.
    When formulas are followed by closing brackets, parentheses, semicolons, quotes, etc.,
    it fails to close math mode and swallows subsequent text/formulas into KaTeX, causing
    "Parser Error: Can't use function '$' in math mode".
    Guarantees code blocks (```...``` and `...`) are never mutated.
    """
    if not text:
        return ""

    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', text)
    allowed_following = set(" \t\r\n?!.,:？！。，：")
    math_pattern = re.compile(
        r'(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$'
        r'|(?<!\\)\$([^\$\n]+?)(?<!\\)\$'
        r'|(?<!\\)\\\(([\s\S]+?)(?<!\\)\\\)'
        r'|(?<!\\)\\\[([\s\S]+?)(?<!\\)\\\]'
    )

    for i in range(0, len(parts), 2):
        part = parts[i]
        if '$' not in part and '\\(' not in part and '\\[' not in part:
            continue

        # 1. Sanitize math operators accidentally wrapped in \\text{...} (e.g. \\text{∑} -> \\sum)
        part = sanitize_math_mode_syntax(part)

        # 2. Absorb enclosing parentheses/brackets into math delimiters: ($formula$) -> $(formula)$
        # In Flutter's Markdown renderer, inline math is wrapped in a WidgetSpan.
        # If parentheses are outside [($formula$)], Flutter treats the transition to/from WidgetSpan
        # as a line-break opportunity. If the formula is near the margin, Flutter can strand '(' at the end
        # of the preceding line and ')' at the start of the next line, splitting a formula across 3 lines.
        # Moving the parentheses inside the formula prevents line splitting and renders them atomically.
        part = re.sub(r'\(\s*(?<!\\)\$([^\$\n]+?)(?<!\\)\$\s*\)', r'$(\1)$', part)
        part = re.sub(r'\[\s*(?<!\\)\$([^\$\n]+?)(?<!\\)\$\s*\]', r'$[\1]$', part)

        # 3. Absorb trailing sentence punctuation (. , ; :) into inline math.
        # In Flutter's Markdown renderer, inline math is rendered inside a WidgetSpan.
        # When punctuation immediately follows outside ($formula$.), Flutter's line-breaking engine
        # treats the boundary between WidgetSpan and TextSpan as a soft-wrap opportunity.
        # If the formula fills the line width, the punctuation and the next word wrap together,
        # resulting in an orphaned period or comma appearing at the very beginning of the new line.
        # Moving the punctuation inside ($formula.$) ensures it remains visually attached to the formula.
        part = re.sub(r'(?<!\$)(?<!\\)\$([^\$\n]+?)(?<!\\)\$(?!\$)([.,;:])(?=\s|$)', r'$\1\2$', part)
        part = re.sub(r'(?<!\\)\\\(([\s\S]+?)(?<!\\)\\\)([.,;:])(?=\s|$)', r'\(\1\2\)', part)

        pos = 0
        buf = []
        for m in math_pattern.finditer(part):
            start, end = m.span()
            buf.append(part[pos:start])
            buf.append(m.group(0))
            pos = end
            if pos < len(part):
                next_char = part[pos]
                if next_char not in allowed_following:
                    buf.append(' ')
        buf.append(part[pos:])
        parts[i] = ''.join(buf)

    return ''.join(parts)


def process_markdown_wikilinks(text: str) -> str:
    """
    Converts [[Note Title]] or [[Target Note|Custom Alias]] WikiLinks
    into standard Markdown links using the zettel://note/ protocol.
    Also normalizes LaTeX formulas and preserves single linebreaks.
    Guarantees code blocks (```...``` and `...`) are never mutated.
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

    # Split text by code blocks (```...```) and inline code (`...`)
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\[\[(.*?)\]\]', replace_wikilink, parts[i])
        parts[i] = normalize_markdown_latex(parts[i])
        parts[i] = preserve_single_linebreaks(parts[i])

    return "".join(parts)


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

    def save_note(self, note_id: Optional[str], content: str, collection: str = "", category: Optional[str] = None) -> Tuple[str, str]:
        """
        Saves or updates a note with sanitized title extraction.
        Returns (note_id, sanitized_title).
        """
        col = collection if category is None else category
        title = sanitize_title(content)
        collection_clean = col.strip() if col else ""

        if note_id:
            self.db.update_note(note_id, title, content, collection_clean)
            return note_id, title
        else:
            new_id = generate_unique_id()
            self.db.insert_note(new_id, title, content, collection_clean)
            return new_id, title

    def rename_note(self, note_id: str, new_title: str, collection: Optional[str] = None, category: Optional[str] = None) -> Tuple[bool, str]:
        """
        Renames a note and updates its first line heading in content.
        Preserves existing collection if collection is None or empty string.
        Returns (success, new_title_or_error_message).
        """
        col = collection if category is None else category
        if not new_title or not new_title.strip():
            return False, "New title cannot be empty or result in an empty sanitized title."

        sanitized = sanitize_title(new_title)
        if not sanitized or (sanitized == "Untitled Note" and not new_title.strip()):
            return False, "New title cannot be empty or result in an empty sanitized title."

        note_data = self.db.get_note(note_id)
        if not note_data:
            return False, "Note not found."

        existing_collection = note_data[3] if isinstance(note_data, (tuple, list)) else (getattr(note_data, "collection", "") or getattr(note_data, "category", ""))
        col_to_save = col if (col is not None and col != "") else (existing_collection or "")

        # Safely extract and update content
        current_content = ""
        if isinstance(note_data, (tuple, list)):
            current_content = note_data[2] if note_data[2] is not None else ""
        elif hasattr(note_data, "content") and note_data.content is not None:
            current_content = note_data.content

        lines = current_content.split('\n')
        heading_updated = False
        for idx, line in enumerate(lines):
            if line.strip():
                lines[idx] = f"# {sanitized}"
                heading_updated = True
                break

        if not heading_updated:
            lines = [f"# {sanitized}"]

        updated_content = '\n'.join(lines)

        self.db.update_note(note_id, sanitized, updated_content, col_to_save)
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
        """Returns all notes metadata tuples (id, title, collection) and sorted unique collections."""
        notes_metadata_from_db, all_collections_from_db = self.db.get_all_notes_metadata()
        metadata_list = [(nid, title, col) for nid, title, col in notes_metadata_from_db]
        collections_list = sorted([col for col in all_collections_from_db if col])
        return metadata_list, collections_list

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

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection and all notes belonging to it."""
        if not collection_name or not collection_name.strip():
            return False
        return self.db.delete_collection(collection_name)

    # Backward compatibility alias
    delete_category = delete_collection

