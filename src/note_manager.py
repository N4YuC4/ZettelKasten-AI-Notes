import re
import uuid

def generate_unique_id():
    return str(uuid.uuid4())

def get_sanitized_title(content):
    first_line = content.split('\n')[0].strip()
    if not first_line:
        return "Untitled Note"

    # 1. Remove Markdown heading syntax (e.g., #, ##, etc.) from the beginning of the line
    sanitized_title = re.sub(r'^#+\s*', '', first_line)

    # 2. Remove other common Markdown formatting characters
    # This regex will remove:
    # - Bold/Italic (**text**, *text*, __text__, _text_)
    # - Inline code (`code`)
    # - Strikethrough (~~text~~)
    # - Links and images (![alt](url) or [text](url))
    # - Other special characters that might interfere with titles
    sanitized_title = re.sub(r'(\*\*|__|\*|_|`|~|!\[.*?\]\(.*?\)|\s*\[.*?]\(.*?\))', '', sanitized_title)
    
    # Remove invalid characters for filenames (though less critical for DB titles)
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', sanitized_title)
    
    # Remove any leading/trailing whitespace that might result from sanitization
    sanitized_title = sanitized_title.strip()

    return sanitized_title[:50] # Limit title length

def save_note(db_manager, note_id, note_content, category_path=""):
    sanitized_title = get_sanitized_title(note_content)
    
    if note_id is None:
        new_note_id = generate_unique_id()
        db_manager.insert_note(new_note_id, sanitized_title, note_content, category_path)
        return new_note_id, sanitized_title # Return ID and title for main.py to update state
    else:
        db_manager.update_note(note_id, sanitized_title, note_content, category_path)
        return note_id, sanitized_title # Return ID and title for main.py to update state

def delete_note(db_manager, note_id):
    try:
        db_manager.delete_note(note_id)
        print(f"Note with ID {note_id} deleted from database.")
        return True
    except Exception as e:
        print(f"Error deleting note from database: {e}")
        return False

def rename_note(db_manager, note_id, new_title, category_path=""):
    sanitized_new_title = get_sanitized_title(new_title)
    if not sanitized_new_title:
        return False, "New title cannot be empty or result in an empty sanitized title."

    # Retrieve current content to update it with new title
    note_data = db_manager.get_note(note_id)
    if note_data:
        current_content = note_data[2] # content is at index 2
        db_manager.update_note(note_id, sanitized_new_title, current_content, category_path)
        print(f"Note with ID {note_id} renamed to {sanitized_new_title}.")
        return True, sanitized_new_title
    else:
        return False, "Note not found."

def load_all_notes_metadata(db_manager):
    notes_metadata_from_db, all_categories_from_db = db_manager.get_all_notes_metadata()
    
    notes_metadata = [] # List of (display_title, note_id, category_path)
    all_categories = set()

    for note_id, title, category in notes_metadata_from_db:
        notes_metadata.append((note_id, title, category))
        if category:
            all_categories.add(category)
            
    return notes_metadata, sorted(list(all_categories_from_db))

def get_note_content(db_manager, note_id):
    note_data = db_manager.get_note(note_id)
    if note_data:
        return note_data[2] # content is at index 2
    return None

def create_category(category_name):
    # With database storage, categories are just a field in the note. 
    # No need to create a physical directory.
    # This function can simply return True if the category name is valid.
    if category_name.strip():
        return True
    return False
