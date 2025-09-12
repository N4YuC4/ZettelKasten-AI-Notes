import re
import uuid
import database_manager

def generate_unique_id():
    return str(uuid.uuid4())

def get_sanitized_title(content):
    first_line = content.split('\n')[0].strip()
    if not first_line:
        return "Untitled Note"

    # Remove Markdown heading syntax (e.g., #, ##, etc.)
    title_without_markdown = re.sub(r'^#+\s*', '', first_line).strip()

    # Remove invalid characters for filenames (though less critical for DB titles)
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', title_without_markdown)
    # Replace spaces with underscores or hyphens for consistency, though not strictly needed for DB
    sanitized_title = sanitized_title.replace(' ', '_')
    return sanitized_title[:50] # Limit title length

def save_note(note_id, note_content, category_path=""):
    sanitized_title = get_sanitized_title(note_content)
    
    if note_id is None:
        new_note_id = generate_unique_id()
        database_manager.insert_note(new_note_id, sanitized_title, note_content, category_path)
        return new_note_id, sanitized_title # Return ID and title for main.py to update state
    else:
        database_manager.update_note(note_id, sanitized_title, note_content, category_path)
        return note_id, sanitized_title # Return ID and title for main.py to update state

def delete_note(note_id):
    try:
        database_manager.delete_note(note_id)
        print(f"Note with ID {note_id} deleted from database.")
        return True
    except Exception as e:
        print(f"Error deleting note from database: {e}")
        return False

def rename_note(note_id, new_title, category_path=""):
    sanitized_new_title = get_sanitized_title(new_title)
    if not sanitized_new_title:
        return False, "New title cannot be empty or result in an empty sanitized title."

    # Retrieve current content to update it with new title
    note_data = database_manager.get_note(note_id)
    if note_data:
        current_content = note_data[2] # content is at index 2
        database_manager.update_note(note_id, sanitized_new_title, current_content, category_path)
        print(f"Note with ID {note_id} renamed to {sanitized_new_title}.")
        return True, sanitized_new_title
    else:
        return False, "Note not found."

def load_all_notes_metadata():
    notes_metadata = [] # List of (display_title, note_id, category_path)
    all_categories = set()

    db_notes = database_manager.get_all_notes_metadata()
    for note_id, title, category in db_notes:
        notes_metadata.append((title.replace("_", " "), note_id, category))
        if category:
            all_categories.add(category)
            
    return notes_metadata, sorted(list(all_categories))

def get_note_content(note_id):
    note_data = database_manager.get_note(note_id)
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