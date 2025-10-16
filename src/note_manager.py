# note_manager.py
#
# This file contains helper functions related to the management of Zettelkasten notes.
# It handles operations such as cleaning note titles, saving, deleting, renaming notes, and
# loading note metadata.

import re # For regular expressions
import uuid # For generating unique IDs
from logger import log_debug # For debug logging function

# The generate_unique_id function creates a unique ID for new notes.
def generate_unique_id():
    return str(uuid.uuid4()) # Create a UUID (Universally Unique Identifier) and convert it to a string

# The get_sanitized_title function extracts a cleaned title from the first line of the note content.
# It removes Markdown headings, content in parentheses, and other Markdown formatting characters.
def get_sanitized_title(content):
    first_line = content.split('\n')[0].strip() # Get the first line of the content and clean up whitespace
    if not first_line:
        return "Untitled Note" # Return a default title if the first line is empty

    # 1. Remove Markdown heading syntax (e.g., #, ##, etc.) from the beginning of the line
    sanitized_title = re.sub(r'^#+\s*', '', first_line)

    # Remove bold/italic markers
    sanitized_title = re.sub(r'(\*\*|__|\*|_)', '', sanitized_title)
    # Remove inline code markers
    sanitized_title = re.sub(r'`', '', sanitized_title)
    # Remove strikethrough markers
    sanitized_title = re.sub(r'~~', '', sanitized_title)
    # Remove image/link syntax (e.g., ![alt](url) or [text](url))
    sanitized_title = re.sub(r'!\\[.*?]\(.*?\\]\)', '', sanitized_title)
    sanitized_title = re.sub(r'\[.*?]\[.*?]', '', sanitized_title)
    # Remove remaining special characters that might be part of markdown or problematic in titles
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', sanitized_title)
    
    # Remove leading/trailing whitespace that may have occurred after cleaning
    sanitized_title = sanitized_title.strip()

    return sanitized_title

# The save_note function saves or updates a note in the database.
# db_manager: The database manager object.
# note_id: The ID of the note to be updated (if None, a new note is created).
# note_content: The content of the note.
# category_path: The category of the note.
def save_note(db_manager, note_id, note_content, category_path=""):
    sanitized_title = get_sanitized_title(note_content) # Get the cleaned title from the content
    
    if not sanitized_title: # If the title is empty after cleaning
        sanitized_title = "Untitled Note" # Assign a default title

    # Check if there is an existing note with this title
    existing_note_id = db_manager.get_note_id_by_title(sanitized_title) # Existing note ID by title

    if existing_note_id: # If an existing note is found
        log_debug(f"DEBUG: Found existing note with sanitized title '{sanitized_title}' and ID '{existing_note_id}'. Updating instead of creating new.")
        # Update the existing note
        db_manager.update_note(existing_note_id, sanitized_title, note_content, category_path)
        return existing_note_id, sanitized_title # Return the existing note ID and title
    elif note_id is None: # If there is no existing note and no ID is provided, create a new note
        new_note_id = generate_unique_id() # Create a new unique ID
        db_manager.insert_note(new_note_id, sanitized_title, note_content, category_path) # Add the new note
        return new_note_id, sanitized_title # Return the new note ID and title
    else:
        # If note_id is provided, update the specific note (this path is for manual edits of existing notes)
        db_manager.update_note(note_id, sanitized_title, note_content, category_path)
        return note_id, sanitized_title # Return the updated note ID and title

# The delete_note function deletes a specific note from the database.
# db_manager: The database manager object.
# note_id: The ID of the note to be deleted.
def delete_note(db_manager, note_id):
    try:
        db_manager.delete_note(note_id) # Delete the note from the database
        log_debug(f"Note with ID {note_id} deleted from database.")
        return True # Indicate success
    except Exception as e:
        log_debug(f"Error deleting note from database: {e}") # Log the error message
        return False # Indicate failure

# The rename_note function renames a specific note.
# db_manager: The database manager object.
# note_id: The ID of the note to be renamed.
# new_title: The new title of the note.
# category_path: The category of the note (currently not used but kept for compatibility).
def rename_note(db_manager, note_id, new_title, category_path=""):
    sanitized_new_title = get_sanitized_title(new_title) # Clean the new title
    if not sanitized_new_title:
        return False, "New title cannot be empty or result in an empty sanitized title."

    note_data = db_manager.get_note(note_id) # Get the current data of the note
    if note_data:
        current_content = note_data[2] # The content of the note (index 2)
        # Update the note with the new title and existing content
        db_manager.update_note(note_id, sanitized_new_title, current_content, category_path)
        log_debug(f"Note with ID {note_id} renamed to {sanitized_new_title}.")
        return True, sanitized_new_title # Indicate success and return the new title
    else:
        return False, "Note not found."

# The load_all_notes_metadata function loads the metadata of all notes and all categories.
# db_manager: The database manager object.
def load_all_notes_metadata(db_manager):
    notes_metadata_from_db, all_categories_from_db = db_manager.get_all_notes_metadata() # Get metadata and categories from the database
    
    notes_metadata = [] # List of note metadata in the format (display_title, note_id, category_path)
    all_categories = set() # A set to store unique categories

    for note_id, title, category in notes_metadata_from_db:
        notes_metadata.append((note_id, title, category)) # Add metadata to the list
        if category:
            all_categories.add(category) # Add the category to the set if it exists
            
    return notes_metadata, sorted(list(all_categories_from_db)) # Return metadata and sorted categories

# The get_note_content function returns the content of a specific note.
# db_manager: The database manager object.
# note_id: The ID of the note whose content is to be retrieved.
def get_note_content(db_manager, note_id):
    note_data = db_manager.get_note(note_id) # Get the note data
    if note_data:
        return note_data[2] # Return the content (index 2)
    return None # Return None if the note is not found

# The create_category function manages the category creation process.
# In the current database schema, since categories are a field of the note,
# this function only checks if the category name is valid.
# category_name: The name of the category to be created.
def create_category(category_name):
    # In database storage, categories are just a field in the note.
    # There is no need to create a physical directory.
    # This function can return True if the category name is valid.
    if category_name.strip(): # If the category name is not empty after stripping whitespace
        return True
    return False
