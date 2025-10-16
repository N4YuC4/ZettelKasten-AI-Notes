# database_manager.py
#
# This file contains the DatabaseManager class, which manages the interaction
# with the application's SQLite database. It provides CRUD (Create, Read, Update, Delete)
# operations for notes, categories, and note links.

import sqlite3 # To work with the SQLite database
import os # For file system operations
from datetime import datetime # For timestamps

# The path to the database file. It is located in the 'db' folder in the application's root directory.
DATABASE_FILE = os.path.join("db", "notes.db")

# The DatabaseManager class manages the SQLite database connection and operations.
class DatabaseManager:
    # The __init__ method establishes the database connection and creates the necessary tables.
    def __init__(self):
        # Ensure the 'db' folder exists, otherwise create it
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        self.conn = sqlite3.connect(DATABASE_FILE) # Connect to the database
        self.conn.execute("PRAGMA foreign_keys = ON") # Enable foreign key constraints
        self.create_notes_table() # Create the notes table
        self.create_note_links_table() # Create the note links table
        self._create_settings_table() # Create the settings table

    # The _create_settings_table method creates a table to store application settings.
    def _create_settings_table(self):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, -- > The setting key (unique)
                value TEXT -- > The setting value
            )
        """)
        self.conn.commit() # Save the changes

    def get_setting(self, key):
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def set_setting(self, key, value):
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    # The create_notes_table method creates a table to store note information.
    def create_notes_table(self):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, -- > The unique ID of the note
                title TEXT NOT NULL, -- > The title of the note (cannot be empty)
                content TEXT, -- > The content of the note
                category TEXT DEFAULT '', -- > The category of the note (default empty)
                created_at TEXT NOT NULL, -- > The creation time
                updated_at TEXT NOT NULL -- > The last update time
            )
        """)
        self.conn.commit() # Save the changes

    # The create_note_links_table method creates a table to store the links between notes.
    def create_note_links_table(self):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_links (
                source_note_id TEXT NOT NULL, -- > The ID of the source note
                target_note_id TEXT NOT NULL, -- > The ID of the target note
                PRIMARY KEY (source_note_id, target_note_id), -- > The combination of the two IDs must be unique
                FOREIGN KEY (source_note_id) REFERENCES notes(id) ON DELETE CASCADE, -- > If the source note is deleted, the link is also deleted
                FOREIGN KEY (target_note_id) REFERENCES notes(id) ON DELETE CASCADE -- > If the target note is deleted, the link is also deleted
            )
        """)
        self.conn.commit() # Save the changes

    # The insert_note_link method adds a link between two notes.
    # source_note_id: The ID of the note where the link starts.
    # target_note_id: The ID of the note where the link ends.
    def insert_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor() # Get the database cursor
        try:
            cursor.execute("""
                INSERT INTO note_links (source_note_id, target_note_id)
                VALUES (?, ?)
            """, (source_note_id, target_note_id)) # Add the link
            self.conn.commit() # Save the changes
            return True # Indicate success
        except sqlite3.IntegrityError:
            # If the link already exists (due to the PRIMARY KEY constraint)
            return False # Indicate failure

    # The note_count method returns the total number of notes in the selected category in the database.
    def note_count(self,category):
        cursor = self.conn.cursor() # Get the database cursor
        if category == "All Notes": # If all notes are selected
            cursor.execute("SELECT COUNT(*) FROM notes") # Query the count of all notes
        else:
            cursor.execute("SELECT COUNT(*) FROM notes WHERE category = ?", (category,)) # Query the number of notes with the specified category
        result = cursor.fetchone() # Get the first result
        return result[0] if result else 0 # Return the count if there is a result, otherwise 0

    # The get_note_links method returns the IDs of all notes linked to a specific note.
    # note_id: The ID of the note whose links are to be queried.
    def get_note_links(self, note_id):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("""
            SELECT target_note_id FROM note_links WHERE source_note_id = ?
            UNION -- > Get notes linked as both source and target
            SELECT source_note_id FROM note_links WHERE target_note_id = ?
        """, (note_id, note_id)) # Execute the query
        return [row[0] for row in cursor.fetchall()] # Return the results as a list

    # The delete_note_link method deletes a specific link between two notes.
    # source_note_id: The ID of the source note.
    # target_note_id: The ID of the target note.
    def delete_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("""
            DELETE FROM note_links WHERE source_note_id = ? AND target_note_id = ?
        """, (source_note_id, target_note_id)) # Delete the link
        self.conn.commit() # Save the changes
        return cursor.rowcount > 0 # Return True if a link was deleted, otherwise False

    # The get_note_id_by_title method returns the ID of a note based on its title.
    # title: The title of the note to search for.
    def get_note_id_by_title(self, title):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("SELECT id FROM notes WHERE title = ?", (title,)) # Query the ID by title
        result = cursor.fetchone() # Get the first result
        return result[0] if result else None # Return the ID if there is a result, otherwise None

    # The insert_note method adds a new note to the database.
    # note_id: The unique ID of the note.
    # title: The title of the note.
    # content: The content of the note.
    # category: The category of the note (optional).
    def insert_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor() # Get the database cursor
        now = datetime.now().isoformat() # Get the current time in ISO format
        cursor.execute("""
            INSERT INTO notes (id, title, content, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (note_id, title, content, category, now, now)) # Add the note
        self.conn.commit() # Save the changes

    # The update_note method updates an existing note.
    # note_id: The ID of the note to be updated.
    # title: The new title.
    # content: The new content.
    # category: The new category (optional).
    def update_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor() # Get the database cursor
        now = datetime.now().isoformat() # Get the current time in ISO format
        cursor.execute("""
            UPDATE notes
            SET title = ?, content = ?, category = ?, updated_at = ?
            WHERE id = ?
        """, (title, content, category, now, note_id)) # Update the note
        self.conn.commit() # Save the changes

    # The delete_note method deletes a specific note from the database.
    # note_id: The ID of the note to be deleted.
    def delete_note(self, note_id):
        try:
            cursor = self.conn.cursor() # Get the database cursor
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,)) # Delete the note
            self.conn.commit() # Save the changes
            return True  # Indicate success
        except sqlite3.Error as e:
            print(f"Database error during note deletion: {e}") # Print the error message
            return False # Indicate failure

    # The delete_category method deletes a specific category and all notes belonging to it.
    # category_name: The name of the category to be deleted.
    def delete_category(self, category_name):
        try:
            cursor = self.conn.cursor() # Get the database cursor
            cursor.execute("DELETE FROM notes WHERE category = ?", (category_name,)) # Delete the notes belonging to the category
            self.conn.commit() # Save the changes
            return True  # Indicate success
        except sqlite3.Error as e:
            print(f"Database error during category deletion: {e}") # Print the error message
            return False  # Indicate failure

    # The get_note method returns all data (ID, title, content, category) of a specific note.
    # note_id: The ID of the note to be retrieved.
    def get_note(self, note_id):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("SELECT id, title, content, category FROM notes WHERE id = ?", (note_id,)) # Query the note
        note = cursor.fetchone() # Get the first result
        return note # Return the note data

    # The get_all_notes_metadata method returns the metadata (ID, title, category) of all notes and
    # all unique category names.
    def get_all_notes_metadata(self):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("SELECT id, title, category FROM notes") # Query the metadata of the notes
        notes_metadata = cursor.fetchall() # Get all results
        all_categories = set() # A set to store unique categories
        for note_id, title, category in notes_metadata:
            if category:
                all_categories.add(category) # Add the category to the set if it exists
        return notes_metadata, all_categories # Return the metadata and categories

    # The create_category method is a placeholder since categories are part of the notes in the current schema.
    # It can be used for separate category management in the future.
    def create_category(self, category_name):
        # For now, it always returns True as it is assumed that the category is implicitly created
        # when a note with this category is saved.
        return True

    # The read_note_content method returns the content of a specific note.
    # note_id: The ID of the note whose content is to be read.
    def read_note_content(self, note_id):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,)) # Query the content
        result = cursor.fetchone() # Get the first result
        return result[0] if result else None # Return the content if there is a result, otherwise None

    # The save_note method saves a note (creates a new one or updates it).
    # note_id: The ID of the note to be updated (if None, a new note is created).
    # note_content: The content of the note.
    # category: The category of the note (optional).
    def save_note(self, note_id, note_content, category=""):
        from uuid import uuid4 # To generate a unique ID
        now = datetime.now().isoformat() # Get the current time in ISO format
        title = note_content.split('\n')[0].strip() # Get the first line as the title

        if not title:
            title = "Untitled Note" # Assign a default title if the title is empty

        if note_id:
            # Update the existing note
            cursor = self.conn.cursor() # Get the database cursor
            cursor.execute("UPDATE notes SET title = ?, content = ?, category = ?, updated_at = ? WHERE id = ?",
                           (title, note_content, category, now, note_id)) # Update the note
            self.conn.commit() # Save the changes
            return note_id, title # Return the note ID and title
        else:
            # Create a new note
            new_note_id = str(uuid4()) # Create a new unique ID
            cursor = self.conn.cursor() # Get the database cursor
            cursor.execute("INSERT INTO notes (id, title, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                           (new_note_id, title, note_content, category, now, now)) # Add the new note
            self.conn.commit() # Save the changes
            return new_note_id, title # Return the new note ID and title

    # The rename_note method updates the title of a note.
    # note_id: The ID of the note to be renamed.
    # new_title: The new title of the note.
    # category: The category of the note (currently not used but kept for compatibility).
    def rename_note(self, note_id, new_title, category=""):
        now = datetime.now().isoformat() # Get the current time in ISO format
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("UPDATE notes SET title = ?, updated_at = ? WHERE id = ?",
                       (new_title, now, note_id)) # Update the title of the note
        self.conn.commit() # Save the changes
        return True, new_title # Indicate success and return the new title

    # The get_all_note_titles_and_ids method returns the titles and IDs of all notes as a dictionary.
    def get_all_note_titles_and_ids(self):
        cursor = self.conn.cursor() # Get the database cursor
        cursor.execute("SELECT title, id FROM notes") # Query the titles and IDs
        return {title: note_id for title, note_id in cursor.fetchall()} # Return as a dictionary

    # The get_all_note_links method returns all note links as pairs of (source_note_id, target_note_id).
    def get_all_note_links(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT source_note_id, target_note_id FROM note_links")
        return cursor.fetchall()

    def bulk_insert_notes(self, notes_data):
        cursor = self.conn.cursor()
        try:
            cursor.executemany("""
                INSERT INTO notes (id, title, content, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, notes_data)
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def bulk_insert_links(self, links_data):
        cursor = self.conn.cursor()
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO note_links (source_note_id, target_note_id)
                VALUES (?, ?)
            """, links_data)
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    # The close_connection method closes the database connection.
    def close_connection(self):
        if self.conn:
            self.conn.close() # Close the connection
            self.conn = None
