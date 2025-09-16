import sqlite3
import os
from datetime import datetime

DATABASE_FILE = os.path.join("db", "notes.db")

class DatabaseManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_notes_table()
        self.create_note_links_table()
        self._create_settings_table()

    def _create_settings_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()


    def create_notes_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                category TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def create_note_links_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_links (
                source_note_id TEXT NOT NULL,
                target_note_id TEXT NOT NULL,
                PRIMARY KEY (source_note_id, target_note_id),
                FOREIGN KEY (source_note_id) REFERENCES notes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def insert_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO note_links (source_note_id, target_note_id)
                VALUES (?, ?)
            """, (source_note_id, target_note_id))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Link already exists
            return False

    def get_note_links(self, note_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT target_note_id FROM note_links WHERE source_note_id = ?
            UNION
            SELECT source_note_id FROM note_links WHERE target_note_id = ?
        """, (note_id, note_id))
        return [row[0] for row in cursor.fetchall()]

    def delete_note_link(self, source_note_id, target_note_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM note_links WHERE source_note_id = ? AND target_note_id = ?
        """, (source_note_id, target_note_id))
        self.conn.commit()
        return cursor.rowcount > 0 # Returns True if a link was deleted, False otherwise

    def get_note_id_by_title(self, title):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM notes WHERE title = ?", (title,))
        result = cursor.fetchone()
        return result[0] if result else None

    def insert_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO notes (id, title, content, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (note_id, title, content, category, now, now))
        self.conn.commit()

    def update_note(self, note_id, title, content, category=""):
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE notes
            SET title = ?, content = ?, category = ?, updated_at = ?
            WHERE id = ?
        """, (title, content, category, now, note_id))
        self.conn.commit()

    def delete_note(self, note_id):
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self.conn.commit()
            return True  # Indicate success
        except sqlite3.Error as e:
            print(f"Database error during note deletion: {e}")
            return False # Indicate failure
    
    def delete_category(self, category_name):
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM notes WHERE category = ?", (category_name,))
            self.conn.commit()
            return True  # Indicate success
        except sqlite3.Error as e:
            print(f"Database error during category deletion: {e}")
            return False  # Indicate failure

    def get_note(self, note_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content, category FROM notes WHERE id = ?", (note_id,))
        note = cursor.fetchone()
        return note

    def get_all_notes_metadata(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, category FROM notes")
        notes_metadata = cursor.fetchall()
        all_categories = set()
        for note_id, title, category in notes_metadata:
            if category:
                all_categories.add(category)
        return notes_metadata, all_categories

    def create_category(self, category_name):
        # In the current schema, categories are part of notes.
        # This method is a placeholder for future dedicated category management.
        # For now, we just return True, assuming the category will be
        # implicitly created when a note with that category is saved.
        return True

    def read_note_content(self, note_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_note(self, note_id, note_content, category=""):
        from uuid import uuid4
        now = datetime.now().isoformat()
        title = note_content.split('\n')[0].strip() # Take the first line as title

        if not title:
            title = "Untitled Note"

        if note_id:
            # Update existing note
            cursor = self.conn.cursor()
            cursor.execute("UPDATE notes SET title = ?, content = ?, category = ?, updated_at = ? WHERE id = ?",
                           (title, note_content, category, now, note_id))
            self.conn.commit()
            return note_id, title
        else:
            # Create new note
            new_note_id = str(uuid4())
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO notes (id, title, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                           (new_note_id, title, note_content, category, now, now))
            self.conn.commit()
            return new_note_id, title

    def rename_note(self, note_id, new_title, category=""):
        now = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("UPDATE notes SET title = ?, updated_at = ? WHERE id = ?",
                       (new_title, now, note_id))
        self.conn.commit()
        return True, new_title # Return success and the new title

    def get_all_note_titles_and_ids(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT title, id FROM notes")
        return {title: note_id for title, note_id in cursor.fetchall()}

    def close_connection(self):
        if self.conn:
            self.conn.close()
