# database_manager.py
#
# Manages SQLite database operations for Zettelkasten notes, categories, settings, and links.
# Thread-safe with thread-local connections and WAL mode.

import sqlite3
import os
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Set, Dict, Any
from logger import log_error, log_debug
from models import Note, NoteMetadata, NoteLink

# Default path to the database file in the 'db' directory
DATABASE_FILE = os.path.join("db", "notes.db")


class DatabaseManager:
    """
    Manages SQLite database operations with thread-local connections.
    """
    def __init__(self, db_path: Optional[str] = None, init_tables: bool = True):
        self.db_path = db_path if db_path is not None else DATABASE_FILE
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._local = threading.local()
        if init_tables:
            self.create_notes_table()
            self.create_note_links_table()
            self._create_settings_table()

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL and busy timeout."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            connection = sqlite3.connect(self.db_path)
            connection.execute("PRAGMA journal_mode = WAL;")
            connection.execute("PRAGMA foreign_keys = ON;")
            connection.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = connection
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        """Returns the thread-local database connection."""
        return self.conn

    def _create_settings_table(self) -> None:
        """Creates settings key-value table if not exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def get_setting(self, key: str) -> Optional[str]:
        """Retrieves a configuration value by key."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def set_setting(self, key: str, value: str) -> None:
        """Sets or replaces a configuration value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

    def create_notes_table(self) -> None:
        """Creates the notes table and category index."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category);")

    def create_note_links_table(self) -> None:
        """Creates the note links junction table and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS note_links (
                    source_note_id TEXT NOT NULL,
                    target_note_id TEXT NOT NULL,
                    PRIMARY KEY (source_note_id, target_note_id),
                    FOREIGN KEY (source_note_id) REFERENCES notes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_note_id) REFERENCES notes(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON note_links(source_note_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON note_links(target_note_id);")

    def insert_note_link(self, source_note_id: str, target_note_id: str) -> bool:
        """Inserts a link between source and target notes. Returns True on success, False if already exists."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO note_links (source_note_id, target_note_id)
                    VALUES (?, ?)
                """, (source_note_id, target_note_id))
                return True
        except sqlite3.IntegrityError:
            return False
        except sqlite3.Error as e:
            log_error(f"Database error during insert note link: {e}")
            return False

    def note_count(self, category: Optional[str] = None) -> int:
        """Returns total notes count, filtered by category if provided."""
        cursor = self.conn.cursor()
        if not category:
            cursor.execute("SELECT COUNT(*) FROM notes")
        else:
            cursor.execute("SELECT COUNT(*) FROM notes WHERE category = ?", (category,))
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_note_links(self, note_id: str) -> List[str]:
        """Returns all note IDs connected to the given note ID (bidirectional)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT target_note_id FROM note_links WHERE source_note_id = ?
            UNION
            SELECT source_note_id FROM note_links WHERE target_note_id = ?
        """, (note_id, note_id))
        return [row[0] for row in cursor.fetchall()]

    def delete_note_link(self, source_note_id: str, target_note_id: str) -> bool:
        """Deletes a link between two notes bidirectionally."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM note_links
                    WHERE (source_note_id = ? AND target_note_id = ?)
                       OR (source_note_id = ? AND target_note_id = ?)
                """, (source_note_id, target_note_id, target_note_id, source_note_id))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            log_error(f"Database error during note link deletion: {e}")
            return False

    def get_note_id_by_title(self, title: str) -> Optional[str]:
        """Looks up a note ID by its title."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM notes WHERE title = ?", (title,))
        result = cursor.fetchone()
        return result[0] if result else None

    def insert_note(self, note_id: str, title: str, content: str, category: str = "") -> None:
        """Inserts a new note record."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (id, title, content, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (note_id, title, content, category, now, now))

    def update_note(self, note_id: str, title: str, content: str, category: str = "") -> None:
        """Updates an existing note record."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE notes
                SET title = ?, content = ?, category = ?, updated_at = ?
                WHERE id = ?
            """, (title, content, category, now, note_id))

    def delete_note(self, note_id: str) -> bool:
        """Deletes a note and cascaded links."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM note_links WHERE source_note_id = ? OR target_note_id = ?", (note_id, note_id))
                cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                return True
        except sqlite3.Error as e:
            log_error(f"Database error during note deletion: {e}")
            return False

    def delete_category(self, category_name: str) -> bool:
        """Deletes all notes in a category and their links using subqueries."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM note_links
                    WHERE source_note_id IN (SELECT id FROM notes WHERE category = ?)
                       OR target_note_id IN (SELECT id FROM notes WHERE category = ?)
                """, (category_name, category_name))
                cursor.execute("DELETE FROM notes WHERE category = ?", (category_name,))
                return True
        except sqlite3.Error as e:
            log_error(f"Database error during category deletion: {e}")
            return False

    def get_note(self, note_id: str) -> Optional[Tuple[str, str, str, str]]:
        """Returns (id, title, content, category) tuple for a note ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content, category FROM notes WHERE id = ?", (note_id,))
        return cursor.fetchone()

    def get_note_model(self, note_id: str) -> Optional[Note]:
        """Returns typed Note model for a note ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content, category, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if row:
            return Note(id=row[0], title=row[1], content=row[2], category=row[3], created_at=row[4], updated_at=row[5])
        return None

    def get_all_notes_metadata(self) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """Returns metadata list [(id, title, category)] and unique category set."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, category FROM notes ORDER BY updated_at DESC")
        notes_metadata = cursor.fetchall()
        all_categories = {category for _, _, category in notes_metadata if category}
        return notes_metadata, all_categories

    def get_all_notes_metadata_models(self) -> List[NoteMetadata]:
        """Returns a list of typed NoteMetadata models."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, category FROM notes ORDER BY updated_at DESC")
        return [NoteMetadata(id=row[0], title=row[1], category=row[2] or "") for row in cursor.fetchall()]

    def create_category(self, category_name: str) -> bool:
        """Category placeholder for backward compatibility."""
        return True

    def read_note_content(self, note_id: str) -> Optional[str]:
        """Returns raw content string of a note."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_note(self, note_id: Optional[str], note_content: str, category: str = "") -> Tuple[str, str]:
        """Direct DB helper to save or update note."""
        from uuid import uuid4
        import note_service
        now = datetime.now().isoformat()
        title = note_service.sanitize_title(note_content)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if note_id:
                cursor.execute("UPDATE notes SET title = ?, content = ?, category = ?, updated_at = ? WHERE id = ?",
                               (title, note_content, category, now, note_id))
                return note_id, title
            else:
                new_note_id = str(uuid4())
                cursor.execute("INSERT INTO notes (id, title, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                               (new_note_id, title, note_content, category, now, now))
                return new_note_id, title

    def rename_note(self, note_id: str, new_title: str, category: str = "") -> Tuple[bool, str]:
        """Direct DB helper to rename note."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE notes SET title = ?, updated_at = ? WHERE id = ?",
                           (new_title, now, note_id))
            return True, new_title

    def get_all_note_titles_and_ids(self) -> Dict[str, str]:
        """Returns {title: note_id} mapping."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT title, id FROM notes")
        return {title: note_id for title, note_id in cursor.fetchall()}

    def get_all_note_links(self) -> List[Tuple[str, str]]:
        """Returns list of all (source_note_id, target_note_id) pairs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT source_note_id, target_note_id FROM note_links")
        return cursor.fetchall()

    def bulk_insert_notes(self, notes_data: List[Tuple]) -> None:
        """Batch inserts note tuples into the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO notes (id, title, content, category, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, notes_data)
        except sqlite3.Error as e:
            log_error(f"Database error during bulk insert notes: {e}")
            raise e

    def bulk_insert_links(self, links_data: List[Tuple[str, str]]) -> None:
        """Batch inserts note link tuples into the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR IGNORE INTO note_links (source_note_id, target_note_id)
                    VALUES (?, ?)
                """, links_data)
        except sqlite3.Error as e:
            log_error(f"Database error during bulk insert links: {e}")
            raise e

    def bulk_insert_notes_and_links(self, notes_data: List[Tuple], links_data: List[Tuple[str, str]]) -> None:
        """Batch inserts note and link tuples atomically within a single transaction."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if notes_data:
                    cursor.executemany("""
                        INSERT INTO notes (id, title, content, category, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, notes_data)
                if links_data:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO note_links (source_note_id, target_note_id)
                        VALUES (?, ?)
                    """, links_data)
        except sqlite3.Error as e:
            log_error(f"Database error during atomic bulk insert notes and links: {e}")
            raise e

    def close_connection(self) -> None:
        """Closes thread-local database connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
