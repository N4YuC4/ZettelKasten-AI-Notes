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
            self.ensure_schema()

    def _init_schema(self, connection: sqlite3.Connection) -> None:
        """Ensures all tables and indexes exist using CREATE TABLE IF NOT EXISTS, migrating legacy schema if needed."""
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                collection TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Automated backward-compatible migration: Rename 'category' column to 'collection' if present
        cursor.execute("PRAGMA table_info(notes);")
        columns = [row[1] for row in cursor.fetchall()]
        if "category" in columns and "collection" not in columns:
            cursor.execute("ALTER TABLE notes RENAME COLUMN category TO collection;")
            cursor.execute("DROP INDEX IF EXISTS idx_notes_category;")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_collection ON notes(collection);")
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        connection.commit()

    def ensure_schema(self) -> None:
        """Ensures all tables and indexes exist on the current connection."""
        self._init_schema(self.conn)

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL and busy timeout."""
        conn = getattr(self._local, 'conn', None)
        # Check if the DB file was deleted on disk behind an existing connection
        if conn is not None and not os.path.exists(self.db_path):
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None
            conn = None

        if conn is None:
            dir_name = os.path.dirname(self.db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            connection = sqlite3.connect(self.db_path)
            connection.execute("PRAGMA journal_mode = WAL;")
            connection.execute("PRAGMA foreign_keys = ON;")
            connection.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = connection
            self._init_schema(connection)
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
        """Creates the notes table and collection index."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    collection TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_collection ON notes(collection);")

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

    def note_count(self, collection: Optional[str] = None, category: Optional[str] = None) -> int:
        """Returns total notes count, filtered by collection if provided."""
        col = collection if collection is not None else category
        cursor = self.conn.cursor()
        if not col:
            cursor.execute("SELECT COUNT(*) FROM notes")
        else:
            cursor.execute("SELECT COUNT(*) FROM notes WHERE collection = ?", (col,))
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

    def insert_note(self, note_id: str, title: str, content: str, collection: str = "", category: Optional[str] = None) -> None:
        """Inserts a new note record."""
        col = collection if category is None else category
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (id, title, content, collection, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (note_id, title, content, col, now, now))

    def update_note(self, note_id: str, title: str, content: str, collection: str = "", category: Optional[str] = None) -> None:
        """Updates an existing note record."""
        col = collection if category is None else category
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE notes
                SET title = ?, content = ?, collection = ?, updated_at = ?
                WHERE id = ?
            """, (title, content, col, now, note_id))

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

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes all notes in a collection and their links using subqueries."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM note_links
                    WHERE source_note_id IN (SELECT id FROM notes WHERE collection = ?)
                       OR target_note_id IN (SELECT id FROM notes WHERE collection = ?)
                """, (collection_name, collection_name))
                cursor.execute("DELETE FROM notes WHERE collection = ?", (collection_name,))
                return True
        except sqlite3.Error as e:
            log_error(f"Database error during collection deletion: {e}")
            return False

    # Backward compatibility alias
    delete_category = delete_collection

    def get_note(self, note_id: str) -> Optional[Tuple[str, str, str, str]]:
        """Returns (id, title, content, collection) tuple for a note ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content, collection FROM notes WHERE id = ?", (note_id,))
        return cursor.fetchone()

    def get_note_model(self, note_id: str) -> Optional[Note]:
        """Returns typed Note model for a note ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content, collection, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if row:
            return Note(id=row[0], title=row[1], content=row[2], collection=row[3], created_at=row[4], updated_at=row[5])
        return None

    def get_all_notes_metadata(self) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """Returns metadata list [(id, title, collection)] and unique collection set."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, collection FROM notes ORDER BY updated_at DESC")
        notes_metadata = cursor.fetchall()
        all_collections = {collection for _, _, collection in notes_metadata if collection}
        return notes_metadata, all_collections

    def get_all_notes_metadata_models(self) -> List[NoteMetadata]:
        """Returns a list of typed NoteMetadata models."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, collection FROM notes ORDER BY updated_at DESC")
        return [NoteMetadata(id=row[0], title=row[1], collection=row[2] or "") for row in cursor.fetchall()]

    def create_collection(self, collection_name: str) -> bool:
        """Collection placeholder for backward compatibility."""
        return True

    # Backward compatibility alias
    create_category = create_collection

    def read_note_content(self, note_id: str) -> Optional[str]:
        """Returns raw content string of a note."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_note(self, note_id: Optional[str], note_content: str, collection: str = "", title: Optional[str] = None, category: Optional[str] = None) -> Tuple[str, str]:
        """Direct DB helper to save or update note without business layer coupling."""
        from uuid import uuid4
        col = collection if category is None else category
        now = datetime.now().isoformat()
        if not title:
            for line in (note_content or "").splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    title = stripped
                    break
            title = title or "Untitled Note"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if note_id:
                cursor.execute("UPDATE notes SET title = ?, content = ?, collection = ?, updated_at = ? WHERE id = ?",
                               (title, note_content, col, now, note_id))
                return note_id, title
            else:
                new_note_id = str(uuid4())
                cursor.execute("INSERT INTO notes (id, title, content, collection, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                               (new_note_id, title, note_content, col, now, now))
                return new_note_id, title

    def rename_note(self, note_id: str, new_title: str, collection: str = "", category: Optional[str] = None) -> Tuple[bool, str]:
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
                    INSERT INTO notes (id, title, content, collection, created_at, updated_at)
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
                        INSERT INTO notes (id, title, content, collection, created_at, updated_at)
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
