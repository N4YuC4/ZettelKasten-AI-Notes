# settings_manager.py
#
# Dedicated SQLite repository for all application configurations and preferences.
# Stored independently in db/settings.db with thread-local connections and WAL mode.

import sqlite3
import os
import threading
from typing import Optional, Dict, Any
from logger import log_error, log_debug
import local_models_catalog

DEFAULT_SETTINGS_DB_FILE = os.path.join("db", "settings.db")


class SettingsManager:
    """
    Thread-safe configuration repository backed by a dedicated SQLite database (db/settings.db).
    Decouples application configuration completely from note vault databases.
    """
    _instance: Optional['SettingsManager'] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else DEFAULT_SETTINGS_DB_FILE
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns thread-local SQLite connection configured with WAL mode."""
        conn = getattr(self._local, 'conn', None)
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
            connection.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = connection
            self._init_schema_on_conn(connection)
        return self._local.conn

    def _init_schema_on_conn(self, connection: sqlite3.Connection) -> None:
        """Creates settings key-value table on the given connection."""
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        connection.commit()

    def _init_schema(self) -> None:
        """Ensures schema exists."""
        self._init_schema_on_conn(self.conn)

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves a configuration value by key, returning default if not found."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                return result[0]
            return default
        except sqlite3.Error as e:
            log_error(f"Error reading setting '{key}': {e}")
            return default

    def set_setting(self, key: str, value: str) -> None:
        """Sets or replaces a configuration key-value pair."""
        try:
            with self.conn as connection:
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else "")
                )
        except sqlite3.Error as e:
            log_error(f"Error writing setting '{key}': {e}")

    def get_all_settings(self) -> Dict[str, str]:
        """Returns all configured settings as a dictionary."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            return dict(cursor.fetchall())
        except sqlite3.Error as e:
            log_error(f"Error loading all settings: {e}")
            return {}

    def remove_setting(self, key: str) -> None:
        """Removes a configuration key from the settings database."""
        try:
            with self.conn as connection:
                cursor = connection.cursor()
                cursor.execute("DELETE FROM settings WHERE key = ?", (key,))
        except sqlite3.Error as e:
            log_error(f"Error removing setting '{key}': {e}")

    def get_defaults(self) -> Dict[str, str]:
        """Returns application factory default values."""
        return {
            "UI_THEME": "Dark",
            "AUTO_SAVE": "True",
            "AI_PROVIDER": "gemini",
            "ACTIVE_LOCAL_MODEL": local_models_catalog.DEFAULT_MODEL_ID,
            "GPU_ACCELERATION": "True",
            "CONFIRM_DELETE_NOTE": "True",
            "CONFIRM_DELETE_COLLECTION": "True",
            "MODELS_DIR": local_models_catalog.get_default_models_dir(),
            "AI_CUSTOM_SYSTEM_PROMPT": "",
            "CUSTOM_DB_PATH": "",
            "GEMINI_API_KEY": "",
            "EMBEDDING_MODEL_ID": local_models_catalog.DEFAULT_EMBEDDING_MODEL_ID,
            "EMBEDDING_SIMILARITY_THRESHOLD": "0.65",
        }

    def reset_to_defaults(self) -> None:
        """Resets all settings in the database to factory defaults."""
        defaults = self.get_defaults()
        try:
            with self.conn as connection:
                cursor = connection.cursor()
                cursor.execute("DELETE FROM settings")
                for key, val in defaults.items():
                    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (key, val))
            log_debug("Settings successfully reset to factory defaults.")
        except sqlite3.Error as e:
            log_error(f"Error resetting settings: {e}")

    def close_connection(self) -> None:
        """Closes thread-local connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> 'SettingsManager':
        """Singleton accessor for convenience."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path=db_path)
            return cls._instance

