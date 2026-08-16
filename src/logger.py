import logging
from logging.handlers import RotatingFileHandler
import os
import sys

# Ensure log directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "debug.log")

# Configure logger
logger = logging.getLogger("ZettelkastenAINotes")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Rotating file handler (5MB max size, 3 backup archives)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def log_debug(msg: str, *args, **kwargs) -> None:
    """Logs debug message to console and debug.log file."""
    logger.debug(msg, *args, **kwargs)

def log_error(msg: str, *args, exc_info=None, **kwargs) -> None:
    """Logs error message to stderr and debug.log file."""
    logger.error(msg, *args, exc_info=exc_info, **kwargs)