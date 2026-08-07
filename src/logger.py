import logging
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
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def log_debug(msg: str) -> None:
    """Logs debug message to console and debug.log file."""
    logger.debug(msg)

def log_error(msg: str) -> None:
    """Logs error message to stderr and debug.log file."""
    logger.error(msg)