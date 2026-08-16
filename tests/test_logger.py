# tests/test_logger.py
import pytest
import logging
import threading
from logging.handlers import RotatingFileHandler
import logger

def test_logger_configured_with_rotating_handler():
    """
    Verifies FLAW-LOG-01: Logger uses RotatingFileHandler to prevent unbounded log growth.
    """
    file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) >= 1
    assert isinstance(file_handlers[0], RotatingFileHandler)
    assert file_handlers[0].maxBytes == 5 * 1024 * 1024
    assert file_handlers[0].backupCount == 3
    assert file_handlers[0].encoding == "utf-8"

def test_log_error_captures_exc_info(caplog):
    """
    Verifies FLAW-LOG-02: log_error logs full tracebacks when exc_info is passed.
    """
    try:
        raise ValueError("Diagnostic Test Error")
    except ValueError:
        logger.log_error("Operation failed", exc_info=True)
    
    # Verify exception message appears in logs
    assert any("Operation failed" in record.message for record in caplog.records)

def test_logger_multithreaded_safe():
    """
    Verifies logger works concurrently across multiple threads without errors.
    """
    errors = []
    def worker(worker_id):
        try:
            for i in range(50):
                logger.log_debug(f"Thread-{worker_id} debug message {i}")
                logger.log_error(f"Thread-{worker_id} error message {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
