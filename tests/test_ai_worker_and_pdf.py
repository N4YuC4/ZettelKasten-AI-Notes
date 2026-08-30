import pytest
import os
import sys
import sqlite3
import threading
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_note_generator_worker import AiNoteGeneratorWorker
import pdf_processor
import database_manager
import note_manager
from gemini_api_client import GeminiAuthError, GeminiRateLimitError, GeminiApiError

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_worker_db.db"
    database_manager.DATABASE_FILE = str(db_path)
    db = database_manager.DatabaseManager()
    yield db
    db.close_connection()

def test_worker_db_init_failure_triggers_on_error(monkeypatch):
    def mock_init(self, init_tables=False):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(database_manager.DatabaseManager, "__init__", mock_init)
    
    errors = []
    worker = AiNoteGeneratorWorker("dummy_text", on_finished=None, on_error=lambda e: errors.append(e))
    worker.run()
    assert len(errors) == 1
    assert "disk I/O error" in errors[0]

def test_worker_atomic_batch_rollback_on_failure(temp_db, monkeypatch):
    notes_payload = [
        {"title": "Batch Note 1", "content": "Content 1", "connections": ["Batch Note 2"], "general_title": "AI Batch"},
        {"title": "Batch Note 2", "content": "Content 2", "connections": [], "general_title": "AI Batch"}
    ]

    
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = notes_payload
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)
    
    def failing_bulk_links(self, links):
        raise sqlite3.OperationalError("Simulated DB Disk Failure during Link Insertion")
    monkeypatch.setattr(database_manager.DatabaseManager, "bulk_insert_links", failing_bulk_links)
    
    errors = []
    finished_results = []
    worker = AiNoteGeneratorWorker(
        "sample extracted text",
        on_finished=lambda notes: finished_results.append(notes),
        on_error=lambda err: errors.append(err)
    )
    worker.run()
    
    assert len(errors) == 1
    assert "Simulated DB Disk Failure" in errors[0]
    assert len(finished_results) == 0
    
    all_notes, _ = note_manager.load_all_notes_metadata(temp_db)
    persisted_titles = [title for _, title, _ in all_notes]
    assert "Batch Note 1" not in persisted_titles
    assert "Batch Note 2" not in persisted_titles

def test_ai_worker_batch_duplicate_title_deduplication(temp_db, monkeypatch):
    notes_payload = [
        {"title": "Summary", "content": "Part 1: Key Findings", "connections": [], "general_title": "AI Batch"},
        {"title": "Summary", "content": "Part 2: Methodology", "connections": [], "general_title": "AI Batch"},
        {"title": "Summary", "content": "Part 3: Conclusions", "connections": [], "general_title": "AI Batch"}
    ]
    
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = notes_payload
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)
    
    finished_notes = []
    worker = AiNoteGeneratorWorker(
        "sample document content",
        on_finished=lambda res: finished_notes.append(res),
        on_error=None
    )
    worker.run()
    
    all_notes, _ = note_manager.load_all_notes_metadata(temp_db)
    ai_notes = [(nid, title) for nid, title, cat in all_notes if cat == "AI Batch"]
    
    assert len(ai_notes) == 3
    distinct_ids = {nid for nid, _ in ai_notes}
    assert len(distinct_ids) == 3
    
    titles = [title for _, title in ai_notes]
    assert len(set(titles)) == 3
    assert "Summary" in titles

def test_ai_worker_rejects_self_referential_links(temp_db, monkeypatch):
    notes_payload = [
        {
            "title": "Machine Learning",
            "content": "ML Overview Content",
            "connections": ["Machine Learning", "Deep Learning"],
            "general_title": "AI Research"
        },
        {
            "title": "Deep Learning",
            "content": "DL Overview Content",
            "connections": ["Machine Learning"],
            "general_title": "AI Research"
        }
    ]
    
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = notes_payload
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)
    
    worker = AiNoteGeneratorWorker(
        "sample input text",
        on_finished=lambda res: None,
        on_error=None
    )
    worker.run()
    
    all_links = temp_db.get_all_note_links()
    for src_id, tgt_id in all_links:
        assert src_id != tgt_id, f"Self-referential link detected: {src_id} -> {tgt_id}"
    assert len(all_links) >= 1

def test_ai_worker_batch_intra_connection_prioritization(temp_db, monkeypatch):
    # Pre-insert an old note with title "Introduction"
    temp_db.insert_note("old-id", "Introduction", "Old Intro Content", "Old Category")

    # AI generates a batch where Note 1 is "Introduction" (will be disambiguated to Introduction (2))
    # and Note 2 is "Details" linking to "Introduction" (should link to the new Note 1, not old-id)
    notes_payload = [
        {"title": "Introduction", "content": "New Intro Content", "connections": [], "general_title": "AI Batch"},
        {"title": "Details", "content": "Details Content", "connections": ["Introduction"], "general_title": "AI Batch"}
    ]

    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = notes_payload
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    worker = AiNoteGeneratorWorker("dummy text", on_finished=None, on_error=None)
    worker.run()

    all_notes, _ = note_manager.load_all_notes_metadata(temp_db)
    details_note_id = [nid for nid, title, _ in all_notes if title == "Details"][0]
    new_intro_id = [nid for nid, title, _ in all_notes if title == "Introduction (2)"][0]

    links = temp_db.get_note_links(details_note_id)
    assert new_intro_id in links
    assert "old-id" not in links

def test_ai_worker_cancellation(temp_db, monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = [
        {"title": "Note 1", "content": "Content 1", "connections": [], "general_title": "AI Batch"}
    ]
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)
    
    worker = AiNoteGeneratorWorker("text", on_finished=None, on_error=None)
    worker.cancel()
    assert worker.is_cancelled() is True
    worker.run()
    
    # DB must have 0 notes inserted
    all_notes, _ = note_manager.load_all_notes_metadata(temp_db)
    assert len(all_notes) == 0

def test_pdf_text_extraction_runs_in_background_worker(monkeypatch):
    extraction_thread_id = None
    main_thread_id = threading.get_ident()
    
    def mock_extract(pdf_path):
        nonlocal extraction_thread_id
        extraction_thread_id = threading.get_ident()
        return "Extracted PDF content text"
        
    monkeypatch.setattr(pdf_processor, "extract_text_from_pdf", mock_extract)
    
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = []
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)
    
    worker = AiNoteGeneratorWorker(
        "dummy.pdf",
        on_finished=lambda notes: None,
        on_error=None
    )
    
    t = threading.Thread(target=worker.run)
    t.start()
    t.join(timeout=2.0)
    
    assert extraction_thread_id is not None
    assert extraction_thread_id != main_thread_id

def test_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pdf_processor.extract_text_from_pdf("non_existent_file.pdf")

def test_pdf_empty_file_raises(tmp_path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        pdf_processor.extract_text_from_pdf(str(empty_file))

def test_pdf_encrypted_file_raises(monkeypatch, tmp_path):
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
    
    mock_reader = MagicMock()
    mock_reader.is_encrypted = True
    mock_reader.pages = [MagicMock()]
    monkeypatch.setattr("pypdf.PdfReader", lambda f: mock_reader)
    
    with pytest.raises(ValueError, match="encrypted"):
        pdf_processor.extract_text_from_pdf(str(dummy_pdf))

def test_pdf_zero_pages_raises(monkeypatch, tmp_path):
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")
    
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = []
    monkeypatch.setattr("pypdf.PdfReader", lambda f: mock_reader)
    
    with pytest.raises(ValueError, match="no pages"):
        pdf_processor.extract_text_from_pdf(str(dummy_pdf))

def test_worker_empty_notes_result(temp_db, monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = []
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    finished_results = []
    errors = []
    worker = AiNoteGeneratorWorker(
        "Some dummy text",
        on_finished=lambda notes: finished_results.append(notes),
        on_error=lambda err: errors.append(err)
    )
    worker.run()

    assert len(finished_results) == 1
    assert finished_results[0] == []
    assert len(errors) == 0

def test_worker_handles_malformed_items(temp_db, monkeypatch):
    malformed_payload = [
        "not a dict",
        None,
        {"title": "Valid Note", "content": "Valid Content", "connections": [None, 123, "NonExistentLink"], "general_title": "AI Batch"}
    ]
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = malformed_payload
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    finished_results = []
    worker = AiNoteGeneratorWorker(
        "Some text",
        on_finished=lambda notes: finished_results.append(notes),
        on_error=None
    )
    worker.run()

    assert len(finished_results) == 1
    all_notes, _ = note_manager.load_all_notes_metadata(temp_db)
    assert len(all_notes) == 1
    assert all_notes[0][1] == "Valid Note"

def test_worker_gemini_typed_errors(monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.side_effect = GeminiAuthError("Auth Failure")
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    errors = []
    worker = AiNoteGeneratorWorker(
        "Some text",
        on_finished=None,
        on_error=lambda err: errors.append(err)
    )
    worker.run()

    assert len(errors) == 1
    assert "Auth Failure" in errors[0]

