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
    
    def failing_bulk_insert(self, notes_data, links_data):
        raise sqlite3.OperationalError("Simulated DB Disk Failure during Link Insertion")
    monkeypatch.setattr(database_manager.DatabaseManager, "bulk_insert_notes_and_links", failing_bulk_insert)
    
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


def test_resolve_target_id_strict_matching():
    from ai_note_generator_worker import _resolve_target_id
    batch_map = {
        "Binaural Beats": "id-1",
        "Transcranial Magnetic Stimulation (TMS)": "id-2",
        "I-Dosers": "id-3",
    }
    global_map = {
        "Placebo Effect": "id-4",
    }

    # 1. Verbatim exact matches
    assert _resolve_target_id("Binaural Beats", batch_map, global_map) == "id-1"
    assert _resolve_target_id("Placebo Effect", batch_map, global_map) == "id-4"

    # 2. Markdown header sanitized exact matches
    assert _resolve_target_id("# Binaural Beats", batch_map, global_map) == "id-1"
    assert _resolve_target_id("## Placebo Effect  ", batch_map, global_map) == "id-4"

    # 3. Case-folded exact match
    assert _resolve_target_id("binaural beats", batch_map, global_map) == "id-1"

    # 4. Article-stripped matching (e.g., 'The ...' <-> '...')
    batch_with_articles = {
        "The Argument from Consciousness (Jefferson's Critique)": "id-art-1",
        "Digital Computers as Discrete State Machines": "id-morph-1",
    }
    assert _resolve_target_id("Argument from Consciousness (Jefferson's Critique)", batch_with_articles, {}) == "id-art-1"
    assert _resolve_target_id("The Argument from Consciousness (Jefferson's Critique)", batch_with_articles, {}) == "id-art-1"

    # 5. Morphological token 1-to-1 matching (singular/plural variations: Computer <-> Computers)
    assert _resolve_target_id("Digital Computer as Discrete State Machines", batch_with_articles, {}) == "id-morph-1"
    assert _resolve_target_id("Digital Computers as Discrete State Machine", batch_with_articles, {}) == "id-morph-1"

    # 6. Strict rejection of substrings / prefixes / partial fuzzy matches to avoid false graph links
    # 'Binaural' is only part of 'Binaural Beats' -> must return None
    assert _resolve_target_id("Binaural", batch_map, global_map) is None
    # 'Transcranial Magnetic Stimulation' missing '(TMS)' -> must return None
    assert _resolve_target_id("Transcranial Magnetic Stimulation", batch_map, global_map) is None
    # Dissimilar or unknown target returns None
    assert _resolve_target_id("Direct Brain Stimulation", batch_map, global_map) is None
    assert _resolve_target_id("Completely Unrelated Topic", batch_map, global_map) is None



def test_worker_unloads_cached_model_on_finish(temp_db, monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = []
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    unload_called = False
    def mock_unload():
        nonlocal unload_called
        unload_called = True

    from local_gguf_client import LocalGgufClient
    monkeypatch.setattr(LocalGgufClient, "unload_cached_model", mock_unload)

    worker = AiNoteGeneratorWorker("dummy text", on_finished=None, on_error=None)
    worker.run()

    assert unload_called is True


def test_worker_passes_n_gpu_layers_based_on_setting(temp_db, monkeypatch):
    import database_manager
    db = database_manager.DatabaseManager(init_tables=False)
    db.set_setting("AI_PROVIDER", "local")
    db.set_setting("GPU_ACCELERATION", "False")

    mock_model_info = MagicMock()
    mock_model_info.display_name = "Mock Model"
    monkeypatch.setattr("local_models_catalog.get_model_by_id", lambda mid: mock_model_info)
    monkeypatch.setattr("model_downloader.ModelDownloader.is_model_downloaded", lambda mi, md: True)
    monkeypatch.setattr("model_downloader.ModelDownloader.get_model_path", lambda mi, md: "/dummy/path.gguf")

    passed_n_gpu_layers = []

    class MockLocalClient:
        def __init__(self, model_path, n_gpu_layers=-1):
            passed_n_gpu_layers.append(n_gpu_layers)

        def generate_zettelkasten_notes(self, text, *args, **kwargs):
            return []

        @classmethod
        def unload_cached_model(cls):
            pass

    monkeypatch.setattr("ai_note_generator_worker.LocalGgufClient", MockLocalClient)

    # 1. GPU_ACCELERATION is False -> n_gpu_layers=0
    worker = AiNoteGeneratorWorker("dummy text", on_finished=None, on_error=None)
    worker.run()
    assert passed_n_gpu_layers == [0]

    # 2. GPU_ACCELERATION is True -> n_gpu_layers=-1
    db.set_setting("GPU_ACCELERATION", "True")
    worker2 = AiNoteGeneratorWorker("dummy text", on_finished=None, on_error=None)
    worker2.run()
    assert passed_n_gpu_layers == [0, -1]


def test_worker_reports_progress(temp_db, monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = [
        {"title": "Note 1", "content": "Content 1", "connections": []}
    ]
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    progress_log = []
    worker = AiNoteGeneratorWorker(
        "sample text",
        on_finished=None,
        on_error=None,
        on_progress=lambda msg: progress_log.append(msg)
    )
    worker.run()
    assert len(progress_log) > 0
    assert any("text" in p.lower() for p in progress_log)


def test_worker_cancel_before_run(temp_db):
    finished = []
    errors = []
    worker = AiNoteGeneratorWorker(
        "sample text",
        on_finished=lambda notes: finished.append(notes),
        on_error=lambda err: errors.append(err)
    )
    worker.cancel()
    assert worker.is_cancelled() is True
    worker.run()
    assert len(finished) == 0
    assert len(errors) == 0


def test_worker_gemini_calls_generate_note_links(temp_db, monkeypatch):
    mock_gemini_client = MagicMock()
    mock_gemini_client.generate_zettelkasten_notes.return_value = [
        {"title": "Note Alpha", "content": "Content Alpha", "connections": [], "general_title": "Topic"},
        {"title": "Note Beta", "content": "Content Beta", "connections": [], "general_title": "Topic"}
    ]
    mock_gemini_client.generate_note_links.return_value = [
        {"title": "Note Alpha", "content": "Content Alpha", "connections": ["Note Beta"], "general_title": "Topic"},
        {"title": "Note Beta", "content": "Content Beta", "connections": ["Note Alpha"], "general_title": "Topic"}
    ]
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini_client)

    progress_messages = []
    worker = AiNoteGeneratorWorker(
        "sample text",
        on_finished=None,
        on_error=None,
        on_progress=lambda m: progress_messages.append(m)
    )
    worker.run()

    mock_gemini_client.generate_zettelkasten_notes.assert_called_once()
    mock_gemini_client.generate_note_links.assert_called_once()
    assert any("Analyzing graph connections" in m for m in progress_messages)


def test_worker_merges_stage1_folgezettel_and_stage2_verweis_links(temp_db, monkeypatch):
    """
    Verifies that:
    1. Tier 1 (Folgezettel / Intra-chunk) connections from Stage 1 are preserved and made bidirectional.
    2. Tier 2 (Verweis / Macro) connections from Stage 2 are merged seamlessly.
    3. Standalone notes with no connections remain unlinked (no artificial/forced links).
    """
    stage1_notes = [
        {"title": "Hardware Architecture", "content": "Three-part computer architecture.", "connections": [], "general_title": "Turing"},
        {"title": "Store Unit", "content": "The memory store sub-component.", "connections": ["Hardware Architecture"], "general_title": "Turing"},
        {"title": "Philosophical Objection", "content": "The theological and mathematical objections.", "connections": [], "general_title": "Turing"},
        {"title": "Standalone Historical Axiom", "content": "Independent historical context.", "connections": [], "general_title": "Turing"}
    ]

    mock_gemini = MagicMock()
    mock_gemini.generate_zettelkasten_notes.return_value = stage1_notes
    def mock_linking(notes, on_progress=None):
        from ai_response_parser import AiResponseParser
        return AiResponseParser.attach_links_to_notes(notes, [(1, 3)])

    mock_gemini.generate_note_links.side_effect = mock_linking
    monkeypatch.setattr("ai_note_generator_worker.GeminiApiClient", lambda: mock_gemini)

    worker = AiNoteGeneratorWorker("dummy text", on_finished=None, on_error=None)
    worker.run()

    # Verify notes and links in database
    title_to_id = temp_db.get_all_note_titles_and_ids()
    id_hw = title_to_id["Hardware Architecture"]
    id_store = title_to_id["Store Unit"]
    id_obj = title_to_id["Philosophical Objection"]
    id_axiom = title_to_id["Standalone Historical Axiom"]

    # Check links
    hw_links = temp_db.get_note_links(id_hw)
    store_links = temp_db.get_note_links(id_store)
    obj_links = temp_db.get_note_links(id_obj)
    axiom_links = temp_db.get_note_links(id_axiom)

    # Hardware Architecture should be linked to Store Unit (Folgezettel) AND Philosophical Objection (Verweis)
    assert id_store in hw_links
    assert id_obj in hw_links

    # Store Unit should be linked to Hardware Architecture (Folgezettel)
    assert id_hw in store_links

    # Philosophical Objection should be linked to Hardware Architecture (Verweis)
    assert id_hw in obj_links

    # Standalone Historical Axiom must have ZERO links (unforced/genuine isolation)
    assert len(axiom_links) == 0

    # Check Markdown wikilinks inside note contents
    hw_content = temp_db.read_note_content(id_hw)
    assert "[[Store Unit]]" in hw_content
    assert "[[Philosophical Objection]]" in hw_content

    store_content = temp_db.read_note_content(id_store)
    assert "[[Hardware Architecture]]" in store_content

    axiom_content = temp_db.read_note_content(id_axiom)
    assert "## Related Notes" not in axiom_content
    assert "[[" not in axiom_content
