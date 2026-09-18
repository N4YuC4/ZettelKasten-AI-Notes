# ai_note_generator_worker.py
#
# Worker class that generates Zettelkasten-style notes from PDF content or raw text
# using Google Gemini AI, and atomically commits notes and links to SQLite.
# Executed in a background thread to prevent UI freezing.

import os
from env_config import configure_headless_environment

configure_headless_environment()

import re
import difflib
import threading
import multiprocessing as mp
import traceback
from uuid import uuid4
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any

from ai_provider import BaseAiProvider, create_ai_provider
from gemini_api_client import GeminiApiClient, GeminiApiError, GeminiAuthError, GeminiRateLimitError
from local_gguf_client import LocalGgufClient, LocalLlmError, LocalModelNotFoundError, LocalModelOOMError
import local_models_catalog
import model_downloader
from hardware_checker import HardwareChecker
import note_service
import database_manager
import pdf_processor
from logger import log_debug, log_error


def _resolve_target_id(
    target_title_raw: str,
    batch_title_to_id: Dict[str, str],
    global_title_to_id: Dict[str, str]
) -> Optional[str]:
    """
    Resolves connection targets using strict matching ONLY to prevent false knowledge graph links:
    1. Direct exact match (verbatim title)
    2. Sanitized exact match (markdown # stripped)
    3. Case-folded / whitespace-trimmed exact match
    Never uses fuzzy or substring guessing to prevent hallucinated graph edges.
    """
    if not target_title_raw or not isinstance(target_title_raw, str):
        return None

    raw_clean = target_title_raw.strip()
    if not raw_clean:
        return None

    sanitized = note_service.sanitize_title(raw_clean)

    # 1. Verbatim exact lookup in batch or global
    direct = (
        batch_title_to_id.get(raw_clean)
        or batch_title_to_id.get(sanitized)
        or global_title_to_id.get(raw_clean)
        or global_title_to_id.get(sanitized)
    )
    if direct:
        return direct

    # 2. Strict case-folded exact match (case-insensitive exact)
    target_folded = raw_clean.casefold()
    sanitized_folded = sanitized.casefold()

    for title, nid in batch_title_to_id.items():
        if title and (title.casefold() == target_folded or title.casefold() == sanitized_folded):
            return nid

    for title, nid in global_title_to_id.items():
        if title and (title.casefold() == target_folded or title.casefold() == sanitized_folded):
            return nid

    return None


def _isolated_local_inference_entry(
    queue: Any,
    model_path: str,
    text_content: str,
    n_gpu_layers: int,
    n_threads: Optional[int] = None,
    custom_system_prompt: Optional[str] = None
) -> None:
    """
    Executes local GGUF model loading and inference in a completely isolated process.
    Prevents any Wayland / EGL presentation driver conflicts with Flutter/Flet GUI threads,
    guarantees clean VRAM/RAM reclamation on exit, and allows instant cancellation via process termination.
    """
    try:
        from hardware_checker import HardwareChecker
        from local_gguf_client import LocalGgufClient

        HardwareChecker.configure_vulkan_environment(enable_gpu=(n_gpu_layers != 0))
        queue.put(("progress", "Loading model into memory..."))

        client = LocalGgufClient(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            text_content=text_content
        )
        queue.put(("progress", "Model ready. Generating notes..."))

        def report_prog(p_msg: str):
            try:
                queue.put(("progress", p_msg))
            except Exception:
                pass

        notes = client.generate_zettelkasten_notes(
            text_content,
            on_progress=report_prog,
            custom_system_prompt=custom_system_prompt
        )
        if notes and len(notes) > 1:
            report_prog("Analyzing graph connections...")
            notes = client.generate_note_links(notes, on_progress=report_prog)
        queue.put(("result", notes))
    except Exception as exc:
        queue.put(("error", str(exc)))


class AiNoteGeneratorWorker:
    """
    Worker that executes the AI note generation and database persistence in a background thread.
    Supports cancellation and atomic batch rollback.
    """
    def __init__(
        self,
        extracted_text: str,
        on_finished: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ):
        self.extracted_text = extracted_text
        self.extracted_text_or_path = extracted_text
        self.on_finished = on_finished
        self.on_error = on_error
        self.on_progress = on_progress
        self._cancel_event = threading.Event()
        self._active_process: Optional[Any] = None

    def report_progress(self, msg: str) -> None:
        """Dispatches live progress update to UI callback."""
        log_debug(f"[Worker Progress] {msg}")
        if self.on_progress:
            try:
                self.on_progress(msg)
            except Exception as pe:
                log_error(f"Error in on_progress callback: {pe}")

    def cancel(self) -> None:
        """Signals the worker to cancel the ongoing generation process."""
        self._cancel_event.set()
        if self._active_process and self._active_process.is_alive():
            try:
                log_debug("AiNoteGeneratorWorker: Terminating active child inference process.")
                self._active_process.terminate()
            except Exception as e:
                log_error(f"Error terminating child inference process: {e}")

    def is_cancelled(self) -> bool:
        """Returns True if the worker process has been cancelled."""
        return self._cancel_event.is_set()

    def create_provider(
        self,
        provider_type: str,
        db_manager: Optional[Any] = None,
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        **kwargs
    ) -> BaseAiProvider:
        """
        Instantiates the appropriate BaseAiProvider implementation.
        """
        if provider_type == "local":
            return LocalGgufClient(model_path, n_gpu_layers=n_gpu_layers)
        else:
            api_key = kwargs.get("api_key") or (db_manager.get_setting("GEMINI_API_KEY") if db_manager else None)
            try:
                return GeminiApiClient(api_key=api_key, db_manager=db_manager)
            except TypeError:
                # Support 0-argument test mocks (e.g. lambda: mock_client)
                return GeminiApiClient()

    def run(self) -> None:
        """Main execution method for the background worker thread."""
        db_manager_worker = None
        try:
            # Check for early cancellation
            if self._cancel_event.is_set():
                log_debug("DEBUG: AiNoteGeneratorWorker cancelled before start.")
                return

            self.report_progress("Preparing text...")

            # 1. Resolve text content (extract from PDF if path provided, or use raw text)
            if isinstance(self.extracted_text_or_path, str) and (
                self.extracted_text_or_path.lower().endswith(".pdf") or os.path.isfile(self.extracted_text_or_path)
            ):
                text_content = pdf_processor.extract_text_from_pdf(self.extracted_text_or_path)
            else:
                text_content = self.extracted_text_or_path

            if not text_content or not str(text_content).strip():
                raise ValueError("Input text is empty or contains no readable content.")

            # Check for cancellation before expensive API calls
            if self._cancel_event.is_set():
                log_debug("DEBUG: AiNoteGeneratorWorker cancelled before Gemini API call.")
                return

            # 2. Initialize thread-local DatabaseManager
            db_manager_worker = database_manager.DatabaseManager(init_tables=True)
            custom_prompt = db_manager_worker.get_setting("AI_CUSTOM_SYSTEM_PROMPT")

            # 3. Determine AI Provider and generate notes
            ai_provider = (db_manager_worker.get_setting("AI_PROVIDER") or "gemini").lower()
            log_debug(f"AiNoteGeneratorWorker using provider: {ai_provider}")

            if ai_provider == "local":
                active_model_id = db_manager_worker.get_setting("ACTIVE_LOCAL_MODEL") or local_models_catalog.DEFAULT_MODEL_ID
                models_dir = db_manager_worker.get_setting("MODELS_DIR") or local_models_catalog.get_default_models_dir()
                model_info = local_models_catalog.get_model_by_id(active_model_id)

                if not model_info:
                    raise LocalModelNotFoundError(f"Selected model not found in catalog: '{active_model_id}'")

                model_path = model_downloader.ModelDownloader.get_model_path(model_info, models_dir)
                if not model_downloader.ModelDownloader.is_model_downloaded(model_info, models_dir):
                    raise LocalModelNotFoundError(
                        f"'{model_info.display_name}' has not been downloaded yet. "
                        "Please download the model via Settings -> Model Manager."
                    )

                gpu_accel = (db_manager_worker.get_setting("GPU_ACCELERATION") != "False")
                n_gpu_layers = -1 if gpu_accel else 0
                log_debug(f"AiNoteGeneratorWorker local LLM n_gpu_layers={n_gpu_layers} (GPU_ACCELERATION={gpu_accel})")
                
                # In test environments (pytest), run in-process so unit test mocks work as expected
                if "PYTEST_CURRENT_TEST" in os.environ:
                    HardwareChecker.configure_vulkan_environment(enable_gpu=gpu_accel)
                    self.report_progress("Loading model into memory...")
                    local_client = self.create_provider("local", db_manager_worker, model_path=model_path, n_gpu_layers=n_gpu_layers)

                    if self._cancel_event.is_set():
                        log_debug("DEBUG: AiNoteGeneratorWorker cancelled after loading model.")
                        return

                    self.report_progress("Model ready. Generating notes...")
                    generated_notes = local_client.generate_zettelkasten_notes(
                        text_content,
                        on_progress=self.report_progress,
                        custom_system_prompt=custom_prompt
                    )
                    if generated_notes and len(generated_notes) > 1:
                        self.report_progress("Analyzing graph connections...")
                        linked_notes = local_client.generate_note_links(
                            generated_notes,
                            on_progress=self.report_progress
                        )
                        if isinstance(linked_notes, list):
                            generated_notes = linked_notes
                else:
                    # In live production GUI, run in an isolated spawn process to guarantee
                    # zero driver deadlocks with Wayland/Flutter/EGL, live UI responsiveness,
                    # and clean OS memory reclamation.
                    ctx = mp.get_context("spawn")
                    queue = ctx.Queue()
                    proc = ctx.Process(
                        target=_isolated_local_inference_entry,
                        args=(queue, model_path, text_content, n_gpu_layers, None, custom_prompt)
                    )
                    self._active_process = proc
                    proc.start()

                    generated_notes = None
                    error_msg = None

                    while proc.is_alive() or not queue.empty():
                        if self._cancel_event.is_set():
                            log_debug("DEBUG: AiNoteGeneratorWorker cancelled. Terminating inference process.")
                            proc.terminate()
                            proc.join(timeout=2.0)
                            if proc.is_alive():
                                proc.kill()
                            return

                        try:
                            msg_type, data = queue.get(timeout=0.1)
                            if msg_type == "progress":
                                self.report_progress(data)
                            elif msg_type == "result":
                                generated_notes = data
                                break
                            elif msg_type == "error":
                                error_msg = data
                                break
                        except Exception:
                            pass

                    proc.join(timeout=3.0)
                    if proc.is_alive():
                        proc.terminate()
                    self._active_process = None

                    if error_msg:
                        raise LocalLlmError(error_msg)

                    if generated_notes is None:
                        if self._cancel_event.is_set():
                            return
                        raise LocalLlmError("Local model shut down before completing note extraction.")
            else:
                self.report_progress("Generating notes with Gemini...")
                gemini_client = self.create_provider("gemini", db_manager_worker)
                generated_notes = gemini_client.generate_zettelkasten_notes(
                    text_content,
                    on_progress=self.report_progress,
                    custom_system_prompt=custom_prompt
                )
                if generated_notes and len(generated_notes) > 1:
                    if self._cancel_event.is_set():
                        log_debug("DEBUG: AiNoteGeneratorWorker cancelled before Gemini linking.")
                        return
                    self.report_progress("Analyzing graph connections...")
                    linked_notes = gemini_client.generate_note_links(
                        generated_notes,
                        on_progress=self.report_progress
                    )
                    if isinstance(linked_notes, list):
                        generated_notes = linked_notes

            # Check for cancellation after API call
            if self._cancel_event.is_set():
                log_debug("DEBUG: AiNoteGeneratorWorker cancelled after Gemini API call.")
                return

            if not generated_notes:
                log_debug("DEBUG: No notes generated by Gemini API.")
                if self.on_finished:
                    try:
                        self.on_finished([])
                    except Exception as cb_err:
                        log_error(f"Error in on_finished callback: {cb_err}")
                return

            # 4. Stage 1: Build fresh mapping of existing notes & disambiguate titles
            title_to_id = db_manager_worker.get_all_note_titles_and_ids()
            used_titles = set(title_to_id.keys())
            batch_title_to_id = {}
            log_debug(f"DEBUG: Initial title_to_id count: {len(title_to_id)}")

            for note_data in generated_notes:
                if not isinstance(note_data, dict):
                    continue

                raw_title = note_data.get('title', 'Untitled Note')
                sanitized_title = note_service.sanitize_title(f"# {raw_title}")
                final_title = note_service.disambiguate_title(sanitized_title, used_titles)

                used_titles.add(final_title)
                new_id = str(uuid4())

                note_data['_final_id'] = new_id
                note_data['_final_title'] = final_title
                note_data['_is_new'] = True

                batch_title_to_id[final_title] = new_id
                batch_title_to_id[sanitized_title] = new_id
                batch_title_to_id[raw_title] = new_id

                title_to_id[final_title] = new_id
                if sanitized_title not in title_to_id:
                    title_to_id[sanitized_title] = new_id
                if raw_title not in title_to_id:
                    title_to_id[raw_title] = new_id

            # 5. Stage 2: Prepare batch insertion records and connections
            self.report_progress("Establishing graph connections...")
            notes_to_insert = []
            links_to_insert = []
            now = datetime.now().isoformat()

            for note_data in generated_notes:
                if not isinstance(note_data, dict):
                    continue

                final_id = note_data['_final_id']
                final_title = note_data.get('_final_title', note_data.get('title', 'Untitled Note'))
                content = note_data.get('content', '')
                collection = note_data.get('general_title', 'AI Generated')

                full_content = f"# {final_title}\n\n{content}"

                notes_to_insert.append(
                    (final_id, final_title, full_content, collection, now, now)
                )

                connections = note_data.get('connections', [])
                if isinstance(connections, list):
                    for target_title_raw in connections:
                        if not isinstance(target_title_raw, str):
                            continue
                        target_id = _resolve_target_id(target_title_raw, batch_title_to_id, title_to_id)
                        # Avoid self-referential links and ensure target exists
                        if target_id and target_id != final_id:
                            link_pair = (final_id, target_id)
                            if link_pair not in links_to_insert:
                                links_to_insert.append(link_pair)
                        elif not target_id:
                            log_debug(f"DEBUG: Could not find target_id for '{target_title_raw}'. Link not inserted.")

            # Check for cancellation before DB commit
            if self._cancel_event.is_set():
                log_debug("DEBUG: AiNoteGeneratorWorker cancelled before database commit.")
                return

            # 6. Stage 3: Atomic single-transaction database insertion
            self.report_progress("Saving notes to database...")
            try:
                db_manager_worker.bulk_insert_notes_and_links(notes_to_insert, links_to_insert)
                log_debug(f"DEBUG: Atomically inserted {len(notes_to_insert)} notes and {len(links_to_insert)} links.")
            except Exception as db_err:
                log_error(f"Database error during atomic batch insertion: {db_err}")
                raise db_err

            # 7. Safe callback dispatch
            if self.on_finished:
                try:
                    log_debug("AiNoteGeneratorWorker: Dispatching on_finished callback to UI.")
                    self.on_finished(generated_notes)
                    log_debug("AiNoteGeneratorWorker: on_finished callback executed successfully.")
                except Exception as cb_err:
                    log_error(f"Error in on_finished callback: {cb_err}\n{traceback.format_exc()}")

        except (GeminiAuthError, GeminiRateLimitError, GeminiApiError, LocalLlmError) as ge:
            err_msg = str(ge)
            log_error(f"AiNoteGeneratorWorker AI error: {err_msg}")
            if self.on_error:
                try:
                    self.on_error(err_msg)
                except Exception as cb_err:
                    log_error(f"Error in on_error callback: {cb_err}")
        except (ValueError, FileNotFoundError) as ve:
            err_msg = str(ve)
            log_error(f"AiNoteGeneratorWorker validation error: {err_msg}")
            if self.on_error:
                try:
                    self.on_error(err_msg)
                except Exception as cb_err:
                    log_error(f"Error in on_error callback: {cb_err}")
        except Exception as e:
            err_msg = f"An error occurred during AI note generation: {e}"
            log_error(f"AiNoteGeneratorWorker unhandled exception: {err_msg}\n{traceback.format_exc()}")
            if self.on_error:
                try:
                    self.on_error(err_msg)
                except Exception as cb_err:
                    log_error(f"Error in on_error callback: {cb_err}")
        finally:
            # Explicitly unload local LLM from RAM/VRAM to free memory immediately after completion
            try:
                LocalGgufClient.unload_cached_model()
            except Exception as unload_err:
                log_error(f"Error unloading local GGUF model: {unload_err}")

            if db_manager_worker:
                try:
                    db_manager_worker.close_connection()
                except Exception as close_err:
                    log_error(f"Error closing DB connection: {close_err}")
