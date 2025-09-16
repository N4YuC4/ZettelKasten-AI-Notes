from PyQt5.QtCore import QObject, pyqtSignal
from gemini_api_client import GeminiApiClient
import note_manager
import database_manager 
from main import log_debug # Import log_debug

class AiNoteGeneratorWorker(QObject):
    finished = pyqtSignal(list) # Emits a list of generated notes or an empty list on failure
    error = pyqtSignal(str) # Emits an error message

    def __init__(self, extracted_text): # Removed db_manager
        super().__init__()
        self.extracted_text = extracted_text
        # self.db_manager = db_manager # Removed

    def run(self):
        # Create a new db_manager instance in this thread
        db_manager_worker = database_manager.DatabaseManager()
        try:
            gemini_client = GeminiApiClient()
            generated_notes = gemini_client.generate_zettelkasten_notes(self.extracted_text)

            if generated_notes:
                # Load existing notes into title_to_id for comprehensive lookup
                title_to_id = db_manager_worker.get_all_note_titles_and_ids()
                log_debug(f"DEBUG: Initial title_to_id: {title_to_id}")
                note_ids = []
                notes_saved_count = 0

                # Phase 1: Save all notes and populate title_to_id with newly saved notes
                for note_data in generated_notes:
                    title = note_data.get('title', 'Untitled Note')
                    content = note_data.get('content', '')
                    category = note_data.get('general_title', 'AI Generated')
                    if title and content:
                        new_note_id, saved_title = note_manager.save_note(db_manager_worker, None, f"# {title}\n\n{content}", category)
                        title_to_id[saved_title] = new_note_id # Add/update with newly saved note
                        note_ids.append(new_note_id)
                        notes_saved_count += 1
                        log_debug(f"DEBUG: Saved note '{saved_title}' with ID '{new_note_id}'")
                log_debug(f"DEBUG: Final title_to_id after new notes: {title_to_id}")

                # Phase 2: Add connections after all notes have been saved
                for i, note_data in enumerate(generated_notes):
                    source_id = note_ids[i]
                    connections = note_data.get('connections', [])
                    for target_title_raw in connections:
                        sanitized_target_title = note_manager.get_sanitized_title(target_title_raw)
                        log_debug(f"DEBUG: Looking for target_title {repr(sanitized_target_title)} (len: {len(sanitized_target_title)}, raw: {repr(target_title_raw)}) in title_to_id.")
                        target_id = title_to_id.get(sanitized_target_title)
                        if target_id:
                            log_debug(f"DEBUG: Found target_id '{target_id}' for '{sanitized_target_title}'. Inserting link from '{source_id}' to '{target_id}'.")
                            log_debug(f"DEBUG: Attempting to insert link: Source ID: {source_id}, Target ID: {target_id}, Raw Target Title: {target_title_raw}")
                            db_manager_worker.insert_note_link(source_id, target_id)
                        else:
                            log_debug(f"DEBUG: Could not find target_id for '{sanitized_target_title}' (raw: '{target_title_raw}'). Link not inserted.")
                self.finished.emit(generated_notes)
            else:
                self.error.emit("AI did not generate any notes from the PDF content.")
        except ValueError as ve:
            self.error.emit(f"API Key Error: {str(ve)}")
        except Exception as e:
            self.error.emit(f"An error occurred during AI note generation: {e}")
        finally:
            # Close the worker's db_manager connection
            db_manager_worker.close_connection()
