# ai_note_generator_worker.py
#
# This file defines a worker class that generates Zettelkasten-style notes from PDF content
# using artificial intelligence (AI) and saves these notes to the database.
# The process is executed in a separate thread to avoid freezing the GUI.

from PyQt5.QtCore import QObject, pyqtSignal # For PyQt signal and object system
from gemini_api_client import GeminiApiClient # For interacting with the Gemini API
import note_manager # For note management functions (saving, title sanitization)
import database_manager # For database operations
from logger import log_debug # For debug logging function
from uuid import uuid4 # For generating unique IDs
from datetime import datetime # For timestamps

# The AiNoteGeneratorWorker class executes the AI note generation process in a separate thread.
# It is derived from QObject to use the signal/slot mechanism.
class AiNoteGeneratorWorker(QObject):
    # finished signal: Emits a list of generated notes when the process is successfully completed.
    finished = pyqtSignal(list) 
    # error signal: Emits an error message when an error occurs during the process.
    error = pyqtSignal(str) 

    # The __init__ method initializes the worker object.
    # extracted_text: The text extracted from the PDF.
    def __init__(self, extracted_text):
        super().__init__()
        self.extracted_text = extracted_text # Store the text to be processed

    # The run method is the main function called when the thread starts.
    # It contains the logic for AI note generation, saving, and linking.
    def run(self):
        # Create a new DatabaseManager instance for this thread.
        # Each thread should have its own database connection.
        db_manager_worker = database_manager.DatabaseManager()
        try:
            gemini_client = GeminiApiClient() # Create the Gemini API client
            # Generate Zettelkasten notes using the Gemini API
            generated_notes = gemini_client.generate_zettelkasten_notes(self.extracted_text)

            if generated_notes: # If notes were successfully generated
                # Load existing notes and their IDs for a comprehensive search
                title_to_id = db_manager_worker.get_all_note_titles_and_ids()
                log_debug(f"DEBUG: Initial title_to_id: {title_to_id}")

                notes_to_insert = []
                links_to_insert = []
                new_note_mappings = {}

                # Stage 1: Prepare note data and temporary IDs
                for note_data in generated_notes:
                    temp_id = str(uuid4())
                    title = note_data.get('title', 'Untitled Note')
                    sanitized_title = note_manager.get_sanitized_title(f"# {title}")
                    new_note_mappings[sanitized_title] = temp_id
                    note_data['_temp_id'] = temp_id

                # Merge both existing and new note titles
                title_to_id.update(new_note_mappings)

                now = datetime.now().isoformat()
                # Stage 2: Create lists for inserting notes and links
                for note_data in generated_notes:
                    source_id = note_data['_temp_id']
                    title = note_data.get('title', 'Untitled Note')
                    content = note_data.get('content', '')
                    category = note_data.get('general_title', 'AI Generated')
                    
                    full_content = f"# {title}\n\n{content}"
                    sanitized_title = note_manager.get_sanitized_title(full_content)

                    notes_to_insert.append(
                        (source_id, sanitized_title, full_content, category, now, now)
                    )

                    connections = note_data.get('connections', [])
                    for target_title_raw in connections:
                        sanitized_target_title = note_manager.get_sanitized_title(target_title_raw)
                        target_id = title_to_id.get(sanitized_target_title)
                        if target_id:
                            links_to_insert.append((source_id, target_id))
                        else:
                            log_debug(f"DEBUG: Could not find target_id for '{sanitized_target_title}' (raw: '{target_title_raw}'). Link not inserted.")

                # Stage 3: Bulk database operations
                if notes_to_insert:
                    db_manager_worker.bulk_insert_notes(notes_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(notes_to_insert)} notes.")

                if links_to_insert:
                    db_manager_worker.bulk_insert_links(links_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(links_to_insert)} links.")

                self.finished.emit(generated_notes)
            else:
                self.error.emit("AI did not generate any notes from the PDF content.")
        except ValueError as ve:
            self.error.emit(f"API Key Error: {str(ve)}")
        except Exception as e:
            self.error.emit(f"An error occurred during AI note generation: {e}")
        finally:
            if db_manager_worker:
                db_manager_worker.close_connection()
