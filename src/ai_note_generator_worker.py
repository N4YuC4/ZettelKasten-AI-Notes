# ai_note_generator_worker.py
#
# This file defines a worker class that generates Zettelkasten-style notes from PDF content
# using artificial intelligence (AI) and saves these notes to the database.
# The process is executed in a separate thread to avoid freezing the GUI.

from gemini_api_client import GeminiApiClient # For interacting with the Gemini API
import note_manager # For note management functions (saving, title sanitization)
import database_manager # For database operations
from logger import log_debug # For debug logging function
from uuid import uuid4 # For generating unique IDs
from datetime import datetime # For timestamps

# The AiNoteGeneratorWorker class executes the AI note generation process in a separate thread.
class AiNoteGeneratorWorker:
    # The __init__ method initializes the worker object.
    # extracted_text: The text extracted from the PDF.
    # on_finished: Callback function triggered when notes generation completes successfully.
    # on_error: Callback function triggered when an error occurs.
    def __init__(self, extracted_text, on_finished, on_error):
        self.extracted_text = extracted_text # Store the text to be processed
        self.on_finished = on_finished
        self.on_error = on_error

    # The run method is the main function called when the thread starts.
    # It contains the logic for AI note generation, saving, and linking.
    def run(self):
        # Create a new DatabaseManager instance for this thread.
        # Each thread should have its own database connection.
        db_manager_worker = database_manager.DatabaseManager(init_tables=False)
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
                now = datetime.now().isoformat()

                # Stage 1: Identify new vs existing notes
                for note_data in generated_notes:
                    title = note_data.get('title', 'Untitled Note')
                    sanitized_title = note_manager.get_sanitized_title(f"# {title}")
                    
                    if sanitized_title in title_to_id:
                        # Note exists, we will append to it later
                        note_data['_final_id'] = title_to_id[sanitized_title]
                        note_data['_is_new'] = False
                    else:
                        # Note is new, assign a new UUID
                        new_id = str(uuid4())
                        note_data['_final_id'] = new_id
                        note_data['_is_new'] = True
                        title_to_id[sanitized_title] = new_id # Update mapping for links

                # Stage 2: Process insertions and updates
                for note_data in generated_notes:
                    final_id = note_data['_final_id']
                    title = note_data.get('title', 'Untitled Note')
                    content = note_data.get('content', '')
                    category = note_data.get('general_title', 'AI Generated')
                    
                    full_content = f"# {title}\n\n{content}"
                    sanitized_title = note_manager.get_sanitized_title(full_content)

                    if note_data['_is_new']:
                        # Insert new note
                        notes_to_insert.append(
                            (final_id, sanitized_title, full_content, category, now, now)
                        )
                    else:
                        # Append to existing note
                        existing_note = db_manager_worker.get_note(final_id)
                        if existing_note:
                            existing_content = existing_note[2]
                            updated_content = existing_content + f"\n\n## AI Eklemeleri\n\n{content}"
                            db_manager_worker.update_note(final_id, sanitized_title, updated_content, category)

                    connections = note_data.get('connections', [])
                    for target_title_raw in connections:
                        sanitized_target_title = note_manager.get_sanitized_title(target_title_raw)
                        target_id = title_to_id.get(sanitized_target_title)
                        if target_id:
                            links_to_insert.append((final_id, target_id))
                        else:
                            log_debug(f"DEBUG: Could not find target_id for '{sanitized_target_title}'. Link not inserted.")

                # Stage 3: Bulk database operations for new notes
                if notes_to_insert:
                    db_manager_worker.bulk_insert_notes(notes_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(notes_to_insert)} notes.")

                if links_to_insert:
                    db_manager_worker.bulk_insert_links(links_to_insert)
                    log_debug(f"DEBUG: Bulk inserted {len(links_to_insert)} links.")

                if self.on_finished:
                    self.on_finished(generated_notes)
            else:
                if self.on_finished:
                    self.on_finished([])
        except ValueError as ve:
            if self.on_error:
                self.on_error(f"API Key Error: {str(ve)}")
        except Exception as e:
            if self.on_error:
                self.on_error(f"An error occurred during AI note generation: {e}")
        finally:
            if db_manager_worker:
                db_manager_worker.close_connection()

