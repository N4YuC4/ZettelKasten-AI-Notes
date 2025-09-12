from PyQt5.QtCore import QObject, pyqtSignal
from gemini_api_client import GeminiApiClient
import note_manager
import database_manager # Added import

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
                notes_saved_count = 0
                for note in generated_notes:
                    title = note.get('title', 'Untitled Note')
                    content = note.get('content', '')
                    category = note.get('general_title', 'AI Generated')
                    if title and content:
                        # Use the worker's own db_manager
                        note_manager.save_note(db_manager_worker, None, f"# {title}\n\n{content}", category)
                        notes_saved_count += 1
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
