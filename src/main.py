import sys
import os
import re
import markdown
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QMainWindow, QAction, QFileDialog, QListWidget, QSplitter, QMessageBox, QInputDialog, QHBoxLayout, QComboBox, QStyle # Added QStyle 
from PyQt5.QtCore import Qt
import database_manager
import pdf_processor
from gemini_api_client import GeminiApiClient

import tkinter as tk
from tkinter import filedialog

import tkinter as tk
from tkinter import filedialog

# ... (rest of your imports)

class ZettelkastenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zettelkasten AI Notes")
        self.setGeometry(100, 100, 1200, 800)

        self.db_manager = database_manager.DatabaseManager()
        self.current_note_id = None
        self.current_note_category = "" # To keep track of the current note's category
        self.displayed_title_to_note_id = {}
        self.note_id_to_category = {}

        self.init_ui()
        self.load_notes()

    def init_ui(self):
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Button Layout
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # Category Selection
        self.category_combo_box = QComboBox()
        self.category_combo_box.addItem("All Notes") # Default option
        self.category_combo_box.currentIndexChanged.connect(self.load_notes) # Reload notes when category changes
        self.button_layout.addWidget(self.category_combo_box)

        # Get standard icons
        style = QApplication.instance().style()

        self.new_category_button = QPushButton(style.standardIcon(QStyle.SP_DirOpenIcon), "New Category") # Using SP_DirOpenIcon for new category
        self.new_category_button.clicked.connect(self.create_new_category)
        self.button_layout.addWidget(self.new_category_button)

        self.delete_category_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Category")
        self.delete_category_button.clicked.connect(self.delete_category)
        self.button_layout.addWidget(self.delete_category_button)

        # Buttons
        self.new_button = QPushButton(style.standardIcon(QStyle.SP_FileIcon), "New Note") # Using SP_FileIcon for new note
        self.new_button.clicked.connect(self.new_note)
        self.button_layout.addWidget(self.new_button)

        self.save_button = QPushButton(style.standardIcon(QStyle.SP_DialogSaveButton), "Save Note")
        self.save_button.clicked.connect(self.save_note)
        self.button_layout.addWidget(self.save_button)

        self.rename_button = QPushButton(style.standardIcon(QStyle.SP_DialogResetButton), "Rename Note") # Using SP_DialogResetButton for rename (no direct rename icon)
        self.rename_button.clicked.connect(self.rename_note)
        self.button_layout.addWidget(self.rename_button)

        self.delete_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Note")
        self.delete_button.clicked.connect(self.delete_note)
        self.button_layout.addWidget(self.delete_button)

        self.generate_notes_button = QPushButton(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Generate Notes from PDF")
        self.generate_notes_button.clicked.connect(self.generate_notes_from_pdf)
        self.button_layout.addWidget(self.generate_notes_button)

        # Splitter for notes list and editor/preview
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Notes List (Left Pane)
        self.notes_list_widget = QListWidget()
        self.notes_list_widget.itemClicked.connect(self.open_selected_note)
        self.splitter.addWidget(self.notes_list_widget)

        # Editor/Preview Splitter (Right Pane)
        self.editor_preview_splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.editor_preview_splitter)

        # Editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Notlarınızı buraya yazın...")
        self.editor.textChanged.connect(self.update_preview) # Connect textChanged signal
        self.editor_preview_splitter.addWidget(self.editor)

        # Preview (Placeholder for now)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Markdown önizlemesi burada görünecek...")
        self.editor_preview_splitter.addWidget(self.preview)

    def create_new_category(self):
        category_name, ok = QInputDialog.getText(self, "New Category", "Enter category name:")
        if ok and category_name:
            # Check if category already exists
            all_notes_metadata, all_categories = self.db_manager.get_all_notes_metadata()
            if category_name in all_categories:
                QMessageBox.warning(self, "New Category", f"Category '{category_name}' already exists.")
                return

            # Create a new note with the new category to make it appear in the dropdown
            self.new_note()
            self.current_note_category = category_name
            self.editor.setPlainText(f"# New note in {category_name}\n\nStart writing your note here...")
            self.save_note()
            QMessageBox.warning(self, "New Category", f"Category '{category_name}' created and a new note added.")
            self.load_notes(category_to_select=category_name) # Pass the new category name
            # self.category_combo_box.setCurrentText(category_name) # This line is no longer needed here

    def new_note(self):
        self.editor.clear()
        self.preview.clear()
        self.current_note_id = None
        self.current_note_category = self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else ""
        self.setWindowTitle("Zettelkasten AI Notes - New Note")

    def save_note(self):
        note_content = self.editor.toPlainText()
        if not note_content.strip():
            QMessageBox.warning(self, "Save Note", "Cannot save empty note.")
            return

        # Use self.current_note_category instead of self.category_combo_box.currentText()
        category_to_save = self.current_note_category if self.current_note_category != "All Notes" else ""

        note_id, display_title = self.db_manager.save_note(
            self.current_note_id, 
            note_content, 
            category_to_save
        )

        if note_id and display_title:
            self.current_note_id = note_id
            self.current_note_category = category_to_save
            self.setWindowTitle(f"Zettelkasten AI Notes - {display_title}")
            self.load_notes(category_to_select=category_to_save)
        else:
            QMessageBox.critical(self, "Error", "Failed to save note.")

    def rename_note(self):
        selected_items = self.notes_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Rename Note", "Please select a note to rename.")
            return

        selected_display_title = selected_items[0].text()
        note_id_to_rename = self.displayed_title_to_note_id.get(selected_display_title)

        if not note_id_to_rename:
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        new_title, ok = QInputDialog.getText(self, "Rename Note", "Enter new title:", 
                                            text=selected_display_title)

        if ok and new_title:
            success, new_display_title = self.db_manager.rename_note(
                note_id_to_rename, 
                new_title, 
                self.note_id_to_category.get(note_id_to_rename, "")
            )

            if success:
                # If the renamed note was the currently open one, update its state
                if self.current_note_id == note_id_to_rename:
                    self.setWindowTitle(f"Zettelkasten AI Notes - {new_display_title}")
                self.load_notes(category_to_select=self.note_id_to_category.get(note_id_to_rename, "")) # Pass the category of the renamed note
            else:
                QMessageBox.critical(self, "Error", f"Failed to rename note: {new_display_title}")
        elif ok and not new_title:
            QMessageBox.warning(self, "Rename Note", "New title cannot be empty.")

    def delete_note(self):
        selected_items = self.notes_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Delete Note", "Please select a note to delete.")
            return

        selected_display_title = selected_items[0].text()
        note_id_to_delete = self.displayed_title_to_note_id.get(selected_display_title)

        if not note_id_to_delete:
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        reply = QMessageBox.question(self, 'Delete Note', 
                                     f"Are you sure you want to delete '{selected_display_title}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            success = self.db_manager.delete_note(note_id_to_delete)
            if success:
                self.new_note() # Clear editor and preview after deletion
                self.load_notes(category_to_select=self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else "") # Reload notes to update the list, try to keep current category
            else:
                QMessageBox.critical(self, "Error", "Failed to delete note.")

    def delete_category(self):
        selected_category = self.category_combo_box.currentText()
        if not selected_category:
            QMessageBox.information(self, "Delete Category", "Please select a category to delete.")
            return

        selected_category

        if not selected_category :
            QMessageBox.critical(self, "Error", f"Could not find category for note: {selected_category}")
            return
        elif selected_category == "All Notes":
            QMessageBox.information(self, "Delete Category", "Cannot delete 'All Notes' category.")
            return

        reply = QMessageBox.question(self, 'Delete Category',
                                     f"Are you sure you want to delete category '{selected_category}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            success = self.db_manager.delete_category(selected_category)
            if success:
                self.new_note()  # Clear editor and preview after deletion
                self.load_notes(category_to_select=self.category_combo_box.currentText() if self.category_combo_box.currentText() != "All Notes" else "")  # Reload notes to update the list, try to keep current category
            else:
                QMessageBox.critical(self, "Error", "Failed to delete category.")




    def generate_notes_from_pdf(self):
        # Hide the main window temporarily to prevent it from interfering with Tkinter's dialog
        self.hide()
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window

        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf")]
        )
        root.destroy() # Destroy the Tkinter root window
        self.show() # Show the main window again

        if pdf_path:
            QMessageBox.information(self, "Processing PDF", "Extracting text from PDF... This may take a moment.")
            extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)

            if extracted_text:
                QMessageBox.information(self, "Processing PDF", "Text extracted. Generating notes with AI... This may take longer.")
                try:
                    gemini_client = GeminiApiClient()
                    generated_notes = gemini_client.generate_zettelkasten_notes(extracted_text)

                    if generated_notes:
                        notes_saved_count = 0
                        for note in generated_notes:
                            title = note.get('title', 'Untitled Note')
                            content = note.get('content', '')
                            if title and content:
                                # Save each generated note as a new note
                                self.db_manager.save_note(None, f"# {title}\n\n{content}", "AI Generated") # Assign to a new category
                                notes_saved_count += 1
                        QMessageBox.information(self, "Notes Generated", f"Successfully generated and saved {notes_saved_count} notes from PDF.")
                        self.load_notes(category_to_select="AI Generated") # Reload notes to show the new ones
                    else:
                        QMessageBox.warning(self, "Notes Generation Failed", "AI did not generate any notes from the PDF content.")
                except ValueError as ve:
                    QMessageBox.critical(self, "API Key Error", str(ve))
                except Exception as e:
                    QMessageBox.critical(self, "AI Generation Error", f"An error occurred during AI note generation: {e}")
            else:
                QMessageBox.warning(self, "PDF Processing Failed", "Could not extract text from the selected PDF file.")
        else:
            QMessageBox.information(self, "PDF Selection Cancelled", "No PDF file selected.")

    def load_notes(self, index=None, category_to_select=None): # Add category_to_select
        print(f"DEBUG: load_notes called with index: {index}, category_to_select: {category_to_select}")

        self.notes_list_widget.clear()

        # Disconnect to prevent recursive calls during clear and repopulate
        try:
            self.category_combo_box.currentIndexChanged.disconnect(self.load_notes)
        except TypeError:
            pass # Signal was not connected, which is fine for the first call or if already disconnected

        self.category_combo_box.blockSignals(True) # Block signals to prevent recursion during clear and add
        self.category_combo_box.clear()
        self.category_combo_box.addItem("All Notes")
        self.displayed_title_to_note_id = {}
        self.note_id_to_category = {}

        all_notes_metadata, all_categories = self.db_manager.get_all_notes_metadata()
        print(f"DEBUG: all_categories: {all_categories}") # Added debug print

        # Populate categories in the combo box
        for category in sorted(list(all_categories)):
            self.category_combo_box.addItem(category)

        # Determine the category to display
        if index is not None and index >= 0 and index < self.category_combo_box.count():
            # If called by a signal, use the index provided by the signal
            self.category_combo_box.setCurrentIndex(index)
            selected_category = self.category_combo_box.itemText(index)
        elif category_to_select: # New: If a category is explicitly passed
            idx = self.category_combo_box.findText(category_to_select)
            print(f"DEBUG: findText result for {category_to_select}: {idx}") # Added debug print
            if idx != -1:
                self.category_combo_box.setCurrentIndex(idx)
                selected_category = category_to_select
            else: # Fallback if category_to_select not found (shouldn't happen if it was just created)
                self.category_combo_box.setCurrentIndex(0)
                selected_category = "All Notes"
        else:
            # Default to "All Notes" if not called by a signal or no category_to_select
            self.category_combo_box.setCurrentIndex(0)
            selected_category = "All Notes"

        self.category_combo_box.blockSignals(False) # Unblock signals
        self.category_combo_box.currentIndexChanged.connect(self.load_notes) # Reconnect after blocking

        print(f"DEBUG: Selected category: {selected_category}")
        
        for note_id, display_title, category_path in all_notes_metadata:
            if category_path:
                self.note_id_to_category[note_id] = category_path

            if selected_category == "All Notes" or category_path == selected_category:
                self.notes_list_widget.addItem(display_title)
                self.displayed_title_to_note_id[display_title] = note_id

    def open_selected_note(self, item):
        selected_display_title = item.text()
        note_id = self.displayed_title_to_note_id.get(selected_display_title)
        note_category = self.note_id_to_category.get(note_id, "")

        if not note_id:
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        self.current_note_id = note_id
        self.current_note_category = note_category
        
        content = self.db_manager.read_note_content(self.current_note_id)
        if content is not None:
            self.editor.setPlainText(content)
            self.setWindowTitle(f"Zettelkasten AI Notes - {selected_display_title}")
        else:
            QMessageBox.critical(self, "Error", f"Could not read content for note: {selected_display_title}")
            self.new_note() # Clear editor if file not found

    def update_preview(self):
        markdown_text = self.editor.toPlainText()
        html = markdown.markdown(markdown_text)
        self.preview.setHtml(html)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyle('Fusion') # Apply Fusion style for a more modern look

    # Load and apply stylesheet
    stylesheet_path = os.path.join(os.path.dirname(__file__), 'style.qss')
    if os.path.exists(stylesheet_path):
        with open(stylesheet_path, "r") as f:
            app.setStyleSheet(f.read())

    ex = ZettelkastenApp()
    ex.show()
    sys.exit(app.exec_())
