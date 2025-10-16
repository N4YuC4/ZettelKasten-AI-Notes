# main.py
#
# This file is the main entry point for the Zettelkasten AI Notes application.
# It provides a desktop application interface using PyQt5 and includes basic functionalities
# such as note management, Markdown preview, category management, creating notes from PDF,
# and linking notes together.

import sys # For system-related functions (e.g., application exit)
import os # For file system operations (e.g., joining file paths)
from dotenv import set_key # For setting environment variables in the .env file
from logger import *
import markdown # To convert Markdown text to HTML

# Required modules for PyQt5 GUI components
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QMainWindow,
    QAction, QListWidget, QSplitter, QMessageBox, QInputDialog, QHBoxLayout,
    QComboBox, QStyle, QMenu, QProgressDialog, QDialog, QLabel, QLineEdit,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, QSettings # For Qt core types, multithreading, and settings

# Other modules of the application
import database_manager # For database operations
import note_manager # For note management functions like saving, loading, renaming, and deleting notes
import pdf_processor # To extract text from PDF files
from ai_note_generator_worker import AiNoteGeneratorWorker # To run the AI note generation process in a separate thread
from mind_map_widget import MindMapWidget # For the mind map widget

# Tkinter library for file selection (not integrated with PyQt5, opens a separate window)
import tkinter as tk
from tkinter import filedialog


# The NoteSelectionDialog class is a dialog box that allows the user to select a note from the existing notes.
# This is especially used when linking notes together.
class NoteSelectionDialog(QDialog):
    # The __init__ method initializes the dialog box.
    # db_manager: DatabaseManager object for database operations.
    # current_note_id: The ID of the current note to be linked (to prevent linking to itself).
    # parent: The parent widget of this dialog box.
    def __init__(self, db_manager, current_note_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Note to Link") # Set the window title
        self.setGeometry(200, 200, 400, 600) # Set the position and size of the window

        self.db_manager = db_manager # Store the database manager
        self.current_note_id = current_note_id # Store the ID of the current note
        self.selected_note_id = None # The ID of the selected note
        self.selected_note_title = None # The title of the selected note

        self.init_ui() # Initialize the user interface
        self.load_notes() # Load the notes

    # The init_ui method creates the user interface of the dialog box.
    def init_ui(self):
        self.main_layout = QVBoxLayout(self) # Main vertical layout

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search notes...") # Placeholder text
        self.search_input.textChanged.connect(self.load_notes) # Reload notes when the text changes
        self.main_layout.addWidget(self.search_input) # Add to the layout

        # Note list widget
        self.notes_list_widget = QListWidget()
        self.notes_list_widget.itemClicked.connect(self.on_note_selected) # Connect the event when an item is clicked
        self.main_layout.addWidget(self.notes_list_widget) # Add to the layout

        # Horizontal layout for buttons
        self.button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select") # Select button
        self.select_button.clicked.connect(self.accept) # Accept the dialog when clicked
        self.select_button.setEnabled(False) # Initially disabled
        self.button_layout.addWidget(self.select_button) # Add to the layout

        self.cancel_button = QPushButton("Cancel") # Cancel button
        self.cancel_button.clicked.connect(self.reject) # Reject the dialog when clicked
        self.button_layout.addWidget(self.cancel_button) # Add to the layout

        self.main_layout.addLayout(self.button_layout) # Add the button layout to the main layout

    # The load_notes method loads notes from the database and lists them.
    # It filters based on the text in the search box.
    def load_notes(self):
        self.notes_list_widget.clear() # Clear the current list
        search_text = self.search_input.text().lower() # Get the search text and convert to lowercase

        # Load metadata of all notes
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(self.db_manager)

        # For each note
        for note_id, title, _ in all_notes_metadata:
            if note_id == self.current_note_id: # Do not show the current note in the list (prevents linking to itself)
                continue
            if search_text in title.lower(): # If the search text is in the title
                item = QListWidgetItem(title) # Create a new list item
                item.setData(Qt.UserRole, note_id) # Store the note ID in the item's data
                self.notes_list_widget.addItem(item) # Add to the list

    # The on_note_selected method is called when a note is selected from the list.
    # item: The selected QListWidgetItem object.
    def on_note_selected(self, item):
        self.selected_note_id = item.data(Qt.UserRole) # Get the ID of the selected note
        self.selected_note_title = item.text() # Get the title of the selected note
        self.select_button.setEnabled(True) # Enable the select button


class ZettelkastenApp(QMainWindow):
    ALL_NOTES = "All Notes"
    # The __init__ method initializes the main application window.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zettelkasten AI Notes") # Set the window title
        self.setGeometry(100, 100, 1200, 800) # Set the position and size of the window

        # Create the database manager object
        self.db_manager = database_manager.DatabaseManager()
        self.current_note_id = None # The ID of the currently open note
        self.current_note_category = "" # The category of the currently open note
        # Dictionary mapping displayed titles to note IDs (for the note list)
        self.displayed_title_to_note_id = {}
        # Dictionary mapping note IDs to categories
        self.note_id_to_category = {}

        self.mind_map_widget = MindMapWidget(self.db_manager)
        self.mind_map_widget.note_selected.connect(self.open_note_by_id)

        self.init_ui() # Initialize the user interface
        self.load_notes() # Load the notes
        self._update_mind_map() # Initially load the mind map

        self.settings = QSettings("Zettelkasten", "AI Notes") # Initialize application settings

        # Load splitter positions
        if self.settings.contains("top_horizontal_splitter_state"):
            self.top_horizontal_splitter.restoreState(self.settings.value("top_horizontal_splitter_state"))
        if self.settings.contains("editor_preview_vertical_splitter_state"):
            self.editor_preview_vertical_splitter.restoreState(self.settings.value("editor_preview_vertical_splitter_state"))
        if self.settings.contains("bottom_horizontal_splitter_state"):
            self.bottom_horizontal_splitter.restoreState(self.settings.value("bottom_horizontal_splitter_state"))

        # Load and apply theme from settings
        saved_theme = self.db_manager.get_setting("UI_THEME")
        if saved_theme:
            self.apply_theme(saved_theme)
        else:
            self.apply_theme("Light") # Default theme

    def closeEvent(self, event):
        # Save splitter positions
        self.settings.setValue("top_horizontal_splitter_state", self.top_horizontal_splitter.saveState())
        self.settings.setValue("editor_preview_vertical_splitter_state", self.editor_preview_vertical_splitter.saveState())
        self.settings.setValue("bottom_horizontal_splitter_state", self.bottom_horizontal_splitter.saveState())
        super().closeEvent(event)

    def apply_theme(self, theme_name):
        stylesheet_path = os.path.join(os.path.dirname(__file__), f'{theme_name.lower()}_theme.qss')
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                QApplication.instance().setStyleSheet(f.read())
            self.db_manager.set_setting("UI_THEME", theme_name)
        else:
            QMessageBox.warning(self, "Theme Error", f"Theme file not found: {stylesheet_path}")

    # The init_ui method creates the user interface of the main window.
    def init_ui(self):
        self.central_widget = QWidget() # The central widget of the main window
        self.setCentralWidget(self.central_widget) # Set the central widget
        self.main_layout = QVBoxLayout(self.central_widget) # Main vertical layout

        # Horizontal layout for buttons
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout) # Add to the main layout

        self.note_count_label = QLabel(f"{self.current_note_category} note count: {self.db_manager.note_count(self.current_note_category)}") # Category label
        self.button_layout.addWidget(self.note_count_label) # Add to the button layout

        # Category selection box (ComboBox)
        self.category_combo_box = QComboBox()
        self.category_combo_box.addItem(self.ALL_NOTES) # Add "All Notes" option by default
        # Call the load_notes method when the selection changes
        self.category_combo_box.currentIndexChanged.connect(self.load_notes)
        self.button_layout.addWidget(self.category_combo_box) # Add to the button layout

        style = QApplication.instance().style() # Get the application style (for icons)

        # New Category Button
        self.new_category_button = QPushButton(style.standardIcon(QStyle.SP_DirOpenIcon), "New Category")
        self.new_category_button.clicked.connect(self.create_new_category) # Create a new category when clicked
        self.button_layout.addWidget(self.new_category_button) # Add to the button layout

        # Delete Category Button
        self.delete_category_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Category")
        self.delete_category_button.clicked.connect(self.delete_category) # Delete the category when clicked
        self.button_layout.addWidget(self.delete_category_button) # Add to the button layout

        # New Note Button
        self.new_button = QPushButton(style.standardIcon(QStyle.SP_FileIcon), "New Note")
        self.new_button.clicked.connect(self.new_note) # Create a new note when clicked
        self.button_layout.addWidget(self.new_button) # Add to the button layout

        # Save Note Button
        self.save_button = QPushButton(style.standardIcon(QStyle.SP_DialogSaveButton), "Save Note")
        self.save_button.clicked.connect(self.save_note) # Save the note when clicked
        self.button_layout.addWidget(self.save_button) # Add to the button layout

        # Rename Note Button
        self.rename_button = QPushButton(style.standardIcon(QStyle.SP_DialogResetButton), "Rename Note")
        self.rename_button.clicked.connect(self.rename_note) # Rename the note when clicked
        self.button_layout.addWidget(self.rename_button) # Add to the button layout

        # Delete Note Button
        self.delete_button = QPushButton(style.standardIcon(QStyle.SP_TrashIcon), "Delete Note")
        self.delete_button.clicked.connect(self.delete_note) # Delete the note when clicked
        self.button_layout.addWidget(self.delete_button) # Add to the button layout

        # Generate AI Notes from PDF Button
        self.generate_notes_button = QPushButton(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Generate AI Notes from PDF")
        self.generate_notes_button.clicked.connect(self.generate_notes_from_pdf) # Generate notes from PDF when clicked
        self.button_layout.addWidget(self.generate_notes_button) # Add to the button layout

        # Link Note Button
        self.link_note_button = QPushButton(style.standardIcon(QStyle.SP_DialogYesButton), "Link Note")
        self.link_note_button.clicked.connect(self.link_note_action) # Link the note when clicked
        self.button_layout.addWidget(self.link_note_button) # Add to the button layout

        # The main splitter has been removed, two separate horizontal splitters will be used instead.

        # Note editor (text area)
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Write your notes here...") # Placeholder text
        self.editor.textChanged.connect(self.update_preview) # Update the preview when the text changes

        # Markdown preview area
        self.preview = QTextEdit()
        self.preview.setReadOnly(True) # Make it read-only
        self.preview.setPlaceholderText("Markdown preview will appear here...") # Placeholder text

        # Note list widget
        self.notes_list_widget = QListWidget()
        self.notes_list_widget.itemClicked.connect(self.open_selected_note) # Open the note when an item is clicked
        self.notes_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # Set the custom context menu policy
        # Call the _show_note_context_menu method when a context menu is requested
        self.notes_list_widget.customContextMenuRequested.connect(self._show_note_context_menu)

        # Linked Notes List
        self.linked_notes_list_widget = QListWidget()
        self.linked_notes_list_widget.itemClicked.connect(self.open_selected_note_from_link) # Open when a linked note is clicked
        self.linked_notes_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # Set the custom context menu policy
        # Call the _show_linked_note_context_menu method when a context menu is requested
        self.linked_notes_list_widget.customContextMenuRequested.connect(self._show_linked_note_context_menu)

        # Vertical splitter for Editor and Preview
        self.editor_preview_vertical_splitter = QSplitter(Qt.Vertical)
        self.editor_preview_vertical_splitter.addWidget(self.editor)
        self.editor_preview_vertical_splitter.addWidget(self.preview)
        self.editor_preview_vertical_splitter.setSizes([400, 400]) # Initial sizes

        # Top horizontal splitter (Note List and Editor/Preview)
        self.top_horizontal_splitter = QSplitter(Qt.Horizontal)
        self.top_horizontal_splitter.addWidget(self.notes_list_widget)
        self.top_horizontal_splitter.addWidget(self.editor_preview_vertical_splitter)
        self.top_horizontal_splitter.setSizes([300, 900]) # Initial sizes
        self.main_layout.addWidget(self.top_horizontal_splitter)

        # Bottom horizontal splitter (Mind Map and Linked Notes)
        self.bottom_horizontal_splitter = QSplitter(Qt.Horizontal)
        self.bottom_horizontal_splitter.addWidget(self.mind_map_widget)
        self.bottom_horizontal_splitter.addWidget(self.linked_notes_list_widget)
        self.bottom_horizontal_splitter.setSizes([600, 600]) # Initial sizes
        self.main_layout.addWidget(self.bottom_horizontal_splitter)

        self.display_linked_notes() # Show linked notes

        self.create_menu_bar() # Create the menu bar

    # The create_menu_bar method creates the application's menu bar.
    def create_menu_bar(self):
        menubar = self.menuBar() # Get the menu bar
        settings_menu = menubar.addMenu("Settings") # Add the "Settings" menu

        # Theme menu
        theme_menu = menubar.addMenu("Theme")
        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self.apply_theme("Light"))
        theme_menu.addAction(light_theme_action)

        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self.apply_theme("Dark"))
        theme_menu.addAction(dark_theme_action)

        # Gemini API Key entry action
        gemini_api_key_action = QAction("Enter Gemini API Key", self)
        gemini_api_key_action.triggered.connect(self._show_api_key_dialog) # Show the API key dialog when clicked
        settings_menu.addAction(gemini_api_key_action) # Add the action to the menu

    # The _show_note_context_menu method shows the context menu that opens when right-clicking on items in the note list.
    # position: The position where the right-click occurred.
    def _show_note_context_menu(self, position):
        item = self.notes_list_widget.itemAt(position) # Get the item at the clicked position

        if item:
            self.notes_list_widget.setCurrentItem(item) # Make the clicked item the current item

            context_menu = QMenu(self) # Create a new context menu

            rename_action = context_menu.addAction("Rename Note") # Add the "Rename Note" action
            delete_action = context_menu.addAction("Delete Note") # Add the "Delete Note" action
            link_action = context_menu.addAction("Link to...") # Add the "Link to..." action

            # Show the menu and get the selected action
            action = context_menu.exec_(self.notes_list_widget.mapToGlobal(position))

            # Call the relevant method based on the selected action
            if action == rename_action:
                self.rename_note()
            elif action == delete_action:
                self.delete_note()
            elif action == link_action:
                self.link_note_action()

    # The _show_api_key_dialog method shows a dialog to enter the Gemini API key
    # and saves the key to the .env file.
    def _show_api_key_dialog(self):
        # Ask the user to enter the API key
        api_key, ok = QInputDialog.getText(self, "Gemini API Key", "Enter your Gemini API Key:")
        if ok and api_key: # If the user clicks OK and the key is not empty
            # Determine the path of the .env file (in the parent directory of main.py)
            dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
            set_key(dotenv_path, "GEMINI_API_KEY", api_key) # Save the key to the .env file
            QMessageBox.information(self, "Gemini API Key", "Gemini API Key saved to .env file. Please restart the application for changes to take effect.")
        elif ok and not api_key: # If the user clicks OK but the key is empty
            QMessageBox.warning(self, "Gemini API Key", "API Key cannot be empty.")

    # The link_note_action method is used to link the selected note to another note.
    def link_note_action(self):
        if not self.current_note_id: # If no current note is selected, give a warning
            QMessageBox.warning(self, "Link Note", "Please select a source note first.")
            return

        # Create and show the note selection dialog
        dialog = NoteSelectionDialog(self.db_manager, self.current_note_id, self)
        if dialog.exec_() == QDialog.Accepted: # If the user selects a note and clicks OK
            target_note_id = dialog.selected_note_id # Get the ID of the target note
            target_note_title = dialog.selected_note_title # Get the title of the target note

            # If the target note ID exists and is different from the current note
            if target_note_id and target_note_id != self.current_note_id:
                # Add the note link to the database
                success = self.db_manager.insert_note_link(self.current_note_id, target_note_id)
                if success:
                    QMessageBox.information(self, "Link Note", f"Successfully linked to '{target_note_title}'.")
                    self._update_views(category_to_select=self.current_note_category) # Update the views
                else:
                    QMessageBox.warning(self, "Link Note", f"Link to '{target_note_title}' already exists or failed to create.")
            elif target_note_id == self.current_note_id: # If trying to link a note to itself
                QMessageBox.warning(self, "Link Note", "Cannot link a note to itself.")

    # The create_new_category method creates a new note category.
    def create_new_category(self):
        # Ask the user to enter the category name
        category_name, ok = QInputDialog.getText(self, "New Category", "Enter category name:")
        if ok and category_name: # If the user clicks OK and the category name is not empty
            # Load existing note metadata and categories
            all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(self.db_manager)
            if category_name in all_categories: # If the category already exists, give a warning
                QMessageBox.warning(self, "New Category", f"Category '{category_name}' already exists.")
                return

            self.new_note() # Create a new empty note
            self.current_note_category = category_name # Set the category of the current note
            # Add default text to the editor
            self.editor.setPlainText(f"# New note in {category_name}\n\nStart writing your note here...")
            self.save_note() # Save the note
            QMessageBox.information(self, "New Category", f"Category '{category_name}' created and a new note added.")

    # The new_note method prepares for creating a new note by clearing the editor and preview.
    def new_note(self):
        self.editor.clear() # Clear the editor
        self.preview.clear() # Clear the preview
        self.current_note_id = None # Reset the current note ID
        # Get the current category from the combo box, leave it empty if it's "All Notes"
        self.current_note_category = self.category_combo_box.currentText() if self.category_combo_box.currentText() != self.ALL_NOTES else ""
        self.setWindowTitle("Zettelkasten AI Notes - New Note") # Update the window title

    # The save_note method saves or updates the content of the current note in the database.
    def save_note(self):
        note_content = self.editor.toPlainText() # Get the text from the editor
        if not note_content.strip(): # If the note content is empty, give a warning
            QMessageBox.warning(self, "Save Note", "Cannot save empty note.")
            return

        # Determine the category to save
        category_to_save = self.current_note_category if self.current_note_category != self.ALL_NOTES else ""

        # Save or update the note
        note_id, display_title = note_manager.save_note(
            self.db_manager,
            self.current_note_id,
            note_content,
            category_to_save
        )

        if note_id and display_title:  # If the note was saved successfully
            self.current_note_id = note_id  # Update the current note ID
            self.current_note_category = category_to_save  # Update the current note category
            self.setWindowTitle(f"Zettelkasten AI Notes - {display_title}")  # Update the window title
            self._update_views(category_to_select=category_to_save)  # Update the views
        else:
            QMessageBox.critical(self, "Error", "Failed to save note.")  # Show an error message if saving fails

    def _update_views(self, category_to_select=None):
        """Updates the note list, linked notes, and mind map."""
        self.load_notes(category_to_select=category_to_select)
        self.display_linked_notes()
        self._update_mind_map()

    # The rename_note method renames the selected note.
    def rename_note(self):
        selected_items = self.notes_list_widget.selectedItems() # Get the selected items
        if not selected_items: # If no item is selected, give a warning
            QMessageBox.information(self, "Rename Note", "Please select a note to rename.")
            return

        selected_display_title = selected_items[0].text() # Get the displayed title of the selected note
        # Get the note ID from the displayed title
        note_id_to_rename = self.displayed_title_to_note_id.get(selected_display_title)

        if not note_id_to_rename: # If the note ID is not found, show an error message
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        # Ask the user to enter the new title (with the current title pre-filled)
        new_title, ok = QInputDialog.getText(self, "Rename Note", "Enter new title:",
                                            text=selected_display_title)

        if ok and new_title: # If the user clicks OK and the new title is not empty
            # Rename the note
            success, new_display_title = note_manager.rename_note(
                self.db_manager,
                note_id_to_rename,
                new_title,
                self.note_id_to_category.get(note_id_to_rename, "")) # Get the category of the note

            if success: # If renaming is successful
                if self.current_note_id == note_id_to_rename: # If the renamed note is the open note
                    self.setWindowTitle(f"Zettelkasten AI Notes - {new_display_title}") # Update the window title
                # Reload the notes and select the relevant category
                self._update_views(category_to_select=self.note_id_to_category.get(note_id_to_rename, ""))
            else:
                QMessageBox.critical(self, "Error", f"Failed to rename note: {new_display_title}") # Show an error message
        elif ok and not new_title: # If the user clicks OK but the new title is empty
            QMessageBox.warning(self, "Rename Note", "New title cannot be empty.")

    # The delete_note method deletes the selected note from the database.
    def delete_note(self):
        if not self.current_note_id: # Check if a note is currently open
            QMessageBox.information(self, "Delete Note", "Please select a note to delete.")
            return

        # Get the title of the current note for the confirmation message
        note_to_delete_title = ""
        for title, note_id in self.displayed_title_to_note_id.items():
            if note_id == self.current_note_id:
                note_to_delete_title = title
                break
        
        if not note_to_delete_title:
            # Fallback if title not found (should not happen in normal operation)
            note_data = self.db_manager.get_note(self.current_note_id)
            if note_data:
                note_to_delete_title = note_data[1]
            else:
                note_to_delete_title = "the selected note"


        # Ask the user to confirm the deletion
        reply = QMessageBox.question(self, 'Delete Note', 
                                     f"Are you sure you want to delete '{note_to_delete_title}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # If the user confirms
            success = note_manager.delete_note(self.db_manager, self.current_note_id) # Delete the note
            if success:
                self.new_note() # Create a new empty note (clear the editor)
                # Reload the notes (with the current category selected)
                self._update_views(category_to_select=self.category_combo_box.currentText() if self.category_combo_box.currentText() != self.ALL_NOTES else "")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete note.") # Show an error message

    # The delete_category method deletes the selected category and all notes belonging to it.
    def delete_category(self):
        selected_category = self.category_combo_box.currentText() # Get the selected category
        if not selected_category or selected_category == self.ALL_NOTES:
            QMessageBox.information(self, "Delete Category", "Please select a valid category to delete.")
            return

        # Ask the user to confirm the deletion
        reply = QMessageBox.question(self, 'Delete Category', 
                                     f"Are you sure you want to delete category '{selected_category}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # If the user confirms
            success = self.db_manager.delete_category(selected_category) # Delete the category
            if success:
                self.new_note() # Create a new empty note (clear the editor)
                # Reload the notes (with the current category selected)
                self._update_views(category_to_select=self.ALL_NOTES)
            else:
                QMessageBox.critical(self, "Error", "Failed to delete category.") # Show an error message

    # The generate_notes_from_pdf method extracts text from a PDF file and uses this text
    # to generate notes with AI. This process is done in a separate thread.
    def generate_notes_from_pdf(self):
        self.hide() # Hide the main window (for the Tkinter file dialog)
        root = tk.Tk() # Create a Tkinter root window
        root.withdraw() # Hide the root window

        # Ask the user to select a PDF file
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf")]
        )
        root.destroy() # Destroy the Tkinter root window
        self.show() # Show the main window again

        if pdf_path: # If a PDF file was selected
            # Show the progress dialog (for PDF text extraction)
            self.loading_dialog = QProgressDialog("Extracting text from PDF... This may take a moment.", None, 0, 0, self)
            self.loading_dialog.setWindowModality(Qt.WindowModal) # Make it a modal window
            self.loading_dialog.setCancelButton(None) # Remove the cancel button
            self.loading_dialog.setWindowTitle("Generating AI Notes From PDF") # Set the title
            self.loading_dialog.show() # Show the dialog
            QApplication.processEvents() # Process GUI events

            extracted_text = pdf_processor.extract_text_from_pdf(pdf_path) # Extract text from the PDF

            if extracted_text: # If text was successfully extracted
                # Update the text of the progress dialog (for AI note generation)
                self.loading_dialog.setLabelText("Generating notes with AI... This may take longer.")
                QApplication.processEvents() # Process GUI events

                self.thread = QThread() # Create a new thread
                self.worker = AiNoteGeneratorWorker(extracted_text) # Create an AI note generator worker
                self.worker.moveToThread(self.thread) # Move the worker to the thread

                # Connect thread signals and worker slots
                self.thread.started.connect(self.worker.run) # Call the worker's run method when the thread starts
                self.worker.finished.connect(self.handle_ai_generation_finished) # Call handle_ai_generation_finished when the worker is finished
                self.worker.error.connect(self.handle_ai_generation_error) # Call handle_ai_generation_error when the worker has an error
                self.worker.finished.connect(self.thread.quit) # Terminate the thread when the worker is finished
                self.worker.error.connect(self.thread.quit) # Terminate the thread when the worker has an error
                self.worker.finished.connect(self.worker.deleteLater) # Delete the worker when it's finished
                self.worker.error.connect(self.worker.deleteLater) # Delete the worker when it has an error
                self.thread.finished.connect(self.thread.deleteLater) # Delete the thread when it's finished

                self.thread.start() # Start the thread
            else:
                QMessageBox.warning(self, "PDF Processing Failed", "Could not extract text from the selected PDF file.")
        else:
            QMessageBox.information(self, "PDF Selection Cancelled", "No PDF file selected.")

    # The load_notes method loads notes and categories from the database and updates the note list.
    # index: Specifies the category ComboBox index to select.
    # category_to_select: Specifies the category name to select.
    def load_notes(self, index=None, category_to_select=None):
        print(f"DEBUG: load_notes called with index: {index}, category_to_select: {category_to_select}")

        self.notes_list_widget.clear() # Clear the note list

        try:
            # Temporarily disconnect the currentIndexChanged signal (to prevent an infinite loop)
            self.category_combo_box.currentIndexChanged.disconnect(self.load_notes)
        except TypeError:
            pass # Don't raise an error if the signal is not already connected

        self.category_combo_box.blockSignals(True) # Block signals
        self.category_combo_box.clear() # Clear the category ComboBox
        self.category_combo_box.addItem(self.ALL_NOTES) # Add the "All Notes" option
        self.displayed_title_to_note_id = {} # Reset the title-ID mapping
        self.note_id_to_category = {} # Reset the ID-Category mapping

        # Load metadata of all notes and all categories
        all_notes_metadata, all_categories = note_manager.load_all_notes_metadata(self.db_manager)
        print(f"DEBUG: all_categories: {all_categories}")

        # Add categories to the ComboBox
        for category in sorted(list(all_categories)):
            self.category_combo_box.addItem(category)

        # Determine the category to select
        if index is not None and index >= 0 and index < self.category_combo_box.count():
            self.category_combo_box.setCurrentIndex(index)
            selected_category = self.category_combo_box.itemText(index)
        elif category_to_select:
            idx = self.category_combo_box.findText(category_to_select) # Find the category name
            print(f"DEBUG: findText result for {category_to_select}: {idx}")
            if idx != -1:
                self.category_combo_box.setCurrentIndex(idx)
                selected_category = category_to_select
            else: # If the category is not found, select "All Notes"
                self.category_combo_box.setCurrentIndex(0)
                selected_category = self.ALL_NOTES
        else: # By default, select "All Notes"
            self.category_combo_box.setCurrentIndex(0)
            selected_category = self.ALL_NOTES

        self.category_combo_box.blockSignals(False) # Re-enable signals
        self.category_combo_box.currentIndexChanged.connect(self.load_notes) # Reconnect the signal

        print(f"DEBUG: Selected category: {selected_category}")
        self.note_count_label.setText(f"{selected_category} Note count: {self.db_manager.note_count(selected_category)}")

        # Filter and add notes to the list
        for note_id, display_title, category_path in all_notes_metadata:
            if category_path:
                self.note_id_to_category[note_id] = category_path # Store the note ID to category mapping

            # Filter by selected category or "All Notes"
            if selected_category == self.ALL_NOTES or category_path == selected_category:
                self.notes_list_widget.addItem(display_title) # Add the note to the list
                self.displayed_title_to_note_id[display_title] = note_id # Store the title-ID mapping
        self._update_mind_map() # Update the mind map after loading the notes

    # The open_selected_note method loads a selected note from the note list into the editor.
    # item: The selected QListWidgetItem object.
    def open_selected_note(self, item):
        log_debug(f"DEBUG: open_selected_note called. Selected item: {item.text()}")
        selected_display_title = item.text() # Get the displayed title of the selected note
        note_id = self.displayed_title_to_note_id.get(selected_display_title) # Get the note ID from the title
        note_category = self.note_id_to_category.get(note_id, "") # Get the category of the note

        if not note_id: # If the note ID is not found, show an error message
            QMessageBox.critical(self, "Error", f"Could not find note ID for note: {selected_display_title}")
            return

        self.current_note_id = note_id # Update the current note ID
        self.current_note_category = note_category # Update the current note category
        log_debug(f"DEBUG: open_selected_note - current_note_id assigned: {self.current_note_id}")
        self._update_views(category_to_select=note_category) # Update the views

        content = note_manager.get_note_content(self.db_manager, self.current_note_id) # Get the note content from the database
        if content is not None: # If content exists
            self.editor.setPlainText(content) # Load the content into the editor
            self.setWindowTitle(f"Zettelkasten AI Notes - {selected_display_title}") # Update the window title
        else:
            QMessageBox.critical(self, "Error", f"Could not read content for note: {selected_display_title}") # Show an error message
            log_debug(f"DEBUG: open_selected_note - could not read content! Note ID: {self.current_note_id}")
            self.new_note() # Create a new empty note

    # The update_preview method converts the Markdown text in the editor to HTML
    # and displays it in the preview area.
    def update_preview(self):
        markdown_text = self.editor.toPlainText() # Get the text from the editor
        html = markdown.markdown(markdown_text) # Convert Markdown to HTML
        self.preview.setHtml(html) # Set the HTML in the preview area

    # The handle_ai_generation_finished method is called when the AI note generation process is complete.
    # generated_notes: The list of generated notes.
    def handle_ai_generation_finished(self, generated_notes):
        if self.loading_dialog: # If the loading dialog is open, close it
            self.loading_dialog.close()
            self.loading_dialog = None

        if generated_notes: # If notes were generated
            log_debug("DEBUG: handle_ai_generation_finished: AI notes processed by the worker. Refreshing the user interface.")
            QMessageBox.information(self, "AI Note Generation", "Notes were successfully generated and saved!")
            # Reload the notes and select the current category
            self._update_views(category_to_select=self.current_note_category)
        else:
            QMessageBox.warning(self, "AI Note Generation", "No notes were generated by the AI.")

    # The handle_ai_generation_error method is called when an error occurs during AI note generation.
    # message: The error message.
    def handle_ai_generation_error(self, message):
        if self.loading_dialog: # If the loading dialog is open, close it
            self.loading_dialog.close()
            self.loading_dialog = None
        QMessageBox.critical(self, "AI Note Generation Error", f"An error occurred during AI note generation: {message}")

    def open_note_by_id(self, note_id):
        if not note_id:
            return

        display_title = None
        for title, nid in self.displayed_title_to_note_id.items():
            if nid == note_id:
                display_title = title
                break

        if display_title:
            items = self.notes_list_widget.findItems(display_title, Qt.MatchExactly)
            if items:
                self.notes_list_widget.setCurrentItem(items[0])
                self.open_selected_note(items[0])
        else:
            QMessageBox.warning(self, "Open Note", "Could not find the selected note in the main list.")


    # The open_selected_note_from_link method is called when a note is selected from the linked notes list.
    # It finds and opens the selected linked note in the main note list.
    # item: The selected QListWidgetItem object.
    def open_selected_note_from_link(self, item):
        selected_note_id = item.data(Qt.UserRole) # Get the ID of the selected note
        self.open_note_by_id(selected_note_id)

    # The _show_linked_note_context_menu method shows the context menu that opens when right-clicking
    # on items in the linked notes list.
    # position: The position where the right-click occurred.
    def _show_linked_note_context_menu(self, position):
        item = self.linked_notes_list_widget.itemAt(position) # Get the item at the clicked position
        if item:
            self.linked_notes_list_widget.setCurrentItem(item) # Make the clicked item the current item
            context_menu = QMenu(self) # Create a new context menu
            unlink_action = context_menu.addAction("Unlink Note") # Add the "Unlink Note" action
            # Show the menu and get the selected action
            action = context_menu.exec_(self.linked_notes_list_widget.mapToGlobal(position))
            if action == unlink_action:
                linked_note_id_to_unlink = item.data(Qt.UserRole)
                if self.current_note_id and linked_note_id_to_unlink:
                    self.unlink_note(self.current_note_id, linked_note_id_to_unlink) # Call the unlink note method
                else:
                    QMessageBox.critical(self, "Error", "Could not determine notes for unlinking.")

    # The display_linked_notes method lists the notes linked to the current note.
    def display_linked_notes(self):
        self.linked_notes_list_widget.clear() # Clear the linked notes list
        if self.current_note_id: # If a current note is selected
            linked_note_ids = self.db_manager.get_note_links(self.current_note_id) # Get the linked note IDs
            if linked_note_ids: # If there are linked notes
                for linked_id in linked_note_ids:
                    note_data = self.db_manager.get_note(linked_id) # Get the data of the linked note
                    if note_data:
                        linked_title = note_data[1] # Get the note title (index 1)
                        item = QListWidgetItem(linked_title) # Create a new list item
                        item.setData(Qt.UserRole, linked_id) # Store the actual note ID in the item's data
                        self.linked_notes_list_widget.addItem(item) # Add to the list
            else:
                self.linked_notes_list_widget.clear()
                self.linked_notes_list_widget.addItem("No linked notes.") # Show a message if there are no linked notes
        else:
            self.linked_notes_list_widget.clear()
            self.linked_notes_list_widget.addItem("Select a note to see its links.") # Show a message if no note is selected

    # The unlink_note method removes the link of the selected linked note from the current note.
    def unlink_note(self, source_note_id, target_note_id):
        if not source_note_id or not target_note_id:
            QMessageBox.critical(self, "Error", "Invalid note IDs for unlinking.")
            return

        # Get the title of the note to be unlinked (only for the confirmation message)
        linked_note_title_to_unlink = self.db_manager.get_note(target_note_id)[1] if self.db_manager.get_note(target_note_id) else "Unknown Note"

        # Ask the user to confirm the unlinking
        reply = QMessageBox.question(self, 'Unlink Note', 
                                     f"Are you sure you want to unlink '{linked_note_title_to_unlink}' from the current note?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes: # If the user confirms
            # Delete the note link from the database
            success = self.db_manager.delete_note_link(source_note_id, target_note_id)
            if success:
                QMessageBox.information(self, "Unlink Note", f"Successfully unlinked '{linked_note_title_to_unlink}'.")
                self._update_views(category_to_select=self.current_note_category)
            else:
                QMessageBox.critical(self, "Error", f"Failed to unlink note: {linked_note_title_to_unlink}.")

    # The open_note_from_mind_map method is called when a note is selected from the mind map.
    def open_note_from_mind_map(self, note_id):
        self.open_note_by_id(note_id)

    # The _update_mind_map method updates the mind map with all notes and links.
    def _update_mind_map(self):
        log_debug("DEBUG: _update_mind_map called.")
        all_notes_metadata, _ = note_manager.load_all_notes_metadata(self.db_manager)
        all_links = self.db_manager.get_all_note_links()

        selected_category = self.category_combo_box.currentText()

        filtered_notes_metadata = []
        filtered_note_ids = set()
        for note_id, title, category_path in all_notes_metadata:
            if selected_category == self.ALL_NOTES or category_path == selected_category:
                filtered_notes_metadata.append((note_id, title, category_path))
                filtered_note_ids.add(note_id)

        filtered_links = []
        for source_id, target_id in all_links:
            if source_id in filtered_note_ids and target_id in filtered_note_ids:
                filtered_links.append((source_id, target_id))

        log_debug(f"DEBUG: _update_mind_map - Number of notes: {len(filtered_notes_metadata)}, Number of links: {len(filtered_links)}, Current note ID: {self.current_note_id}")
        self.mind_map_widget.update_map(filtered_notes_metadata, filtered_links, self.current_note_id)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = ZettelkastenApp()
    main_app.show()
    sys.exit(app.exec_())