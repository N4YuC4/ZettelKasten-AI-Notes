# Zettelkasten AI Notes

![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A desktop application built with `PyQt5` for efficient knowledge management using the Zettelkasten method. It allows users to create, edit, and link notes, supports `Markdown` for rich content, and offers a real-time preview. It stores all data in a local `SQLite` database. A key feature is its ability to use `Google Gemini AI` to generate Zettelkasten-style notes from PDF documents and automatically suggest relevant links between them. This project was developed for the Artificial Intelligence Hackathon organized by Pupilica.

## Key Features

*   **Intuitive User Interface:** A clean and responsive `GUI` for seamless note management.
*   **Note Creation and Management:** Easily create, save, rename, and delete individual notes.
*   **Markdown Support with Live Preview:** Write your notes in `Markdown` and see the rendered output in real-time.
*   **Categorization:** Organize notes into custom categories for better filtering and navigation.
*   **AI-Powered Note Generation from PDF:** Extract text from PDF documents and use `Google Gemini AI` to automatically generate Zettelkasten-style notes with suggested relevant links.
*   **Mind Map Visualization:** View your notes and their connections as an interactive mind map. This helps you visualize the relationships between your ideas and navigate your knowledge base more effectively.
*   **Smart Note Linking:** Create explicit links between notes to build a rich, interconnected graph of your knowledge.
*   **Theming:** Choose between light and dark themes to customize the application's appearance.
*   **SQLite Database:** Robust and reliable local storage for all your notes and their relationships.

## How it Works

The `main.py` file initializes the `PyQt5` application and sets up the main window (`ZettelkastenApp`). It connects to the `SQLite` database via `database_manager.py`. When a user interacts with the application (e.g., types in the editor, clicks `save`, selects a category), `ZettelkastenApp` handles the events, updates the `UI`, and calls the appropriate methods in `database_manager.py` to perform `CRUD` (Create, Read, Update, Delete) operations on the notes in the database. The first line of a note's content is automatically used as its title.

AI note generation is managed by `ai_note_generator_worker.py`, which runs in a separate `thread` to avoid freezing the `UI`. It sends the text extracted from a PDF to the `Google Gemini API` via `gemini_api_client.py` and then processes the response to create and link the notes.

The mind map visualization is managed by `mind_map_widget.py`, which uses a `force-directed graph layout` to display the notes and their connections.

## Screenshots
*Coming soon...*

## Installation

### Prerequisites
*   Python 3.8 or higher
*   `pip` (Python package installer)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Zettelkasten-AI-Notes.git
    cd Zettelkasten-AI-Notes
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key:**
    The application requires a `Google Gemini API Key` to generate AI notes.
    *   Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a file named `.env` in the root directory of the project (e.g., `Zettelkasten-AI-Notes/.env`).
    *   Add your API key to this file in the following format:
        ```
        GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    *   Alternatively, you can enter the API key directly within the application via the `Settings -> Enter Gemini API Key` menu option.

## Usage

1.  **Start the application:**
    ```bash
    python src/main.py
    ```

2.  **Basic Note Management:**
    *   **New Note:** Click the "New Note" button to clear the editor and start a new note.
    *   **Save Note:** Write your `Markdown` content in the editor. The first line will automatically become the note's title. Click the "Save Note" button to save it to the database.
    *   **Rename Note:** Select a note from the list, then click the "Rename Note" button to change its title.
    *   **Delete Note:** Select a note and click the "Delete Note" button to remove it.

3.  **Category Management:**
    *   **Create New Category:** Click "New Category", enter a name, and a new note will be created within that category.
    *   **Filter by Category:** Use the category dropdown menu to view notes belonging to a specific category or "All Notes".
    *   **Delete Category:** Select a category from the dropdown and click the "Delete Category" button. This will delete the category and all notes associated with it.

4.  **Markdown Preview:**
    As you type in the left editor pane, the right pane will show a live `Markdown` preview of your note.

5.  **AI Note Generation from PDF:**
    *   Click the "Generate AI Notes from PDF" button.
    *   Select a PDF file from your system.
    *   The application will extract the text from the PDF and send it to `Gemini AI` for note generation.
    *   The generated notes will be automatically saved and added to your note list under the "AI Generated" category or a general title provided by the AI.

6.  **Note Linking:**
    *   **Link Notes:** Select a note from the main list. Click the "Link Note" button or right-click the note and select "Link to...". A dialog will appear allowing you to search for and select another note.
    *   **View Linked Notes:** When a note is selected in the main list, its linked notes will appear in the "Linked Notes List" pane.
    *   **Open Linked Note:** Click on a linked note in the "Linked Notes List" to open it in the editor.
    *   **Unlink Note:** Right-click a note in the "Linked Notes List" and select "Unlink Note" to remove the link.

7.  **Mind Map Visualization:**
    *   The mind map at the bottom of the window displays all your notes as `nodes` and their links as `connections`.
    *   Click on a `node` in the mind map to open the corresponding note in the editor.
    *   Use the mouse wheel to `zoom in` and `zoom out` of the view, and right-click and drag to `pan`.

8.  **Theming:**
    *   Go to the "Theme" menu and select "Light Theme" or "Dark Theme" to change the application's appearance.

## Project Structure
```
Zettelkasten-AI-Notes/
├── src/
│   ├── ai_note_generator_worker.py # Manages AI note generation in a separate thread
│   ├── database_manager.py         # Manages all SQLite database operations
│   ├── gemini_api_client.py        # Interfaces with the Google Gemini API
│   ├── main.py                     # Main application entry point and GUI logic
│   ├── note_manager.py             # Manages note-related operations (save, rename, delete, sanitize)
│   ├── pdf_processor.py            # Extracts text content from PDF files
│   ├── mind_map_widget.py          # The mind map visualization widget
│   ├── dark_theme.qss              # Stylesheet for the dark theme
│   ├── light_theme.qss             # Stylesheet for the light theme
│   └── logger.py                   # A simple logger for debugging
├── .gitignore                      # Git ignore file
├── LICENSE                         # Project license
├── README.md                       # Project overview and setup instructions
└── requirements.txt                # Python dependencies
```

## Contributing

1.  `fork` the repository.
2.  Create a new `branch` (`git checkout -b feature/NewFeature`).
3.  Make your changes.
4.  `commit` your changes (`git commit -m 'Add a new feature'`).
5.  `push` to the `branch` (`git push origin feature/NewFeature`).
6.  Open a `Pull Request`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please open an `issue` on the GitHub repository.