import google.generativeai as genai
from dotenv import load_dotenv
import os
import json # Import json
from database_manager import DatabaseManager # Import DatabaseManager
from main import log_debug # Import log_debug

class GeminiApiClient:
    def __init__(self):
        load_dotenv() # Load environment variables from .env file
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable in a .env file or via the application's menu (Settings -> Enter Gemini API Key).")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_zettelkasten_notes(self, text_content):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Args:
            text_content (str): The text content from which to generate notes.
        Returns:
            list: A list of generated notes, where each note is a dictionary
                  with 'general_title', 'title', 'content' and 'connections' keys.
        """
        prompt = f"""You are an AI assistant specialized in generating Zettelkasten-style notes. all notes should be concise, focused on a single idea, and formatted in markdown. Make sure markdown formatting is perfectly correct.     
The notes language should be what documents language is.
From the following text, extract key concepts, arguments, and insights.
For each insight, create a concise Zettelkasten note. Each note should be self-contained and atomic.

Format your output as a JSON array of objects, where each object has a 'general_title', 'title', 'content' and 'connections' key.

The 'general_title' should be a single title about what all notes talking about.

Aim to create a rich network of *exceptionally relevant and semantically robust* interconnected notes. For every single connection, there MUST be a crystal-clear, undeniable conceptual link between the core ideas of the two notes. Connections should represent relationships such as:
-   **Elaboration/Detail:** One note provides essential, deeper insight or specific examples for a concept introduced in another.
-   **Causality (Cause/Effect):** One note directly causes or is a direct consequence of Note B.
-   **Dependency (Prerequisite/Follow-up):** Understanding Note A is absolutely necessary to grasp Note B, or Note B logically extends from Note A.
-   **Direct Comparison/Contrast:** Notes directly compare or highlight fundamental differences between specific concepts.
-   **Concrete Example/Direct Application:** One note provides a specific, illustrative example or a practical application of a principle in Note B.
-   **Direct Contradiction/Alternative Perspective:** Notes present opposing, mutually exclusive, or significantly different viewpoints on the SAME specific concept.
Strictly avoid connections based on mere keyword overlap, general thematic association, or weak inferential links. If a note does not have an *unquestionably strong, direct, and conceptually indispensable* connection to another note within the generated set, you MUST NOT create a connection. Prioritize the highest possible quality and precision of connections, even if it means fewer links. **If there is any doubt about the strength or directness of a connection, DO NOT create it.** Focus on creating only the most essential and undeniable links.

Each 'connections' key must contain a list of *exact titles* of other related notes that are also being generated in this JSON array. These titles must precisely match the 'title' field of the notes you generate. If a note has no *strong and direct* relevant connections within the generated set, this list can be empty.

Example:
[
  {{"general_title": "Zettelkasten Method", "title": "Concept of Zettelkasten", "content": "Zettelkasten is a personal knowledge management and note-taking method used in research and study. It consists of individual notes with unique IDs, interconnected by links.", "connections": []}},
  {{"general_title": "Zettelkasten Method", "title": "Atomic Notes Principle", "content": "Each Zettelkasten note should contain only one idea or concept to ensure atomicity and reusability.", "connections": []}}
]

Text to process:
{text_content}
"""
        try:
            response = self.model.generate_content(prompt)
            notes_json_str = response.text
            log_debug(f"DEBUG: Raw Gemini API response (full, len={len(notes_json_str)}): {notes_json_str}")
            
            try:
                notes = json.loads(notes_json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try stripping markdown code block delimiters
                if notes_json_str.startswith('```json') and notes_json_str.endswith('```'):
                    notes_json_str = notes_json_str[len('```json\n'):-len('\n```')]
                    notes = json.loads(notes_json_str)
                else:
                    raise # Re-raise if stripping doesn't help

            return notes
        except json.JSONDecodeError as jde:
            log_debug(f"Error parsing JSON from Gemini API: {jde}")
            return []
        except Exception as e:
            log_debug(f"Error generating notes with Gemini API: {e}")
            return []

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # You need to set your GEMINI_API_KEY environment variable
    # For example: export GEMINI_API_KEY="YOUR_API_KEY"
    # client = GeminiApiClient()
    # dummy_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    # generated_notes = client.generate_zettelkasten_notes(dummy_text)
    # if generated_notes:
    #     log_debug("Generated Notes:")
    #     for note in generated_notes:
    #         log_debug(f"Title: {note['title']}")
    #         log_debug(f"Content: {note['content']}")
    #         log_debug("---")
    pass
