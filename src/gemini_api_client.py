import google.generativeai as genai
from dotenv import load_dotenv
import os

class GeminiApiClient:
    def __init__(self, api_key=None):
        # Construct absolute path to .env file in the project root
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        load_dotenv(dotenv_path=env_path)

        # If an API key is not passed directly, try to get it from the environment.
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        # Raise an error if the API key is still not found.
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it as an environment variable or pass it to the constructor.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_zettelkasten_notes(self, text_content):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Args:
            text_content (str): The text content from which to generate notes.
        Returns:
            list: A list of generated notes, where each note is a dictionary
                  with 'title' and 'content' keys.
        """
        prompt = f"""You are an AI assistant specialized in generating Zettelkasten-style notes.
From the following text, extract key concepts, arguments, and insights.
For each insight, create a concise Zettelkasten note. Each note should be self-contained and atomic.

Format your output as a JSON array of objects, where each object has a  'title', 'content' and 'connections' key.

each 'connections' key should contain a list of titles of related notes (can be empty if no connections) but make sure to include all relevant connections.

Example:
[
  {{"title": "Concept of Zettelkasten", "content": "Zettelkasten is a personal knowledge management and note-taking method used in research and study. It consists of individual notes with unique IDs, interconnected by links.", "connections": []}},
  {{"title": "Atomic Notes Principle", "content": "Each Zettelkasten note should contain only one idea or concept to ensure atomicity and reusability.", "connections": []}}
]

Text to process:
{text_content}
"""
        try:
            response = self.model.generate_content(prompt)
            notes_json_str = response.text
            # Strip markdown code block delimiters if present
            if notes_json_str.startswith('```json') and notes_json_str.endswith('```'):
                notes_json_str = notes_json_str[len('```json\n'):-len('\n```')]
            print(f"DEBUG: Raw Gemini API response (stripped): {notes_json_str}")
            import json
            notes = json.loads(notes_json_str)
            return notes
        except Exception as e:
            print(f"Error generating notes with Gemini API: {e}")
            return []

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # You need to set your GEMINI_API_KEY environment variable
    # For example: export GEMINI_API_KEY="YOUR_API_KEY"
    # client = GeminiApiClient()
    # dummy_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    # generated_notes = client.generate_zettelkasten_notes(dummy_text)
    # if generated_notes:
    #     print("Generated Notes:")
    #     for note in generated_notes:
    #         print(f"Title: {note['title']}")
    #         print(f"Content: {note['content']}")
    #         print("---")
    pass
