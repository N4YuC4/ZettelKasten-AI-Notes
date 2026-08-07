# gemini_api_client.py
#
# This file contains the GeminiApiClient class, which interacts with the Google Gemini API
# to generate Zettelkasten-style notes. It extracts key concepts, arguments, and insights
# from the given text content and transforms them into structured JSON format notes.

from google import genai # Modern Google GenAI API library
from dotenv import load_dotenv # To load environment variables from a .env file
import os # For accessing environment variables
import json # To process JSON data
import re # For regex operations
from logger import log_debug # For debug logging function

# The GeminiApiClient class communicates with the Gemini API and manages note generation requests.
class GeminiApiClient:
    # The __init__ method initializes the API client, loads the API key, and configures the model.
    def __init__(self):
        load_dotenv() # Load environment variables from the .env file
        api_key = os.getenv("GEMINI_API_KEY") # Get the API key from environment variables

        if not api_key: # If the API key is not found, raise an error
            raise ValueError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable in a .env file or via the application's menu (Settings -> Enter Gemini API Key).")

        self.client = genai.Client(api_key=api_key) # Configure the Google GenAI client with the key
        self.model_name = 'gemma-4-31b-it'

    def _parse_notes_json(self, notes_json_str):
        """Helper method to parse and clean JSON output returned from Gemini API."""
        if not notes_json_str:
            return []

        # Clean control characters and null bytes from start/end
        notes_json_str_cleaned = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', notes_json_str)

        try:
            # 1. Attempt direct parsing on cleaned string
            return json.loads(notes_json_str_cleaned, strict=False)
        except json.JSONDecodeError:
            pass

        # 2. Extract JSON from Markdown code blocks
        cleaned_str = notes_json_str_cleaned
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_str)
        if match:
            cleaned_str = match.group(1).strip()
        elif cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str.replace("```json", "", 1).strip()
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3].strip()

        cleaned_str = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', cleaned_str)

        # 3. Try raw_decode to ignore trailing garbage
        for start_char in ('[', '{'):
            start_idx = cleaned_str.find(start_char)
            if start_idx != -1:
                try:
                    decoder = json.JSONDecoder()
                    notes, _ = decoder.raw_decode(cleaned_str[start_idx:])
                    return notes
                except json.JSONDecodeError:
                    pass

        # 4. Try fixing unescaped backslashes
        try:
            fixed_str = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', cleaned_str)
            for start_char in ('[', '{'):
                start_idx = fixed_str.find(start_char)
                if start_idx != -1:
                    try:
                        decoder = json.JSONDecoder()
                        notes, _ = decoder.raw_decode(fixed_str[start_idx:])
                        return notes
                    except json.JSONDecodeError:
                        pass
            return json.loads(fixed_str, strict=False)
        except json.JSONDecodeError:
            pass

        # 5. Last resort: find outermost array brackets
        array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned_str, re.DOTALL)
        if array_match:
            fixed_array = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', array_match.group(0))
            return json.loads(fixed_array, strict=False)

        return []

    # The generate_zettelkasten_notes method generates Zettelkasten-style notes from the given text content.
    def generate_zettelkasten_notes(self, text_content):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Args:
            text_content (str): The text content from which to generate notes.
        Returns:
            list: A list of generated notes, where each note is a dictionary
                  with 'general_title', 'title', 'content' and 'connections' keys.
        """
        try:
            prompt = f"""You are an AI assistant specialized in generating Zettelkasten-style notes.
Extract key concepts, arguments, and insights from the text below.
Create a set of concise, self-contained, and atomic Zettelkasten notes.
Notes must be in the same language as the input text. Make sure markdown formatting is correct.

Each note in the JSON array must have a 'general_title', 'title', 'content', and 'connections' list.
The 'general_title' should represent the overall topic of the document.
The 'connections' list should contain the exact titles of other related notes in this generated set. If a note has no clear relations, the connections list should be empty.

Example structure:
[
  {{"general_title": "Zettelkasten Method", "title": "Concept of Zettelkasten", "content": "Zettelkasten is a personal knowledge management and note-taking method used in research and study. It consists of individual notes with unique IDs, interconnected by links.", "connections": []}},
  {{"general_title": "Zettelkasten Method", "title": "Atomic Notes Principle", "content": "Each Zettelkasten note should contain only one idea or concept to ensure atomicity and reusability.", "connections": []}}
]

Text to process:
{text_content}
"""
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            notes_json_str = response.text
            log_debug(f"DEBUG: Raw Gemini API response (full, len={len(notes_json_str)}): {notes_json_str}")
            return self._parse_notes_json(notes_json_str)
        except Exception as e:
            log_debug(f"Error generating notes with Gemini API: {e}")
            return []

# This block provides an example usage when the file is run directly (for testing purposes).
if __name__ == '__main__':
    # Example usage (for testing purposes)
    # You need to set the GEMINI_API_KEY environment variable.
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
