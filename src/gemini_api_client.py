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
import traceback # For detailed error tracing
from logger import log_debug, log_error # For logging functions

load_dotenv() # Load environment variables from .env on module import


# Custom Typed Exceptions for Gemini API operations
class GeminiApiError(Exception):
    """Base exception for Gemini API client errors."""
    pass


class GeminiAuthError(GeminiApiError):
    """Exception raised for API key or authentication errors."""
    pass


class GeminiRateLimitError(GeminiApiError):
    """Exception raised when API rate limit or quota is exceeded."""
    pass


# The GeminiApiClient class communicates with the Gemini API and manages note generation requests.
class GeminiApiClient:
    # The __init__ method initializes the API client, loads the API key, and configures the model.
    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") # Get the API key from environment variables

        if not api_key: # If the API key is not found, raise typed GeminiAuthError
            raise GeminiAuthError("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable in a .env file or via the application's menu (Settings -> Enter Gemini API Key).")

        self.client = genai.Client(api_key=api_key) # Configure the Google GenAI client with the key
        self.model_name = 'gemma-4-31b-it'


    def _parse_notes_json(self, notes_json_str):
        """Helper method to parse and clean JSON output returned from Gemini API."""
        if not notes_json_str or not isinstance(notes_json_str, str):
            return []

        def normalize_result(data):
            """Normalizes parsed JSON data (list or dict) into a list of note dictionaries."""
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, dict):
                        result.append(item)
                return result
            elif isinstance(data, dict):
                # Check for standard wrapper keys
                for key in ('notes', 'zettelkasten', 'data', 'items', 'result', 'generated_notes'):
                    if key in data and isinstance(data[key], list):
                        return [item for item in data[key] if isinstance(item, dict)]
                # Check if the dict itself is a single note
                if 'title' in data or 'content' in data:
                    return [data]
                # Check if it's a dict containing note dicts as values
                dict_values = [v for v in data.values() if isinstance(v, dict)]
                if dict_values:
                    return dict_values
            return []

        # Clean control characters and null bytes from start/end
        notes_json_str_cleaned = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', notes_json_str)

        # 1. Attempt direct parsing on cleaned string
        try:
            parsed = json.loads(notes_json_str_cleaned, strict=False)
            normalized = normalize_result(parsed)
            if normalized:
                return normalized
        except json.JSONDecodeError:
            pass

        # 2. Extract JSON from Markdown code blocks (```json ... ``` or ``` ... ```)
        cleaned_str = notes_json_str_cleaned
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_str)
        for block in code_blocks:
            block_cleaned = re.sub(r'^[\s\x00-\x1f\x7f-\x9f]+|[\s\x00-\x1f\x7f-\x9f]+$', '', block)
            try:
                parsed = json.loads(block_cleaned, strict=False)
                normalized = normalize_result(parsed)
                if normalized:
                    return normalized
            except json.JSONDecodeError:
                pass
            # Try raw_decode on code block
            for start_char in ('[', '{'):
                start_idx = block_cleaned.find(start_char)
                if start_idx != -1:
                    try:
                        decoder = json.JSONDecoder()
                        parsed, _ = decoder.raw_decode(block_cleaned[start_idx:])
                        normalized = normalize_result(parsed)
                        if normalized:
                            return normalized
                    except json.JSONDecodeError:
                        pass

        # 3. Try raw_decode to ignore trailing garbage on full string
        for start_char in ('[', '{'):
            start_idx = cleaned_str.find(start_char)
            if start_idx != -1:
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(cleaned_str[start_idx:])
                    normalized = normalize_result(parsed)
                    if normalized:
                        return normalized
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
                        parsed, _ = decoder.raw_decode(fixed_str[start_idx:])
                        normalized = normalize_result(parsed)
                        if normalized:
                            return normalized
                    except json.JSONDecodeError:
                        pass
            parsed = json.loads(fixed_str, strict=False)
            normalized = normalize_result(parsed)
            if normalized:
                return normalized
        except json.JSONDecodeError:
            pass

        # 5. Fallback: search for outermost array or object brackets
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", cleaned_str)
        if array_match:
            try:
                fixed_array = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', array_match.group(0))
                parsed = json.loads(fixed_array, strict=False)
                normalized = normalize_result(parsed)
                if normalized:
                    return normalized
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r"\{\s*\"[\s\S]*\}\s*", cleaned_str)
        if obj_match:
            try:
                fixed_obj = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', obj_match.group(0))
                parsed = json.loads(fixed_obj, strict=False)
                normalized = normalize_result(parsed)
                if normalized:
                    return normalized
            except json.JSONDecodeError:
                pass

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
        Raises:
            GeminiAuthError: If authentication or API key validation fails.
            GeminiRateLimitError: If rate limit / quota is exceeded.
            GeminiApiError: If generation or response parsing fails.
        """
        if not text_content or not text_content.strip():
            return []

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
<document_content>
{text_content}
</document_content>
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except Exception as e:
            err_msg = str(e)
            log_error(f"Gemini API request error: {err_msg}\n{traceback.format_exc()}")
            lower_msg = err_msg.lower()
            if any(term in lower_msg for term in ["401", "403", "api_key_invalid", "permission_denied", "unauthorized", "api key"]):
                raise GeminiAuthError(f"Gemini API authentication failed: {err_msg}") from e
            elif any(term in lower_msg for term in ["429", "resource_exhausted", "quota_exceeded", "rate limit", "too many requests"]):
                raise GeminiRateLimitError(f"Gemini API rate limit exceeded: {err_msg}") from e
            else:
                raise GeminiApiError(f"Gemini API error: {err_msg}") from e

        if not response or not hasattr(response, 'text') or not response.text:
            log_debug("DEBUG: Gemini API returned empty response text.")
            return []

        notes_json_str = response.text
        log_debug(f"DEBUG: Raw Gemini API response (full, len={len(notes_json_str)}): {notes_json_str}")

        parsed_notes = self._parse_notes_json(notes_json_str)
        if not parsed_notes and notes_json_str.strip():
            log_error(f"Failed to parse notes from Gemini API response: {notes_json_str[:300]}")
            raise GeminiApiError(f"Failed to parse notes from Gemini API response: {notes_json_str[:200]}")

        return parsed_notes

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

