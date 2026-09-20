# gemini_api_client.py
#
# This file contains the GeminiApiClient class, which interacts with the Google Gemini API
# to generate Zettelkasten-style notes. It extracts key concepts, arguments, and insights
# from the given text content and transforms them into structured JSON format notes.

from google import genai # Modern Google GenAI API library
import os # For accessing environment variables
import json # To process JSON data
import re # For regex operations
import traceback # For detailed error tracing
from logger import log_debug, log_error # For logging functions


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


import time
from typing import Optional, Callable, List, Dict, Any, Tuple, Set
from ai_response_parser import AiResponseParser
from prompt_templates import build_note_extraction_prompt, build_graph_linking_prompt
from ai_provider import BaseAiProvider
import semantic_chunker

# The GeminiApiClient class communicates with the Gemini API and manages note generation requests.
class GeminiApiClient(BaseAiProvider):
    # The __init__ method initializes the API client, loads the API key, and configures the model.
    def __init__(self, api_key=None, db_manager=None, settings_manager=None):
        if not api_key:
            if settings_manager is not None:
                api_key = settings_manager.get_setting("GEMINI_API_KEY")
            elif db_manager is not None:
                api_key = db_manager.get_setting("GEMINI_API_KEY")
            else:
                try:
                    from settings_manager import SettingsManager
                    api_key = SettingsManager.get_instance().get_setting("GEMINI_API_KEY")
                except Exception:
                    pass
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise GeminiAuthError("Gemini API Key not found. Please enter your Gemini API Key in Settings.")

        self.client = genai.Client(api_key=api_key) # Configure the Google GenAI client with the key
        self.model_name = 'gemma-4-31b-it'

    def _parse_notes_json(self, notes_json_str):
        """Helper method to parse and clean JSON output returned from Gemini API."""
        return AiResponseParser.parse_notes_json(notes_json_str)

    def _execute_inference(
        self,
        chunk_text: str,
        previous_notes_json: Optional[str] = None,
        existing_titles: Optional[List[str]] = None,
        unified_general_title: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Executes a single Gemini API completion on chunk_text with optional chained context and 429 backoff."""
        prompt = build_note_extraction_prompt(
            chunk_text=chunk_text,
            previous_notes_json=previous_notes_json,
            existing_titles=existing_titles,
            unified_general_title=unified_general_title,
            custom_system_prompt=custom_system_prompt
        )

        max_retries = 3
        backoff_delay = 2.0
        response = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                break
            except Exception as e:
                err_msg = str(e)
                log_error(f"Gemini API request error (attempt {attempt + 1}/{max_retries}): {err_msg}\n{traceback.format_exc()}")
                lower_msg = err_msg.lower()
                if any(term in lower_msg for term in ["401", "403", "api_key_invalid", "permission_denied", "unauthorized", "api key"]):
                    raise GeminiAuthError(f"Gemini API authentication failed: {err_msg}") from e
                elif any(term in lower_msg for term in ["429", "resource_exhausted", "quota_exceeded", "rate limit", "too many requests"]):
                    if attempt < max_retries - 1:
                        log_debug(f"Gemini rate limit hit, backing off for {backoff_delay}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    raise GeminiRateLimitError(f"Gemini API rate limit exceeded: {err_msg}") from e
                else:
                    raise GeminiApiError(f"Gemini API error: {err_msg}") from e

        if not response or not hasattr(response, 'text') or not response.text:
            log_debug("DEBUG: Gemini API returned empty response text.")
            return []

        notes_json_str = response.text
        log_debug(f"DEBUG: Raw Gemini API response (len={len(notes_json_str)}): {notes_json_str[:300]}")

        parsed_notes = self._parse_notes_json(notes_json_str)
        if not parsed_notes and notes_json_str.strip():
            log_error(f"Failed to parse notes from Gemini API response: {notes_json_str[:300]}")
            raise GeminiApiError(f"Failed to parse notes from Gemini API response: {notes_json_str[:200]}")

        return parsed_notes

    # The generate_zettelkasten_notes method generates Zettelkasten-style notes from the given text content.
    def generate_zettelkasten_notes(
        self,
        text_content,
        on_progress: Optional[Callable[[str], None]] = None,
        custom_system_prompt: Optional[str] = None
    ):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Uses universal dynamic semantic chunking with chained context for large documents.
        """
        if not text_content or not str(text_content).strip():
            return []

        # Sanitize prompt delimiter tags to prevent prompt injection breakouts
        sanitized_text = (
            str(text_content)
            .replace("<document_content>", "")
            .replace("</document_content>", "")
            .strip()
        )
        if not sanitized_text:
            return []

        # Safe high ceiling to protect against corrupt/infinite memory consumption
        MAX_SAFE_CHARS = 2_000_000
        if len(sanitized_text) > MAX_SAFE_CHARS:
            sanitized_text = sanitized_text[:MAX_SAFE_CHARS]

        total_tokens = semantic_chunker.estimate_tokens(sanitized_text)
        max_chunk_tokens = semantic_chunker.DEFAULT_EXTRACTION_CHUNK_TOKENS

        # 1. Single pass if within chunk budget
        if total_tokens <= max_chunk_tokens:
            if on_progress:
                on_progress("AI is extracting notes...")
            return self._execute_inference(sanitized_text, custom_system_prompt=custom_system_prompt)

        # 2. Document exceeds budget -> dynamic semantic chunking with Chained JSON Context
        log_debug(
            f"Document size (~{total_tokens} tokens) exceeds single-pass budget ({max_chunk_tokens} tokens). "
            "Splitting document into semantic chunks with chained context for Gemini..."
        )
        chunks = semantic_chunker.chunk_text(
            sanitized_text,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=semantic_chunker.DEFAULT_OVERLAP_TOKENS
        )
        log_debug(f"Document split into {len(chunks)} chunks for sequential chained Gemini processing.")

        all_notes: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        all_seen_titles_list: List[str] = []
        unified_general_title: Optional[str] = None
        previous_chunk_json: Optional[str] = None

        for idx, chunk in enumerate(chunks):
            progress_msg = f"AI is extracting notes (Part {idx + 1}/{len(chunks)})..."
            log_debug(f"{progress_msg} ({len(chunk)} chars)")
            if on_progress:
                on_progress(progress_msg)
            try:
                chunk_notes = self._execute_inference(
                    chunk_text=chunk,
                    previous_notes_json=previous_chunk_json,
                    existing_titles=all_seen_titles_list if all_seen_titles_list else None,
                    unified_general_title=unified_general_title,
                    custom_system_prompt=custom_system_prompt,
                )
                new_chunk_notes: List[Dict[str, Any]] = []
                for note in chunk_notes:
                    if not unified_general_title and note.get("general_title"):
                        unified_general_title = note["general_title"].strip()

                    title = note.get("title", "").strip()
                    title_key = title.lower()
                    if title_key and title_key not in seen_titles:
                        seen_titles.add(title_key)
                        all_seen_titles_list.append(title)
                        all_notes.append(note)
                        new_chunk_notes.append(note)
                    elif not title_key:
                        all_notes.append(note)
                        new_chunk_notes.append(note)

                # Prepare previous_chunk_json for subsequent chunk (pure concept reference: id, title, brief content)
                if new_chunk_notes:
                    ref_notes = [
                        {
                            "id": n.get("id", i + 1),
                            "title": n.get("title", ""),
                            "content": (n.get("content", "")[:250] + "...") if len(n.get("content", "")) > 250 else n.get("content", "")
                        }
                        for i, n in enumerate(new_chunk_notes)
                        if n.get("title")
                    ]
                    clean_export = {
                        "general_title": unified_general_title or "",
                        "notes": ref_notes
                    }
                    previous_chunk_json = json.dumps(clean_export, ensure_ascii=False, indent=2)

                # Gentle inter-chunk pacing to respect RPM quotas
                if idx < len(chunks) - 1:
                    time.sleep(1.0)

            except Exception as ce:
                log_error(f"Error processing chunk {idx + 1}/{len(chunks)} in Gemini: {ce}")
                if not all_notes and idx == len(chunks) - 1:
                    raise

        # Guarantee sequential global 1-based unique IDs across all chunks
        for idx, note in enumerate(all_notes):
            note["id"] = idx + 1
            if unified_general_title:
                note["general_title"] = unified_general_title

        return all_notes

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 of the Two-Stage Pipeline: Global Knowledge Graph Linking using Gemini API.
        Takes the complete set of notes generated in Stage 1 with full titles and contents.
        Queries Gemini with the numbered notes map to discover genuine conceptual connections.
        Attaches discovered connections directly to each note's 'connections' list using AiResponseParser.
        """
        if not notes or len(notes) <= 1:
            return notes

        log_debug(f"Starting Stage 2: Gemini Global Knowledge Graph Linking for {len(notes)} notes...")
        if on_progress:
            on_progress("Analyzing conceptual links between notes...")

        # Prepare numbered notes export for prompt input
        full_notes_payload = [
            {
                "id": idx + 1,
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for idx, n in enumerate(notes)
            if n.get("title")
        ]

        # Safety ceiling if massive document note payload exceeds comfortable limits
        notes_json_str = json.dumps(full_notes_payload, ensure_ascii=False, indent=2)
        if len(notes_json_str) > 100_000:
            full_notes_payload = [
                {
                    "id": idx + 1,
                    "title": n.get("title", ""),
                    "content": (n.get("content", "")[:300] + "...") if len(n.get("content", "")) > 300 else n.get("content", "")
                }
                for idx, n in enumerate(notes)
                if n.get("title")
            ]

        user_prompt = build_graph_linking_prompt(full_notes_payload)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
            )
            raw_content = response.text if response and hasattr(response, 'text') and response.text else ""
            log_debug(f"Gemini Stage 2 linking response received (len={len(raw_content)} chars): {raw_content[:400]}")
            pairs = AiResponseParser.parse_links_json(raw_content)
            log_debug(f"Discovered {len(pairs)} raw link pairs from Gemini linking pass.")

            return AiResponseParser.attach_links_to_notes(notes, pairs)

        except Exception as e:
            log_error(f"Gemini Stage 2 global linking failed (non-fatal, continuing with notes without links): {e}\n{traceback.format_exc()}")
            return notes

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

