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
from prompt_templates import (
    build_note_extraction_prompt,
    build_graph_linking_prompt,
    build_candidate_pairs_verification_prompt,
    partition_candidate_pairs,
)
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
        custom_system_prompt: Optional[str] = None,
        rag_notes: Optional[List[Dict[str, Any]]] = None,
        global_concept_map: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes a single Gemini API completion on chunk_text with optional Two-Tier RAG context and 429 backoff."""
        prompt = build_note_extraction_prompt(
            chunk_text=chunk_text,
            previous_notes_json=previous_notes_json,
            existing_titles=existing_titles,
            unified_general_title=unified_general_title,
            custom_system_prompt=custom_system_prompt,
            rag_notes=rag_notes,
            global_concept_map=global_concept_map,
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
            if AiResponseParser.is_valid_empty_notes_response(notes_json_str):
                log_debug("Gemini API explicitly returned valid empty notes array (no new concepts in chunk).")
                return []
            log_error(f"Failed to parse notes from Gemini API response: {notes_json_str[:300]}")
            raise GeminiApiError(f"Failed to parse notes from Gemini API response: {notes_json_str[:200]}")

        return parsed_notes

    # The generate_zettelkasten_notes method generates Zettelkasten-style notes from the given text content.
    def generate_zettelkasten_notes(
        self,
        text_content,
        on_progress: Optional[Callable[[str], None]] = None,
        custom_system_prompt: Optional[str] = None,
        semantic_memory_service: Optional[Any] = None,
        reranker_service: Optional[Any] = None,
    ):
        """
        Generates Zettelkasten-style notes from the given text content using the Gemini API.
        Uses universal dynamic semantic chunking with an in-memory vector RAG pool (NoteRagPool)
        with Two-Tier retrieval (Global Concept Map + Reranked Focal Notes).
        The local embedding and reranker models are strictly mandatory.
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

        # Enforce mandatory embedding service & reranker service & initialize RAG pool
        from note_rag_pool import NoteRagPool
        from semantic_memory_service import SemanticMemoryService
        from reranker_service import RerankerService

        memory_service = semantic_memory_service or SemanticMemoryService()
        rank_service = reranker_service or RerankerService()
        rag_pool = NoteRagPool(
            semantic_memory_service=memory_service,
            reranker_service=rank_service,
            ai_provider=self
        )

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

        # 2. Document exceeds budget -> dynamic semantic chunking with vector RAG pool
        log_debug(
            f"Document size (~{total_tokens} tokens) exceeds single-pass budget ({max_chunk_tokens} tokens). "
            "Splitting document into semantic chunks with vector RAG pool for Gemini..."
        )
        chunks = semantic_chunker.chunk_text(
            sanitized_text,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=semantic_chunker.DEFAULT_OVERLAP_TOKENS
        )
        log_debug(f"Document split into {len(chunks)} chunks for sequential RAG Gemini processing.")

        unified_general_title: Optional[str] = None

        for idx, chunk in enumerate(chunks):
            progress_msg = f"AI is extracting notes (Part {idx + 1}/{len(chunks)})..."
            log_debug(f"{progress_msg} ({len(chunk)} chars)")
            if on_progress:
                on_progress(progress_msg)
            try:
                # Retrieve two-tier context: Tier 1 global concept map + Tier 2 reranked focal notes
                tier1_concepts, tier2_focal_notes = rag_pool.retrieve_two_tier_context(
                    query_chunk=chunk,
                    total_budget=5200,
                    top_k_focal=8
                )
                chunk_notes = self._execute_inference(
                    chunk_text=chunk,
                    rag_notes=tier2_focal_notes if tier2_focal_notes else None,
                    global_concept_map=tier1_concepts if tier1_concepts else None,
                    existing_titles=rag_pool.get_all_titles() if len(rag_pool) > 0 and not tier1_concepts else None,
                    unified_general_title=unified_general_title,
                    custom_system_prompt=custom_system_prompt,
                )
                new_chunk_notes: List[Dict[str, Any]] = []
                for note in chunk_notes:
                    if not unified_general_title and note.get("general_title"):
                        unified_general_title = note["general_title"].strip()

                    note["_chunk_id"] = idx
                    new_chunk_notes.append(note)

                # Collect new notes into RAG pool (indexes embeddings immediately)
                if new_chunk_notes:
                    rag_pool.add_notes(new_chunk_notes)

                # Gentle inter-chunk pacing to respect RPM quotas
                if idx < len(chunks) - 1:
                    time.sleep(1.0)

            except Exception as ce:
                log_error(f"Error processing chunk {idx + 1}/{len(chunks)} in Gemini: {ce}")
                if len(rag_pool) == 0 and idx == len(chunks) - 1:
                    raise

        # Retrieve all accumulated notes from RAG pool
        all_notes = rag_pool.get_all_notes()
        for idx, note in enumerate(all_notes):
            note["id"] = idx + 1
            if unified_general_title:
                note["general_title"] = unified_general_title

        return all_notes

    def generate_note_links(
        self,
        notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None,
        similarity_threshold: Optional[float] = None,
        semantic_memory_service: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 of the Two-Stage Pipeline: Global Knowledge Graph Linking using Gemini API.
        When SemanticMemoryService (Microsoft Harrier 0.6B) is available:
        Performs Pure Semantic Vector Linking using statistical Z-score dynamic thresholding
        (threshold = max(quality_floor, mean + 1.8 * std)), establishing genuine knowledge graph
        connections in milliseconds with 0 Gemini API calls.
        Falls back to legacy direct Gemini prompt if the local embedding model is not yet installed.
        """
        if not notes or len(notes) <= 1:
            return notes

        log_debug(f"Starting Stage 2: Gemini Global Knowledge Graph Linking for {len(notes)} notes...")
        if on_progress:
            on_progress("Analyzing conceptual links between notes...")

        # 1. Attempt Pure Semantic Linking via SemanticMemoryService
        memory_service = semantic_memory_service
        if memory_service is None:
            try:
                from semantic_memory_service import SemanticMemoryService
                memory_service = SemanticMemoryService()
            except Exception as e:
                log_debug(f"SemanticMemoryService init error in Gemini client: {e}")
                memory_service = None

        if memory_service and memory_service.is_model_available():
            try:
                # Ensure notes are consolidated & deduplicated before computing knowledge graph links
                if hasattr(memory_service, "consolidate_and_deduplicate_notes"):
                    deduped = memory_service.consolidate_and_deduplicate_notes(notes, ai_provider=self)
                    if isinstance(deduped, list) and (not deduped or isinstance(deduped[0], dict)):
                        notes = deduped
                if on_progress:
                    on_progress("Computing semantic connections with Harrier embedding...")
                link_pairs, embeddings, eff_threshold = memory_service.compute_semantic_links(
                    notes, similarity_threshold=similarity_threshold, cross_chunk_only=True
                )
                if embeddings is not None and len(embeddings) == len(notes):
                    for idx, note in enumerate(notes):
                        note["_embedding"] = embeddings[idx]

                log_debug(
                    f"Gemini Stage 2 pure semantic linking discovered {len(link_pairs)} connections "
                    f"(effective threshold: {eff_threshold:.4f}). Attaching links..."
                )
                return AiResponseParser.attach_links_to_notes(notes, link_pairs)
            except Exception as e:
                log_error(f"Semantic linking error in Gemini client (falling back to direct prompt): {e}")

        # Fallback legacy prompt (if embedding model is not yet installed or failed)
        full_notes_payload = [
            {
                "id": idx + 1,
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for idx, n in enumerate(notes)
            if n.get("title")
        ]

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
            log_debug(f"Gemini Stage 2 legacy linking response received (len={len(raw_content)} chars): {raw_content[:400]}")
            pairs = AiResponseParser.parse_links_json(raw_content)
            log_debug(f"Discovered {len(pairs)} raw link pairs from Gemini legacy linking pass.")

            return AiResponseParser.attach_links_to_notes(notes, pairs)

        except Exception as e:
            log_error(f"Gemini Stage 2 legacy linking failed (non-fatal, continuing with notes without links): {e}\n{traceback.format_exc()}")
            return notes

        return notes

    def synthesize_note_cluster(
        self,
        cluster_notes: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """Synthesizes an N-way duplicate cluster into a single cohesive note via Gemini API."""
        if not cluster_notes or len(cluster_notes) < 2:
            return cluster_notes[0] if cluster_notes else None

        from prompt_templates import SYSTEM_INSTRUCTION_SYNTHESIS, build_note_synthesis_prompt

        prompt = build_note_synthesis_prompt(cluster_notes)
        full_content = f"{SYSTEM_INSTRUCTION_SYNTHESIS}\n\n{prompt}"
        log_debug(f"GeminiApiClient: Executing N-way synthesis for {len(cluster_notes)} notes...")
        if on_progress:
            on_progress(f"Synthesizing {len(cluster_notes)} overlapping notes into unified concept...")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_content,
            )
            raw_content = response.text if response and hasattr(response, 'text') and response.text else ""
            if raw_content:
                parsed = AiResponseParser.parse_synthesized_note(raw_content)
                if parsed:
                    log_debug(f"GeminiApiClient: Successfully synthesized cluster into '{parsed['title']}'")
                    return parsed
        except Exception as e:
            log_error(f"GeminiApiClient: Failed to synthesize note cluster: {e}\n{traceback.format_exc()}")

        return None

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

