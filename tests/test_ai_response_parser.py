import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai_response_parser import AiResponseParser
from prompt_templates import (
    SYSTEM_INSTRUCTION_EXTRACTION,
    SYSTEM_INSTRUCTION_LINKING,
    build_note_extraction_prompt,
    build_graph_linking_prompt,
    build_chained_context_block,
)


def test_clean_connections_filters_citations_and_markers():
    raw_conns = [
        "[1]",
        "[38]",
        "[Smith et al., 2021]",
        "Figure 1",
        "Table 2",
        "Madde 5",
        "Chapter 3",
        "12:30",
        "123",
        "Valid Concept",
        "Self Note",  # should be filtered if current_title is "Self Note"
        "valid concept",  # duplicate case-insensitive
    ]
    cleaned = AiResponseParser.clean_connections(raw_conns, current_title="Self Note")
    assert cleaned == ["Valid Concept"]


def test_parse_notes_json_unified_schema():
    payload = """
    {
      "general_title": "Quantum Mechanics",
      "notes": [
        {"id": 1, "title": "Superposition", "content": "Particle in multiple states."},
        {"id": 2, "title": "Entanglement", "content": "Spooky action at a distance."}
      ],
      "links": [
        {"source": 1, "target": 2}
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(payload)
    assert len(notes) == 2
    assert notes[0]["general_title"] == "Quantum Mechanics"
    assert notes[0]["title"] == "Superposition"
    assert "Entanglement" in notes[0]["connections"]
    assert "Superposition" in notes[1]["connections"]


def test_parse_notes_json_code_block_and_backslash_repairs():
    payload = r"""
    Here are the notes:
    ```json
    {
      "general_title": "Math Theory",
      "notes": [
        {"id": 1, "title": "Euler Formula", "content": "e^{i\pi} + 1 = 0 \alpha \beta"}
      ]
    }
    ```
    """
    notes = AiResponseParser.parse_notes_json(payload)
    assert len(notes) == 1
    assert notes[0]["title"] == "Euler Formula"


def test_parse_notes_json_truncated_array_salvaging():
    # Simulate LLM hitting token limit before closing bracket
    truncated = """
    {
      "general_title": "Biology",
      "notes": [
        {"id": 1, "title": "Cell Theory", "content": "All organisms are composed of cells."},
        {"id": 2, "title": "DNA Replication", "content": "Process of copying double-stranded DNA molecule
    """
    notes = AiResponseParser.parse_notes_json(truncated)
    assert len(notes) >= 1
    assert notes[0]["title"] == "Cell Theory"


def test_parse_links_json():
    response = """
    ```json
    {
      "links": [
        {"source": 1, "target": 2},
        {"source": "Concept A", "target": "Concept B"}
      ]
    }
    ```
    """
    pairs = AiResponseParser.parse_links_json(response)
    assert len(pairs) == 2
    assert pairs[0] == (1, 2)
    assert pairs[1] == ("Concept A", "Concept B")


def test_parse_links_json_truncated_and_verbose():
    # Simulates model outputting verbose explanations and getting cut off at max_tokens
    truncated_response = """
    {
      "links": [
        {"source": 1, "target": 2, "relationship": "prerequisite", "reason": "Concept 1 is foundational for Concept 2."},
        {"source": 2, "target": 3, "relationship": "cause-effect", "reason": "Concept 2 leads to the mechanism in Concept 3."},
        {"source": 3, "target": 5, "relationship": "contrast", "reason": "Incomplete text that gets truncated mid-way...
    """
    pairs = AiResponseParser.parse_links_json(truncated_response)
    assert len(pairs) >= 2
    assert (1, 2) in pairs
    assert (2, 3) in pairs


def test_parse_links_json_regex_fallback_dirty_text():
    dirty_response = """
    Here are the conceptual connections I discovered after reviewing the notes:
    {"source": 1, "target": 4} because of foundational similarity.
    Additionally, {"source": "Note 2", "target": "Note 3"}.
    Hope this helps!
    """
    pairs = AiResponseParser.parse_links_json(dirty_response)
    assert (1, 4) in pairs
    assert ("Note 2", "Note 3") in pairs



def test_attach_links_to_notes():
    notes = [
        {"id": 1, "title": "Concept 1", "content": "Text 1", "connections": []},
        {"id": 2, "title": "Concept 2 (General)", "content": "Text 2", "connections": []},
        {"id": 3, "title": "Concept 3", "content": "Text 3", "connections": []}
    ]
    pairs = [
        (1, 2),  # integer ID
        ("2", "3"),  # string digit ID
        ("Concept 1", "Concept 3"),  # title string
        (1, 1),  # self-link (should be ignored)
        (1, "[99]"),  # forbidden pattern / non-existent (should be ignored)
    ]
    res = AiResponseParser.attach_links_to_notes(notes, pairs)
    assert len(res) == 3
    assert "Concept 2 (General)" in res[0]["connections"]
    assert "Concept 3" in res[0]["connections"]
    assert "Concept 1" in res[1]["connections"]
    assert "Concept 3" in res[1]["connections"]
    assert "Concept 1" in res[2]["connections"]
    assert "Concept 2 (General)" in res[2]["connections"]



def test_prompt_templates_generation():
    assert "Zettelkasten" in SYSTEM_INSTRUCTION_EXTRACTION
    assert "knowledge graphs" in SYSTEM_INSTRUCTION_LINKING
    assert "deeply elaborated" in SYSTEM_INSTRUCTION_EXTRACTION
    assert "Atomicity means ONE distinct concept per note, NOT brevity" in SYSTEM_INSTRUCTION_EXTRACTION
    assert "conceptual_analysis" in SYSTEM_INSTRUCTION_EXTRACTION
    assert "relationship_logic" in SYSTEM_INSTRUCTION_LINKING

    prompt = build_note_extraction_prompt("This is source text.", unified_general_title="Physics")
    assert "<document_content>" in prompt
    assert "This is source text." in prompt
    assert 'Set strictly to "Physics"' in prompt
    assert "DEPTH OVER BREVITY" in prompt
    assert "Strictly avoid shallow 1-2 sentence abstracts" in prompt
    assert "Core Definition & Theoretical Basis" in prompt
    assert "Internal Mechanics & Step-by-Step Logic" in prompt
    assert "Context, Nuances & Boundary Conditions" in prompt
    assert "conceptual_analysis" in prompt
    assert "core_thesis" in prompt
    assert "Mathematical & Scientific Formulations" in prompt
    assert r"\frac" in prompt
    assert r"\times" in prompt
    assert "Multi-Metric & Multi-Formula Layout" in prompt
    assert "Concept Name" in prompt
    assert "Performance Evaluation Metrics" in prompt
    assert "Matthews Correlation Coefficient (MCC)" in prompt

    linking_prompt = build_graph_linking_prompt([{"id": 1, "title": "T1", "content": "C1"}])
    assert "<all_notes>" in linking_prompt
    assert '"title": "T1"' in linking_prompt
    assert "relationship_logic" in linking_prompt


def test_parse_notes_json_with_conceptual_analysis_and_internal_think_tags():
    payload = """
    {
      "general_title": "AI Reasoning Models",
      "conceptual_analysis": {
        "core_thesis": "Modern reasoning models like DeepSeek-R1 use reinforcement learning with internal deliberation tokens.",
        "atomic_breakdown": "Separate the training methodology from the test-time scaling behavior."
      },
      "notes": [
        {
          "id": 1,
          "title": "Chain of Thought Deliberation",
          "content": "The architecture emits <think> tokens during inference to structure its internal search process before generating the final answer.</think>"
        }
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(payload)
    assert len(notes) == 1
    assert notes[0]["title"] == "Chain of Thought Deliberation"
    # Content must retain the user's authentic <think> and </think> text because it's part of the technical note
    assert "<think>" in notes[0]["content"]
    assert "</think>" in notes[0]["content"]


def test_parse_notes_json_outer_think_block_bypassed():
    payload = """
    <think>
    I am analyzing this document. The core ideas appear to be Quantum Computing and Superposition.
    Let me output the JSON now.
    </think>
    ```json
    {
      "general_title": "Quantum Physics",
      "conceptual_analysis": {
        "core_thesis": "Superposition is fundamental.",
        "atomic_breakdown": "Isolate superposition into a single note."
      },
      "notes": [
        {"id": 1, "title": "Superposition Principle", "content": "A physical system exists partly in all theoretically possible states."}
      ]
    }
    ```
    """
    notes = AiResponseParser.parse_notes_json(payload)
    assert len(notes) == 1
    assert notes[0]["title"] == "Superposition Principle"


def test_parse_links_json_with_relationship_logic_and_outer_think():
    payload = """
    <think>
    Analyzing links:
    Note 1 provides the mathematical foundation for Note 2.
    </think>
    {
      "links": [
        {
          "relationship_logic": "Note 1 is the theoretical basis for the mechanism in Note 2.",
          "source": 1,
          "target": 2
        },
        {
          "relationship_logic": "Note 2 contrasts directly with Note 3.",
          "source": 2,
          "target": 3
        }
      ]
    }
    """
    pairs = AiResponseParser.parse_links_json(payload)
    assert len(pairs) == 2
    assert (1, 2) in pairs
    assert (2, 3) in pairs


def test_parse_links_json_regex_fallback_with_relationship_logic():
    dirty_payload = """
    Here are the links:
    {"relationship_logic": "Direct dependency", "source": 1, "target": 5}
    {"relationship_logic": "Contrasting mechanism", "source": "Alpha", "target": "Beta"}
    """
    pairs = AiResponseParser.parse_links_json(dirty_payload)
    assert (1, 5) in pairs
    assert ("Alpha", "Beta") in pairs


def test_parse_notes_json_preserves_latex_commands():
    # Simulates raw unescaped LaTeX generated by LLMs in JSON strings:
    # \times, \beta, \text, \frac, \rho
    raw_payload = r"""
    {
      "general_title": "Mathematics",
      "notes": [
        {
          "id": 1,
          "title": "Formulas",
          "content": "Model dimensions: $3\times3$ and $2\times2$. Regression: $\beta=0.75, R^2=0.56$. Metric: $\text{TP}+\text{TN}$. Fraction: $\frac{1}{2}$. Correlation: $\rho$."
        }
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(raw_payload)
    assert len(notes) == 1
    content = notes[0]["content"]
    assert r"$3\times3$" in content
    assert r"$2\times2$" in content
    assert r"$\beta=0.75" in content
    assert r"$\text{TP}+\text{TN}$" in content
    assert r"$\frac{1}{2}$" in content
    assert r"$\rho$" in content
    assert "\t" not in content
    assert "\x08" not in content



