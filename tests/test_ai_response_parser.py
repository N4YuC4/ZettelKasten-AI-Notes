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
    assert "NATURAL STRUCTURAL BRANCHING & LINKS (FOLGEZETTEL)" in prompt
    assert "top-level 'links' array using the integer note IDs" in prompt
    assert "Folgezettel" in SYSTEM_INSTRUCTION_EXTRACTION

    linking_prompt = build_graph_linking_prompt([{"id": 1, "title": "T1", "content": "C1"}])
    assert "<all_notes>" in linking_prompt
    assert '"title": "T1"' in linking_prompt
    assert "relationship_logic" in linking_prompt
    assert "CONCISE IN-JSON THINKING" in linking_prompt
    assert "STRICT CONCISENESS & DENSITY" in linking_prompt
    assert "10-15 words" in linking_prompt
    assert "THOROUGH CROSS-SECTION BRIDGES (VERWEIS)" in linking_prompt
    assert "RESPECT STANDALONE NOTES" in linking_prompt

    # Test candidate pairs verification prompt
    from prompt_templates import build_candidate_pairs_verification_prompt
    pairs = [(1, 14, 0.85), (1, 86, 0.72)]
    notes_map = {
        1: {"title": "Note 1", "content": "Content 1"},
        14: {"title": "Note 14", "content": "Content 14"},
        86: {"title": "Note 86", "content": "Content 86"},
        99: {"title": "Unrelated Note", "content": "Should not appear"}
    }
    batch_prompt = build_candidate_pairs_verification_prompt(pairs, notes_map)
    assert "<candidate_notes>" in batch_prompt
    assert "<pairs_to_evaluate>" in batch_prompt
    assert "Note 1" in batch_prompt
    assert "Note 14" in batch_prompt
    assert "Note 86" in batch_prompt
    assert "Unrelated Note" not in batch_prompt
    assert "Score: 0.85" in batch_prompt
    assert "Score: 0.72" in batch_prompt



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


def test_repair_truncated_json_various_cutoffs():
    # 1. Truncated mid-string of an incomplete item
    raw1 = '{"general_title": "Physics", "notes": [{"id": 1, "title": "Gravity", "content": "Attraction between masses."}, {"id": 2, "title": "Incomplete", "content": "This sentence was cut off'
    rep1 = AiResponseParser.repair_truncated_json(raw1)
    notes1 = AiResponseParser.parse_notes_json(rep1)
    assert len(notes1) == 1
    assert notes1[0]["title"] == "Gravity"

    # 2. Truncated at comma after complete note
    raw2 = '{"general_title": "Physics", "notes": [{"id": 1, "title": "Gravity", "content": "Attraction."}, '
    rep2 = AiResponseParser.repair_truncated_json(raw2)
    notes2 = AiResponseParser.parse_notes_json(rep2)
    assert len(notes2) == 1
    assert notes2[0]["title"] == "Gravity"

    # 3. Truncated inside key definition
    raw3 = '{"general_title": "Physics", "notes": [{"id": 1, "title": "Gravity", "content": "Attraction."}, {"id": 2, "tit'
    rep3 = AiResponseParser.repair_truncated_json(raw3)
    notes3 = AiResponseParser.parse_notes_json(rep3)
    assert len(notes3) == 1
    assert notes3[0]["title"] == "Gravity"


def test_parse_notes_json_truncated_with_raw_newlines_strict_false():
    # Simulates real-world Turkish LLM output that was truncated with multi-line Markdown
    truncated_multiline = """{
  "general_title": "CIEDE2000 Renk Analizinde Parlama Hatasını Azaltma",
  "conceptual_analysis": {
    "core_thesis": "Bu çalışma endüstriyel renk analizlerini inceler.",
    "atomic_breakdown": "1. Parlama hatası. 2. CIEDE2000."
  },
  "notes": [
    {
      "id": 1,
      "title": "Parlama Hatasını Azaltma Mekanizması",
      "content": "Bu mekanizma speküler yansımayı filtreler.\n\nİkinci paragrafta gerçek satır sonu vardır.\nFormül: $L=50$."
    },
    {
      "id": 2,
      "title": "CIEDE2000 Renk Uzayı Dönüşümü",
      "content": "Lab uzayında renk farkı hesaplaması.\nÇok satırlı markdown analizi."
    },
    {
      "id": 3,
      "title": "Yarım Kalan Not",
      "content": "Bu not max_tokens yüzünden yarım kaldı ve tırnağı yok
"""
    notes = AiResponseParser.parse_notes_json(truncated_multiline)
    assert len(notes) >= 2
    titles = [n["title"] for n in notes]
    assert "Parlama Hatasını Azaltma Mekanizması" in titles
    assert "CIEDE2000 Renk Uzayı Dönüşümü" in titles
    # Ensure raw newlines inside markdown are preserved
    assert "\n" in notes[0]["content"]


def test_parse_notes_json_resilient_skips_malformed_item():
    # Note 1 has an unescaped bad quote or syntax issue, but Note 2 is valid
    raw = """
    {
      "general_title": "AI Logic",
      "notes": [
        { "id": 1, "title": "Bad Note", "broken_key": , },
        { "id": 2, "title": "Good Note", "content": "Valid content successfully recovered." }
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(raw)
    assert len(notes) >= 1
    assert any(n["title"] == "Good Note" for n in notes)


def test_attach_links_to_notes_prevents_chunk_id_collisions():
    # Simulates notes coming from two separate chunks, each originally having "id": 1, 2
    note1 = {"id": 1, "title": "Imitation Game", "content": "Turing test"}
    note2 = {"id": 2, "title": "Digital Computer", "content": "Discrete state machine"}
    note3 = {"id": 1, "title": "Memory Architecture", "content": "RAM and storage"}
    note4 = {"id": 2, "title": "Operating Unit", "content": "ALU and registers"}

    notes = [note1, note2, note3, note4]
    # Link Note 1 ("Imitation Game") to Note 2 ("Digital Computer") and Note 4 ("Operating Unit")
    pairs = [(1, 2), (1, 4), ("3", "4")]

    linked = AiResponseParser.attach_links_to_notes(notes, pairs)

    # Note 1 (1-based index 1) must link to Note 2 and Note 4
    assert "Digital Computer" in linked[0]["connections"]
    assert "Operating Unit" in linked[0]["connections"]

    # Note 2 (1-based index 2) must link back to Note 1
    assert "Imitation Game" in linked[1]["connections"]

    # Note 3 (1-based index 3) must link to Note 4
    assert "Operating Unit" in linked[2]["connections"]

    # Note 4 (1-based index 4) must link back to Note 1 and Note 3
    assert "Imitation Game" in linked[3]["connections"]
    assert "Memory Architecture" in linked[3]["connections"]


def test_attach_links_to_notes_resolves_article_and_morphological_titles():
    notes = [
        {"id": 1, "title": "The Argument from Consciousness (Jefferson's Critique)", "content": "Critique body"},
        {"id": 2, "title": "Digital Computers as Discrete State Machines", "content": "Computers body"},
        {"id": 3, "title": "Solipsism Objection", "content": "Solipsism body"}
    ]
    # Pairs reference target by title string with dropped 'The' and singular 'Computer'
    pairs = [
        ("Solipsism Objection", "Argument from Consciousness (Jefferson's Critique)"),
        ("Solipsism Objection", "Digital Computer as Discrete State Machines")
    ]

    linked = AiResponseParser.attach_links_to_notes(notes, pairs)

    # Note 3 should be linked to Note 1 and Note 2
    assert "The Argument from Consciousness (Jefferson's Critique)" in linked[2]["connections"]
    assert "Digital Computers as Discrete State Machines" in linked[2]["connections"]

    # Note 1 and 2 should be linked back to Note 3
    assert "Solipsism Objection" in linked[0]["connections"]
    assert "Solipsism Objection" in linked[1]["connections"]


def test_stage1_intra_chunk_numeric_connections_resolve_canonically():
    chunk_json = """
    {
      "general_title": "Turing Paper",
      "conceptual_analysis": {
        "core_thesis": "Machine intelligence analysis.",
        "atomic_breakdown": "1. Three-Part Architecture. 2. Store Unit. 3. Executive Unit."
      },
      "notes": [
        {
          "id": 1,
          "title": "Three-Part Architecture of Digital Computers",
          "content": "Store, Executive Unit, and Control Unit architecture.",
          "connections": []
        },
        {
          "id": 2,
          "title": "Store Unit Component",
          "content": "Store unit functions as internal memory.",
          "connections": [1]
        },
        {
          "id": 3,
          "title": "Executive Unit Execution",
          "content": "Executive unit performs mathematical operations.",
          "connections": [1, 2]
        }
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(chunk_json)
    assert len(notes) == 3

    # Note 1 is the root axiom -> 0 connections initially
    assert notes[0]["connections"] == []

    # Note 2 connects to Note 1 via integer ID 1 -> resolved to exact canonical title of Note 1
    assert notes[1]["connections"] == ["Three-Part Architecture of Digital Computers"]

    # Note 3 connects to Note 1 and Note 2 via integer IDs [1, 2] -> resolved to exact canonical titles
    assert "Three-Part Architecture of Digital Computers" in notes[2]["connections"]
    assert "Store Unit Component" in notes[2]["connections"]


def test_build_chained_context_block_and_prompt_strict_deduplication():
    # Verify context block warns against paraphrasing
    block = build_chained_context_block(
        existing_titles=["The Imitation Game Structure"],
        unified_general_title="Computing Machinery"
    )
    assert "STRICTLY FORBIDDEN TO RE-EXTRACT OR PARAPHRASE" in block
    assert "The Imitation Game Structure" in block

    # Verify prompt includes deduplication and overlap clauses
    prompt = build_note_extraction_prompt(
        chunk_text="Sample text on machine learning.",
        existing_titles=["The Imitation Game Structure"],
        unified_general_title="Computing Machinery"
    )
    assert "STRICT CONCEPT DEDUPLICATION" in prompt
    assert "NOTE ON SECTION OVERLAP" in prompt
    assert "DEDICATED NEW NOTES (NO DUPLICATES)" in prompt
    assert "Never re-extract or paraphrase existing concepts under altered titles across sections" in SYSTEM_INSTRUCTION_EXTRACTION


def test_contrastive_candidate_audit_and_section_echo_prompt():
    prompt = build_note_extraction_prompt(
        chunk_text="Regional convergence testing with Swamy RCM model.",
        existing_titles=["Beta Convergence", "Swamy Random Coefficients Model"],
        unified_general_title="Regional Economics"
    )
    # 1. Verify contrastive deliberation and candidate audit clauses
    assert "CONCEPTUAL ANALYSIS & CONTRASTIVE AUDIT" in prompt
    assert "candidate_audit" in prompt
    assert "Criterion of Distinctness" in prompt
    assert "Criterion of Redundancy (Section-Echo)" in prompt
    assert "PURE CONCEPTUAL TITLES" in prompt
    assert "ACADEMIC SECTION-ECHO RULE" in prompt
    assert "PERMISSION TO OMIT (EMPTY NOTES ALLOWED)" in prompt

    # 2. Verify AiResponseParser parses responses containing candidate_audit without issues
    mock_llm_json = """
    {
      "general_title": "Regional Economics",
      "conceptual_analysis": {
        "core_thesis": "Analyzes regional convergence using empirical econometric modeling.",
        "candidate_audit": [
          {
            "candidate": "Conditional Beta Convergence",
            "contrast_with_existing": "Distinct model with structural variables.",
            "status": "APPROVED"
          },
          {
            "candidate": "Swamy RCM Definition",
            "contrast_with_existing": "Redundant echo of established model.",
            "status": "REJECTED_DUPLICATE"
          }
        ],
        "atomic_breakdown": "1. Conditional Beta Convergence."
      },
      "notes": [
        {
          "id": 1,
          "title": "Conditional Beta Convergence",
          "content": "Deep analysis of conditional convergence conditioned on investment ratios.",
          "connections": ["Beta Convergence"]
        }
      ]
    }
    """
    parsed = AiResponseParser.parse_notes_json(mock_llm_json)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Conditional Beta Convergence"
    assert parsed[0]["connections"] == ["Beta Convergence"]


def test_empty_notes_array_does_not_create_untitled_notes_from_conceptual_analysis():
    # When a chunk deliberately returns "notes": [] (omission permission),
    # the parser must return [] and NOT convert conceptual_analysis into an Untitled Note.
    empty_notes_json = """
    {
      "general_title": "Regional Economics",
      "conceptual_analysis": {
        "core_thesis": "Document analyzes econometric convergence.",
        "candidate_audit": [
          {
            "candidate": "Beta Convergence",
            "contrast_with_existing": "Already in vault.",
            "status": "REJECTED_DUPLICATE"
          }
        ],
        "atomic_breakdown": ""
      },
      "notes": []
    }
    """
    notes = AiResponseParser.parse_notes_json(empty_notes_json)
    assert notes == []


def test_meta_syntax_placeholders_and_empty_notes_dropped():
    # Verifies that raw meta-placeholders like <Concept Name> or empty content are dropped
    raw_payload = """
    {
      "general_title": "Test",
      "notes": [
        {"id": 1, "title": "<Canonical name of concept>", "content": "Some content."},
        {"id": 2, "title": "Valid Concept", "content": "Valid explanation."},
        {"id": 3, "title": "Empty Note", "content": ""},
        {"id": 4, "title": "Untitled Note", "content": "Some content."}
      ]
    }
    """
    notes = AiResponseParser.parse_notes_json(raw_payload)
    assert len(notes) == 1
    assert notes[0]["title"] == "Valid Concept"


def test_is_valid_empty_notes_response():
    # 1. Valid empty notes responses
    valid_empty_1 = '{"general_title": "Physics", "notes": []}'
    valid_empty_2 = '```json\n{"general_title": "Economics", "conceptual_analysis": {"core_thesis": "Test"}, "notes": []}\n```'
    valid_empty_3 = '<think>Analysis here...</think>\n{"notes": []}'
    valid_empty_4 = '{"items": []}'
    valid_empty_5 = '[]'

    assert AiResponseParser.is_valid_empty_notes_response(valid_empty_1) is True
    assert AiResponseParser.is_valid_empty_notes_response(valid_empty_2) is True
    assert AiResponseParser.is_valid_empty_notes_response(valid_empty_3) is True
    assert AiResponseParser.is_valid_empty_notes_response(valid_empty_4) is True
    assert AiResponseParser.is_valid_empty_notes_response(valid_empty_5) is True

    # 2. Non-empty notes responses
    non_empty = '{"notes": [{"title": "Concept", "content": "Content"}]}'
    assert AiResponseParser.is_valid_empty_notes_response(non_empty) is False

    # 3. Invalid syntax or arbitrary strings
    assert AiResponseParser.is_valid_empty_notes_response("I cannot answer this.") is False
    assert AiResponseParser.is_valid_empty_notes_response("") is False
    assert AiResponseParser.is_valid_empty_notes_response(None) is False


def test_synthesize_note_contents_preserves_unique_details():
    # 1. Identical notes -> No duplication, untouched
    n1 = "# Lightness\n\nLightness L* calculation fails in low luminance regions.\n\n## Related Notes\n- [[CMC]]"
    n2 = "# Lightness\n\nLightness L* calculation fails in low luminance regions.\n\n## Related Notes\n- [[Other]]"
    res1 = AiResponseParser.synthesize_note_contents(n1, n2)
    assert "###" not in res1
    assert res1.strip() == n1.strip()

    # 2. Duplicate note with complementary unique knowledge -> Novel details synthesized
    n_base = "# Yüzey Pürüzlülüğü Etkisi\n\nTekstil kumaşlarındaki mikro gölgeler spektrofotometre okumalarında hataya yol açar.\n\n## Related Notes\n- [[CMC]]"
    n_incoming = "# Yüzey Pürüzlülüğü\n\nTekstil kumaşlarındaki mikro gölgeler spektrofotometre okumalarında varyansa yol açar.\n\nAyrıca otomotiv boyalarındaki portakal kabuğu efekti (orange peel) de ölçümlerde benzer dalgalanmalara neden olur.\n\n## Related Notes\n- [[Boya]]"

    res2 = AiResponseParser.synthesize_note_contents(n_base, n_incoming)
    assert "### Ek Gözlemler ve Tamamlayıcı Detaylar" in res2
    assert "orange peel" in res2
    assert "## Related Notes" in res2
    # Verify related notes still at the end
    assert res2.endswith("- [[CMC]]") or "- [[CMC]]" in res2


def test_parse_synthesized_note_json_and_markdown():
    # 1. Clean JSON response
    payload_json = """
    ```json
    {
      "title": "CIEDE2000 Parlaklık Analizi",
      "content": "Karanlık bölgelerde formül yetersiz kalır.",
      "connections": ["CMC Formülü", "[1]"]
    }
    ```
    """
    res = AiResponseParser.parse_synthesized_note(payload_json)
    assert res is not None
    assert res["title"] == "CIEDE2000 Parlaklık Analizi"
    assert res["content"] == "Karanlık bölgelerde formül yetersiz kalır."
    assert res["connections"] == ["CMC Formülü"]  # [1] citation filtered out!

    # 2. Raw Markdown fallback response
    payload_md = """
    # Birleşik Konsept

    Bu içerik birleşik konseptin gövdesidir.

    ## Related Notes
    - [[Kromatik Düzenleme]]
    """
    res_md = AiResponseParser.parse_synthesized_note(payload_md)
    assert res_md is not None
    assert res_md["title"] == "Birleşik Konsept"
    assert "Bu içerik birleşik konseptin gövdesidir." in res_md["content"]
    assert "Kromatik Düzenleme" in res_md["connections"]


def test_char_ngram_overlap_multilingual():
    """Tests language-agnostic character n-gram overlap on agglutinative suffixes."""
    # Turkish suffix variations:
    # "Ayrım boşluğunun bulunmaması..." vs "Ayrım boşluğu bulunmadığında..."
    s1 = "Ayrım boşluğunun bulunmaması insan gözünün parlaklık ve ton algı hassasiyetini belirgin şekilde artırır."
    s2 = "Ayrım boşluğu bulunmadığında gözün parlaklık ve ton algı hassasiyetinde belirgin bir artış görülür."

    # Naive word token overlap drops significantly due to suffixes
    w1 = set(AiResponseParser.normalize_tokens(s1))
    w2 = set(AiResponseParser.normalize_tokens(s2))
    tok_overlap = len(w1 & w2) / max(1, min(len(w1), len(w2)))
    assert tok_overlap < 0.50

    # Character n-gram overlap captures the morphological roots cleanly (>0.60)
    char_overlap = AiResponseParser.char_ngram_overlap(s1, s2, n=4)
    assert char_overlap > 0.60

    # Completely dissimilar text should have near zero 4-gram overlap (<0.10)
    s_diff = "Kuantum dalga fonksiyonu Hilbert uzayında izole parçacığın olasılık genliğini betimler."
    assert AiResponseParser.char_ngram_overlap(s1, s_diff, n=4) < 0.10


def test_is_duplicate_concept_precision_and_recall():
    """
    Verifies that is_duplicate_concept strictly distinguishes autonomous domain concepts
    while reliably merging genuine duplicate re-articulations.
    """
    body_ciede = "CIEDE2000 formülü, CIE tarafından standartlaştırılmış renk farkı metriğidir. Parlaklık, kroma ve ton sapmalarını hesaplar. " * 3
    body_cmc = "CMC (l:c) formülü, 1984 yılında Colour Measurement Committee tarafından geliştirilmiş tekstil tolerans metriğidir. " * 3
    body_gap = "Renk numuneleri arasında fiziksel ayrım boşluğu bulunmadığında insan gözünün algı hassasiyeti belirgin artar. " * 3
    body_3d = "Üç boyutlu kavisli yüzeyler ve geometrik nesneler ışığı düzlemsel numunelerden farklı saçarak algıyı etkiler. " * 3
    body_textile = "Tekstil endüstrisinde kumaş dokusu, lif yönü ve yüzey pürüzlülüğü spektrofotometre ölçümlerinde varyans yaratır. " * 3
    body_math = "Matematiksel süreksizlikler özellikle kroma ve ton açısı hesaplamalarında trigonometrik tanımsızlıklar doğurur. " * 3
    body_cie94 = "CIE94 formülü, 1994 yılında önerilen kroma ve ton ağırlıklandırma fonksiyonlarına sahip renk farkı metriğidir. " * 3

    # 1. Distinct concepts in the same document must NEVER be merged
    distinct_pairs = [
        ("CIEDE2000 Renk Farkı Formülü", body_ciede, "CMC (l:c) Formülü", body_cmc, 0.75),
        ("CIEDE2000 Renk Farkı Formülü", body_ciede, "Ayrım Boşluğu (Gap Effect / No-Separation) ve Hassasiyeti", body_gap, 0.72),
        ("CIEDE2000 Renk Farkı Formülü", body_ciede, "Üç Boyutlu Objeler", body_3d, 0.60),
        ("CIEDE2000 Renk Farkı Formülü", body_ciede, "Tekstil Endüstrisi ve Yüzey Pürüzlülüğü", body_textile, 0.58),
        ("CIEDE2000 Renk Farkı Formülü", body_ciede, "Matematiksel Süreksizlikler", body_math, 0.65),
        ("CIE94 Renk Farkı Metriği", body_cie94, "CIEDE2000 Renk Farkı Metriği", body_ciede, 0.92),  # Distinct digits (94 vs 2000)
        ("Deney Protokolü 1", body_ciede, "Deney Protokolü 2", body_ciede, 0.95),  # Distinct index numbers
    ]
    for t1, c1, t2, c2, sim in distinct_pairs:
        is_dup, _, _ = AiResponseParser.is_duplicate_concept(t1, t2, c1, c2, sim)
        assert is_dup is False, f"False merge between distinct concepts: '{t1}' and '{t2}'"

    # 2. True duplicate concepts must be recognized and merged
    body_ciede_alt = "CIEDE2000 matematiksel renk farkı formülasyonu, görsel algıdaki elipsoit toleransları modellemek için geliştirilmiştir. " * 3
    duplicate_pairs = [
        # Exact canonical title match with different qualifier
        ("Ayrım Boşluğu ve Hassasiyeti", "Ayrım Boşluğu (Gap Effect No-Separation) ve Hassasiyeti", body_gap, body_gap, 0.55),
        # Paraphrased title with shared key entity and very high semantic similarity
        ("CIEDE2000 Lightness Error", "CIEDE2000 Dark Patch Flaws", body_ciede, body_ciede_alt, 0.98),
        # Substring / strong title similarity with high semantic similarity
        ("Chromatic Adaptation Transform", "Chromatic Adaptation Mechanism", body_ciede, body_ciede_alt, 0.96),
        ("Kromatik Uyum Mekanizması", "Kromatik Uyum Modeli", body_ciede, body_ciede_alt, 0.97),
        # Core concept identity with word permutation and parenthetical qualifier
        ("Swamy’nin Tesadüfi Katsayılar Modeli (RCM)", "Tesadüfi Katsayılar Modeli (Swamy’nin RCM)", body_ciede, body_ciede_alt, 0.36),
        # Core concept identity with generic descriptive modifier (Ekonomik Büyüme vs Büyüme)
        ("Adam Smith’in Büyüme Teorisi", "Adam Smith’in Ekonomik Büyüme Teorisi", body_ciede, body_ciede_alt, 0.22),
    ]
    for t1, t2, c1, c2, sim in duplicate_pairs:
        is_dup, match_score, overlap = AiResponseParser.is_duplicate_concept(t1, t2, c1, c2, sim)
        assert is_dup is True, f"Failed to merge true duplicate concept: '{t1}' and '{t2}'"
        assert match_score > 0.50


def test_parse_notes_json_with_list_ids_and_list_links():
    # Simulates real-world Chunk 6 output where model returned list IDs and list targets
    raw_payload = """{
      "general_title": "Economics Thesis",
      "notes": [
        {
          "id": 1,
          "title": "Capital Accumulation",
          "content": "Capital accumulation drives long-run growth.",
          "connections": [2]
        },
        {
          "id": [2],
          "title": "Technological Convergence",
          "content": "Technological convergence across developing economies.",
          "connections": [1, [3]]
        },
        {
          "id": 3,
          "title": "Total Factor Productivity",
          "content": "TFP represents efficiency beyond capital and labor inputs.",
          "connections": []
        }
      ],
      "links": [
        {"source": 1, "target": [2, 3]},
        {"source": [2, 3], "target": 1}
      ]
    }"""
    notes = AiResponseParser.parse_notes_json(raw_payload)
    assert len(notes) == 3
    # Check that note 2 was unwrapped and assigned a valid ID
    assert notes[1]["id"] == 2 or isinstance(notes[1]["id"], int)
    assert notes[1]["title"] == "Technological Convergence"

    # Note 1 should connect to Technological Convergence and Total Factor Productivity
    conns_1 = notes[0]["connections"]
    assert "Technological Convergence" in conns_1
    assert "Total Factor Productivity" in conns_1

    # Note 2 should connect to Capital Accumulation and Total Factor Productivity
    conns_2 = notes[1]["connections"]
    assert "Capital Accumulation" in conns_2
    assert "Total Factor Productivity" in conns_2


def test_parse_links_json_with_unrolled_lists():
    raw_links_str = """{
      "links": [
        {"source": 1, "target": [2, 3]},
        {"source": [4, 5], "target": [6, 7]}
      ]
    }"""
    pairs = AiResponseParser.parse_links_json(raw_links_str)
    # Expected unrolled pairs: (1, 2), (1, 3), (4, 6), (4, 7), (5, 6), (5, 7)
    assert (1, 2) in pairs
    assert (1, 3) in pairs
    assert (4, 6) in pairs
    assert (4, 7) in pairs
    assert (5, 6) in pairs
    assert (5, 7) in pairs
    assert len(pairs) == 6


def test_attach_links_to_notes_with_list_endpoints_and_ids():
    notes = [
        {"id": [1], "title": "Alpha Concept", "content": "Alpha content", "connections": []},
        {"id": [2], "title": "Beta Concept", "content": "Beta content", "connections": []},
        {"id": 3, "title": "Gamma Concept", "content": "Gamma content", "connections": []},
    ]
    # Link pairs with list values
    pairs = [
        ([1], [2, 3]),
        (2, 3),
    ]
    attached = AiResponseParser.attach_links_to_notes(notes, pairs)
    assert len(attached) == 3

    # Alpha should be linked to Beta and Gamma
    assert "Beta Concept" in attached[0]["connections"]
    assert "Gamma Concept" in attached[0]["connections"]
    # Beta should be linked to Alpha and Gamma
    assert "Alpha Concept" in attached[1]["connections"]
    assert "Gamma Concept" in attached[1]["connections"]
    # Gamma should be linked to Alpha and Beta
    assert "Alpha Concept" in attached[2]["connections"]
    assert "Beta Concept" in attached[2]["connections"]


def test_extract_body_paragraphs_leading_blank_lines():
    """Verifies that leading blank lines before the main title do not prevent extract_body_paragraphs from skipping the title."""
    content = "\n\n   \n# Main Concept Title\n\nFirst body paragraph explaining the core idea.\n\nSecond body paragraph with details.\n\n## Related Notes\n- [[Other Note]]"
    paras = AiResponseParser.extract_body_paragraphs(content)
    assert len(paras) == 2
    assert paras[0] == "First body paragraph explaining the core idea."
    assert paras[1] == "Second body paragraph with details."
    assert not any("# Main Concept Title" in p for p in paras)





