# prompt_templates.py
#
# Centralized prompt templates and Zettelkasten rules for both Cloud (Gemini) and Local (GGUF) models.
# Enforces strict language matching, atomicity, structural citation filtering, and unified JSON schemas.

from typing import Optional, List, Dict, Any, Tuple
import json


SYSTEM_INSTRUCTION_EXTRACTION = (
    "You are an expert knowledge architect and research scholar specialized in Niklas Luhmann's Zettelkasten method. "
    "Your objective is to extract key concepts, theories, mechanisms, and arguments from the given text into deeply elaborated, "
    "analytically thorough, yet strictly atomic and self-contained notes. "
    "Before extracting notes, reason systematically about the document's core thesis and the precise conceptual boundaries of each idea "
    "inside the 'conceptual_analysis' block. "
    "Atomicity means ONE distinct concept per note, NOT brevity. Do not write shallow summaries; explain the inner workings, "
    "causal logic, nuances, and specific details of each idea in full depth. "
    "When explaining scientific, mathematical, or technical concepts, use rigorous LaTeX notation, clean structured bullet points, "
    "and display math blocks ($$ ... $$) for standalone equations to ensure high legibility and typographical elegance. "
    "Output must be a valid JSON object. "
    "Always generate all titles, contents, and collections in the exact same language as the source text."
)

SYSTEM_INSTRUCTION_LINKING = (
    "You are an expert AI specialized in Niklas Luhmann's Zettelkasten method and knowledge graphs. "
    "Your task is to analyze a completed set of atomic Zettelkasten notes from a document and discover "
    "genuine, meaningful conceptual links between them (such as prerequisite, cause-effect, contrast, or conceptual complement). "
    "For every connection, deliberate on the underlying conceptual rationale in 'relationship_logic' before selecting the note IDs. "
    "Connect notes using their integer note IDs (id). "
    "Do NOT make forced or superficial connections. Return a valid JSON object with a 'links' array containing 'relationship_logic', 'source', and 'target'."
)


def build_chained_context_block(
    previous_notes_json: Optional[str] = None,
    existing_titles: Optional[List[str]] = None,
    unified_general_title: Optional[str] = None
) -> str:
    """Builds reference context block from preceding document chunks to prevent duplicate concepts."""
    if not (previous_notes_json or existing_titles or unified_general_title):
        return ""

    context_sections = []
    if unified_general_title:
        context_sections.append(f'DOCUMENT TOPIC: "{unified_general_title}"')
    if existing_titles:
        titles_list_str = ", ".join(f'"{t}"' for t in existing_titles)
        context_sections.append(f"EXISTING TITLES (DO NOT DUPLICATE):\n[{titles_list_str}]")
    if previous_notes_json:
        context_sections.append(f"PREVIOUS NOTES (REFERENCE):\n```json\n{previous_notes_json}\n```")

    return (
        "\n=== PREVIOUS SECTION CONTEXT ===\n"
        + "\n\n".join(context_sections)
        + "\n================================\n\n"
    )


def build_note_extraction_prompt(
    chunk_text: str,
    previous_notes_json: Optional[str] = None,
    existing_titles: Optional[List[str]] = None,
    unified_general_title: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
) -> str:
    """
    Builds standard Zettelkasten extraction prompt used across all AI providers.
    Enforces unified JSON schema with 'general_title', 'conceptual_analysis', and 'notes',
    instructing models to deliberate conceptually before generating atomic notes.
    Accepts optional custom_system_prompt with strict core rule precedence.
    """
    chained_context_block = build_chained_context_block(
        previous_notes_json=previous_notes_json,
        existing_titles=existing_titles,
        unified_general_title=unified_general_title
    )

    custom_instructions_block = ""
    if custom_system_prompt and custom_system_prompt.strip():
        custom_instructions_block = f"""
4. USER CUSTOM INSTRUCTIONS & PRECEDENCE:
USER-DEFINED INSTRUCTIONS:
{custom_system_prompt.strip()}

PRECEDENCE & CONFLICT RESOLUTION RULE:
The core system rules above (valid JSON schema, strict language matching, and atomic note isolation) are mandatory, foundational, and inviolable. If any user-defined instruction conflicts with or contradicts these core system rules, the core system rules MUST strictly prevail and the conflicting user instruction must be disregarded.
"""

    collection_rule = (
        f'- \'general_title\': Set strictly to "{unified_general_title}".'
        if unified_general_title
        else "- 'general_title': Overarching topic of the document in the document's language."
    )

    return f"""You are an expert knowledge architect and researcher specialized in Niklas Luhmann's Zettelkasten method.
Your task is to extract deeply analyzed, highly detailed, yet strictly atomic Zettelkasten notes from the text below.
Output MUST be a valid JSON object with 'general_title', 'conceptual_analysis', and 'notes'.

{chained_context_block}
RULES AND GUIDELINES:

1. CONCEPTUAL ANALYSIS & DELIBERATION (REASONING):
   - Before generating atomic notes, deliberate and formulate your reasoning inside 'conceptual_analysis':
     * 'core_thesis': Articulate the overarching thesis, theoretical foundation, and central argument of the document.
     * 'atomic_breakdown': Plan the discrete conceptual boundaries—identify which specific concepts must be separated into independent atomic notes without overlap.
   - Express all analytical deliberation exclusively inside this JSON object. Do not output any commentary or tags outside the JSON.
   - Focus purely on understanding the text and formulating atomic notes. (Do not concern yourself with graph linking or note connections in this extraction stage).

2. STRICT LANGUAGE MATCHING:
   - Output all 'general_title', 'conceptual_analysis', 'title', and 'content' in the EXACT SAME LANGUAGE as the source document.
   - NEVER MIX LANGUAGES: If the document is in English, every title, collection, and content MUST be in English. If the document is in Turkish, everything MUST be in Turkish.

3. ATOMIC ZETTELKASTEN NOTES:
   - ATOMIC SCOPE: Each note represents ONE specific concept, definition, mechanism, or argument. Never merge multiple disparate concepts into one note. If the text covers multiple sub-mechanisms or concepts, create separate atomic notes for each.
   - DEPTH OVER BREVITY: "Atomic" does NOT mean brief, superficial, or an executive summary. Strictly avoid shallow 1-2 sentence abstracts. Delve deeply and thoroughly into every topic.
   - CONTENT STRUCTURE: The 'content' of each note must be an autonomous, self-contained, and comprehensive exposition covering:
     a. Core Definition & Theoretical Basis: Precise explanation of what the concept is and its foundational premise.
     b. Internal Mechanics & Step-by-Step Logic: How it works, underlying causal relationships, operational dynamics, formulas, or logical steps described in the text.
     c. Context, Nuances & Boundary Conditions: Specific caveats, exceptions, prerequisites, trade-offs, and critical distinctions from adjacent ideas.
     d. Substance & Specifics: Retain concrete terminology, parameters, and evidence from the source text. Avoid vague generalities or conversational filler (do NOT use phrases like "The author explains..." or "This section describes..."). Write directly with intellectual authority.
     e. Mathematical & Scientific Formulations:
        - Format all mathematical, statistical, and algorithmic expressions using standard LaTeX/KaTeX notation.
        - Syntax & Operators: Use proper LaTeX structures (e.g., \\frac{{a}}{{b}} for fractions, \\sqrt{{...}} for roots, \\times for multiplication, and native math symbols like \\sum, \\prod, \\partial, \\chi^2).
        - Display Equations: Separate core or standalone equations into dedicated display math blocks ($$ ... $$) with empty lines before and after. Reserve inline math ($ ... $) strictly for variables, parameters, or short formulas inside running text.
        - Multi-Metric & Multi-Formula Layout: When presenting multiple performance metrics, parameters, or equations, structure them as a clean Markdown bulleted list with bold metric titles (see few-shot example below).
   - 'id': Sequential integer starting from 1 (1, 2, 3...).
   - 'title': Descriptive conceptual name in the document's language (e.g., 'Contract Termination for Just Cause', 'Hyperbolic Embedding Geometry'). Never use structural labels, numbers, or citations as titles (e.g., 'Chapter 1', 'Section 2', 'Article 5', 'Figure 3', '[12]').
   - 'content': Complete, self-contained, deeply articulated explanation of the concept in the document's language adhering to the content structure above.
{collection_rule}

4. DEDICATED NEW NOTES (NO DUPLICATES):
   - Do NOT duplicate or summarize notes that already exist in previous sections. Only extract new, distinct concepts.

JSON FORMAT:
{{
  "general_title": "Document Topic",
  "conceptual_analysis": {{
    "core_thesis": "Comprehensive articulation of the overarching thesis, foundational premise, and primary theoretical framework presented in the document.",
    "atomic_breakdown": "Systematic breakdown of the distinct concepts to be extracted into individual atomic notes, defining clear thematic boundaries to prevent overlap."
  }},
  "notes": [
    {{
      "id": 1,
      "title": "Concept Name",
      "content": "Comprehensive and deeply elaborated exposition of the foundational theoretical model. Explains the core premise, underlying assumptions, and causal mechanisms governing the system in exhaustive detail.\\n\\nThe primary operational dynamics are defined by the state transition formulation:\\n\\n$$S_{{t+1}} = \\alpha S_t + (1 - \\alpha) \\sum_{{i=1}}^N W_i X_{{i,t}}$$\\n\\nwhere $S_t$ denotes the latent state vector at time $t$, $W_i$ represents the normalized weighting matrix satisfying $\\sum W_i = 1$, and $\\alpha \\in [0, 1]$ serves as the decay coefficient. Under non-stationary conditions, the model accommodates heteroscedastic shocks through adaptive scaling, preventing catastrophic divergence while preserving gradient flow across temporal horizons."
    }},
    {{
      "id": 2,
      "title": "Performance Evaluation Metrics",
      "content": "Detailed analysis of the quantitative evaluation framework employed to assess model fidelity and classification robustness. When evaluating imbalanced datasets, multi-criteria performance requires isolating specific diagnostic dimensions:\\n\\n- **Sensitivity (Recall)**: Measures the proportion of actual positives correctly identified: $\\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}}$\\n- **Specificity**: Quantifies the true negative detection rate: $\\frac{{\\text{{TN}}}}{{\\text{{TN}} + \\text{{FP}}}}$\\n- **Matthews Correlation Coefficient (MCC)**:\\n  $$\\text{{MCC}} = \\frac{{\\text{{TP}} \\times \\text{{TN}} - \\text{{FP}} \\times \\text{{FN}}}}{{\\sqrt{{(\\text{{TP}} + \\text{{FP}})(\\text{{TP}} + \\text{{FN}})(\\text{{TN}} + \\text{{FP}})(\\text{{TN}} + \\text{{FN}})}}}}$$\\n\\nMCC is robust against severe class skew because it incorporates all four quadrants of the confusion matrix into an orthogonal correlation measure ranging from $-1$ to $+1$."
    }}
  ]
}}
{custom_instructions_block}
Text to process:
<document_content>
{chunk_text}
</document_content>
"""


def build_graph_linking_prompt(notes_payload: List[Dict[str, Any]]) -> str:
    """Builds prompt for Stage 2 global knowledge graph linking pass."""
    notes_json_str = json.dumps(notes_payload, ensure_ascii=False, indent=2)

    return f"""Below are all Zettelkasten notes extracted from the document with their 'id', 'title', and complete contents:

<all_notes>
{notes_json_str}
</all_notes>

TASK:
Analyze all the notes and descriptions above holistically.
Discover genuine conceptual links (such as prerequisite, complementary, cause-effect, or logical continuation) between notes using their integer note IDs ('id').

RULES:
1. DELIBERATE RELATIONSHIP LOGIC FIRST:
   - For each genuine connection, formulate the precise conceptual justification in 'relationship_logic' BEFORE specifying 'source' and 'target'. Explain why these two notes share a vital theoretical dependency, causal relationship, contrast, or functional complement.
2. The 'source' and 'target' fields MUST strictly contain the matching notes' integer 'id' values (1, 2, 3...).
3. STRICTLY DIFFERENT IDS (NO SELF-LOOPS): 'source' and 'target' MUST be two DIFFERENT note IDs (source != target). NEVER link a note to itself (e.g., {{"source": 1, "target": 1}} is INVALID and FORBIDDEN).
4. DO NOT FORCE CONNECTIONS: Only connect notes that share an authentic conceptual link. Standalone definitions or axioms may remain unconnected. If there are no links, return an empty array: {{"links": []}}.
5. Avoid duplicate bidirectional links (if 1 -> 2 is established, do not also write 2 -> 1).
6. STRICT JSON OUTPUT:
   - Output ONLY the JSON object with the 'links' array containing 'relationship_logic', 'source', and 'target'.
   - Do NOT output any markdown commentary or text outside the JSON.

FEW-SHOT EXAMPLE:
{{
  "links": [
    {{
      "relationship_logic": "Note 1 establishes the foundational theoretical axiom upon which the operational mechanism in Note 2 is constructed.",
      "source": 1, "target": 2
    }},
    {{
      "relationship_logic": "Note 2 specifies the primary algorithmic pipeline whose failure states and mathematical boundaries are analyzed in Note 4.",
      "source": 2, "target": 4
    }}
  ]
}}
"""

