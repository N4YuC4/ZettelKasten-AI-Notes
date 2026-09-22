# prompt_templates.py
#
# Centralized prompt templates and Zettelkasten rules for both Cloud (Gemini) and Local (GGUF) models.
# Enforces strict language matching, atomicity, structural citation filtering, and unified JSON schemas.

from typing import Optional, List, Dict, Any, Tuple, Callable
import json


SYSTEM_INSTRUCTION_EXTRACTION = (
    "You are an expert knowledge architect and research scholar specialized in Niklas Luhmann's Zettelkasten method. "
    "Your objective is to extract key concepts, theories, mechanisms, and arguments from the given text into deeply elaborated, "
    "analytically thorough, yet strictly atomic and self-contained notes. "
    "Before extracting notes, perform contrastive deliberation inside 'conceptual_analysis': articulate the core thesis, "
    "audit prospective candidate concepts against existing titles to eliminate recurring section echoes, and verify distinctness. "
    "Atomicity means ONE distinct concept per note, NOT brevity. Do not write shallow summaries; explain the inner workings, "
    "causal logic, nuances, and specific details of each idea in full depth. "
    "When explaining scientific, mathematical, or technical concepts, use rigorous LaTeX notation, clean structured bullet points, "
    "and display math blocks ($$ ... $$) for standalone equations to ensure high legibility and typographical elegance. "
    "After all notes are fully written, record natural structural dependencies (Folgezettel) in a top-level 'links' array using integer note IDs (id), while keeping standalone definitions unlinked. "
    "Output must be a valid JSON object. "
    "Never re-extract or paraphrase existing concepts under altered titles across sections; each note must represent a genuinely novel, independent concept. "
    "Always generate all titles, contents, and collections in the exact same language as the source text. "
    "Resist any language drift caused by foreign-language tables, citations, or code fragments, while preserving canonical international technical terms and acronyms."
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
        context_sections.append(
            f'DOCUMENT TOPIC (CANONICAL NARRATIVE LANGUAGE ANCHOR): "{unified_general_title}"'
        )
    if existing_titles:
        titles_list_str = ", ".join(f'"{t}"' for t in existing_titles)
        context_sections.append(
            f"EXISTING TITLES (STRICTLY FORBIDDEN TO RE-EXTRACT OR PARAPHRASE):\n[{titles_list_str}]"
        )
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
        f'- \'general_title\': Set strictly to "{unified_general_title}". All notes must strictly match the narrative language of this overarching topic.'
        if unified_general_title
        else "- 'general_title': Overarching topic of the document in the document's primary narrative language."
    )

    return f"""You are an expert knowledge architect and researcher specialized in Niklas Luhmann's Zettelkasten method.
Your task is to extract deeply analyzed, highly detailed, yet strictly atomic Zettelkasten notes from the text below.
Output MUST be a valid JSON object with 'general_title', 'conceptual_analysis', and 'notes'.

{chained_context_block}
RULES AND GUIDELINES:

1. CONCEPTUAL ANALYSIS & CONTRASTIVE AUDIT (REASONING):
   - Before generating atomic notes, deliberate and formulate your reasoning inside 'conceptual_analysis':
     * 'core_thesis': Articulate the overarching thesis, theoretical foundation, and central argument of the document in 2-3 concise sentences.
     * 'candidate_audit': Contrast each prospective concept from this chunk against PREVIOUS NOTES and EXISTING TITLES:
       - Criterion of Distinctness: Does this concept introduce an independent, distinct theoretical mechanism, rival hypothesis, or different operational model that stands on its own (e.g., Conditional Beta Convergence vs Absolute Beta Convergence)? If YES -> status: "APPROVED".
       - Criterion of Redundancy (Section-Echo): Does this concept merely repeat, define, summarize, or empirically apply an existing theory already covered in earlier sections under a slightly varied name (e.g., repeating the definition, framework, or empirical estimation of an already established model)? If YES -> status: "REJECTED_DUPLICATE" and do NOT extract it into notes.
     * 'atomic_breakdown': List only the APPROVED concepts (3-7 concept names, or empty if all prospective candidates were duplicates) to be extracted into notes without overlap.
   - Keep 'conceptual_analysis' focused and brief to preserve maximum generation token budget for the atomic notes themselves.
   - Express all analytical deliberation exclusively inside this JSON object. Do not output any commentary or tags outside the JSON.

2. STRICT LANGUAGE MATCHING:
   - NEVER MIX LANGUAGES: The entire document possesses ONE canonical narrative language (the author's primary language for explanatory prose in the document body). Every note title, analytical deliberation in 'conceptual_analysis', and explanatory exposition in 'content' MUST be written strictly in this primary narrative language.
   - IMMUNITY TO EMBEDDED FOREIGN FRAGMENTS (NO DRIFT): Individual text chunks often contain embedded foreign-language elements—such as statistical or regression tables, variable names, programming code snippets, foreign abstracts, or international bibliographic citations (e.g., '[Author, Year]'). You must NEVER switch the note's prose, reasoning, or title into that foreign language. Articulate the findings, mechanisms, and theories presented in those tables or citations entirely within the document's primary narrative language.
   - PRESERVATION OF CANONICAL TECHNICAL TERMS & ACRONYMS: Do NOT awkwardly force-translate established, universally recognized international technical terms, discipline-specific model names, scientific nomenclature, or standard acronyms (e.g., retain authentic terms like 'CRISPR-Cas9', 'ANOVA', 'Random Effects', 'Beta-Convergence', 'API', 'GMM' as conventionally used in academic literature). While preserving these technical names and acronyms authentic to their discipline, write all surrounding sentences, verbs, and explanations strictly in the document's primary narrative language.

3. ATOMIC ZETTELKASTEN NOTES:
   - ATOMIC SCOPE: Each note represents ONE specific concept, distinct model, or independent theoretical mechanism. If the text covers genuinely distinct mechanisms, extensions, or competing hypotheses with different theoretical premises (e.g., Absolute vs Conditional convergence, or distinct econometric estimation models), create separate atomic notes for each. However, internal dimensions of the same theory (its bare definition, narrative introduction, and specific empirical sample table) belong together in a single comprehensive note and must NOT be fragmented across multiple shallow notes.
   - PURE CONCEPTUAL TITLES: 'title' must be the canonical, universal name of the concept in the document's language (e.g., 'Conditional Beta Convergence', 'Swamy Random Coefficients Model'). Do NOT include narrative section qualifiers or stylistic markers in titles (e.g., avoid appending 'Theoretical Foundations of...', 'Definition and Application of...', 'Framework of...', 'Overview of...'). Never use structural labels, numbers, or citations as titles (e.g., 'Chapter 1', 'Section 2', 'Article 5', 'Figure 3', '[12]').
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
        - Multi-Metric & Multi-Formula Layout: When presenting multiple performance metrics (e.g., Performance Evaluation Metrics like Matthews Correlation Coefficient (MCC), Sensitivity, Specificity), parameters, or equations, structure them as a clean Markdown bulleted list with bold metric titles.
   - 'id': Sequential integer starting from 1 (1, 2, 3...).
   - 'content': Complete, self-contained, deeply articulated explanation of the concept in the document's language adhering to the content structure above.
{collection_rule}

4. NATURAL STRUCTURAL BRANCHING & LINKS (FOLGEZETTEL):
   - Only AFTER all notes are completely written in 'notes', specify authentic structural relationships in a top-level 'links' array using the integer note IDs ('source' and 'target').
   - Link sub-components to their architectural system (e.g., Note 2 belongs to Note 1: {{"source": 1, "target": 2}}), direct objections to the premises they critique, or sequential mechanisms.
   - If notes in this chunk are independent definitions or axioms with no structural dependency between them, provide an empty list: "links": []. Never invent forced or artificial connections.

5. DEDICATED NEW NOTES (NO DUPLICATES) & STRICT CONCEPT DEDUPLICATION (SECTION-ECHO FILTERING):
   - ACADEMIC SECTION-ECHO RULE: Technical and academic papers repeatedly reference, define, and re-summarize the same core theories across Introduction, Literature Review, Methodology, and Discussion sections. If a theory or concept was already extracted in a previous chunk, its reappearance in subsequent methodology or results sections is a narrative echo, NOT a new concept. Do not generate duplicate notes for recurring theories unless a genuinely new model or distinct theoretical formulation is introduced.
   - NEVER bypass deduplication by creating slight variations or paraphrased synonyms of existing titles (e.g., creating 'X Setup' when 'X Structure' already exists, or appending 'Application' / 'Model' to an established topic).
   - PERMISSION TO OMIT (EMPTY NOTES ALLOWED): If the text in this chunk primarily elaborates, tests, or continues concepts that are already captured in PREVIOUS NOTES or EXISTING TITLES without introducing genuinely new models or theories, output an empty notes array: "notes": []. A concise set of truly distinct, high-fidelity notes is vastly superior to redundant, fragmented variations.
   - NOTE ON SECTION OVERLAP: The beginning of this text may contain a brief sentence overlap from the preceding section to maintain narrative continuity. Do NOT extract notes from repeated introductory text if the underlying concept was already extracted in previous sections.
   - Only extract genuine, distinct, and newly introduced concepts that appear for the first time in this text.

JSON FORMAT:
{{
  "general_title": "Document Topic",
  "conceptual_analysis": {{
    "core_thesis": "<Synthesize the overarching thesis, foundational premise, and theoretical framework in 2-3 concise sentences in the document's language>",
    "candidate_audit": [
      {{
        "candidate": "<Primary candidate concept discovered in this text>",
        "contrast_with_existing": "<Theoretical justification explaining why this concept introduces a distinct, independent mechanism not covered in existing notes>",
        "status": "APPROVED"
      }},
      {{
        "candidate": "<Secondary candidate concept repeating an existing idea>",
        "contrast_with_existing": "<Identifies that this idea is an empirical application or narrative restatement of an already extracted concept>",
        "status": "REJECTED_DUPLICATE"
      }}
    ],
    "atomic_breakdown": "<Approved concept names to extract into notes>"
  }},
  "notes": [
    {{
      "id": 1,
      "title": "Concept Name",
      "content": "<Comprehensive, deeply elaborated exposition of the concept covering its foundational definition, internal causal mechanisms, mathematical/technical equations in LaTeX ($$ ... $$) if applicable, and boundary conditions. Fully articulated in the language of the source document without meta-references.>"
    }},
    {{
      "id": 2,
      "title": "<Canonical name of the second distinct concept, preserving established technical terms/acronyms>",
      "content": "<Detailed self-contained analysis of the second distinct concept adhering to the atomic exposition structure above, formatted in LaTeX and clean Markdown.>"
    }}
  ],
  "links": [
    {{
      "source": 1,
      "target": 2
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
1. DELIBERATE RELATIONSHIP LOGIC FIRST (CONCISE IN-JSON THINKING):
   - For each genuine connection, formulate the precise conceptual justification in 'relationship_logic' BEFORE specifying 'source' and 'target'. Explain why these two notes share a vital theoretical dependency, causal relationship, contrast, or functional complement.
   - STRICT CONCISENESS & DENSITY: Express the reasoning strictly in ONE concise sentence (maximum 10-15 words). State directly the underlying theoretical link, prerequisite, or causal dependency without conversational filler (e.g., "Note 1 defines Beta convergence which Note 8 tests regionally.").
2. The 'source' and 'target' fields MUST strictly contain the matching notes' integer 'id' values (1, 2, 3...).
3. STRICTLY DIFFERENT IDS (NO SELF-LOOPS): 'source' and 'target' MUST be two DIFFERENT note IDs (source != target). NEVER link a note to itself (e.g., {{"source": 1, "target": 1}} is INVALID and FORBIDDEN).
4. THOROUGH CROSS-SECTION BRIDGES (VERWEIS):
   - Focus on discovering macro conceptual bridges between different sections, themes, or distant arguments (e.g., connecting a philosophical argument in Section 6 with a computational constraint defined in Section 3).
   - Only establish links where an authentic conceptual relationship exists:
     * Foundational dependencies and sequential continuations (e.g., Note B builds upon or elaborates the mechanism in Note A).
     * Objections, critiques, and counter-arguments (e.g., philosophical or mathematical objections linked to the specific premises they dispute).
     * Structural component relationships across sections (e.g., sub-units or modules linked to their overarching theoretical system).
     * Analogies, comparative frameworks, and learning models.
   - RESPECT STANDALONE NOTES: Do NOT create forced, artificial, or superficial connections just to link every note. If a note is a self-contained definition, axiom, or historical aside that does not directly interface with other notes, leave it unlinked.
5. Avoid duplicate bidirectional links (if 1 -> 2 is established, do not also write 2 -> 1).
6. STRICT JSON OUTPUT:
   - Output ONLY the JSON object with the 'links' array containing 'relationship_logic', 'source', and 'target'.
   - Do NOT output any markdown commentary or text outside the JSON.

FEW-SHOT EXAMPLE:
{{
  "links": [
    {{
      "relationship_logic": "Note 1 establishes the theoretical axiom operationalized in Note 2.",
      "source": 1, "target": 2
    }},
    {{
      "relationship_logic": "Note 2 specifies the algorithm whose boundary conditions are analyzed in Note 4.",
      "source": 2, "target": 4
    }}
  ]
}}
"""


def build_candidate_pairs_verification_prompt(
    candidate_pairs: List[Tuple[Any, Any, float]],
    notes_map: Dict[Any, Dict[str, Any]]
) -> str:
    """
    Builds a high-precision batch verification prompt for Stage 2 Knowledge Graph Linking.
    Contains ONLY the notes participating in pre-filtered candidate pairs with full un-truncated content.
    The LLM evaluates only the proposed semantic candidate pairs for genuine Zettelkasten relationships.
    """
    # 1. Collect distinct note IDs appearing in candidate pairs
    active_note_ids = set()
    for src, tgt, _ in candidate_pairs:
        active_note_ids.add(src)
        active_note_ids.add(tgt)

    # 2. Build compact note reference payload containing full text
    relevant_notes = []
    for nid in sorted(active_note_ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        note = notes_map.get(nid)
        if note:
            relevant_notes.append({
                "id": nid,
                "title": note.get("title", ""),
                "content": note.get("content", "")
            })

    notes_json_str = json.dumps(relevant_notes, ensure_ascii=False, indent=2)

    # 3. Format candidate pairs checklist
    pairs_lines = []
    for idx, (src, tgt, score) in enumerate(candidate_pairs):
        src_title = notes_map.get(src, {}).get("title", f"Note {src}")
        tgt_title = notes_map.get(tgt, {}).get("title", f"Note {tgt}")
        pairs_lines.append(f"{idx + 1}. [ID {src}: \"{src_title}\"] <---> [ID {tgt}: \"{tgt_title}\"] (Score: {score:.2f})")

    pairs_checklist_str = "\n".join(pairs_lines)

    return f"""Below are the notes and semantically matched candidate pairs identified across the document:

<candidate_notes>
{notes_json_str}
</candidate_notes>

<pairs_to_evaluate>
{pairs_checklist_str}
</pairs_to_evaluate>

TASK:
Evaluate the proposed candidate pairs above based on Niklas Luhmann's Zettelkasten principles.
For each candidate pair, determine if there is an authentic conceptual link (such as prerequisite, cause-effect, theoretical dependency, contrast, or functional elaboration).

RULES:
1. DELIBERATE RELATIONSHIP LOGIC FIRST:
   - For each confirmed connection, formulate the concise conceptual justification in 'relationship_logic' (STRICTLY ONE concise sentence, max 10-15 words) BEFORE writing 'source' and 'target'.
   - If a candidate pair is only superficially similar or lacks a meaningful conceptual connection, REJECT IT and omit it from the output.
2. ONLY EVALUATE PROPOSED PAIRS:
   - Verify only connections among the proposed candidate pairs listed above.
3. STRICT INTEGER IDS:
   - The 'source' and 'target' fields MUST contain the matching note integer 'id' values (e.g., 1, 14).
   - 'source' != 'target' (never link a note to itself).
4. STRICT JSON OUTPUT:
   - Output ONLY the JSON object with the 'links' array containing 'relationship_logic', 'source', and 'target'.
   - Do NOT output markdown explanations or preamble outside the JSON.

FEW-SHOT EXAMPLE:
{{
  "links": [
    {{
      "relationship_logic": "Note 1 establishes the theoretical axiom operationalized in Note 14.",
      "source": 1,
      "target": 14
    }}
  ]
}}
"""


def partition_candidate_pairs(
    candidate_pairs: List[Tuple[Any, Any, float]],
    notes_map: Dict[Any, Dict[str, Any]],
    max_prompt_tokens: int,
    max_pairs_per_batch: int = 30,
    count_tokens_fn: Optional[Callable[[str], int]] = None
) -> List[List[Tuple[Any, Any, float]]]:
    """
    Partitions candidate pairs into manageable batches such that:
    1. Each batch contains at most `max_pairs_per_batch` candidate pairs.
    2. The verification prompt constructed with full note contents does not exceed `max_prompt_tokens`.
    """
    if not candidate_pairs:
        return []

    batches: List[List[Tuple[Any, Any, float]]] = []
    current_batch: List[Tuple[Any, Any, float]] = []

    def _estimate_tokens(prompt: str) -> int:
        if count_tokens_fn is not None:
            return count_tokens_fn(prompt)
        return int(len(prompt) / 3.5)

    for pair in candidate_pairs:
        trial_batch = current_batch + [pair]

        # Check pair count threshold
        if len(trial_batch) > max_pairs_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = [pair]
                continue

        # Check prompt token budget threshold
        if max_prompt_tokens > 0:
            trial_note_ids = {src for src, tgt, _ in trial_batch} | {tgt for src, tgt, _ in trial_batch}
            trial_notes_map = {nid: notes_map[nid] for nid in trial_note_ids if nid in notes_map}
            trial_prompt = build_candidate_pairs_verification_prompt(trial_batch, trial_notes_map)
            trial_tokens = _estimate_tokens(trial_prompt) + 200

            if trial_tokens > max_prompt_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = [pair]
                    continue
                else:
                    # Single pair alone exceeds budget, keep it as its own batch
                    current_batch = [pair]
                    batches.append(current_batch)
                    current_batch = []
                    continue

        current_batch.append(pair)

    if current_batch:
        batches.append(current_batch)

    return batches


