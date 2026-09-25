# prompt_templates.py
#
# Centralized prompt templates and Zettelkasten rules for both Cloud (Gemini) and Local (GGUF) models.
# Enforces strict language matching, atomicity, structural citation filtering, and unified JSON schemas.

from typing import Optional, List, Dict, Any, Tuple, Callable
import json


SYSTEM_INSTRUCTION_EXTRACTION = (
    "You are an expert knowledge architect, research scholar, and deterministic knowledge compiler specialized in Niklas Luhmann's Zettelkasten method. "
    "Your objective is to extract key concepts, theories, mechanisms, and arguments from the given text into deeply elaborated, "
    "analytically thorough, yet strictly atomic and self-contained notes. "
    "Before extracting notes, perform contrastive deliberation inside 'conceptual_analysis': articulate the core thesis, "
    "audit prospective candidate concepts against existing titles to eliminate recurring section echoes, and verify distinctness. "
    "Atomicity means ONE distinct concept per note, NOT brevity. Do not write shallow summaries; explain the inner workings, "
    "causal logic, nuances, specific parameters, and empirical findings in full depth. "
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

SYSTEM_INSTRUCTION_SYNTHESIS = (
    "You are an expert knowledge synthesizer and Zettelkasten architect. "
    "Your task is to consolidate multiple overlapping or duplicate notes describing the same core concept "
    "across different sections of a document into a SINGLE, unified, comprehensive, and atomic Zettelkasten note. "
    "STRICT LANGUAGE INVARIANCE: Every note title, section header, and explanatory prose MUST be written strictly "
    "in the primary narrative language of the source notes. Never translate titles or content into English. "
    "Do NOT simply summarize or shorten the notes. Preserve all distinct technical parameters, mathematical formulas, "
    "empirical findings, examples, and causal explanations from all input variants. "
    "Eliminate verbatim repetitions and harmonize the prose into a seamless, organic, and authoritative Markdown document. "
    "Output must be a valid JSON object matching the schema: "
    '{"title": "<Canonical Title in source language>", "content": "<Markdown body in source language without title header>", "connections": ["Target Note 1", ...]}'
)


def build_note_synthesis_prompt(cluster_notes: List[Dict[str, Any]]) -> str:
    """
    Builds the user prompt for N-way multi-note synthesis across a cluster of duplicate notes.
    """
    variants_text = []
    all_connections = set()
    for idx, note in enumerate(cluster_notes):
        title = note.get("title", f"Note {idx+1}")
        content = note.get("content", "")
        conns = note.get("connections", [])
        if isinstance(conns, list):
            for c in conns:
                if isinstance(c, str) and c.strip():
                    all_connections.add(c.strip())
        variants_text.append(
            f"--- VARIANT {idx+1} (Title: {title}) ---\n{content.strip()}"
        )

    joined_variants = "\n\n".join(variants_text)
    conns_str = ", ".join(f'"{c}"' for c in sorted(all_connections)) if all_connections else "None"

    return f"""The following {len(cluster_notes)} notes were extracted from different sections of the same document, describing the same core concept:

{joined_variants}

Existing Connections across all variants: [{conns_str}]

TASK:
Synthesize these {len(cluster_notes)} variants into a SINGLE, definitive, and deeply elaborated atomic Zettelkasten note.

RULES:
1. TITLE: Choose or formulate the most accurate, canonical, and concise concept title strictly in the EXACT SAME LANGUAGE as the source notes. NEVER translate a non-English title into English.
2. SYNTHESIS: Seamlessly integrate all distinct arguments, technical nuances, experimental findings, and equations from every variant into a single, cohesive narrative.
3. PRESERVATION: Never omit concrete parameters, figures, or specific domain examples. Do NOT create a shallow summary.
4. ATOMICITY & DEDUPLICATION: Merge identical points into clear, well-structured paragraphs or bullet points without repetition.
5. CONNECTIONS: Consolidate relevant outgoing wikilink references in the "connections" array.
6. STRICT LANGUAGE INVARIANCE: Maintain the exact same primary language as the source notes throughout the entire note (both title and content).
7. OUTPUT: Provide your response as a valid JSON object with keys "title", "content", and "connections".
"""



def build_chained_context_block(
    previous_notes_json: Optional[str] = None,
    existing_titles: Optional[List[str]] = None,
    unified_general_title: Optional[str] = None,
    rag_notes: Optional[List[Dict[str, Any]]] = None,
    global_concept_map: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Builds reference context block from preceding document chunks to prevent duplicate concepts."""
    if not (previous_notes_json or existing_titles or unified_general_title or rag_notes or global_concept_map):
        return ""

    context_sections = []
    if unified_general_title:
        context_sections.append(
            f'DOCUMENT TOPIC (CANONICAL NARRATIVE LANGUAGE ANCHOR): "{unified_general_title}"'
        )

    # Tier 1: Panoramic Concept Map of ALL accumulated notes
    if global_concept_map:
        lines = []
        for c in global_concept_map:
            cid = c.get("id", "?")
            ctitle = c.get("title", "")
            cmech = c.get("core_mechanism", "")
            if cmech:
                lines.append(f'- [ID {cid}] "{ctitle}": {cmech}')
            else:
                lines.append(f'- [ID {cid}] "{ctitle}"')
        concept_map_str = "\n".join(lines)
        context_sections.append(
            "TIER 1 — GLOBAL CONCEPT INVENTORY (ALL PREVIOUSLY EXTRACTED NOTES):\n"
            "The following list represents EVERY concept already compiled from earlier sections of this document.\n"
            "Review this inventory carefully; any prospective concept sharing an underlying mechanism with any item below is a DUPLICATE and MUST NOT be re-extracted:\n"
            f"{concept_map_str}"
        )
    elif existing_titles:
        titles_list_str = ", ".join(f'"{t}"' for t in existing_titles)
        context_sections.append(
            f"EXISTING TITLES (STRICTLY FORBIDDEN TO RE-EXTRACT OR PARAPHRASE):\n[{titles_list_str}]"
        )

    # Tier 2: Focal notes retrieved via Vector Embeddings + Qwen3 Reranker
    if rag_notes:
        rag_payload = [
            {
                "id": n.get("id"),
                "title": n.get("title", ""),
                "content": n.get("content", "")
            }
            for n in rag_notes
            if n.get("title")
        ]
        rag_json = json.dumps(rag_payload, ensure_ascii=False, indent=2)
        context_sections.append(
            "PREVIOUS NOTES (REFERENCE) [TIER 2 — FOCAL NOTES WITH FULL CONTENT]:\n"
            "The following notes are the most semantically relevant to this specific text segment (retrieved via neural embeddings and cross-encoder reranking).\n"
            "Examine their full theoretical explanations and mathematical formulas to avoid re-extracting their mechanisms and to link new concepts to them:\n"
            f"```json\n{rag_json}\n```"
        )
    elif previous_notes_json:
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
    rag_notes: Optional[List[Dict[str, Any]]] = None,
    global_concept_map: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Builds standard Zettelkasten extraction prompt used across all AI providers.
    Enforces unified JSON schema with 'general_title', 'conceptual_analysis', and 'notes',
    instructing models to deliberate conceptually before generating atomic notes.
    Accepts optional custom_system_prompt with strict core rule precedence.
    Supports Two-Tier RAG: Tier 1 global concept map + Tier 2 focal notes with full content.
    """
    chained_context_block = build_chained_context_block(
        previous_notes_json=previous_notes_json,
        existing_titles=existing_titles,
        unified_general_title=unified_general_title,
        rag_notes=rag_notes,
        global_concept_map=global_concept_map,
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

    return f"""You are an expert knowledge architect, research scholar, and deterministic knowledge compiler specialized in Niklas Luhmann's Zettelkasten method.
Your task is to extract deeply analyzed, highly detailed, yet strictly atomic Zettelkasten notes from the text below across any scientific, technological, philosophical, legal, historical, or academic domain.
Output MUST be a valid JSON object with 'general_title', 'conceptual_analysis', and 'notes'.

{chained_context_block}
RULES AND GUIDELINES:

1. CONCEPTUAL ANALYSIS & CONTRASTIVE AUDIT (FIRST-PRINCIPLES DELIBERATION):
   - Before generating atomic notes, deliberate and formulate your reasoning inside 'conceptual_analysis':
     * 'core_thesis': Articulate the overarching thesis, theoretical foundation, or central argument of the document in 2-3 concise sentences in the document's language.
     * 'candidate_audit': Evaluate every prospective candidate concept using the First Principles of Conceptual Identity and Orthogonality against PREVIOUS NOTES (provided with full content) and EXISTING TITLES:
       - Criterion of Distinctness (Orthogonality & Operational Autonomy): Does this concept introduce an independent theoretical mechanism, sectoral threshold framework, empirical experiment/proof, or systematic decision architecture with distinct operational parameters? If YES -> status: "APPROVED".
       - Criterion of Redundancy (Section-Echo) & Subsumption Test: Does this candidate merely repeat the exact same invariant mechanism already captured in an existing note without presenting any novel empirical findings, distinct mathematical parameters, sectoral rules, or boundary conditions? If it is a 100% duplicate echo -> status: "REJECTED_DUPLICATE".
     * 'atomic_breakdown': List ALL APPROVED concepts to be extracted into notes without artificial omission.
   - Keep 'conceptual_analysis' focused and brief to preserve maximum generation token budget for the atomic notes themselves.
   - Express all analytical deliberation exclusively inside this JSON object. Do not output any commentary or tags outside the JSON.

2. STRICT LANGUAGE MATCHING:
   - NEVER MIX LANGUAGES: The entire document possesses ONE canonical narrative language (the author's primary language for explanatory prose in the document body). Every note title, analytical deliberation in 'conceptual_analysis', and explanatory exposition in 'content' MUST be written strictly in this primary narrative language.
   - IMMUNITY TO EMBEDDED FOREIGN FRAGMENTS (NO DRIFT): Individual text chunks often contain embedded foreign-language elements—such as statistical or regression tables, variable names, programming code snippets, foreign abstracts, or international bibliographic citations (e.g., '[Author, Year]'). You must NEVER switch the note's prose, reasoning, or title into that foreign language. Articulate the findings, mechanisms, and theories presented in those tables or citations entirely within the document's primary narrative language.
   - PRESERVATION OF CANONICAL TECHNICAL TERMS & ACRONYMS: Do NOT awkwardly force-translate established, universally recognized international technical terms, discipline-specific model names, scientific nomenclature, or standard acronyms (e.g., retain authentic terms like 'CRISPR-Cas9', 'ANOVA', 'Random Effects', 'Beta-Convergence', 'API', 'GMM', 'Transformer', 'Nash Equilibrium', 'mRNA' as conventionally used in academic literature). While preserving these technical names and acronyms authentic to their discipline, write all surrounding sentences, verbs, and explanations strictly in the document's primary narrative language.

3. ATOMIC ZETTELKASTEN NOTES:
   - COMPREHENSIVE SCOPE: Each note represents ONE specific concept, distinct model, sectoral framework, empirical validation, or procedural methodology. Extract ALL distinct concepts present in this text segment without artificial omission. Do NOT limit extraction to only 1 or 2 notes if the text contains multiple valuable and distinct concepts.
   - PURE CONCEPTUAL TITLES: 'title' must be the canonical, universal name of the concept in the document's language in clean plain text (e.g., 'Conditional Beta Convergence', 'Swamy Random Coefficients Model').
     * NEVER use LaTeX math syntax, Greek symbols in math delimiters, or dollar signs inside titles (e.g., write 'Beta Convergence', NEVER 'Beta ($beta$) Convergence' or '$\beta$-Yakınsama').
     * NEVER append parenthetical aliases, alternate spellings, or narrative qualifiers in titles (e.g., write 'Thomas Malthus', NEVER 'Thomas Robert Malthus (T. R. Malthus)').
     * Do NOT include narrative section qualifiers or stylistic markers in titles (avoid 'Theoretical Foundations of...', 'Definition and Application of...', 'Framework of...', 'Overview of...').
     * Never use structural labels, numbers, or citations as titles (e.g., 'Chapter 1', 'Section 2', 'Article 5', 'Figure 3', '[12]').
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
   - Only AFTER all notes are completely written in 'notes', specify authentic structural relationships in a top-level 'links' array using the integer note IDs ('source' and 'target') or exact titles.
   - You may link new notes to any of the PREVIOUS NOTES (using their integer 'id' or title) if the new concept is an authentic continuation, dependency, sub-mechanism, or critique of a previously established note.
   - Link sub-components to their architectural system (e.g., Note 2 belongs to Note 1: {{"source": 1, "target": 2}}), direct objections to the premises they critique, or sequential mechanisms.
   - If notes in this chunk are independent definitions or axioms with no structural dependency, provide an empty list: "links": []. Never invent forced or artificial connections.

5. DEDICATED NEW NOTES (NO DUPLICATES) & STRICT CONCEPT DEDUPLICATION (SECTION-ECHO FILTERING):
   - FIRST PRINCIPLES OF CONCEPTUAL IDENTITY:
     In Niklas Luhmann's Zettelkasten, an atomic note represents an autonomous knowledge unit.
     Avoid verbatim duplicates: Never re-extract the exact same general definition under cosmetic title variations if no new operational parameters, formulas, or findings are introduced.
   - ACADEMIC SECTION-ECHO RULE & RAG DEDUPLICATION:
     Authors naturally re-introduce, summarize, and echo the same foundational concepts across different sections. Examine PREVIOUS NOTES to filter out pure verbatim echoes.
     However, distinct sectoral adaptation models (e.g. clinical dentistry tolerances, medical display calibration, liquid food analysis), concrete empirical experiments (e.g. dataset bias proofs, controlled benchmark tests), and procedural quality-control architectures MUST be extracted as distinct knowledge cards so no vital scientific knowledge is omitted.
   - PERMISSION TO OMIT (EMPTY NOTES ALLOWED) — THE CARDINALITY CONTRACT:
     When a chunk contains only transitional filler or repeats already extracted concepts with zero new empirical, sectoral, or theoretical substance, output an empty notes array:
     "notes": []
     and set "atomic_breakdown": [] inside 'conceptual_analysis'.
   - NOTE ON SECTION OVERLAP: The beginning of this text may contain a brief sentence overlap from the preceding section to maintain narrative continuity. Do NOT extract notes from repeated introductory text if the underlying concept was already extracted in previous sections.
   - Only extract genuine, distinct, and newly introduced concepts that appear in this text.

JSON FORMAT:
{{
  "general_title": "Document Topic",
  "conceptual_analysis": {{
    "core_thesis": "<Synthesize the overarching thesis, foundational premise, and theoretical framework in 2-3 concise sentences in the document's language>",
    "candidate_audit": [
      {{
        "candidate": "<Primary candidate concept discovered in this text>",
        "matched_existing_concept": "NONE_ORTHOGONALLY_NOVEL",
        "contrast_with_existing": "<Demonstrates that this concept introduces a fundamentally independent, orthogonal mechanism or sectoral framework not subsumed by any existing note>",
        "status": "APPROVED"
      }},
      {{
        "candidate": "<Secondary candidate concept sharing an identical mechanism with an existing note without new evidence>",
        "matched_existing_concept": "<Exact title of the matching PREVIOUS NOTE or EXISTING TITLE>",
        "contrast_with_existing": "<Identifies that this idea is an identical verbatim echo without new parameters or evidence>",
        "status": "REJECTED_DUPLICATE"
      }}
    ],
    "atomic_breakdown": "<Approved concept names to extract into notes, or empty array [] if all rejected>"
  }},
  "notes": [
    {{
      "id": 1,
      "title": "Concept Name",
      "content": "<Comprehensive, deeply elaborated exposition of the concept covering its foundational definition, internal causal mechanisms, mathematical/technical equations in LaTeX ($$ ... $$) if applicable, and boundary conditions. Fully articulated in the language of the source document without meta-references.>"
    }},
    {{
      "id": 2,
      "title": "<Canonical name of second distinct concept, sectoral framework, or empirical proof>",
      "content": "<Detailed self-contained analysis of the second distinct concept adhering to the atomic exposition structure above, formatted in LaTeX and clean Markdown.>"
    }},
    {{
      "id": 3,
      "title": "<Canonical name of third distinct concept, methodology, or decision procedure>",
      "content": "<Detailed self-contained analysis of the third distinct concept adhering to the atomic exposition structure above, formatted in LaTeX and clean Markdown.>"
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


