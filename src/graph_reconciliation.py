"""
Graph reconciliation and link normalization utilities for Zettelkasten AI Notes.

Provides shared, reusable primitives for:
- Canonical undirected link-pair ordering
- Transitive title redirect resolution (loop-safe)
- Wikilink remapping in note bodies
- Connection list normalization (self-link elimination and deduplication)
"""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def canonical_pair(u: Any, v: Any) -> Tuple[Any, Any]:
    """
    Returns a deterministic canonical ordering for an undirected pair (u, v).
    Ensures that (u, v) and (v, u) map to the exact same tuple regardless of initial ordering.
    """
    return (u, v) if str(u) <= str(v) else (v, u)


def resolve_transitive_redirect(title: str, redirects: Mapping[str, str]) -> str:
    """
    Follows redirect chains transitively until reaching the terminal canonical title.
    Guards against circular references using a visited set.

    Example:
        redirects = {"A": "B", "B": "C"}
        resolve_transitive_redirect("A", redirects) -> "C"
    """
    curr = title
    visited: Set[str] = set()
    while curr in redirects and curr not in visited:
        visited.add(curr)
        curr = redirects[curr]
    return curr


def resolve_all_terminal_redirects(redirects: Mapping[str, str]) -> Dict[str, str]:
    """
    Resolves all keys in a redirects mapping to their terminal canonical targets.
    """
    return {
        old_title: resolve_transitive_redirect(old_title, redirects)
        for old_title in redirects
    }


def remap_markdown_wikilinks(content: str, terminal_redirects: Mapping[str, str]) -> str:
    """
    Remaps wikilinks in markdown content for any redirected titles.
    Preserves custom aliases (e.g. [[Old Title|Custom Alias]] -> [[New Title|Custom Alias]]).
    """
    if not content or not terminal_redirects:
        return content

    result = content
    for old_t, new_t in terminal_redirects.items():
        if old_t and new_t and old_t != new_t:
            result = re.sub(
                r'\[\[' + re.escape(old_t) + r'(\]\]|\|)',
                lambda m, nt=new_t: f"[[{nt}{m.group(1)}",
                result
            )
    return result


def remap_note_connections(
    connections: Iterable[Any],
    terminal_redirects: Mapping[str, str],
    current_title: str = ""
) -> List[str]:
    """
    Normalizes a list of note connections:
    1. Remaps any redirected title to its terminal canonical title.
    2. Strips self-links (case-insensitive match against current_title).
    3. Discards duplicates (case-insensitive) while preserving first-seen order.
    """
    if not connections:
        return []

    title_clean = current_title.strip().casefold() if current_title else ""
    new_conns: List[str] = []
    seen_conns: Set[str] = set()

    for c in connections:
        if not isinstance(c, str):
            continue
        resolved = terminal_redirects.get(c, c)
        resolved_lower = resolved.strip().casefold()
        if not resolved_lower:
            continue
        if title_clean and resolved_lower == title_clean:
            continue
        if resolved_lower in seen_conns:
            continue
        seen_conns.add(resolved_lower)
        new_conns.append(resolved)

    return new_conns


def remap_notes_links_and_content(
    notes: Sequence[Dict[str, Any]],
    terminal_redirects: Mapping[str, str],
    skip_indices: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """
    Applies connection normalization and markdown wikilink remapping to a list of note dictionaries.
    Optionally skips indices (e.g. suppressed duplicate notes).
    """
    skipped = skip_indices or set()
    cleaned_notes: List[Dict[str, Any]] = []

    for idx, note in enumerate(notes):
        if idx in skipped:
            continue

        note_copy = dict(note)
        current_title = note_copy.get("title", "")

        # 1. Remap and clean connections
        orig_conns = note_copy.get("connections", [])
        if isinstance(orig_conns, list):
            note_copy["connections"] = remap_note_connections(
                orig_conns, terminal_redirects, current_title=current_title
            )

        # 2. Remap body wikilinks
        content = note_copy.get("content", "")
        if isinstance(content, str) and content:
            note_copy["content"] = remap_markdown_wikilinks(content, terminal_redirects)

        cleaned_notes.append(note_copy)

    return cleaned_notes
