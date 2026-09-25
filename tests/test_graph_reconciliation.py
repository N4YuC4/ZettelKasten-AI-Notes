import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from graph_reconciliation import (
    canonical_pair,
    resolve_transitive_redirect,
    resolve_all_terminal_redirects,
    remap_markdown_wikilinks,
    remap_note_connections,
    remap_notes_links_and_content,
)


def test_canonical_pair_ordering():
    assert canonical_pair(1, 2) == (1, 2)
    assert canonical_pair(2, 1) == (1, 2)
    assert canonical_pair("beta", "alpha") == ("alpha", "beta")
    assert canonical_pair("alpha", "beta") == ("alpha", "beta")
    assert canonical_pair("10", "2") == ("10", "2")  # Lexicographical string check
    assert canonical_pair(u="x", v="y") == canonical_pair(u="y", v="x")


def test_resolve_transitive_redirect_direct_and_chained():
    redirects = {
        "A": "B",
        "B": "C",
        "C": "Final",
        "Unrelated": "Other",
    }
    assert resolve_transitive_redirect("A", redirects) == "Final"
    assert resolve_transitive_redirect("B", redirects) == "Final"
    assert resolve_transitive_redirect("C", redirects) == "Final"
    assert resolve_transitive_redirect("Final", redirects) == "Final"
    assert resolve_transitive_redirect("NonExistent", redirects) == "NonExistent"


def test_resolve_transitive_redirect_cycle_protection():
    redirects = {
        "A": "B",
        "B": "C",
        "C": "A",
    }
    # Should safely terminate without an infinite loop
    result = resolve_transitive_redirect("A", redirects)
    assert result in {"A", "B", "C"}


def test_resolve_all_terminal_redirects():
    redirects = {
        "Old Title": "Mid Title",
        "Mid Title": "Terminal Title",
        "Standalone": "Direct Target",
    }
    terminal_map = resolve_all_terminal_redirects(redirects)
    assert terminal_map["Old Title"] == "Terminal Title"
    assert terminal_map["Mid Title"] == "Terminal Title"
    assert terminal_map["Standalone"] == "Direct Target"


def test_remap_markdown_wikilinks():
    content = "Check [[Old Note]] and [[Old Note|alias]], plus [[Untouched Note]]."
    redirects = {"Old Note": "Modern Note"}
    result = remap_markdown_wikilinks(content, redirects)
    assert result == "Check [[Modern Note]] and [[Modern Note|alias]], plus [[Untouched Note]]."


def test_remap_markdown_wikilinks_no_op_on_identity():
    content = "See [[Same Title]] here."
    redirects = {"Same Title": "Same Title"}
    assert remap_markdown_wikilinks(content, redirects) == content


def test_remap_note_connections_dedup_and_self_links():
    connections = ["Target A", "target a", "Self Note", "Target B", "Old Target"]
    redirects = {"Old Target": "Target C"}
    cleaned = remap_note_connections(
        connections,
        terminal_redirects=redirects,
        current_title="Self Note"
    )
    assert cleaned == ["Target A", "Target B", "Target C"]


def test_remap_notes_links_and_content():
    notes = [
        {
            "title": "Note 1",
            "content": "Refers to [[Note 2]].",
            "connections": ["Note 2", "note 1"],
        },
        {
            "title": "Note 2",
            "content": "Merged into Note 3.",
            "connections": ["Note 1"],
        },
    ]
    redirects = {"Note 2": "Note 3"}
    # Note 2 is skipped (e.g. duplicate)
    result = remap_notes_links_and_content(notes, redirects, skip_indices={1})
    assert len(result) == 1
    assert result[0]["title"] == "Note 1"
    assert result[0]["content"] == "Refers to [[Note 3]]."
    assert result[0]["connections"] == ["Note 3"]

