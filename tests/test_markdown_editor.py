# test_markdown_editor.py
#
# Unit tests for MarkdownEditorWidget (Kaynak ve Okuma modları).

import pytest
import os
import sys
from unittest.mock import MagicMock

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from markdown_editor_widget import (
    MarkdownEditorWidget,
    process_markdown_wikilinks,
    preserve_single_linebreaks,
    calculate_document_stats,
    toggle_task_in_text,
)


def test_preserve_single_linebreaks():
    sample = "First line\nSecond line\n\n```python\nprint(1)\nprint(2)\n```\nLast line"
    res = preserve_single_linebreaks(sample)
    lines = res.split("\n")
    assert lines[0] == "First line  "
    assert lines[1] == "Second line  "
    assert lines[2] == ""
    assert lines[3] == "```python"
    assert lines[4] == "print(1)"
    assert lines[5] == "print(2)"
    assert lines[6] == "```"
    assert lines[7] == "Last line  "


def test_process_markdown_wikilinks():
    sample_text = "Check out [[Note Title]] and [[Other Note|Custom Alias]]."
    processed = process_markdown_wikilinks(sample_text)
    assert "[🔗 Note Title](zettel://note/Note%20Title)" in processed
    assert "[🔗 Custom Alias](zettel://note/Other%20Note)" in processed

    # Plain text without wikilinks should retain content
    plain_text = "No links here."
    assert "No links here." in process_markdown_wikilinks(plain_text)

    # Empty string
    assert process_markdown_wikilinks("") == ""


def test_calculate_document_stats():
    text = "Line 1 with three words.\nLine 2 is here.\nLine 3."
    stats = calculate_document_stats(text)
    assert stats["words"] == 11
    assert stats["chars"] == len(text)
    assert stats["lines"] == 3
    assert stats["reading_time_min"] == 1

    # Empty stats
    empty_stats = calculate_document_stats("")
    assert empty_stats["words"] == 0
    assert empty_stats["lines"] == 0
    assert empty_stats["reading_time_min"] == 0


def test_toggle_task_in_text():
    sample_text = "- [ ] Task 1\n- [x] Task 2\n- [ ] Task 3"
    
    # Toggle first task: from unchecked to checked
    res1 = toggle_task_in_text(sample_text, 0)
    assert res1.startswith("- [x] Task 1")

    # Toggle second task: from checked to unchecked
    res2 = toggle_task_in_text(sample_text, 1)
    assert "- [ ] Task 2" in res2

    # Invalid index should return original text
    res_inv = toggle_task_in_text(sample_text, 99)
    assert res_inv == sample_text


def test_markdown_editor_widget_initialization():
    changed_contents = []
    clicked_wikilinks = []

    editor = MarkdownEditorWidget(
        on_content_change=lambda c: changed_contents.append(c),
        on_wikilink_clicked=lambda title: clicked_wikilinks.append(title),
        get_all_notes_callback=lambda: [("1", "Test Note", "General")],
    )

    assert editor.current_mode == "source"
    assert editor.get_value() == ""

    # Set value
    editor.set_value("# My Title\n\nContent here with [[Linked Note]]")
    assert "# My Title" in editor.get_value()

    # Mode switching
    editor.set_mode("reading")
    assert editor.current_mode == "reading"
    assert editor.main_view_area.content == editor.reading_scroll_view
    assert editor.toolbar.visible is False
    assert "[🔗 Linked Note](zettel://note/Linked%20Note)" in editor.reading_markdown_view.value

    editor.set_mode("source")
    assert editor.current_mode == "source"
    assert editor.main_view_area.content == editor.editor_field
    assert editor.toolbar.visible is True

    # Test toggle_mode (Ctrl+E)
    editor.toggle_mode()
    assert editor.current_mode == "reading"
    editor.toggle_mode()
    assert editor.current_mode == "source"


def test_markdown_editor_formatting_helpers():
    editor = MarkdownEditorWidget()
    editor.set_value("Hello World")

    # Select all and wrap with Bold
    editor._selection_start = 0
    editor._selection_end = 11
    editor.wrap_selection("**", "**", "")
    assert editor.get_value() == "**Hello World**"

    # Prepend Heading
    editor._selection_start = 0
    editor._selection_end = len(editor.get_value())
    editor.format_heading(1)
    assert editor.get_value().startswith("# **Hello World**")

    # Insert Code Block
    editor.set_value("")
    editor.insert_code_block()
    assert "```python" in editor.get_value()

    # Insert Table
    editor.set_value("")
    editor.insert_table_template()
    assert "| Başlık 1 |" in editor.get_value()

    # Insert Link
    editor.set_value("")
    editor.insert_link()
    assert "[Bağlantı Metni](https://example.com)" in editor.get_value()

    # Prepend Numbered list
    editor.set_value("First item\nSecond item")
    editor._selection_start = 0
    editor._selection_end = len(editor.get_value())
    editor.prepend_numbered_list()
    assert "1. First item" in editor.get_value()
    assert "2. Second item" in editor.get_value()


def test_wikilink_tap_dispatch():
    clicked = []
    editor = MarkdownEditorWidget(on_wikilink_clicked=lambda t: clicked.append(t))
    editor.set_value("Sample text with [[My Target Note]]")

    class FakeEvent:
        data = "zettel://note/My%20Target%20Note"

    editor._handle_link_tap(FakeEvent())
    assert len(clicked) == 1
    assert clicked[0] == "My Target Note"


def test_link_tap_uri_allowlist():
    editor = MarkdownEditorWidget()
    mock_page = MagicMock()
    editor._get_page = MagicMock(return_value=mock_page)

    class FakeEvent:
        def __init__(self, d):
            self.data = d

    # 1. Trusted HTTPS scheme - should invoke launch_url
    editor._handle_link_tap(FakeEvent("https://example.com/guide"))
    # 2. Blocked file/javascript/data schemes - should not crash or call launch_url for blocked schemes
    editor._handle_link_tap(FakeEvent("file:///etc/passwd"))
    editor._handle_link_tap(FakeEvent("javascript:alert(1)"))
    editor._handle_link_tap(FakeEvent("data:text/html,malicious"))
    editor._handle_link_tap(FakeEvent(""))

