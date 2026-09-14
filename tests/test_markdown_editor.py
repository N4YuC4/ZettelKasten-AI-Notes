# test_markdown_editor.py
#
# Unit tests for MarkdownEditorWidget (Source and Reading modes).

import pytest
import os
import sys
from unittest.mock import MagicMock

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from markdown_editor_widget import (
    MarkdownEditorWidget,
    process_markdown_wikilinks,
    normalize_markdown_latex,
    sanitize_math_mode_syntax,
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


def test_normalize_markdown_latex():
    # Empty string
    assert normalize_markdown_latex("") == ""

    # Formula inside parentheses: closing $ followed by ')' must have space inserted
    text = r"Neoklasik ($\frac{\partial F}{\partial K} > 0$ ve $\frac{\partial^2 F}{\partial K^2} < 0$)."
    normalized = normalize_markdown_latex(text)
    assert r"$\frac{\partial^2 F}{\partial K^2} < 0$ )" in normalized

    # Enclosing parentheses and brackets are absorbed into math delimiters: ($formula$) -> $(formula)$
    text_paren = r"Satisfaction scale ($\beta=0.75, R^2=0.56$), indicating that"
    norm_paren = normalize_markdown_latex(text_paren)
    assert r"$(\beta=0.75, R^2=0.56),$" in norm_paren

    # Punctuation with brackets, quotes:
    text_punc = r'[$y = 2$]; "$z = 3$".'
    norm_punc = normalize_markdown_latex(text_punc)
    assert r"$[y = 2];$" in norm_punc
    assert r'$z = 3$ "' in norm_punc

    # Trailing sentence punctuation (. , ; :) is absorbed into the formula
    # to prevent Flutter from wrapping the punctuation onto a new line by itself
    text_sentence = r"Recall is $TP / (TP + FN)$. Specificity is $TN / (TN + FP)$, and precision."
    norm_sentence = normalize_markdown_latex(text_sentence)
    assert r"$TP / (TP + FN).$" in norm_sentence
    assert r"$TN / (TN + FP),$" in norm_sentence

    # Inline math followed by text or space without punctuation
    text_ok = r"$a = b$ and $c = d$"
    norm_ok = normalize_markdown_latex(text_ok)
    assert norm_ok == text_ok

    # Code block must be protected
    code_text = "```python\n($x = 1$)\n```\nOutside: ($y = 2$)."
    norm_code = normalize_markdown_latex(code_text)
    assert "```python\n($x = 1$)\n```" in norm_code
    assert "$(y = 2).$" in norm_code

    # Inline code must be protected
    inline_text = "Use `($foo$)` variable with ($bar = 1$)."
    norm_inline = normalize_markdown_latex(inline_text)
    assert "`($foo$)`" in norm_inline
    assert "$(bar = 1).$" in norm_inline

    # Math operators inside \\text{} must be sanitized to valid math mode
    math_text = r"$H = \frac{\text{∑}(x_i - \bar{x})}{\text{Hello}}$"
    norm_math = normalize_markdown_latex(math_text)
    assert r"\text{∑}" not in norm_math
    assert r"\sum" in norm_math
    assert r"\text{Hello}" in norm_math


def test_sanitize_math_mode_syntax():
    assert sanitize_math_mode_syntax(r"\text{∑}_{i=1}^N") == r" \sum _{i=1}^N"
    assert sanitize_math_mode_syntax(r"\text{\sum}") == r" \sum "
    assert sanitize_math_mode_syntax(r"\text{∏}") == r" \prod "
    assert sanitize_math_mode_syntax(r"\text{√}") == r" \sqrt "
    assert sanitize_math_mode_syntax(r"\text{Standard text}") == r"\text{Standard text}"


def test_preserve_single_linebreaks_with_math_blocks():
    # Single-line $$ must not toggle in_code indefinitely
    text = "Line 1\n$$ E = mc^2 $$\nLine 2\nLine 3"
    res = preserve_single_linebreaks(text)
    lines = res.split("\n")
    assert lines[0] == "Line 1  "
    assert lines[1] == "$$ E = mc^2 $$  "
    assert lines[2] == "Line 2  "
    assert lines[3] == "Line 3  "

    # Multiline $$ block
    text_multi = "Intro\n$$\nE = mc^2\n$$\nOutro"
    res_multi = preserve_single_linebreaks(text_multi)
    lines_multi = res_multi.split("\n")
    assert lines_multi[0] == "Intro  "
    assert lines_multi[1] == "$$"
    assert lines_multi[2] == "E = mc^2"
    assert lines_multi[3] == "$$"
    assert lines_multi[4] == "Outro  "


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
    assert "| Header 1 |" in editor.get_value()

    # Insert Link
    editor.set_value("")
    editor.insert_link()
    assert "[Link Text](https://example.com)" in editor.get_value()

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

