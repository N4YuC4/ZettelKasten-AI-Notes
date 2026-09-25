"""
JSON repair and LaTeX sanitization engine for Zettelkasten AI Notes.

Repairs malformed or truncated JSON emitted by LLMs:
- Sanitizes unescaped backslashes in LaTeX formulas (KaTeX compatibility)
- Closes dangling strings, removes truncated keys/colons/commas, and balances unclosed brackets
"""

import re


def sanitize_latex_escapes(raw_json: str) -> str:
    """
    Repairs unescaped backslashes in LaTeX formulas within raw JSON responses.
    Prevents JSON decoders from mangling LaTeX commands (e.g. converting \\times
    or \\text to TAB [0x09], or \\beta to BACKSPACE [0x08]).
    """
    if not raw_json or not isinstance(raw_json, str):
        return ""

    # 1. Protect all single backslashes inside math blocks ($...$ and $$...$$)
    def fix_math_block(match: re.Match) -> str:
        block = match.group(0)
        # Replace any single backslash that is not already double-escaped or preceding a quote
        return re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', block)

    fixed = re.sub(r'\$\$[\s\S]+?\$\$|\$[^\$\n]+?\$', fix_math_block, raw_json)

    # 2. Also repair known LaTeX commands that might appear without math dollar delimiters
    latex_keywords = (
        r'(?:times|beta|alpha|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|omega|'
        r'frac|sqrt|text|sum|prod|int|cdot|leq|geq|neq|approx|infty|pm|partial|nabla|'
        r'mathbf|mathrm|mathit|vec|hat|bar|tilde|left|right)'
    )
    fixed = re.sub(r'(?<!\\)\\(' + latex_keywords + r'\b)', r'\\\\\1', fixed)

    # 3. Sanitize math operators accidentally wrapped inside \text{...} (e.g. \text{∑} -> \sum)
    # Prevents KaTeX from failing with "Can't use function X in text mode"
    math_text_ops = {
        '∑': r'\\sum', r'\sum': r'\\sum',
        '∏': r'\\prod', r'\prod': r'\\prod',
        '∫': r'\\int', r'\int': r'\\int',
        '√': r'\\sqrt', r'\sqrt': r'\\sqrt',
        '±': r'\\pm', r'\pm': r'\\pm',
        '≤': r'\\le', r'\le': r'\\le',
        '≥': r'\\ge', r'\ge': r'\\ge',
        '≠': r'\\ne', r'\ne': r'\\ne',
        '≈': r'\\approx', r'\approx': r'\\approx',
        '×': r'\\times', r'\times': r'\\times',
        '÷': r'\\div', r'\div': r'\\div',
    }

    def fix_text_operators(m: re.Match) -> str:
        inner = m.group(1).strip()
        if inner in math_text_ops:
            return f" {math_text_ops[inner]} "
        res = inner
        for sym, repl in math_text_ops.items():
            if sym in res:
                res = res.replace(sym, f" {repl} ")
        if res != inner:
            return res
        return m.group(0)

    fixed = re.sub(r'\\text\{([^}]*)\}', fix_text_operators, fixed)

    return fixed


def repair_truncated_json(raw: str) -> str:
    """
    Repairs truncated JSON strings caused by LLMs hitting max_tokens limits.
    Closes dangling string escapes, removes trailing incomplete keys/colons/commas,
    and balances unclosed brackets ('[' -> ']' and '{' -> '}').
    """
    if not raw or not isinstance(raw, str):
        return ""

    s = raw.strip()

    # 1. Strip trailing dangling backslash if cut off mid-escape
    if s.endswith("\\"):
        s = s[:-1]

    # 2. Check if stopped inside an unclosed string
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str

    # If cut off mid-string, the current element is incomplete.
    # Discard the incomplete element back to before its unclosed '{' brace.
    if in_str:
        last_open_brace = s.rfind("{")
        last_close_brace = s.rfind("}")
        if last_open_brace > last_close_brace:
            s = s[:last_open_brace].rstrip(" \t\r\n,")
        else:
            s += '"'

    # 3. Strip incomplete trailing key, colon, or comma
    # e.g., , "title": " or , "tit" or , {
    s = re.sub(r',\s*"[^"]*"\s*:\s*"[^"]*$', '', s)
    s = re.sub(r',\s*"[^"]*"\s*:\s*$', '', s)
    s = re.sub(r',\s*"[^"]*"\s*$', '', s)
    s = re.sub(r',\s*\{[^{}]*$', '', s)
    s = s.rstrip(' \t\r\n,:/')

    # 4. Recalculate bracket stack after stripping
    stack = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()

    # 5. Append missing closing brackets in reverse order
    while stack:
        s += stack.pop()

    return s

