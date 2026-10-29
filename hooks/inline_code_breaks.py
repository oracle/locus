# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""mkdocs hook: let long identifiers in inline code wrap at their separators.

A phone column is about forty characters wide. An inline code span such as
``examples/notebook_55_gsar_typed_grounding.py`` or
``oracleagentmemory.core.OracleAgentMemory`` is wider than that, and without a
break opportunity the browser cuts it at an arbitrary letter, leaving a lone
``y`` in its own chip on the next line. This inserts ``<wbr>`` after each
``/``, ``.``, ``_``, ``-`` and ``,`` inside inline ``<code>`` spans (and
mkdocstrings member names) of thirty-six or more characters, so a wrap lands on a
separator. Code inside ``<pre>`` blocks is left alone: it scrolls or wraps on
its own rules. ``<wbr>`` has no visible width and is dropped by the Markdown
twins that hooks/llms_txt.py writes, which read the Markdown, not this HTML.
"""

from __future__ import annotations

import re
from typing import Any


# A chip shorter than this fits a phone line on its own, so it moves to the
# next line whole; only longer chips get break opportunities (36 characters is
# about 300px in the inline code face, against a 328px column at 360px).
MIN_CHARS = 36
_SEPARATORS = re.compile(r"([/._,-])(?=\S)")
_PRE = re.compile(r"<pre\b.*?</pre>", re.DOTALL)
_INLINE_CODE = re.compile(r"(<code(?:\s[^>]*)?>)([^<]{" + str(MIN_CHARS) + r",})(</code>)")
_MEMBER_NAME = re.compile(
    r'(<span class="doc doc-object-name[^"]*">)([^<]{' + str(MIN_CHARS) + r",})(</span>)"
)


def _break(text: str) -> str:
    if "<wbr>" in text:
        return text
    return _SEPARATORS.sub(r"\1<wbr>", text)


def _rewrite_segment(segment: str) -> str:
    segment = _INLINE_CODE.sub(lambda m: m.group(1) + _break(m.group(2)) + m.group(3), segment)
    return _MEMBER_NAME.sub(lambda m: m.group(1) + _break(m.group(2)) + m.group(3), segment)


def add_break_opportunities(html: str) -> str:
    """Return ``html`` with ``<wbr>`` after separators in long inline code."""
    out: list[str] = []
    pos = 0
    for block in _PRE.finditer(html):
        out.append(_rewrite_segment(html[pos : block.start()]))
        out.append(block.group(0))
        pos = block.end()
    out.append(_rewrite_segment(html[pos:]))
    return "".join(out)


def on_page_content(html: str, page: Any, config: Any, files: Any) -> str:
    return add_break_opportunities(html)
