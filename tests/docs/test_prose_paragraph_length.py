# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""No docs page carries a wall of text.

A prose paragraph that runs past ``MAX_WORDS`` words is about six rendered
lines, and the one fact a reader came for is lost inside it. The docs keep
one idea per paragraph: parallel facts go in a list, name/value facts in a
table. This test counts the words of every prose paragraph under ``docs/``
and names the ones over the limit with their line numbers, so the fix is
to split the paragraph the failure points at.

Code fences, tables, lists, headings, HTML, macro lines and admonition
markers are not prose and are skipped; an admonition's body is prose and
counts. ``ALLOWED`` names the paragraphs that are long on purpose, by page
and opening words; an entry that no longer matches a long paragraph fails
too, so the list cannot go stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MAX_WORDS = 90

# (page relative to docs/, the paragraph's opening words): long on purpose.
ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        # The admonition is titled "The whole thing, in one paragraph".
        ("research/policy-blindness.md", "An agent admission gate has two halves"),
    }
)

_NOT_PROSE = re.compile(r"^(#|\||[-*+] |\d+\. |!!!|\?\?\?|<|\{\{|\{%|:   |\[\^|\!\[)")
_FENCE = re.compile(r"^\s*(```|~~~)")

PAGES = sorted(p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md"))


def _paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return ``(first_line, word_count, opening_words)`` of every prose paragraph."""
    out: list[tuple[int, int, str]] = []
    para: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        if para and not _NOT_PROSE.match(para[0].lstrip()):
            words = " ".join(line.strip() for line in para).split()
            out.append((start, len(words), " ".join(words[:7])))
        para.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            flush()
            continue
        if not para:
            start = number
        para.append(line)
    flush()
    return out


def _long_paragraphs(page: str) -> list[tuple[int, int, str]]:
    text = (DOCS / page).read_text(encoding="utf-8")
    return [item for item in _paragraphs(text) if item[1] > MAX_WORDS]


def _allowed(page: str, opening: str) -> bool:
    return any(
        page == allowed_page and opening.startswith(start) for allowed_page, start in ALLOWED
    )


@pytest.mark.parametrize("page", PAGES)
def test_no_prose_paragraph_runs_past_the_limit(page: str) -> None:
    offenders = [
        f"line {line}: {words} words ({opening} …)"
        for line, words, opening in _long_paragraphs(page)
        if not _allowed(page, opening)
    ]
    assert not offenders, (
        f"docs/{page} has {len(offenders)} prose paragraph(s) over {MAX_WORDS} words; "
        "split each into short paragraphs, a list or a table:\n  " + "\n  ".join(offenders)
    )


def test_every_allowed_entry_still_names_a_long_paragraph() -> None:
    stale = [
        (page, start)
        for page, start in ALLOWED
        if not any(opening.startswith(start) for _, _, opening in _long_paragraphs(page))
    ]
    assert not stale, f"ALLOWED entries that no longer match a long paragraph: {stale}"
