# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Tests for the diagram checker (``scripts/check_diagrams.py``).

The whole tree must pass, the class vocabulary must come from the
``.tl-dia`` rules alone, and each rule the checker enforces must be
reported when a minimal valid diagram breaks it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.check_diagrams import (
    STYLESHEET,
    check_diagram,
    diagram_vocabulary,
    main,
)


VOCABULARY = diagram_vocabulary(STYLESHEET.read_text(encoding="utf-8"))

MINIMAL = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40" role="img" aria-labelledby="title desc">
  <title id="title">One node</title>
  <desc id="desc">A single node named A.</desc>
  <g class="node primary"><rect x="10" y="10" width="80" height="20" rx="10"/><text x="50" y="24">A</text></g>
</svg>
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "diagram.svg"
    path.write_text(text, encoding="utf-8")
    return path


def test_every_diagram_in_the_tree_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].endswith("diagrams pass")
    assert not out[-1].startswith("0/")


def test_vocabulary_is_read_from_the_stylesheet() -> None:
    assert {
        "node",
        "primary",
        "optional",
        "end",
        "edge",
        "dashed",
        "label",
        "label-bg",
    } <= VOCABULARY
    assert {"on-lane", "store", "decision", "lane", "lane-title", "note", "arrowhead"} <= VOCABULARY
    # Page chrome and the wrapper's own variants are not diagram vocabulary.
    assert not VOCABULARY & {"tl-dia", "tl-dia--tap", "tl-dia__expand", "md-typeset", "tl-glyph"}


def test_vocabulary_ignores_comments_wrapper_variants_and_unrelated_rules() -> None:
    css = """
    /* .tl-dia .commented {} */
    .md-typeset .tl-dia { display: block }
    .md-typeset .tl-dia--tap .chrome, .tl-dia__expand .more { color: red }
    .tl-dia .ok rect, [data-md-color-scheme="slate"] .tl-dia .also.fine text { fill: none }
    .unrelated .thing { fill: red }
    """
    assert diagram_vocabulary(css) == {"ok", "also", "fine"}


def test_the_minimal_diagram_passes(tmp_path: Path) -> None:
    assert check_diagram(_write(tmp_path, MINIMAL), VOCABULARY) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ('<rect x="10"', '<rect style="fill:red" x="10"', "presentation attribute 'style'"),
        ('<rect x="10"', '<rect fill="#f00" x="10"', "presentation attribute 'fill'"),
        ('<text x="50"', '<text font-size="9" x="50"', "presentation attribute 'font-size'"),
        ('class="node primary"', 'class="node mystery"', "class 'mystery'"),
        (
            '<desc id="desc">A single node named A.</desc>',
            '<desc id="desc"></desc>',
            '<desc id="desc">',
        ),
        ('<title id="title">', '<title id="heading">', '<title id="title">'),
        ('role="img" ', "", 'role="img"'),
        ('aria-labelledby="title desc"', 'aria-labelledby="title"', "aria-labelledby"),
    ],
    ids=["style", "fill", "font-size", "unknown-class", "empty-desc", "title-id", "role", "aria"],
)
def test_each_rule_is_reported(tmp_path: Path, old: str, new: str, expected: str) -> None:
    broken = MINIMAL.replace(old, new, 1)
    assert broken != MINIMAL
    problems = check_diagram(_write(tmp_path, broken), VOCABULARY)
    assert any(expected in problem for problem in problems), problems


def test_a_double_hyphen_inside_a_comment_is_not_well_formed(tmp_path: Path) -> None:
    problems = check_diagram(_write(tmp_path, "<!-- run --exec --><svg/>"), VOCABULARY)
    assert problems[0].startswith("not well-formed XML")


def test_exit_status_follows_the_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _write(tmp_path, MINIMAL)
    assert main([str(good)]) == 0
    bad = tmp_path / "bad.svg"
    bad.write_text(
        MINIMAL.replace('<rect x="10"', '<rect stroke="red" x="10"', 1), encoding="utf-8"
    )
    assert main([str(bad)]) == 1
    assert "0/1 diagrams pass" in capsys.readouterr().out
