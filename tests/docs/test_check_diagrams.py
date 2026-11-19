# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Every site diagram passes the static checks in scripts/check_diagrams.py."""

from __future__ import annotations

from pathlib import Path

from scripts.check_diagrams import CSS, DIAGRAMS, check, known_classes, main


CLASSES = known_classes(CSS.read_text(encoding="utf-8"))


def test_every_diagram_passes() -> None:
    assert main([]) == 0


def test_the_vocabulary_is_read_from_the_stylesheet() -> None:
    assert {"node", "primary", "edge", "label-bg", "on-lane", "lane-title"} <= CLASSES


def test_inline_styling_unknown_classes_and_a_missing_title_fail(tmp_path: Path) -> None:
    svg = (DIAGRAMS / "agent-loop.svg").read_text(encoding="utf-8")
    bad = (
        svg.replace('<path class="edge"', '<path class="edge" style="stroke:red"', 1)
        .replace('<text class="label"', '<text class="lable" font-size="9"', 1)
        .replace('<title id="title">The agent loop</title>', "")
    )
    path = tmp_path / "bad.svg"
    path.write_text(bad, encoding="utf-8")

    problems = check(path, CLASSES)

    assert "<path> has inline styling attribute style=" in problems
    assert "<text> has inline styling attribute font-size=" in problems
    assert any('class "lable"' in p for p in problems)
    assert any("<title" in p for p in problems)
    assert main([str(path)]) == 1


def test_a_double_hyphen_in_a_comment_is_not_well_formed(tmp_path: Path) -> None:
    path = tmp_path / "comment.svg"
    path.write_text("<!-- run --exec --><svg/>", encoding="utf-8")
    assert check(path, CLASSES)[0].startswith("not well-formed XML")
