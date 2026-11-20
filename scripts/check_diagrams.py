#!/usr/bin/env python
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Check the site's diagrams against the static rules in docs/diagrams/README.md.

    python scripts/check_diagrams.py                          # every docs/diagrams/*.svg
    python scripts/check_diagrams.py docs/diagrams/agent-loop.svg

A diagram fails when it is not well-formed XML, when an element carries a
presentation attribute (rule 1: ``fill``, ``stroke``, ``style``, ``color`` or
any ``font-*``), when it uses a class that no ``.tl-dia`` descendant rule in
``docs/stylesheets/locus.css`` styles, or when the accessibility markup of
rule 2 is missing (``role="img"``, ``aria-labelledby="title desc"``, a
non-empty ``<title id="title">`` and ``<desc id="desc">``). Layout (rule 6:
nothing overlaps or clips, in both themes and at phone width) needs a browser
and stays a manual check under ``mkdocs serve``. Exit status 1 when any
diagram fails.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = REPO_ROOT / "docs" / "diagrams"
STYLESHEET = REPO_ROOT / "docs" / "stylesheets" / "locus.css"
SVG_NS = "http://www.w3.org/2000/svg"
PRESENTATION_ATTRIBUTES = frozenset({"fill", "stroke", "style", "color"})

# The wrapper class itself, not its BEM variants (.tl-dia--tap, .tl-dia__expand).
_WRAPPER = re.compile(r"\.tl-dia(?![\w-])")
_CLASS = re.compile(r"\.([A-Za-z_][\w-]*)")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RULE_HEAD = re.compile(r"([^{}]+)\{")


def diagram_vocabulary(css: str) -> frozenset[str]:
    """Class names that follow the ``.tl-dia`` wrapper in a selector."""
    vocabulary: set[str] = set()
    for head in _RULE_HEAD.findall(_COMMENT.sub("", css)):
        for selector in head.split(","):
            wrappers = list(_WRAPPER.finditer(selector))
            if wrappers:
                tail = selector[wrappers[-1].end() :]
                # The wrapper's own variants (.tl-dia__expand, .tl-dia--tap) are
                # page chrome around the figure, not vocabulary inside it.
                vocabulary.update(c for c in _CLASS.findall(tail) if not c.startswith("tl-dia"))
    return frozenset(vocabulary)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def check_diagram(path: Path, vocabulary: frozenset[str]) -> list[str]:
    """Problems with one diagram, in document order; empty when it passes."""
    try:
        root = ET.parse(path).getroot()  # noqa: S314 - repository files, not untrusted input
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]

    problems: list[str] = []
    if root.get("role") != "img":
        problems.append('<svg> needs role="img" (rule 2)')
    if root.get("aria-labelledby") != "title desc":
        problems.append('<svg> needs aria-labelledby="title desc" (rule 2)')
    for name in ("title", "desc"):
        child = root.find(f"{{{SVG_NS}}}{name}")
        if child is None or child.get("id") != name or not (child.text or "").strip():
            problems.append(f'<svg> needs a non-empty <{name} id="{name}"> child (rule 2)')

    for element in root.iter():
        tag = _local_name(element.tag)
        for attribute in element.attrib:
            if attribute in PRESENTATION_ATTRIBUTES or attribute.startswith("font-"):
                problems.append(
                    f"<{tag}> carries the presentation attribute {attribute!r} (rule 1)"
                )
        for cls in (element.get("class") or "").split():
            if cls not in vocabulary:
                problems.append(
                    f"<{tag}> uses class {cls!r}, which no .tl-dia rule in locus.css styles"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check site diagrams against the static rules in docs/diagrams/README.md."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="SVG files to check (default: every docs/diagrams/*.svg)",
    )
    args = parser.parse_args(argv)
    paths: list[Path] = args.paths or sorted(DIAGRAMS_DIR.glob("*.svg"))
    vocabulary = diagram_vocabulary(STYLESHEET.read_text(encoding="utf-8"))

    failing = 0
    for path in paths:
        problems = check_diagram(path, vocabulary)
        if problems:
            failing += 1
            for problem in problems:
                print(f"{path}: {problem}")
    print(f"{len(paths) - failing}/{len(paths)} diagrams pass")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
