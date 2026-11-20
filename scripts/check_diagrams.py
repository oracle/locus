# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Check site diagrams against the static rules in docs/diagrams/README.md.

Usage: python scripts/check_diagrams.py [docs/diagrams/NAME.svg ...]

With no arguments every ``docs/diagrams/*.svg`` is checked. A diagram fails
on inline styling attributes, a class no ``.tl-dia`` rule in
``docs/stylesheets/locus.css`` defines, or a missing ``role="img"``,
``aria-labelledby="title desc"``, ``<title id="title">`` or
``<desc id="desc">``. Layout (clipping, overlap, both themes, phone width)
needs a browser and is not checked here. Exits 1 when any diagram fails.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"
CSS = ROOT / "docs" / "stylesheets" / "locus.css"
SVG_NS = "{http://www.w3.org/2000/svg}"
STYLING = {"fill", "stroke", "style", "color"}  # plus every font-* attribute


def known_classes(css: str) -> set[str]:
    """Classes named after ``.tl-dia `` in a selector (not ``.tl-dia:hover``, ``.tl-dia__*``)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    classes: set[str] = set()
    for selector in re.findall(r"([^{}]+)\{", css):
        for part in selector.split(","):
            _, scoped, rest = part.partition(".tl-dia ")
            if scoped:
                classes.update(re.findall(r"\.([a-zA-Z][\w-]*)", rest))
    return classes


def check(path: Path, classes: set[str]) -> list[str]:
    try:
        root = ET.parse(path).getroot()  # noqa: S314 — repository files, not untrusted input
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    problems: list[str] = []
    if root.get("role") != "img":
        problems.append('<svg> lacks role="img"')
    if root.get("aria-labelledby") != "title desc":
        problems.append('<svg> lacks aria-labelledby="title desc"')
    for tag in ("title", "desc"):
        el = root.find(f"{SVG_NS}{tag}")
        if el is None or el.get("id") != tag or not (el.text or "").strip():
            problems.append(f'missing a non-empty <{tag} id="{tag}"> under <svg>')
    for el in root.iter():
        name = el.tag.removeprefix(SVG_NS)
        problems.extend(
            f"<{name}> has inline styling attribute {attr}="
            for attr in el.attrib
            if attr in STYLING or attr.startswith("font-")
        )
        problems.extend(
            f'<{name}> uses class "{cls}", which no .tl-dia rule in locus.css defines'
            for cls in (el.get("class") or "").split()
            if cls not in classes
        )
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(DIAGRAMS.glob("*.svg"))
    classes = known_classes(CSS.read_text(encoding="utf-8"))
    failed = 0
    for path in paths:
        problems = check(path, classes)
        failed += bool(problems)
        for problem in problems:
            print(f"{path}: {problem}")
    print(f"{len(paths) - failed} of {len(paths)} diagram(s) pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
