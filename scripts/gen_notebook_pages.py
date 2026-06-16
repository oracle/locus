#!/usr/bin/env python3
# ruff: noqa: T201
"""Generate one docs/notebooks/<id>.md per examples/notebook_*.py.

Each generated page renders:
  - H1 = "Notebook NN: <Title>"  (parsed from the .py docstring)
  - the rest of the docstring as the page body
  - a "## Source" section that includes the .py via pymdownx.snippets

Run after editing notebooks::

    python scripts/gen_notebook_pages.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
OUT = ROOT / "docs" / "notebooks"


def _parse(path: Path) -> tuple[int, str, str]:
    """Return (number, title, body) parsed from the .py docstring."""
    src = path.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    m = re.match(r"notebook_(\d+)_", path.name)
    num = int(m.group(1)) if m else 0
    first, _, rest = doc.partition("\n")
    # ``Notebook NN: Title`` — strip the leading prefix for the H1
    title_match = re.match(r"^\s*Notebook\s+\d+\s*[:.]?\s*(.+?)\.?\s*$", first)
    title = title_match.group(1) if title_match else first.strip()
    return num, title, rest.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[int, str, str]] = []
    for py in sorted(EX.glob("notebook_*.py")):
        try:
            num, title, body = _parse(py)
        except SyntaxError:
            continue
        if not title:
            continue
        slug = py.stem  # e.g. notebook_14_basic_agent
        page = OUT / f"{slug}.md"
        page.write_text(
            f"# Notebook {num:02d}: {title}\n\n"
            f"{body}\n\n"
            f"## Source\n\n"
            f"```python\n"
            f'--8<-- "examples/{py.name}"\n'
            f"```\n",
            encoding="utf-8",
        )
        pages.append((num, slug, title))
    print(f"wrote {len(pages)} notebook pages to {OUT}")
    print()
    print("# add to mkdocs.yml under `- Notebooks:`:")
    for num, slug, title in pages:
        print(f"  - {num:02d} · {title}: notebooks/{slug}.md")


if __name__ == "__main__":
    main()
