# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Every notebook page follows the series template.

A notebook page shows the reader, in this order, what the example builds, the
functions that carry the point (quoted by name so the excerpt follows the
code), how to run it, and the full source collapsed at the end. The checks
here are the ones a reviewer would otherwise repeat on every page: the
sections exist and are ordered, each quoted symbol exists in its file, and
the collapsed include points at a real example.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hooks import build_metadata


ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs" / "notebooks").glob("notebook_*.md"))
ROOTS = [ROOT]

_SOURCE = re.compile(
    r"^\{\{\s*locus_source\s+(?P<path>[^\s:]+)::(?P<symbol>[A-Za-z_][\w.]*)\s*\}\}", re.MULTILINE
)
_INCLUDE = re.compile(r'^\s*--8<-- "(?P<path>[^"]+)"', re.MULTILINE)


def _headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, re.MULTILINE)]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_notebook_page_follows_the_template(page: Path) -> None:
    text = page.read_text()
    heads = _headings(text)
    for required in ("Key lines", "Run it", "Source"):
        assert required in heads, f"{page.name}: missing '## {required}'"
    order = [heads.index(h) for h in ("Key lines", "Run it", "Source")]
    assert order == sorted(order), f"{page.name}: sections out of order: {heads}"
    if "Output" in heads:
        assert heads.index("Run it") < heads.index("Output") < heads.index("Source"), (
            f"{page.name}: '## Output' must sit between 'Run it' and 'Source'"
        )

    symbols = _SOURCE.findall(text)
    assert symbols, f"{page.name}: '## Key lines' has no {{{{ locus_source … }}}} excerpt"
    for path, symbol in symbols:
        build_metadata.render_source(path, symbol, ROOTS)  # raises if the file or symbol is missing

    run = text[text.index("## Run it") :]
    assert re.search(r"^```bash\n.*python ", run, re.MULTILINE | re.DOTALL), (
        f"{page.name}: '## Run it' needs a fenced bash block"
    )

    source = text[text.index("## Source") :]
    assert source.lstrip().startswith("## Source\n\n??? ") or "??? " in source.split("\n", 3)[2], (
        f"{page.name}: the source include must sit in a collapsed '???' block"
    )
    include = _INCLUDE.search(source)
    assert include, f"{page.name}: no --8<-- include under '## Source'"
    assert any((r / include.group("path")).is_file() for r in ROOTS), (
        f"{page.name}: include target missing"
    )

    # Leftover checks apply to prose only: program output and code may draw rules.
    prose = re.sub(r"\n## Output\n.*?<!-- notebook-output:end -->\n", "\n", text, flags=re.DOTALL)
    prose = re.sub(r"^ *`{3,4}.*?^ *`{3,4}$", "", prose, flags=re.DOTALL | re.MULTILINE)
    assert "::\n" not in prose and "──" not in prose, f"{page.name}: RST or ASCII-drawing leftovers"
    assert (
        "bundled mock model is the default" not in text
        and "defaults to the bundled mock" not in text
    ), f"{page.name}: stale default-provider sentence"


_OUTPUT = re.compile(r"\n## Output\n(?P<body>.*?)<!-- notebook-output:end -->\n", re.DOTALL)
#: What a pasted run must never carry: where it ran, as whom, and when.
_IDENTIFYING = re.compile(
    r"ocid1\.|oraclecloud\.com|opc-request-id|/Users/|/home/|OCI profile:"
    r"|\b20\d\d-\d\d-\d\d[T ]\d\d:\d\d:\d\d\.\d"
)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_output_section_is_closed_and_anonymous(page: Path) -> None:
    """An Output section is one sentence, one fenced block and the end marker.

    Sections captured offline are rewritten by ``scripts/notebook_output.py``;
    the ones captured from a live model are pasted by hand, so the shape and
    the absence of anything naming the account, the machine or the moment of
    the run are checked here for both.
    """
    text = page.read_text()
    if "\n## Output\n" not in text:
        return
    match = _OUTPUT.search(text)
    assert match, f"{page.name}: '## Output' has no '<!-- notebook-output:end -->' marker"
    body = match.group("body")
    assert re.fullmatch(r"\n[^\n]+:\n\n(`{3,4})text\n.*\n\1\n", body, re.DOTALL), (
        f"{page.name}: '## Output' must be one sentence ending in ':' and one fenced text block"
    )
    leak = _IDENTIFYING.search(body)
    assert leak is None, f"{page.name}: the captured output carries {leak.group(0)!r}"
