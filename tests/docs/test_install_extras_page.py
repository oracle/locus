# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The install-extras page lists every extra in ``pyproject.toml`` and names real packages.

``docs/how-to/install-extras.md`` is the reference a reader uses to pick a
``pip install "locus-sdk[...]"`` line. Every extra ``pyproject.toml`` defines
has one row, every row names an extra that exists, the packages a row lists
are the extra's own dependencies, and the page is reachable from the nav.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "how-to" / "install-extras.md"
ROW = re.compile(r"^\| `([a-z0-9-]+)` \| (.*?) \| (.*?) \|\s*$")


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    return {name: [_dep_name(spec) for spec in specs] for name, specs in extras.items()}


def _dep_name(spec: str) -> str:
    return _normalise(re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0])


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def _rows() -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match and match.group(1) != "Extra":
            rows[match.group(1)] = (match.group(2), match.group(3))
    return rows


def test_every_extra_has_a_row_and_every_row_is_an_extra() -> None:
    extras, rows = _extras(), _rows()
    assert sorted(rows) == sorted(extras), {
        "missing rows": sorted(set(extras) - set(rows)),
        "rows for unknown extras": sorted(set(rows) - set(extras)),
    }


def test_the_packages_a_row_names_belong_to_that_extra() -> None:
    extras = _extras()
    for name, (_, packages) in _rows().items():
        listed = {_normalise(p) for p in re.findall(r"`([^`]+)`", packages)}
        assert listed, f"{name}: no packages listed"
        assert listed <= set(extras[name]), {
            "extra": name,
            "listed but not a dependency": sorted(listed - set(extras[name])),
        }


def test_the_combined_install_example_names_real_extras() -> None:
    extras = _extras()
    text = PAGE.read_text(encoding="utf-8")
    examples = re.findall(r"locus-sdk\[([a-z0-9,\- ]+)\]", text)
    assert examples
    for example in examples:
        for extra in example.split(","):
            assert extra.strip() in extras, extra


def test_the_page_is_in_the_how_to_nav() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "how-to/install-extras.md" in nav
