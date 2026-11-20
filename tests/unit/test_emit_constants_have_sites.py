# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Every ``EV_*`` constant in ``locus.observability.emit`` is emitted somewhere.

The check parses every module under ``locus`` (except ``emit.py`` itself)
and records two things: the ``EV_*`` names the code reads, and the string
literals it passes as the first argument of a call or as an ``event_type=``
keyword. A constant has a site when its name or its value shows up in
either set, so ``emit(EV_X, ...)`` and ``emit("x.y", ...)`` both count while
a mention in a docstring or comment does not.

``NO_SITE_YET`` mirrors the "Declared but not emitted" list in
``docs/concepts/sse-events.md``. Give one of those constants an emit site
and this test tells you to take it off the list; add a new constant without
a site and it tells you to add one.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


emit_module = importlib.import_module("locus.observability.emit")
EMIT_FILE = Path(emit_module.__file__).resolve()
PACKAGE_ROOT = EMIT_FILE.parents[1]

NO_SITE_YET = frozenset(
    {
        "EV_A2A_TASK_PROCESSING",
        "EV_GRAPH_NODE_ROUTED",
        "EV_RAG_EMBEDDING_GENERATED",
        "EV_RAG_STORE_SEARCH",
        "EV_RAG_STORE_UPSERT",
    }
)


def declared_constants() -> dict[str, str]:
    """``EV_*`` names bound to strings in the emit module."""
    return {
        name: value
        for name, value in vars(emit_module).items()
        if name.startswith("EV_") and isinstance(value, str)
    }


def _event_literals(call: ast.Call) -> list[str]:
    found: list[str] = []
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        found.append(call.args[0].value)
    found.extend(
        kw.value.value
        for kw in call.keywords
        if kw.arg == "event_type"
        and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, str)
    )
    return found


def emit_sites() -> tuple[set[str], set[str]]:
    """``(names, literals)`` referenced by the package outside ``emit.py``."""
    names: set[str] = set()
    literals: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.resolve() == EMIT_FILE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Name) and node.id.startswith("EV_"):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and node.attr.startswith("EV_"):
                names.add(node.attr)
            elif isinstance(node, ast.Call):
                literals.update(_event_literals(node))
    return names, literals


def constants_without_a_site() -> set[str]:
    names, literals = emit_sites()
    return {
        name
        for name, value in declared_constants().items()
        if name not in names and value not in literals
    }


def test_the_scan_finds_real_sites() -> None:
    names, literals = emit_sites()
    # One emitted through its constant, one that was a literal for a long time.
    assert "EV_RESEARCH_EXECUTE_STARTED" in names
    assert "EV_MEMORY_MANAGER_INJECTED" in names
    assert "memory.manager.extract_failed" in literals


def test_no_site_yet_names_are_real_constants() -> None:
    assert declared_constants().keys() >= NO_SITE_YET


def test_every_constant_is_emitted_or_listed() -> None:
    assert constants_without_a_site() == set(NO_SITE_YET)
