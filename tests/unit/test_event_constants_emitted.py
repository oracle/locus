# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Every ``EV_*`` constant in ``locus.observability.emit`` has an emit site.

A site is any module under ``src/locus`` other than ``emit.py`` that names the
constant or quotes its string value, so a literal ``emit("x.y", ...)`` counts.
The scan is textual: it proves a name is referenced, not that the line runs.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path


# ``locus.observability.emit`` is also the name of the re-exported function.
emit_module = importlib.import_module("locus.observability.emit")

SRC = Path(emit_module.__file__).resolve().parents[1]

# Declared but not emitted, as listed under "Declared but not emitted" in
# docs/concepts/sse-events.md. Remove a name here once it gains an emit site.
NOT_EMITTED = {
    "EV_A2A_TASK_PROCESSING",
    "EV_GRAPH_NODE_ROUTED",
    "EV_RAG_EMBEDDING_GENERATED",
    "EV_RAG_STORE_SEARCH",
    "EV_RAG_STORE_UPSERT",
}


def _constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(emit_module).items()
        if name.startswith("EV_") and isinstance(value, str)
    }


def _without_site() -> set[str]:
    emit_file = Path(emit_module.__file__).resolve()
    sources = [p.read_text(encoding="utf-8") for p in SRC.rglob("*.py") if p.resolve() != emit_file]
    return {
        name
        for name, value in _constants().items()
        if not any(
            re.search(rf"\b{name}\b", text) or f'"{value}"' in text or f"'{value}'" in text
            for text in sources
        )
    }


def test_scan_sees_the_constants() -> None:
    assert len(_constants()) > 50
    assert _constants().keys() >= NOT_EMITTED


def test_every_constant_has_an_emit_site() -> None:
    assert _without_site() == NOT_EMITTED
