# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``AGENTS.md``-style memory file loading.

Loads one or more Markdown instruction files at startup and joins
them so :func:`create_deepagent` can prepend the result to the
agent's system prompt. Layered (base → user → project), matching
the convention at <https://agents.md/>.

Missing paths are skipped silently so a default list with optional
locations can be passed without ``FileNotFoundError`` blowing up the
factory.
"""

from __future__ import annotations

from pathlib import Path


_SEPARATOR = "\n\n---\n\n"


def load_agents_md(paths: list[str] | tuple[str, ...]) -> str:
    """Read each existing path and join its contents.

    Args:
        paths: List of file paths in load order. ``~`` is expanded.
            Missing paths are skipped (intentional — callers can pass
            optional layered defaults like ``["~/AGENTS.md", "./AGENTS.md"]``
            without checking each one).

    Returns:
        Joined Markdown string with each file labelled and separated
        by ``---``. Empty string if no paths exist.
    """
    parts: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        parts.append(f"# Memory: {path.name}\n\n{content.strip()}")
    return _SEPARATOR.join(parts)
