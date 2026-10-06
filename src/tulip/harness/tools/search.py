# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Finding things: glob by name, grep by content.

When the workspace has ripgrep (``BackendCapabilities.has_rg``) both run it
through the workspace's own shell: it is faster, it reads ``.gitignore``,
and ``grep`` stops as soon as it has a page of results instead of reading a
million matches to print three hundred. Otherwise — or when ripgrep refuses
a pattern Python accepts, such as a look-behind — they fall back to the
backend's own walk, which skips the same noise directories.
"""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime

from tulip.deepagent.backends.protocol import BackendError
from tulip.harness.backend import SKIP_DIRS
from tulip.harness.tools.common import HarnessContext, outside
from tulip.tools.decorator import Tool, tool


__all__ = ["GLOB_LIMIT", "GREP_LIMIT", "SEARCH_TIMEOUT", "make_glob", "make_grep"]

#: How many results glob and grep return per call. ``offset`` pages past it.
GLOB_LIMIT = 500
GREP_LIMIT = 300

#: Seconds a search may take.
SEARCH_TIMEOUT = 60

_MARK = "__tulip_rg_exit="


def _rg_excludes() -> str:
    """The directories the walk skips, for when they are not ignored."""
    out = ["--hidden", "--glob", "'!.git'"]
    for name in sorted(SKIP_DIRS - {".git"}):
        out += ["--glob", shlex.quote(f"!{name}")]
    return " ".join(out)


def _page(items: list[str], offset: int, limit: int) -> str:
    """One page of results, and how to ask for the next."""
    shown = items[offset : offset + limit]
    if not shown:
        return f"{len(items):,} results — offset {offset} is past the end"
    rest = len(items) - (offset + len(shown))
    tail = f"\n... {rest:,} more — call again with offset={offset + len(shown)}"
    return "\n".join(shown) + (tail if rest > 0 else "")


def _rel(path: str, top: str) -> str:
    prefix = top.rstrip("/") + "/"
    return path.removeprefix(prefix) if path.startswith(prefix) else path


def _ripgrep(h: HarnessContext, command: str, cwd: str, keep: int | None) -> list[str] | None:
    """Run ripgrep in the workspace; its lines, or None when it could not search.

    The exit status survives the pipe through ``head`` by way of descriptor
    4, so "found nothing" (1) and "could not run this search" (2) stay
    different things.
    """
    head = f" | head -n {keep}" if keep else ""
    script = (
        f"exec 3>&1; s=$( {{ {{ {command} 2>/dev/null; echo $? >&4; }}{head} >&3; }} 4>&1 ); "
        f'echo "{_MARK}$s"'
    )
    try:
        result = h.backend.exec(script, timeout=SEARCH_TIMEOUT, cwd=cwd)
    except BackendError:
        return None
    lines = result.output.decode("utf-8", errors="replace").splitlines()
    status = lines.pop() if lines and lines[-1].startswith(_MARK) else ""
    code = status.removeprefix(_MARK).strip()
    if result.timed_out or not code.isdigit():
        return None
    # 0 found, 1 found nothing, 141 stopped by ``head`` once it had enough.
    if int(code) not in (0, 1, 141) and not lines:
        return None
    return [line.removeprefix("./") for line in lines if line]


def make_glob(h: HarnessContext) -> Tool:
    """The ``glob`` tool over ``h``."""

    def glob(pattern: str, path: str = ".", offset: int = 0) -> str:
        """Find files by name pattern, newest first. Honours .gitignore.

        Args:
            pattern: A glob like ``**/*.py`` or ``test_*.py``.
            path: Directory to search from.
            offset: Skip this many results, to page through a long listing.
        """
        try:
            top = h.backend.resolve(path)
        except BackendError:
            return outside(path, h.backend.capabilities.root)
        caps = h.backend.capabilities
        found: list[str] | None = None
        if caps.has_rg and caps.can_exec:
            found = _ripgrep(
                h,
                f"rg --files --sortr modified {_rg_excludes()} --glob {shlex.quote(pattern)} .",
                top,
                None,
            )
        if found is None:
            try:
                infos = h.backend.glob(pattern, path=path, timeout_s=SEARCH_TIMEOUT)
            except BackendError:
                infos = []
            oldest = datetime.min.replace(tzinfo=UTC)
            infos.sort(key=lambda i: (i.modified_at or oldest, i.path), reverse=True)
            found = [_rel(i.path, top) for i in infos]
        if not found:
            return f"no files matching {pattern}"
        return _page(found, max(0, int(offset)), GLOB_LIMIT)

    return tool(idempotent=True)(glob)


def make_grep(h: HarnessContext) -> Tool:
    """The ``grep`` tool over ``h``."""

    def grep(pattern: str, path: str = ".", glob_filter: str = "*", offset: int = 0) -> str:
        """Search file contents with a regular expression. Honours .gitignore.

        Args:
            pattern: Python regular expression.
            path: Directory or file to search.
            glob_filter: Only search files matching this name pattern.
            offset: Skip this many matches, to page through a long result.
        """
        try:
            re.compile(pattern)
        except re.error as exc:
            return f"bad regex: {exc}"
        caps = h.backend.capabilities
        try:
            target = h.backend.resolve(path)
        except BackendError:
            return outside(path, caps.root)
        start = max(0, int(offset))
        want = start + GREP_LIMIT
        hits: list[str] | None = None
        if caps.has_rg and caps.can_exec:
            where = _rel(target, caps.root) if target != caps.root else "."
            argv = "rg --line-number --no-heading --with-filename --color never "
            argv += _rg_excludes()
            if glob_filter and glob_filter != "*":
                argv += f" --glob {shlex.quote(glob_filter)}"
            argv += f" --regexp {shlex.quote(pattern)} -- {shlex.quote(where)}"
            # One more than the page, to know there is another.
            raw = _ripgrep(h, argv, caps.root, want + 1)
            if raw is not None:
                hits = []
                for line in raw:
                    name, _, rest = line.partition(":")
                    number, _, text = rest.partition(":")
                    hits.append(f"{name}:{number}: {text.strip()[:200]}")
        if hits is None:
            # No ripgrep, or a pattern its engine refused (look-behind,
            # back-references): the walk reads it the way Python does.
            try:
                matches = h.backend.grep(
                    pattern, path=path, glob_filter=glob_filter or "*", limit=want + 1
                )
            except BackendError:
                matches = []
            hits = [f"{_rel(m.path, caps.root)}:{m.line}: {m.text.strip()[:200]}" for m in matches]
        if not hits:
            return f"no matches for {pattern}"
        # One more than the page was collected only to know there are more;
        # how many more is not worth reading the rest of the tree to find out.
        shown = hits[start:want]
        if len(hits) > want:
            return "\n".join(shown) + (
                f"\n... more matches — narrow the pattern, or call again with offset={want}"
            )
        if not shown:
            return f"no matches for {pattern} past offset {start}"
        return "\n".join(shown)

    return tool(idempotent=True)(grep)
