# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``StateBackend`` — in-memory ephemeral filesystem-as-memory.

The default backend when ``create_deepagent(enable_filesystem=True)``
is set without an explicit backend. Holds a ``dict[path, bytes]`` and
a parallel metadata map. Per-instance scope: the agent's scratchspace
dies with the backend object — no disk side effects, no cleanup
contract, no path-traversal risk.

Thread-safe via a single ``threading.Lock`` so concurrent tool calls
inside one agent run can't race. Concurrency is rare (locus's loop is
single-threaded) but cheap to guarantee.
"""

from __future__ import annotations

import fnmatch
import re
import threading
from datetime import UTC, datetime

from locus.deepagent.backends.protocol import (
    BackendError,
    FileInfo,
    Match,
)


def _normalize(path: str) -> str:
    """Canonicalize a path for keying: absolute, no trailing slash
    except for the root."""
    if not path:
        raise BackendError("invalid_path", path)
    if not path.startswith("/"):
        raise BackendError("invalid_path", path)
    if path == "/":
        return "/"
    return path.rstrip("/")


class StateBackend:
    """In-memory dict-of-paths-to-bytes backend.

    Files are stored as UTF-8 text. Directories are implicit: a file
    at ``"/notes/draft.md"`` makes ``"/notes"`` a directory in
    listings even though no explicit entry exists for it.
    """

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._mtime: dict[str, datetime] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read / write / edit
    # ------------------------------------------------------------------

    def read(self, path: str, *, offset: int = 0, limit: int = 100) -> str:
        path = _normalize(path)
        with self._lock:
            if path not in self._files:
                if self._is_implicit_dir(path):
                    raise BackendError("is_directory", path)
                raise BackendError("file_not_found", path)
            content = self._files[path]
        lines = content.splitlines()
        offset = max(offset, 0)
        if offset >= len(lines):
            return ""
        end = offset + limit if limit > 0 else len(lines)
        chunk = lines[offset:end]
        # cat -n style line numbers (1-based, right-aligned 6).
        return "\n".join(f"{offset + i + 1:6d}\t{ln}" for i, ln in enumerate(chunk))

    def write(self, path: str, contents: str) -> None:
        path = _normalize(path)
        if not isinstance(contents, str):
            contents = str(contents)
        with self._lock:
            if self._is_implicit_dir(path):
                raise BackendError("is_directory", path)
            self._files[path] = contents
            self._mtime[path] = datetime.now(UTC)

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        path = _normalize(path)
        with self._lock:
            if path not in self._files:
                raise BackendError("file_not_found", path)
            content = self._files[path]
            occurrences = content.count(old_str)
            if occurrences == 0:
                msg = f"old_str not found in {path}"
                raise ValueError(msg)
            if occurrences > 1:
                msg = (
                    f"old_str matches {occurrences} times in {path}; "
                    "provide a longer / more specific snippet so the match is unique"
                )
                raise ValueError(msg)
            self._files[path] = content.replace(old_str, new_str, 1)
            self._mtime[path] = datetime.now(UTC)

    def exists(self, path: str) -> bool:
        try:
            path = _normalize(path)
        except BackendError:
            return False
        with self._lock:
            return path in self._files or self._is_implicit_dir(path)

    # ------------------------------------------------------------------
    # ls / glob / grep
    # ------------------------------------------------------------------

    def ls(self, path: str = "/", *, recursive: bool = False) -> list[FileInfo]:
        path = _normalize(path)
        with self._lock:
            files = dict(self._files)
            mtimes = dict(self._mtime)

        prefix = path if path.endswith("/") else path + "/"
        if path == "/":
            prefix = "/"

        if path != "/" and path not in files and not self._is_implicit_dir(path, files):
            raise BackendError("file_not_found", path)

        out: list[FileInfo] = []
        seen_dirs: set[str] = set()
        for fp, content in files.items():
            if not fp.startswith(prefix) and fp != path:
                continue
            rest = fp[len(prefix) :] if prefix != "/" else fp[1:]
            if not rest:
                continue
            if "/" in rest and not recursive:
                # Surface the immediate child directory once.
                child_dir = prefix + rest.split("/", 1)[0]
                if child_dir not in seen_dirs:
                    seen_dirs.add(child_dir)
                    out.append(FileInfo(path=child_dir, is_dir=True))
                continue
            out.append(
                FileInfo(
                    path=fp,
                    is_dir=False,
                    size=len(content.encode("utf-8")),
                    modified_at=mtimes.get(fp),
                )
            )
        out.sort(key=lambda fi: fi.path)
        return out

    def glob(
        self,
        pattern: str,
        *,
        path: str = "/",
        timeout_s: float = 20.0,  # noqa: ARG002 — in-memory match is O(n); honored as a contract surface
    ) -> list[FileInfo]:
        path = _normalize(path)
        with self._lock:
            files = list(self._files.items())
            mtimes = dict(self._mtime)
        prefix = path if path.endswith("/") else path + "/"
        if path == "/":
            prefix = "/"
        out: list[FileInfo] = []
        for fp, content in files:
            if not fp.startswith(prefix) and fp != path:
                continue
            rel = fp[len(prefix) :] if prefix != "/" else fp[1:]
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fp, pattern):
                out.append(
                    FileInfo(
                        path=fp,
                        is_dir=False,
                        size=len(content.encode("utf-8")),
                        modified_at=mtimes.get(fp),
                    )
                )
        out.sort(key=lambda fi: fi.path)
        return out

    def grep(
        self,
        pattern: str,
        *,
        path: str = "/",
        recursive: bool = True,
    ) -> list[Match]:
        path = _normalize(path)
        regex = re.compile(pattern)
        with self._lock:
            files = dict(self._files)
        prefix = path if path.endswith("/") else path + "/"
        if path == "/":
            prefix = "/"
        out: list[Match] = []
        for fp, content in files.items():
            in_scope = fp == path or fp.startswith(prefix)
            if not in_scope:
                continue
            if not recursive and prefix != "/":
                rel = fp[len(prefix) :]
                if "/" in rel:
                    continue
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    out.append(Match(path=fp, line=i, text=line))
        out.sort(key=lambda m: (m.path, m.line))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_implicit_dir(self, path: str, files: dict[str, str] | None = None) -> bool:
        files = files if files is not None else self._files
        prefix = path if path.endswith("/") else path + "/"
        return any(fp.startswith(prefix) for fp in files)
