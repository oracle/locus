# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""An in-memory workspace: files in a dict, and no shell.

:class:`MemoryBackend` is the deepagent
:class:`~tulip.deepagent.backends.state.StateBackend` with the workspace
surface added, for tests and for agents that only ever read and write files.
It has no shell: :meth:`~MemoryBackend.exec` and the background methods raise
:class:`~tulip.harness.backend.ExecUnsupportedError`, and
:func:`~tulip.harness.toolset.build_harness` leaves the shell tools out.

Content is kept as text, the way ``StateBackend`` keeps it, decoded with
``surrogateescape`` so arbitrary bytes round-trip exactly — an image written
through :meth:`~MemoryBackend.write_bytes` reads back byte for byte.
"""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Callable, Mapping

from tulip.deepagent.backends.protocol import BackendError, FileInfo, Match
from tulip.deepagent.backends.state import StateBackend
from tulip.harness.backend import (
    MAX_LINE_CHARS,
    MAX_READ_BYTES,
    SKIP_DIRS,
    BackendCapabilities,
    ExecResult,
    ExecUnsupportedError,
    FileStat,
    JobOutput,
    JobStatus,
    LineWindow,
    take_window,
)
from tulip.harness.local import glob_match


__all__ = ["MemoryBackend"]

_LABEL = "memory"


def _to_text(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape")


def _to_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


def _noise(path: str, top: str) -> bool:
    """Whether ``path`` is under one of :data:`SKIP_DIRS` below ``top``."""
    rel = path[len(top) :] if top != "/" else path
    return any(part in SKIP_DIRS for part in rel.split("/")[:-1])


class MemoryBackend(StateBackend):
    """Files in memory, rooted at ``/``; no shell.

    Args:
        files: Initial contents, ``{path: text or bytes}``.
    """

    def __init__(self, files: Mapping[str, str | bytes] | None = None) -> None:
        super().__init__()
        self._capabilities = BackendCapabilities(
            isolated=True, label=_LABEL, has_rg=False, root="/", can_exec=False
        )
        for path, content in (files or {}).items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            self.write_bytes(path, data)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    # ------------------------------------------------------------- paths --

    def resolve(self, path: str) -> str:
        if not isinstance(path, str):
            raise BackendError("invalid_path", str(path))
        # Relative paths resolve against the root; ``..`` cannot climb above
        # ``/`` in a normalized absolute path, so nothing escapes.
        return posixpath.normpath(posixpath.join("/", path or "."))

    # ------------------------------------------------------------- files --

    def read_bytes(self, path: str) -> bytes:
        key = self.resolve(path)
        with self._lock:
            if key in self._files:
                return _to_bytes(self._files[key])
            directory = self._is_implicit_dir(key) or key == "/"
        raise BackendError("is_directory" if directory else "file_not_found", path)

    def write_bytes(self, path: str, data: bytes) -> None:
        self.write(self.resolve(path), _to_text(data))

    def write(self, path: str, contents: str) -> None:
        super().write(self.resolve(path), contents)

    def read(self, path: str, *, offset: int = 0, limit: int = 100) -> str:
        return super().read(self.resolve(path), offset=offset, limit=limit)

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        super().edit(self.resolve(path), old_str, new_str)

    def exists(self, path: str) -> bool:
        return super().exists(self.resolve(path))

    def remove(self, path: str) -> None:
        key = self.resolve(path)
        with self._lock:
            if key in self._files:
                del self._files[key]
                self._mtime.pop(key, None)
                return
            directory = self._is_implicit_dir(key) or key == "/"
        raise BackendError("is_directory" if directory else "file_not_found", path)

    def stat(self, path: str) -> FileStat:
        key = self.resolve(path)
        with self._lock:
            content = self._files.get(key)
            mtime = self._mtime.get(key)
            directory = content is None and (key == "/" or self._is_implicit_dir(key))
        if directory:
            return FileStat(size=0, mtime=0.0, is_dir=True)
        if content is None:
            raise BackendError("file_not_found", path)
        data = _to_bytes(content)
        return FileStat(
            size=len(data),
            mtime=mtime.timestamp() if mtime else 0.0,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        max_bytes: int = MAX_READ_BYTES,
        max_line_chars: int = MAX_LINE_CHARS,
    ) -> LineWindow:
        data = self.read_bytes(path)
        return take_window(
            data.splitlines(keepends=True),
            offset=offset,
            limit=limit,
            max_bytes=max_bytes,
            max_line_chars=max_line_chars,
        )

    def ls(
        self, path: str = ".", *, recursive: bool = False, depth: int | None = None
    ) -> list[FileInfo]:
        key = self.resolve(path)
        with self._lock:
            content = self._files.get(key)
            mtime = self._mtime.get(key)
        if content is not None:
            return [FileInfo(path=key, size=len(_to_bytes(content)), modified_at=mtime)]
        entries = super().ls(key, recursive=recursive)
        if recursive:
            entries = [e for e in entries if not _noise(e.path, key)]
            # StateBackend lists files only; a tree view also wants the
            # directories they imply.
            dirs: set[str] = set()
            for info in entries:
                parent = posixpath.dirname(info.path)
                while parent not in (key, "/") and parent.startswith(key.rstrip("/") + "/"):
                    dirs.add(parent)
                    parent = posixpath.dirname(parent)
            entries = entries + [FileInfo(path=d, is_dir=True) for d in dirs]
            if depth is not None:
                base = 0 if key == "/" else key.count("/")
                entries = [e for e in entries if e.path.count("/") - base <= depth]
        return sorted(entries, key=lambda i: i.path)

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
        timeout_s: float = 20.0,
    ) -> list[FileInfo]:
        key = self.resolve(path)
        prefix = "/" if key == "/" else key + "/"
        with self._lock:
            files = {p: c for p, c in self._files.items() if p.startswith(prefix)}
            mtimes = dict(self._mtime)
        return [
            FileInfo(path=p, size=len(_to_bytes(c)), modified_at=mtimes.get(p))
            for p, c in sorted(files.items())
            if glob_match(p[len(prefix) :], posixpath.basename(p), pattern) and not _noise(p, key)
        ]

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        recursive: bool = True,
        glob_filter: str = "*",
        limit: int | None = None,
    ) -> list[Match]:
        key = self.resolve(path)
        if not self.exists(key):
            raise BackendError("file_not_found", path)
        hits = [
            m
            for m in super().grep(pattern, path=key, recursive=recursive)
            if glob_match(posixpath.basename(m.path), posixpath.basename(m.path), glob_filter)
            # StateBackend recurses from the root whatever it is told.
            and (recursive or m.path == key or posixpath.dirname(m.path) == key)
            and not _noise(m.path, key)
        ]
        return hits[:limit] if limit is not None else hits

    # ------------------------------------------------------------- shell --

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        on_output: Callable[[bytes], None] | None = None,
    ) -> ExecResult:
        raise ExecUnsupportedError(_LABEL)

    def start_background(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        foreground: bool = False,
    ) -> JobStatus:
        raise ExecUnsupportedError(_LABEL)

    def wait_background(self, handle: str, timeout: float) -> JobStatus:
        raise ExecUnsupportedError(_LABEL)

    def read_background(self, handle: str, since: int = 0) -> JobOutput:
        raise ExecUnsupportedError(_LABEL)

    def write_background(self, handle: str, data: bytes, *, close: bool = False) -> None:
        raise ExecUnsupportedError(_LABEL)

    def kill_background(self, handle: str) -> JobStatus:
        raise ExecUnsupportedError(_LABEL)

    def release_background(self, handle: str) -> None:
        raise ExecUnsupportedError(_LABEL)

    def list_background(self) -> list[JobStatus]:
        return []
