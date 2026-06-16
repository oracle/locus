# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``FilesystemBackend`` — rooted on-disk filesystem-as-memory.

Opt-in alternative to :class:`StateBackend`. Lets the agent's
scratchspace persist on real disk inside a caller-supplied root.
Useful for inspecting the agent's notes after a run, sharing a
scratchspace across runs (with the same ``root``), or when the
research surface is too large to keep in memory.

Path safety: every operation resolves the requested path *under the
root* and asserts it stays inside. Symlinks that escape the root are
rejected. Absolute or ``..``-prefixed inputs that would land outside
are surfaced as ``BackendError("invalid_path", ...)``.
"""

from __future__ import annotations

import fnmatch
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from locus.deepagent.backends.protocol import (
    BackendError,
    FileInfo,
    Match,
)


class FilesystemBackend:
    """On-disk backend rooted at a caller-supplied directory.

    The agent sees ``/`` as the root; on disk this maps to ``root``.
    All resolved targets must remain under ``root`` — symlink
    traversal and absolute-path escapes are rejected.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve(strict=False)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path resolution + safety
    # ------------------------------------------------------------------

    def _resolve(self, agent_path: str) -> Path:
        """Map an agent-visible POSIX path under ``self._root``.

        Rejects empty / non-absolute inputs and any resolved target
        that would escape the root (via symlink, ``..``, or other
        path tricks).
        """
        if not agent_path or not agent_path.startswith("/"):
            raise BackendError("invalid_path", agent_path)
        # Strip the leading ``/`` so Path.joinpath doesn't treat it as
        # an absolute path that overrides the root.
        relative = agent_path.lstrip("/")
        candidate = (self._root / relative).resolve(strict=False)
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:  # pragma: no cover — defence in depth
            raise BackendError("invalid_path", agent_path) from exc
        # Reject symlinks that resolve outside the root.
        if candidate.is_symlink():
            target = candidate.resolve(strict=False)
            try:
                target.relative_to(self._root)
            except ValueError as exc:
                raise BackendError("invalid_path", agent_path) from exc
        return candidate

    # ------------------------------------------------------------------
    # Read / write / edit
    # ------------------------------------------------------------------

    def read(self, path: str, *, offset: int = 0, limit: int = 100) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            content = target.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
        lines = content.splitlines()
        offset = max(offset, 0)
        if offset >= len(lines):
            return ""
        end = offset + limit if limit > 0 else len(lines)
        chunk = lines[offset:end]
        return "\n".join(f"{offset + i + 1:6d}\t{ln}" for i, ln in enumerate(chunk))

    def write(self, path: str, contents: str) -> None:
        target = self._resolve(path)
        if target.exists() and target.is_dir():
            raise BackendError("is_directory", path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                contents if isinstance(contents, str) else str(contents), encoding="utf-8"
            )
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            content = target.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
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
        try:
            target.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def exists(self, path: str) -> bool:
        try:
            target = self._resolve(path)
        except BackendError:
            return False
        return target.exists()

    # ------------------------------------------------------------------
    # ls / glob / grep
    # ------------------------------------------------------------------

    def _file_info(self, target: Path, agent_path: str) -> FileInfo:
        is_dir = target.is_dir()
        try:
            stat = target.stat()
            size = None if is_dir else stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        except (FileNotFoundError, PermissionError):
            size = None
            mtime = None
        return FileInfo(path=agent_path, is_dir=is_dir, size=size, modified_at=mtime)

    def _to_agent_path(self, target: Path) -> str:
        rel = target.relative_to(self._root).as_posix()
        return "/" + rel if rel != "." else "/"

    def ls(self, path: str = "/", *, recursive: bool = False) -> list[FileInfo]:
        target = self._resolve(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        out: list[FileInfo] = []
        if target.is_file():
            out.append(self._file_info(target, path))
            return out
        try:
            entries = sorted(target.iterdir() if not recursive else target.rglob("*"))
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
        for entry in entries:
            out.append(self._file_info(entry, self._to_agent_path(entry)))
        out.sort(key=lambda fi: fi.path)
        return out

    def glob(
        self,
        pattern: str,
        *,
        path: str = "/",
        timeout_s: float = 20.0,  # noqa: ARG002 — local FS glob is fast; honored as a contract surface
    ) -> list[FileInfo]:
        target = self._resolve(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        out: list[FileInfo] = []
        # Walk the subtree; match against either the relative-to-`path`
        # path or the full agent path. Mirrors how the StateBackend
        # behaves so the two are interchangeable for tools.
        try:
            walked = list(target.rglob("*"))
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
        for entry in walked:
            if entry.is_dir():
                continue
            agent_path = self._to_agent_path(entry)
            try:
                rel = entry.relative_to(target).as_posix()
            except ValueError:
                rel = agent_path
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(agent_path, pattern):
                out.append(self._file_info(entry, agent_path))
        out.sort(key=lambda fi: fi.path)
        return out

    def grep(
        self,
        pattern: str,
        *,
        path: str = "/",
        recursive: bool = True,
    ) -> list[Match]:
        target = self._resolve(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        regex = re.compile(pattern)
        out: list[Match] = []
        if target.is_file():
            self._grep_one(target, regex, out)
            return out
        iterator = target.rglob("*") if recursive else target.iterdir()
        try:
            entries = sorted(iterator)
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
        for entry in entries:
            if entry.is_file():
                self._grep_one(entry, regex, out)
        out.sort(key=lambda m: (m.path, m.line))
        return out

    def _grep_one(self, target: Path, regex: re.Pattern[str], out: list[Match]) -> None:
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            return
        agent_path = self._to_agent_path(target)
        for i, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                out.append(Match(path=agent_path, line=i, text=line))
