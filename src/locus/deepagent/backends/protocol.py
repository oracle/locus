# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Backend protocol for the deepagent filesystem-as-memory toolset.

A backend is an addressable byte/text store the agent can use as a
scratchspace through six file-like tools (``write_file``, ``read_file``,
``ls``, ``edit_file``, ``glob``, ``grep``). The protocol is small and
typed so a backend can be a Python dict (``StateBackend``), a real
directory on disk (``FilesystemBackend``), or any other store that
maps absolute paths to bytes (object storage, sandbox VM, …).

Error contract: backends raise :class:`BackendError` with one of four
standardized codes (``file_not_found``, ``permission_denied``,
``is_directory``, ``invalid_path``). The deepagent FS tools catch
these and return the code as the tool's string output, so the LLM
sees a stable failure label and prompts can teach it to retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable


ErrorCode = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]


class BackendError(Exception):
    """Raised by backend ops to signal a standardized failure.

    The ``code`` is one of four LLM-friendly labels matched
    byte-for-byte against the deepagents catalogue so prompts that
    handle errors transfer cleanly between the two stacks.
    """

    def __init__(self, code: ErrorCode, path: str = "") -> None:
        self.code: ErrorCode = code
        self.path = path
        super().__init__(f"{code}: {path}" if path else code)


@dataclass(frozen=True)
class FileInfo:
    """Best-effort metadata for one path returned by ``ls`` / ``glob``."""

    path: str
    is_dir: bool = False
    size: int | None = None
    modified_at: datetime | None = None


@dataclass(frozen=True)
class Match:
    """One ``grep`` hit — single-line text plus 1-based line number."""

    path: str
    line: int
    text: str


@runtime_checkable
class BackendProtocol(Protocol):
    """Storage substrate the deepagent FS tools delegate to.

    All paths are absolute, ``/``-rooted, POSIX-style strings (e.g.
    ``"/notes/draft.md"``). Backends that target a real filesystem
    map them under a configured root with traversal guards.
    """

    def read(self, path: str, *, offset: int = 0, limit: int = 100) -> str:
        """Return up to ``limit`` lines starting at ``offset`` (1-based).

        ``offset=0`` means "from the beginning". Returns an empty string
        when the file is empty or ``offset`` is past the last line.
        Raises ``BackendError("file_not_found", path)`` if no entry
        exists, ``BackendError("is_directory", path)`` if ``path``
        names a directory.
        """
        ...

    def write(self, path: str, contents: str) -> None:
        """Create or overwrite ``path``. Parent directories are
        implicitly created (no ``mkdir -p`` step needed)."""
        ...

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        """In-place replace ``old_str`` with ``new_str``.

        Mirrors deepagents' contract: the match must be unique. Raises
        ``BackendError("file_not_found", path)`` if the file is
        missing; raises a ``ValueError`` when ``old_str`` doesn't
        match exactly once (zero or multiple matches both reject so
        the agent can retry with a more specific snippet).
        """
        ...

    def exists(self, path: str) -> bool: ...

    def ls(self, path: str = "/", *, recursive: bool = False) -> list[FileInfo]: ...

    def glob(
        self,
        pattern: str,
        *,
        path: str = "/",
        timeout_s: float = 20.0,
    ) -> list[FileInfo]: ...

    def grep(
        self,
        pattern: str,
        *,
        path: str = "/",
        recursive: bool = True,
    ) -> list[Match]: ...
