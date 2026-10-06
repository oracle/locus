# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The workspace a coding harness works in, as one typed surface.

A coding agent needs two things from the place it works: files it can read
and change, and a shell. Where that place is varies. It can be the host
(:class:`~tulip.harness.local.LocalBackend`), a dict in memory for tests
(:class:`~tulip.harness.state.MemoryBackend`), any sandbox that can run a
command and move a file (:class:`~tulip.harness.session.SessionBackend`), or
an NVIDIA OpenShell sandbox
(:class:`~tulip.harness.openshell.OpenShellBackend`). The tools in
:mod:`tulip.harness.tools` are written once against :class:`WorkspaceBackend`
and behave the same on all of them.

:class:`WorkspaceBackend` extends the deepagent
:class:`~tulip.deepagent.backends.protocol.BackendProtocol`, so a workspace
also serves the deepagent filesystem tools. Two things differ:

- **Paths are the workspace's own.** A path is relative to
  :attr:`BackendCapabilities.root`, or absolute inside it, and it names the
  same file the workspace's shell sees. A model that reads ``src/app.py`` and
  then runs ``pytest src/app.py`` must be talking about one file, so the file
  tools cannot have a private namespace the shell does not share. A path that
  resolves outside the root raises ``BackendError("invalid_path")``.
- **A recursive walk skips the noise.** ``ls``, ``glob`` and ``grep`` never
  descend into :data:`SKIP_DIRS` (``.git``, ``node_modules`` and the like).
  On a real repository those hold most of the files and none of the answers.

The shell is :meth:`WorkspaceBackend.exec` for a command the caller waits on,
and the ``*_background`` methods for one it does not. They are kept as plain
methods rather than a separate object so that a remote backend can implement
all of them with nothing more than ``exec``: a job is a process group, a file
of output and, when it takes input, a FIFO, all under a state directory in
the workspace (see :mod:`tulip.harness.session`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tulip.deepagent.backends.protocol import (
    BackendError,
    BackendProtocol,
    FileInfo,
    Match,
    replace_in_file,
)


__all__ = [
    "MAX_LINE_CHARS",
    "MAX_READ_BYTES",
    "SKIP_DIRS",
    "BackendCapabilities",
    "BackendError",
    "ExecResult",
    "ExecUnsupportedError",
    "FileStat",
    "JobError",
    "JobOutput",
    "JobStatus",
    "LineWindow",
    "WorkspaceBackend",
    "WorkspaceTextMixin",
    "take_window",
]

#: What one windowed read returns at most, in bytes of formatted lines. A
#: model that pulls a 40 MB log into context has already lost, and the
#: truncation notice is more useful to it than the rest of the bytes. It caps
#: what one call returns, not which files can be read.
MAX_READ_BYTES = 200_000

#: A line longer than this is cut, with its length noted. One minified bundle
#: line would otherwise spend the whole window.
MAX_LINE_CHARS = 2_000

#: Directories never worth walking. Saves minutes on a real repository.
SKIP_DIRS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}
)


class ExecUnsupportedError(BackendError):
    """The backend has no shell — :class:`~tulip.harness.state.MemoryBackend`.

    Raised rather than returned so a caller that built shell tools over a
    file-only backend finds out on the first call, not from a model that
    concludes the command printed nothing.
    """

    def __init__(self, label: str) -> None:
        super().__init__("permission_denied", "")
        self.label = label

    def __str__(self) -> str:
        return f"the {self.label} workspace has no shell: it stores files only"


class JobError(RuntimeError):
    """A background command could not be started, found or written to.

    The message is written for the model: it says what happened and what to
    do instead.
    """


@dataclass(frozen=True)
class BackendCapabilities:
    """What a workspace is, for the prompt, the gate and the evidence.

    Attributes:
        isolated: Whether commands run somewhere other than the host the
            harness runs on. A policy can refuse unisolated execution
            outright; the prompt says it either way.
        label: Names the workspace in evidence records and in the prompt
            (``"UNISOLATED: host shell"``, ``"openshell:team/sbx-1"``).
        has_rg: Whether ripgrep is on the workspace's ``PATH``. With it,
            ``glob`` and ``grep`` honour ``.gitignore`` and stop early.
        root: The directory relative paths resolve against, in the
            workspace's own namespace.
        can_exec: Whether :meth:`WorkspaceBackend.exec` and the background
            methods work at all.
    """

    isolated: bool
    label: str
    has_rg: bool
    root: str
    can_exec: bool = True


@dataclass(frozen=True)
class FileStat:
    """A file as it is now: enough to tell whether it changed.

    ``sha256`` is empty when the backend did not hash the file — it is a
    directory, or larger than the backend is willing to read just to stat
    it. Comparisons then fall back to size and modification time.
    """

    size: int
    mtime: float
    sha256: str = ""
    is_dir: bool = False


@dataclass(frozen=True)
class ExecResult:
    """What a command did, for a caller that waited on it.

    Attributes:
        exit_code: The command's exit status, or ``None`` when it did not
            finish (timed out and was killed, or the transport lost it).
        output: Standard output and standard error as one stream, in the
            order they were written. Long output keeps its start and its most
            recent part; ``truncated`` says the middle went.
        truncated: Whether bytes were dropped from ``output``.
        timed_out: Whether the command ran out of time.
        duration_s: Wall-clock seconds from start to end.
    """

    exit_code: int | None
    output: bytes
    truncated: bool = False
    timed_out: bool = False
    duration_s: float = 0.0


@dataclass(frozen=True)
class JobStatus:
    """A background command, running or finished."""

    handle: str
    command: str
    running: bool
    exit_code: int | None = None
    elapsed_s: float = 0.0
    has_stdin: bool = False
    pid: int | None = None

    def describe(self) -> str:
        """One line: what it is, and whether it is still going."""
        if self.running:
            return f"{self.handle} is running ({self.elapsed_s:.0f}s): {self.command}"
        return f"{self.handle} exited {self.exit_code} after {self.elapsed_s:.0f}s: {self.command}"


@dataclass(frozen=True)
class JobOutput:
    """What a background command printed from an offset on.

    ``offset`` is where the next read should start. Offsets are absolute
    byte positions in everything the command ever wrote, so they stay valid
    when older output has been dropped; ``lost`` counts the bytes asked for
    that were already gone, and ``data`` says where they were.
    """

    status: JobStatus
    data: bytes
    offset: int
    lost: int = 0


@dataclass(frozen=True)
class LineWindow:
    """A window of a text file, as numbered lines.

    ``lines`` are already formatted the way a model reads a file — the
    1-based line number right-aligned in six columns, a tab, the text — with
    over-long lines cut and their length noted. ``total`` is the file's line
    count, so a caller can say how much is left.
    """

    lines: tuple[str, ...]
    total: int
    first: int = 0
    full: bool = field(default=False, compare=False)


def take_window(
    lines: Iterable[bytes | tuple[bytes, int]],
    *,
    offset: int,
    limit: int,
    max_bytes: int = MAX_READ_BYTES,
    max_line_chars: int = MAX_LINE_CHARS,
    first_index: int = 0,
) -> LineWindow:
    """Number and cut the lines of a window, counting the rest.

    ``lines`` yields raw lines, line ending included or not, or
    ``(prefix, length)`` pairs for a line the backend already cut — a remote
    backend does not ship a megabyte-long line just to report its length.
    Iterating to the end is how the total is known; a backend that already
    knows it (a remote one that ran ``awk`` for it) passes only the window
    and replaces the total. ``first_index`` is the 0-based index of the first
    line yielded.
    """
    offset = max(0, offset)
    limit = max(1, limit)
    window: list[str] = []
    used = 0
    total = first_index
    full = False
    for index, raw in enumerate(lines, start=first_index):
        total = index + 1
        if index < offset or full:
            continue
        if isinstance(raw, tuple):
            prefix, length = raw
            line = prefix.decode("utf-8", errors="replace").rstrip("\r\n")
        else:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            length = len(line)
        if length > max_line_chars:
            line = f"{line[:max_line_chars]} … [line is {length:,} characters]"
        entry = f"{index + 1:6d}\t{line}"
        if used + len(entry) > max_bytes and window:
            full = True
            continue
        window.append(entry)
        used += len(entry) + 1
        full = len(window) >= limit
    return LineWindow(lines=tuple(window), total=total, first=offset, full=full)


@runtime_checkable
class WorkspaceBackend(BackendProtocol, Protocol):
    """Files and a shell, wherever the workspace is.

    Every method is synchronous: tool bodies run on a worker thread, and a
    backend that talks to a remote sandbox blocks on its transport anyway.
    Errors on files raise :class:`BackendError` with the deepagent codes;
    errors on background commands raise :class:`JobError`.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        """What this workspace is. Cheap to read; may be computed once."""
        ...

    # ------------------------------------------------------------- files --

    def resolve(self, path: str) -> str:
        """``path`` as an absolute path inside the root, or ``invalid_path``."""
        ...

    def read_bytes(self, path: str) -> bytes:
        """The file's bytes, exactly as stored."""
        ...

    def write_bytes(self, path: str, data: bytes) -> None:
        """Create or replace a file, creating parent directories.

        An existing file keeps its permissions: a script that was executable
        is still executable after an edit.
        """
        ...

    def remove(self, path: str) -> None:
        """Delete a file. A directory raises ``is_directory``."""
        ...

    def stat(self, path: str) -> FileStat:
        """The file's size, modification time and content hash."""
        ...

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        max_bytes: int = MAX_READ_BYTES,
        max_line_chars: int = MAX_LINE_CHARS,
    ) -> LineWindow:
        """A numbered window of a text file, without loading all of it."""
        ...

    def ls(
        self, path: str = ".", *, recursive: bool = False, depth: int | None = None
    ) -> list[FileInfo]:
        """Entries under ``path``; ``depth`` bounds a recursive listing."""
        ...

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
        timeout_s: float = 20.0,
    ) -> list[FileInfo]:
        """Files under ``path`` whose relative path or name matches."""
        ...

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        recursive: bool = True,
        glob_filter: str = "*",
        limit: int | None = None,
    ) -> list[Match]:
        """Lines matching a regular expression; stops after ``limit`` hits."""
        ...

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
        """Run a shell command line and wait for it, at most ``timeout`` seconds.

        Standard input is closed, so a command that waits for input fails
        fast instead of hanging. At the timeout the command and everything it
        started are killed. ``on_output`` receives output as it arrives, when
        the transport can stream it, and all of it at the end otherwise.
        """
        ...

    def start_background(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        foreground: bool = False,
    ) -> JobStatus:
        """Start a command nobody waits on, and return its handle.

        A background command gets a standard input that
        :meth:`write_background` writes to. ``foreground=True`` is for a
        caller that is about to wait on it with :meth:`wait_background`: it
        gets no standard input, and it does not count against the cap on
        background commands, which exists for the ones nobody is watching.
        """
        ...

    def wait_background(self, handle: str, timeout: float) -> JobStatus:
        """Wait up to ``timeout`` seconds for a command to end."""
        ...

    def read_background(self, handle: str, since: int = 0) -> JobOutput:
        """What the command printed from byte ``since`` on."""
        ...

    def write_background(self, handle: str, data: bytes, *, close: bool = False) -> None:
        """Send ``data`` to the command's input; ``close`` ends the input."""
        ...

    def kill_background(self, handle: str) -> JobStatus:
        """Stop the command and everything it started."""
        ...

    def release_background(self, handle: str) -> None:
        """Forget a finished command; its handle stops working."""
        ...

    def list_background(self) -> list[JobStatus]:
        """Every command that still has a handle, oldest first."""
        ...


class WorkspaceTextMixin:
    """The deepagent text operations, written once over the byte operations.

    A workspace backend implements bytes, stat and windows; this gives it the
    ``read`` / ``write`` / ``edit`` / ``exists`` the deepagent filesystem
    tools call, with the same contract they have on the deepagent backends.
    """

    def read_bytes(self, path: str) -> bytes:  # pragma: no cover - provided by the backend
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def stat(self, path: str) -> FileStat:  # pragma: no cover
        raise NotImplementedError

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        max_bytes: int = MAX_READ_BYTES,
        max_line_chars: int = MAX_LINE_CHARS,
    ) -> LineWindow:  # pragma: no cover
        raise NotImplementedError

    def read(self, path: str, *, offset: int = 0, limit: int = 100) -> str:
        return "\n".join(self.read_lines(path, offset=offset, limit=limit).lines)

    def write(self, path: str, contents: str) -> None:
        self.write_bytes(path, contents.encode("utf-8"))

    def edit(self, path: str, old_str: str, new_str: str) -> None:
        content = self.read_bytes(path).decode("utf-8", errors="replace")
        self.write_bytes(path, replace_in_file(path, content, old_str, new_str).encode("utf-8"))

    def exists(self, path: str) -> bool:
        try:
            self.stat(path)
        except BackendError:
            return False
        return True
