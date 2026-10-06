# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The host as a workspace: files under a root, and the host's shell.

:class:`LocalBackend` is not isolated, and says so in its capabilities: the
shell is the machine the harness runs on, and a command can reach anything
the process can. Files are confined to the root by
:func:`tulip.tools.path_safety.safe_resolve` — ``..``, absolute paths and
symlinks that leave it are refused — but a shell command is not confined by
anything here. Isolation is the job of the sandbox the agent runs in; use
:class:`~tulip.harness.openshell.OpenShellBackend` or another
:class:`~tulip.harness.session.SessionBackend` when it matters.

Every command is a :class:`Shell`: a process in its own process group, with a
thread draining its output into a buffer as it is written. :meth:`exec`
waits on one; a background command is one nobody waits on, read later by
handle. One mechanism for both, so a command a caller stopped waiting on can
be kept running without losing a byte of what it printed.

The buffer keeps the start of the output and the most recent part of it. A
dev server left running for an hour prints more than anyone will read; the
first lines (the port it bound, the error it started with) and the latest
ones are the two worth keeping.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import itertools
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tulip.core.errors import ValidationError
from tulip.deepagent.backends.protocol import BackendError, FileInfo, Match
from tulip.harness.backend import (
    MAX_LINE_CHARS,
    MAX_READ_BYTES,
    SKIP_DIRS,
    BackendCapabilities,
    ExecResult,
    FileStat,
    JobError,
    JobOutput,
    JobStatus,
    LineWindow,
    WorkspaceTextMixin,
    take_window,
)
from tulip.tools.path_safety import safe_resolve


__all__ = ["LocalBackend", "Output", "glob_match"]

#: Bytes from the start of a command's output that are always kept.
HEAD_KEEP = 64_000

#: Bytes of the most recent output kept after the head. Older output between
#: the two is dropped, and a reader is told how much.
TAIL_KEEP = 2_000_000

#: Background commands that may be alive at once. A model that starts a
#: server on every turn would otherwise fill the machine with them.
MAX_BACKGROUND = 16

#: Finished commands kept so their output can still be read. Older ones are
#: forgotten first.
MAX_FINISHED = 32

#: How long a killed process group gets to go quietly before SIGKILL.
KILL_GRACE_SECONDS = 2.0

#: Files larger than this are not hashed by :meth:`LocalBackend.stat`; size
#: and modification time stand in. Hashing a 2 GB log to learn whether it
#: changed costs more than the read it guards.
HASH_LIMIT = 32 * 1024 * 1024

#: A grep walk skips files larger than this: the ceiling is for the generated
#: and the vendored, not for logs.
GREP_MAX_FILE_BYTES = MAX_READ_BYTES * 50


def glob_match(rel: str, name: str, pattern: str) -> bool:
    """fnmatch, plus the reading of ``**/`` that includes the top level.

    ``**/*.py`` means "at any depth", and the top level is a depth. fnmatch
    reads it as "under at least one directory" and misses every top-level
    file — setup.py, conftest.py, the module being looked for.
    """
    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
        return True
    return pattern.startswith("**/") and glob_match(rel, name, pattern[3:])


class Output:
    """A command's output: the head kept whole, then a bounded tail.

    Offsets are absolute byte positions in everything the command ever
    wrote, so a reader that asks for "what is new since 5,120" gets the same
    answer whether or not older output has since been dropped.
    """

    def __init__(self, head_keep: int = HEAD_KEEP, tail_keep: int = TAIL_KEEP) -> None:
        self._head_keep = head_keep
        self._tail_keep = tail_keep
        self._head = bytearray()
        self._tail = bytearray()
        #: Absolute offset of ``_tail[0]``.
        self._tail_start = 0
        self.total = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            room = self._head_keep - len(self._head)
            if room > 0:
                self._head += chunk[:room]
                chunk = chunk[room:]
                # The tail starts where the head ends, and is empty until then.
                self._tail_start = len(self._head)
            if chunk:
                self._tail += chunk
                excess = len(self._tail) - self._tail_keep
                if excess > 0:
                    del self._tail[:excess]
                    self._tail_start += excess
            self.total = self._tail_start + len(self._tail)

    def since(self, offset: int) -> tuple[bytes, int, int]:
        """Bytes written from ``offset`` on, the offset after them, and bytes lost.

        Bytes lost are the ones the reader asked for that had already been
        dropped from the middle; the data says where they were.
        """
        with self._lock:
            offset = max(0, min(offset, self.total))
            head = bytes(self._head[offset:])
            pos = max(offset, len(self._head))
            lost = max(0, self._tail_start - pos)
            pos = max(pos, self._tail_start)
            tail = bytes(self._tail[pos - self._tail_start :])
            total = self.total
        note = f"\n... {lost:,} bytes of output were dropped here ...\n".encode() if lost else b""
        return head + note + tail, total, lost


@dataclass
class Shell:
    """One command, running or finished, and everything it printed."""

    id: str
    command: str
    proc: subprocess.Popen[bytes]
    output: Output = field(default_factory=Output)
    started: float = field(default_factory=time.monotonic)
    #: Started for a caller that waits on it; exempt from the background cap.
    foreground: bool = False
    ended: float | None = None
    on_output: Callable[[bytes], None] | None = None
    _reader: threading.Thread | None = field(default=None, repr=False)

    def drain(self) -> None:
        stream = self.proc.stdout
        if stream is None:  # pragma: no cover - spawn always asks for a pipe
            return
        fd = stream.fileno()
        with contextlib.suppress(OSError, ValueError):
            while True:
                # Straight from the descriptor: returns whatever is there, so
                # output reaches a reader as it is written, not a buffer later.
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                self.output.append(chunk)
                if self.on_output is not None:
                    with contextlib.suppress(Exception):
                        self.on_output(chunk)
        # End of output: the shell and everything that held its stdout are
        # gone, or have closed it. Reap the shell so the exit code is known.
        with contextlib.suppress(OSError):
            self.proc.wait()
        self.ended = time.monotonic()

    @property
    def running(self) -> bool:
        return self.ended is None

    @property
    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    def wait(self, timeout: float) -> bool:
        """Wait for the output to end and the shell to exit. Returns whether it did."""
        if self._reader is not None:
            self._reader.join(timeout)
        return not self.running

    def status(self) -> JobStatus:
        return JobStatus(
            handle=self.id,
            command=self.command,
            running=self.running,
            exit_code=None if self.running else self.proc.returncode,
            elapsed_s=self.elapsed,
            has_stdin=self.proc.stdin is not None,
            pid=self.proc.pid,
        )

    def kill(self) -> None:
        kill_group(self.proc)
        # The reader ends once the group is gone and the pipe closes.
        self.wait(KILL_GRACE_SECONDS)


def kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Stop a command and everything it started.

    The shell's children — a test runner's workers, a dev server, ``sleep``
    in a loop — are in its process group, not its process. Killing only the
    shell leaves them running and holding the output pipe open, which is how
    a timed-out command keeps a caller waiting anyway.
    """
    if not hasattr(os, "killpg"):  # pragma: no cover - Windows has no process groups
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=KILL_GRACE_SECONDS)
    # SIGKILL the group even when the shell itself has exited: a child that
    # ignored SIGTERM is still in it.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


def _kill_all(shells: dict[str, Shell]) -> None:
    """Kill every command still running; run at exit and on :meth:`close`."""
    for shell in list(shells.values()):
        if shell.running:
            kill_group(shell.proc)


class LocalBackend(WorkspaceTextMixin):
    """Files under ``root`` on this machine, and this machine's shell.

    Args:
        root: The workspace directory. Created if missing. Relative paths
            resolve against it; nothing outside it can be read or written
            through the file methods.
        env: Variables added to every command's environment, on top of this
            process's own.
        max_background: How many background commands may run at once.

    Commands still running when the backend is closed, garbage collected or
    the interpreter exits are killed with their process groups. Each command
    runs in its own session, which keeps it out of a terminal's hangup and a
    supervisor's ``kill -- -pgid``, so nothing else would stop them.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
    ) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._env = dict(env or {})
        self._max_background = max_background
        self._shells: dict[str, Shell] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._capabilities = BackendCapabilities(
            isolated=False,
            label="UNISOLATED: host shell",
            has_rg=shutil.which("rg") is not None,
            root=str(self._root),
        )
        self._finalizer = weakref.finalize(self, _kill_all, self._shells)

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def close(self) -> None:
        """Kill every command still running."""
        self._finalizer()

    # ------------------------------------------------------------- paths --

    def _path(self, path: str) -> Path:
        try:
            return safe_resolve(self._root, path or ".")
        except ValidationError as exc:
            raise BackendError("invalid_path", path) from exc

    def resolve(self, path: str) -> str:
        return str(self._path(path))

    # ------------------------------------------------------------- files --

    def read_bytes(self, path: str) -> bytes:
        target = self._path(path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise BackendError("file_not_found", path) from exc
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self._path(path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Opened in place rather than replaced, so the file keeps its
            # mode and owner — an executable script stays executable.
            with target.open("wb") as fh:
                fh.write(data)
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def remove(self, path: str) -> None:
        target = self._path(path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            target.unlink()
        except FileNotFoundError as exc:
            raise BackendError("file_not_found", path) from exc
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def stat(self, path: str) -> FileStat:
        target = self._path(path)
        try:
            info = target.stat()
        except FileNotFoundError as exc:
            raise BackendError("file_not_found", path) from exc
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc
        if target.is_dir():
            return FileStat(size=0, mtime=info.st_mtime, is_dir=True)
        digest = ""
        if info.st_size <= HASH_LIMIT:
            with contextlib.suppress(OSError):
                h = hashlib.sha256()
                with target.open("rb") as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        h.update(block)
                digest = h.hexdigest()
        return FileStat(size=info.st_size, mtime=info.st_mtime, sha256=digest)

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        max_bytes: int = MAX_READ_BYTES,
        max_line_chars: int = MAX_LINE_CHARS,
    ) -> LineWindow:
        target = self._path(path)
        if target.is_dir():
            raise BackendError("is_directory", path)
        try:
            # Streamed line by line, so a 2 GB log costs one window of memory
            # rather than the whole file; only the line count needs the rest.
            with target.open("rb") as fh:
                return take_window(
                    fh,
                    offset=offset,
                    limit=limit,
                    max_bytes=max_bytes,
                    max_line_chars=max_line_chars,
                )
        except FileNotFoundError as exc:
            raise BackendError("file_not_found", path) from exc
        except PermissionError as exc:
            raise BackendError("permission_denied", path) from exc

    def _info(self, target: Path) -> FileInfo | None:
        try:
            st = target.stat()
        except OSError:
            return None
        return FileInfo(
            path=str(target),
            is_dir=target.is_dir(),
            size=None if target.is_dir() else st.st_size,
            modified_at=datetime.fromtimestamp(st.st_mtime, UTC),
        )

    def _walk(self, top: Path, depth: int | None) -> list[Path]:
        """Every entry under ``top``, never descending into :data:`SKIP_DIRS`."""
        out: list[Path] = []
        base = len(top.parts)
        for dirpath, dirnames, filenames in os.walk(top):
            here = Path(dirpath)
            level = len(here.parts) - base
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            out.extend(here / d for d in dirnames)
            out.extend(here / n for n in sorted(filenames))
            if depth is not None and level + 1 >= depth:
                dirnames[:] = []
        return out

    def ls(
        self, path: str = ".", *, recursive: bool = False, depth: int | None = None
    ) -> list[FileInfo]:
        target = self._path(path)
        if not target.exists():
            raise BackendError("file_not_found", path)
        if target.is_file():
            info = self._info(target)
            return [info] if info else []
        if recursive:
            entries = self._walk(target, depth)
        else:
            entries = sorted(target.iterdir())
        infos = [self._info(p) for p in entries]
        return sorted((i for i in infos if i is not None), key=lambda i: i.path)

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
        timeout_s: float = 20.0,
    ) -> list[FileInfo]:
        top = self._path(path)
        if not top.is_dir():
            raise BackendError("file_not_found", path)
        deadline = time.monotonic() + timeout_s
        out: list[FileInfo] = []
        for entry in self._walk(top, None):
            if time.monotonic() > deadline:
                break
            if entry.is_dir():
                continue
            if glob_match(entry.relative_to(top).as_posix(), entry.name, pattern):
                info = self._info(entry)
                if info is not None:
                    out.append(info)
        return out

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        recursive: bool = True,
        glob_filter: str = "*",
        limit: int | None = None,
    ) -> list[Match]:
        rx = re.compile(pattern)
        top = self._path(path)
        if top.is_file():
            targets = [top]
        elif top.is_dir():
            walked = self._walk(top, None if recursive else 1)
            targets = [p for p in walked if fnmatch.fnmatch(p.name, glob_filter) and p.is_file()]
        else:
            raise BackendError("file_not_found", path)
        hits: list[Match] = []
        for f in targets:
            try:
                # Read a line at a time, so size costs time rather than memory.
                if f.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
                with f.open("rb") as fh:
                    for i, raw in enumerate(fh, 1):
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if rx.search(line):
                            hits.append(Match(path=str(f), line=i, text=line))
                            if limit is not None and len(hits) >= limit:
                                return hits
            except OSError:
                continue
        return hits

    # ------------------------------------------------------------- shell --

    def _spawn(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        stdin: bool,
        foreground: bool,
        on_output: Callable[[bytes], None] | None = None,
    ) -> Shell:
        workdir = self._path(cwd) if cwd else self._root
        merged = {**os.environ, **self._env, **(env or {})}
        proc = subprocess.Popen(  # noqa: S602 - the tool is "run a command"; the caller gated it
            command,
            shell=True,
            cwd=workdir,
            env=merged,
            # A command nobody will type into gets no stdin: a child that
            # inherits one either swallows input meant for someone else or
            # blocks on a read nobody will satisfy.
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # One stream, in the order it was written. Stdout then stderr
            # reads as if every warning happened after the run finished.
            stderr=subprocess.STDOUT,
            # Its own process group, so a timeout can kill what it started.
            start_new_session=True,
        )
        shell = Shell(
            id=f"sh{next(self._ids)}",
            command=command,
            proc=proc,
            foreground=foreground,
            on_output=on_output,
        )
        shell._reader = threading.Thread(  # noqa: SLF001 - the shell's own reader, set once
            target=shell.drain, name=f"tulip-harness-{shell.id}", daemon=True
        )
        return shell

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        on_output: Callable[[bytes], None] | None = None,
    ) -> ExecResult:
        shell = self._spawn(
            command, cwd=cwd, env=env, stdin=False, foreground=True, on_output=on_output
        )
        assert shell._reader is not None  # noqa: S101, SLF001 - set by _spawn
        shell._reader.start()  # noqa: SLF001
        finished = shell.wait(max(0.0, timeout))
        if not finished:
            shell.kill()
        data, _, lost = shell.output.since(0)
        return ExecResult(
            exit_code=shell.proc.returncode if finished else None,
            output=data,
            truncated=lost > 0,
            timed_out=not finished,
            duration_s=shell.elapsed,
        )

    def _get(self, handle: str) -> Shell:
        with self._lock:
            shell = self._shells.get(handle.strip())
        if shell is None:
            raise JobError(self._unknown(handle))
        return shell

    def _unknown(self, handle: str) -> str:
        known = self.list_background()
        if not known:
            return f"no background command {handle!r} — none have been started"
        lines = "\n".join(f"  {s.describe()}" for s in known)
        return f"no background command {handle!r}. These exist:\n{lines}"

    def start_background(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        foreground: bool = False,
    ) -> JobStatus:
        if not foreground:
            alive = [s for s in self._snapshot() if s.running and not s.foreground]
            if len(alive) >= self._max_background:
                raise JobError(
                    f"{self._max_background} background commands are already running — "
                    "stop one with kill_shell first"
                )
        shell = self._spawn(command, cwd=cwd, env=env, stdin=not foreground, foreground=foreground)
        with self._lock:
            self._shells[shell.id] = shell
            self._forget_old_locked()
        assert shell._reader is not None  # noqa: S101, SLF001 - set by _spawn
        shell._reader.start()  # noqa: SLF001
        return shell.status()

    def _snapshot(self) -> list[Shell]:
        with self._lock:
            return list(self._shells.values())

    def _forget_old_locked(self) -> None:
        finished = [s for s in self._shells.values() if not s.running]
        for shell in finished[: max(0, len(finished) - MAX_FINISHED)]:
            self._shells.pop(shell.id, None)

    def wait_background(self, handle: str, timeout: float) -> JobStatus:
        shell = self._get(handle)
        shell.wait(max(0.0, timeout))
        return shell.status()

    def read_background(self, handle: str, since: int = 0) -> JobOutput:
        shell = self._get(handle)
        data, offset, lost = shell.output.since(since)
        return JobOutput(status=shell.status(), data=data, offset=offset, lost=lost)

    def write_background(self, handle: str, data: bytes, *, close: bool = False) -> None:
        shell = self._get(handle)
        if not shell.running:
            raise JobError(f"{shell.status().describe()} — it cannot take input")
        stdin = shell.proc.stdin
        if stdin is None:
            raise JobError(
                f"{shell.id} was not started as a background command, so it has no stdin"
            )
        try:
            if data:
                stdin.write(data)
                stdin.flush()
            if close:
                stdin.close()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise JobError(
                f"{shell.id} is not reading its input any more ({type(exc).__name__})"
            ) from exc

    def kill_background(self, handle: str) -> JobStatus:
        shell = self._get(handle)
        if shell.running:
            shell.kill()
        return shell.status()

    def release_background(self, handle: str) -> None:
        with self._lock:
            self._shells.pop(handle.strip(), None)

    def list_background(self) -> list[JobStatus]:
        return [s.status() for s in self._snapshot()]
