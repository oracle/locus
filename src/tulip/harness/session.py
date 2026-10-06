# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Any sandbox that can run a command and move a file, as a workspace.

Sandboxes differ in almost everything — how they start, how they
authenticate, what their SDK calls things — but every one of them can run a
command line and copy a file in and out. :class:`SessionLike` is exactly
that, and nothing more, so a bring-your-own sandbox plugs in with a
three-method adapter. :class:`SessionBackend` builds the whole
:class:`~tulip.harness.backend.WorkspaceBackend` on it:

- **Files** move with ``upload_file`` / ``download_file``. Everything else
  about them — stat, windows of a long file, listings, search — is a short
  POSIX shell script run with ``exec``, so a 2 GB log is windowed where it
  lives rather than copied out to show a page of it.
- **Commands** run through ``exec`` with the workspace root as the working
  directory and standard input closed.
- **Background commands** are a process group the sandbox owns, started with
  ``setsid`` and recorded under a per-backend state directory: the group id,
  the output as a file, the exit status when it ends, and — for a command
  that takes input — a FIFO held open by a keeper process, so the command
  sees end of input only when the caller closes it. Every operation on one is
  a single ``exec``; nothing has to stay connected while it runs.

The scripts need what any Linux image has: ``sh``, coreutils or busybox
(``stat -c``, ``tail``, ``head``, ``base64``, ``timeout``, ``setsid``,
``mkfifo``, ``sha256sum``), ``find``, ``grep`` and ``awk``. Search uses
ripgrep when the sandbox has it, and ``grep -P`` (Perl-compatible, the
closest to Python's ``re``) otherwise, falling back to ``grep -E``.

Paths are checked against the root lexically — ``..`` cannot climb out —
but symlinks are not resolved on this side. A link inside the sandbox that
points elsewhere in the sandbox is the sandbox's business: everything the
backend can reach is already inside the isolation boundary.
"""

from __future__ import annotations

import base64
import math
import posixpath
import shlex
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from tulip.deepagent.backends.protocol import BackendError, ErrorCode, FileInfo, Match
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
from tulip.harness.local import HEAD_KEEP, MAX_BACKGROUND, TAIL_KEEP, glob_match


__all__ = ["SessionBackend", "SessionLike"]

#: How long an internal operation (stat, a listing, a status check) may take.
OP_TIMEOUT = 60.0

#: Files larger than this are not hashed by :meth:`SessionBackend.stat`.
HASH_LIMIT = 32 * 1024 * 1024

#: Marks a line ``read_lines`` cut on the sandbox side; followed by its length.
_CUT = "\x01"

_q = shlex.quote

# Exit codes the scripts below use to say what went wrong.
_NOT_FOUND = 20
_IS_DIR = 21
_DENIED = 22
_CODES: dict[int, ErrorCode] = {_NOT_FOUND: "file_not_found", _IS_DIR: "is_directory"}


@runtime_checkable
class SessionLike(Protocol):
    """The three things a sandbox must do to be a workspace.

    ``exec`` runs ``command`` with a POSIX shell and returns its exit status
    (``None`` when it did not finish) and its standard output and standard
    error as one stream. It should stop the command at ``timeout`` seconds.
    ``upload_file`` creates or replaces a file — the parent directory exists
    by then — and ``download_file`` returns one; both raise on failure, with
    any exception.
    """

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]: ...

    def upload_file(self, path: str, data: bytes) -> None: ...

    def download_file(self, path: str) -> bytes: ...


@dataclass
class _Job:
    handle: str
    command: str
    directory: str
    has_stdin: bool
    foreground: bool
    started: float
    pid: int | None = None
    ended: float | None = None
    exit_code: int | None = None

    def status(self, running: bool, exit_code: int | None) -> JobStatus:
        if not running and self.ended is None:
            self.ended = time.monotonic()
            self.exit_code = exit_code
        end = self.ended if self.ended is not None else time.monotonic()
        return JobStatus(
            handle=self.handle,
            command=self.command,
            running=running,
            exit_code=None if running else self.exit_code,
            elapsed_s=end - self.started,
            has_stdin=self.has_stdin,
            pid=self.pid,
        )


def _status_script(directory: str) -> str:
    """Prints ``r -`` while the job runs, ``x <code>`` once it has ended.

    A job whose process group is gone without writing its status (killed
    from outside) has ended too; its code is unknown, reported as -1. A
    finished job's keeper is stopped here, so it does not outlive the job.
    """
    d = _q(directory)
    return (
        f'd={d}; if [ -e "$d/exit" ]; then printf "x %s\\n" "$(cat "$d/exit")"; '
        f'elif [ -s "$d/pid" ] && ! kill -0 "$(cat "$d/pid")" 2>/dev/null; then echo "x -1"; '
        f'else echo "r -"; fi; '
        f'if [ -e "$d/exit" ] && [ -s "$d/keeper" ]; then '
        f'kill "$(cat "$d/keeper")" 2>/dev/null; rm -f "$d/keeper"; fi\n'
    )


def _parse_status(line: str) -> tuple[bool, int | None]:
    state, _, code = line.strip().partition(" ")
    if state != "x":
        return True, None
    try:
        return False, int(code)
    except ValueError:
        return False, -1


def _cap(data: bytes) -> tuple[bytes, bool]:
    """Keep the head and the most recent tail of a command's output."""
    if len(data) <= HEAD_KEEP + TAIL_KEEP:
        return data, False
    lost = len(data) - HEAD_KEEP - TAIL_KEEP
    note = f"\n... {lost:,} bytes of output were dropped here ...\n".encode()
    return data[:HEAD_KEEP] + note + data[-TAIL_KEEP:], True


class SessionBackend(WorkspaceTextMixin):
    """A :class:`SessionLike` sandbox as a workspace.

    Args:
        session: The sandbox, adapted to :class:`SessionLike`.
        root: The workspace directory inside the sandbox.
        label: Names the workspace in evidence and in the prompt.
        isolated: Whether the sandbox is isolated from the host the harness
            runs on. Leave it ``True`` for a real sandbox; a session that
            runs commands on the host itself must say ``False``.
        state_dir: Where background commands keep their state, inside the
            sandbox. Each backend uses its own subdirectory.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
    """

    def __init__(
        self,
        session: SessionLike,
        *,
        root: str = "/workspace",
        label: str = "session",
        isolated: bool = True,
        state_dir: str = "/tmp/.tulip-harness",  # noqa: S108 - inside the sandbox, per-backend subdir
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
    ) -> None:
        if not root.startswith("/"):
            raise ValueError(f"root must be an absolute path in the sandbox, got {root!r}")
        self._session = session
        self._root = posixpath.normpath(root)
        self._label = label
        self._isolated = isolated
        self._jobs_dir = posixpath.join(state_dir, uuid.uuid4().hex[:12])
        self._env = dict(env or {})
        self._max_background = max_background
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._has_rg: bool | None = None

    @property
    def session(self) -> SessionLike:
        return self._session

    @property
    def capabilities(self) -> BackendCapabilities:
        if self._has_rg is None:
            code, _ = self._run("command -v rg >/dev/null 2>&1")
            self._has_rg = code == 0
        return BackendCapabilities(
            isolated=self._isolated, label=self._label, has_rg=self._has_rg, root=self._root
        )

    # ---------------------------------------------------------- transport --

    def _run(self, script: str, timeout: float = OP_TIMEOUT) -> tuple[int | None, bytes]:
        """Run one of this module's own scripts."""
        return self._session.exec(script, timeout=timeout)

    def _exec_script(
        self,
        script: str,
        timeout: float,
        on_output: Callable[[bytes], None] | None,
    ) -> tuple[int | None, bytes]:
        """Run a caller's command. A streaming transport overrides this."""
        code, output = self._session.exec(script, timeout=timeout)
        if on_output is not None and output:
            on_output(output)
        return code, output

    def _fail(self, code: int | None, path: str) -> BackendError:
        return BackendError(_CODES.get(code or 0, "permission_denied"), path)

    # ------------------------------------------------------------- paths --

    def resolve(self, path: str) -> str:
        if not isinstance(path, str) or "\0" in path:
            raise BackendError("invalid_path", str(path))
        joined = posixpath.normpath(posixpath.join(self._root, path or "."))
        if self._root in ("/", joined) or joined.startswith(self._root + "/"):
            return joined
        raise BackendError("invalid_path", path)

    # ------------------------------------------------------------- files --

    def _kind(self, target: str) -> str:
        code, out = self._run(
            f'p={_q(target)}; if [ -d "$p" ]; then echo d; elif [ ! -e "$p" ]; then echo n; '
            'elif [ ! -r "$p" ]; then echo x; else echo f; fi'
        )
        return out.decode(errors="replace").strip() if code == 0 else "x"

    def read_bytes(self, path: str) -> bytes:
        target = self.resolve(path)
        try:
            return self._session.download_file(target)
        except Exception as exc:  # noqa: BLE001 - a session raises its own types; classified below
            kinds: dict[str, ErrorCode] = {"d": "is_directory", "n": "file_not_found"}
            raise BackendError(kinds.get(self._kind(target), "permission_denied"), path) from exc

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.resolve(path)
        code, _ = self._run(
            f'p={_q(target)}; [ -d "$p" ] && exit {_IS_DIR}; '
            f"mkdir -p {_q(posixpath.dirname(target))} || exit {_DENIED}"
        )
        if code != 0:
            raise self._fail(code, path)
        try:
            self._session.upload_file(target, data)
        except Exception as exc:  # noqa: BLE001 - a session raises its own types
            raise BackendError("permission_denied", path) from exc

    def remove(self, path: str) -> None:
        target = self.resolve(path)
        code, _ = self._run(
            f'p={_q(target)}; if [ -d "$p" ]; then exit {_IS_DIR}; '
            f'elif [ -e "$p" ] || [ -L "$p" ]; then rm -f "$p" || exit {_DENIED}; '
            f"else exit {_NOT_FOUND}; fi"
        )
        if code != 0:
            raise self._fail(code, path)

    def stat(self, path: str) -> FileStat:
        target = self.resolve(path)
        code, out = self._run(
            f'p={_q(target)}; if [ -d "$p" ]; then echo d; exit 0; fi; '
            'if [ ! -e "$p" ]; then echo n; exit 0; fi; '
            's=$(stat -c %s "$p" 2>/dev/null || wc -c < "$p"); '
            'm=$(stat -c %Y "$p" 2>/dev/null || echo 0); h=; '
            f'if [ "$s" -le {HASH_LIMIT} ]; then '
            'h=$( (sha256sum "$p" 2>/dev/null || shasum -a 256 "$p" 2>/dev/null) | cut -d" " -f1); fi; '
            'echo "f $s $m ${h:--}"'
        )
        fields = out.decode(errors="replace").split()
        if code != 0 or not fields:
            raise BackendError("permission_denied", path)
        if fields[0] == "d":
            return FileStat(size=0, mtime=0.0, is_dir=True)
        if fields[0] != "f" or len(fields) < 4:
            raise BackendError("file_not_found", path)
        digest = "" if fields[3] == "-" else fields[3]
        return FileStat(size=int(fields[1]), mtime=float(fields[2]), sha256=digest)

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        max_bytes: int = MAX_READ_BYTES,
        max_line_chars: int = MAX_LINE_CHARS,
    ) -> LineWindow:
        target = self.resolve(path)
        offset, limit = max(0, offset), max(1, limit)
        # The count and the window in one round trip. Lines longer than the
        # cut are shortened in the sandbox, with their length after a marker,
        # so a minified bundle costs a page of transfer rather than megabytes.
        code, out = self._run(
            f'p={_q(target)}; [ -d "$p" ] && exit {_IS_DIR}; [ -e "$p" ] || exit {_NOT_FOUND}; '
            f'[ -r "$p" ] || exit {_DENIED}; awk "END{{print NR}}" "$p"; '
            f"awk -v s={offset} -v e={offset + limit} -v m={max_line_chars} "
            '\'NR>s && NR<=e { if (length($0) > m) printf "%s\\001%d\\n", substr($0, 1, m), '
            'length($0); else print }\' "$p"'
        )
        if code != 0:
            raise self._fail(code, path)
        first, _, body = out.partition(b"\n")
        lines: list[bytes | tuple[bytes, int]] = []
        for raw in body.splitlines():
            prefix, cut, length = raw.rpartition(_CUT.encode())
            lines.append((prefix, int(length)) if cut and length.isdigit() else raw)
        window = take_window(
            lines,
            offset=offset,
            limit=limit,
            max_bytes=max_bytes,
            max_line_chars=max_line_chars,
            first_index=offset,
        )
        total = int(first.strip() or b"0")
        return LineWindow(lines=window.lines, total=total, first=offset, full=window.full)

    def _listing(self, target: str, *, recursive: bool, depth: int | None) -> list[FileInfo]:
        prune = ""
        if recursive:
            names = " -o ".join(f"-name {_q(d)}" for d in sorted(SKIP_DIRS))
            prune = f"\\( -type d \\( {names} \\) -prune \\) -o"
        max_depth = "" if recursive and depth is None else f"-maxdepth {depth or 1}"
        code, out = self._run(
            f'p={_q(target)}; [ -e "$p" ] || exit {_NOT_FOUND}; '
            f'if [ -d "$p" ]; then find "$p" -mindepth 1 {max_depth} {prune} '
            "-exec stat -c '%F|%s|%Y|%n' {} +; "
            "else stat -c '%F|%s|%Y|%n' \"$p\"; fi"
        )
        if code == _NOT_FOUND:
            raise BackendError("file_not_found", target)
        infos: list[FileInfo] = []
        for raw in out.decode(errors="replace").splitlines():
            kind, size, mtime, name = (raw.split("|", 3) + ["", "", "", ""])[:4]
            if not name.startswith("/"):
                continue
            is_dir = kind.startswith("directory")
            infos.append(
                FileInfo(
                    path=name,
                    is_dir=is_dir,
                    size=None if is_dir or not size.isdigit() else int(size),
                    modified_at=(
                        datetime.fromtimestamp(int(mtime), UTC) if mtime.isdigit() else None
                    ),
                )
            )
        return sorted(infos, key=lambda i: i.path)

    def ls(
        self, path: str = ".", *, recursive: bool = False, depth: int | None = None
    ) -> list[FileInfo]:
        return self._listing(self.resolve(path), recursive=recursive, depth=depth)

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
        timeout_s: float = 20.0,
    ) -> list[FileInfo]:
        top = self.resolve(path)
        prefix = top.rstrip("/") + "/"
        return [
            info
            for info in self._listing(top, recursive=True, depth=None)
            if not info.is_dir
            and glob_match(info.path.removeprefix(prefix), posixpath.basename(info.path), pattern)
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
        target = self.resolve(path)
        excludes = " ".join(f"--exclude-dir={_q(d)}" for d in sorted(SKIP_DIRS))
        include = f"--include={_q(glob_filter)}" if glob_filter and glob_filter != "*" else ""
        # One directory level without -r: its entries as arguments, and
        # -d skip so the subdirectories among them are not errors.
        where = '-r "$p"' if recursive else '-d skip $( [ -d "$p" ] && echo "$p"/* || echo "$p" )'
        head = f"| head -n {limit}" if limit else ""
        script = (
            f'p={_q(target)}; [ -e "$p" ] || exit {_NOT_FOUND}; '
            # -P reads a pattern the way Python's re does; a grep without it
            # (busybox) gets the extended syntax instead.
            "if printf x | grep -P x >/dev/null 2>&1; then m=-P; else m=-E; fi; "
            f'grep "$m" -nIH {excludes} {include} -e {_q(pattern)} {where} 2>/dev/null {head}'
        )
        code, out = self._run(script)
        if code == _NOT_FOUND:
            raise BackendError("file_not_found", path)
        hits: list[Match] = []
        for raw in out.decode("utf-8", errors="replace").splitlines():
            name, _, rest = raw.partition(":")
            number, _, text = rest.partition(":")
            if name and number.isdigit():
                hits.append(Match(path=name, line=int(number), text=text))
        return hits

    # ------------------------------------------------------------- shell --

    def _prelude(self, cwd: str | None, env: Mapping[str, str] | None) -> str:
        workdir = self.resolve(cwd) if cwd else self._root
        exports = "".join(f"export {k}={_q(v)}\n" for k, v in {**self._env, **(env or {})}.items())
        return f"cd {_q(workdir)} || exit 1\nexec < /dev/null\n{exports}"

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        on_output: Callable[[bytes], None] | None = None,
    ) -> ExecResult:
        started = time.monotonic()
        code, output = self._exec_script(self._prelude(cwd, env) + command, timeout, on_output)
        duration = time.monotonic() - started
        # ``timeout`` exits 124, and so does a sandbox that enforces the
        # deadline itself; a command that exits 124 well before the deadline
        # chose to, and is not reported as timed out.
        timed_out = code is None or (code == 124 and duration >= timeout * 0.9)
        data, truncated = _cap(output)
        return ExecResult(
            exit_code=None if timed_out else code,
            output=data,
            truncated=truncated,
            timed_out=timed_out,
            duration_s=duration,
        )

    # -------------------------------------------------- background commands --

    def _job(self, handle: str) -> _Job:
        with self._lock:
            job = self._jobs.get(handle.strip())
        if job is None:
            known = self.list_background()
            if not known:
                raise JobError(f"no background command {handle!r} — none have been started")
            lines = "\n".join(f"  {s.describe()}" for s in known)
            raise JobError(f"no background command {handle!r}. These exist:\n{lines}")
        return job

    def _statuses(self, jobs: Iterable[_Job]) -> dict[str, tuple[bool, int | None]]:
        jobs = list(jobs)
        if not jobs:
            return {}
        script = "".join(f"echo {_q(j.handle)}; {_status_script(j.directory)}" for j in jobs)
        _, out = self._run(script)
        lines = out.decode(errors="replace").splitlines()
        found: dict[str, tuple[bool, int | None]] = {}
        for name, state in zip(lines[::2], lines[1::2], strict=False):
            found[name.strip()] = _parse_status(state)
        return found

    def start_background(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        foreground: bool = False,
    ) -> JobStatus:
        if not foreground:
            with self._lock:
                candidates = [j for j in self._jobs.values() if not j.foreground and not j.ended]
            alive = [h for h, (running, _) in self._statuses(candidates).items() if running]
            if len(alive) >= self._max_background:
                raise JobError(
                    f"{self._max_background} background commands are already running — "
                    "stop one with kill_shell first"
                )
        with self._lock:
            handle = f"sh{self._next_id}"
            self._next_id += 1
        directory = posixpath.join(self._jobs_dir, handle)
        workdir = self.resolve(cwd) if cwd else self._root
        d = _q(directory)
        assignments = " ".join(f"{k}={_q(v)}" for k, v in {**self._env, **(env or {})}.items())
        envs = f"env {assignments} " if assignments else ""
        source = "$1/in" if not foreground else "/dev/null"
        # The job: its own session, so its process group is the unit that is
        # killed; its pid written first, its exit status written last and
        # renamed into place, so a reader never sees half of it.
        job = (
            'echo $$ > "$1/pid"; cd "$2" || exit 1; '
            f'sh -c "$3" < "{source}" > "$1/out" 2>&1; '
            'echo $? > "$1/exit.tmp"; mv "$1/exit.tmp" "$1/exit"'
        )
        keeper = ""
        if not foreground:
            # The keeper holds the FIFO's write end open, so the command sees
            # end of input only when the caller closes it, not after every
            # write.
            keeper = (
                'mkfifo "$d/in" || exit 1; '
                'setsid sh -c \'echo $$ > "$1/keeper"; exec tail -f /dev/null\' keeper "$d" '
                '> "$d/in" 2>/dev/null < /dev/null & '
            )
        script = (
            f'd={d}; mkdir -p "$d" || exit 1; : > "$d/out"; {keeper}'
            f'{envs}setsid sh -c {_q(job)} job "$d" {_q(workdir)} {_q(command)} '
            "> /dev/null 2>&1 < /dev/null & "
            'i=0; while [ ! -s "$d/pid" ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i+1)); done; '
            'cat "$d/pid" 2>/dev/null'
        )
        code, out = self._run(script)
        if code != 0:
            raise JobError(f"could not start {command!r} in the {self._label} workspace")
        text = out.decode(errors="replace").strip()
        record = _Job(
            handle=handle,
            command=command,
            directory=directory,
            has_stdin=not foreground,
            foreground=foreground,
            started=time.monotonic(),
            pid=int(text) if text.isdigit() else None,
        )
        with self._lock:
            self._jobs[handle] = record
        return record.status(running=True, exit_code=None)

    def wait_background(self, handle: str, timeout: float) -> JobStatus:
        job = self._job(handle)
        if job.ended is not None:
            return job.status(running=False, exit_code=job.exit_code)
        seconds = max(0, math.ceil(timeout))
        d = _q(job.directory)
        _, out = self._run(
            f"d={d}; end=$(( $(date +%s) + {seconds} )); "
            'while [ ! -e "$d/exit" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 0.2; done; '
            + _status_script(job.directory),
            timeout=seconds + OP_TIMEOUT,
        )
        running, code = _parse_status(out.decode(errors="replace").strip().splitlines()[-1:][0])
        return job.status(running=running, exit_code=code)

    def read_background(self, handle: str, since: int = 0) -> JobOutput:
        job = self._job(handle)
        d = _q(job.directory)
        span = HEAD_KEEP + TAIL_KEEP
        _, out = self._run(
            _status_script(job.directory)
            + f'd={d}; t=$(wc -c < "$d/out" | tr -d " "); echo "$t"; s={max(0, since)}; '
            '[ "$s" -gt "$t" ] && s=$t; n=$((t - s)); '
            f'if [ "$n" -gt {span} ]; then tail -c +$((s + 1)) "$d/out" | head -c {HEAD_KEEP}; '
            f'printf "\\n... %s bytes of output were dropped here ...\\n" $((n - {span})); '
            f'tail -c {TAIL_KEEP} "$d/out"; '
            'elif [ "$n" -gt 0 ]; then tail -c +$((s + 1)) "$d/out" | head -c "$n"; fi'
        )
        state, _, rest = out.partition(b"\n")
        total_line, _, data = rest.partition(b"\n")
        running, code = _parse_status(state.decode(errors="replace"))
        total = int(total_line.strip() or b"0")
        lost = max(0, total - max(0, since) - span) if total - since > span else 0
        return JobOutput(
            status=job.status(running=running, exit_code=code),
            data=data,
            offset=max(total, 0),
            lost=lost,
        )

    def write_background(self, handle: str, data: bytes, *, close: bool = False) -> None:
        job = self._job(handle)
        d = _q(job.directory)
        payload = base64.b64encode(data).decode("ascii")
        write = (
            'timeout 5 sh -c \'printf %s "$1" | base64 -d > "$2/in"\' w '
            f'{_q(payload)} "$d" || exit 33; '
            if data
            else ""
        )
        closing = 'kill "$(cat "$d/keeper")" 2>/dev/null; ' if close else ""
        code, _ = self._run(
            f'd={d}; [ -e "$d/exit" ] && exit 30; [ -p "$d/in" ] || exit 31; {write}{closing}true'
        )
        if code == 30:
            status = self.read_background(handle, since=0).status
            raise JobError(f"{status.describe()} — it cannot take input")
        if code == 31:
            raise JobError(
                f"{job.handle} was not started as a background command, so it has no stdin"
            )
        if code != 0:
            raise JobError(f"{job.handle} is not reading its input any more")

    def kill_background(self, handle: str) -> JobStatus:
        job = self._job(handle)
        d = _q(job.directory)
        # TERM to the whole group, a grace period, then KILL — the same
        # sequence as on the host, run where the processes are.
        _, out = self._run(
            f'd={d}; if [ ! -e "$d/exit" ]; then pg=$(cat "$d/pid" 2>/dev/null); '
            'if [ -n "$pg" ]; then kill -s TERM -- "-$pg" 2>/dev/null; i=0; '
            'while kill -0 -- "-$pg" 2>/dev/null && [ $i -lt 20 ]; do sleep 0.1; i=$((i+1)); done; '
            'kill -s KILL -- "-$pg" 2>/dev/null; fi; '
            '[ -e "$d/exit" ] || echo -15 > "$d/exit"; fi; '
            'if [ -s "$d/keeper" ]; then kill "$(cat "$d/keeper")" 2>/dev/null; fi; '
            'rm -f "$d/in"; ' + _status_script(job.directory)
        )
        running, code = _parse_status(out.decode(errors="replace").strip().splitlines()[-1:][0])
        return job.status(running=running, exit_code=code)

    def release_background(self, handle: str) -> None:
        with self._lock:
            job = self._jobs.pop(handle.strip(), None)
        if job is not None:
            d = _q(job.directory)
            self._run(
                f'd={d}; if [ -s "$d/keeper" ]; then kill "$(cat "$d/keeper")" 2>/dev/null; fi; '
                'rm -rf "$d"'
            )

    def list_background(self) -> list[JobStatus]:
        with self._lock:
            jobs = list(self._jobs.values())
        states = self._statuses(j for j in jobs if j.ended is None)
        out: list[JobStatus] = []
        for job in jobs:
            running, code = states.get(job.handle, (False, job.exit_code))
            out.append(job.status(running=running, exit_code=code))
        return out

    def close(self) -> None:
        """Kill every background command and remove this backend's state."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.ended is None:
                self.kill_background(job.handle)
        with self._lock:
            self._jobs.clear()
        self._run(f"rm -rf {_q(self._jobs_dir)}")
