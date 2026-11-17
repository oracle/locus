# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The conformance kit: does a backend keep the sandbox contract?

The harness tools are written once against
:class:`~locus.harness.backend.WorkspaceBackend` and assume the same
behaviour from every backend: a byte written is the byte read back, a path
cannot leave the root, a missing file is ``file_not_found`` and a directory
is ``is_directory``, a command runs in the root with its input closed and is
killed at its deadline, a background command takes input until it is
closed. The kit checks each of those on a real backend, and the
:class:`~locus.harness.sandbox.SandboxBackend` protocol over them: lifecycle
transitions, declared limits, evidence hooks.

Run it from a test against the backend you are writing::

    from locus.harness.testing import assert_backend_conformance


    def test_conformance():
        assert_backend_conformance(lambda: MyBackend.create(...))

The factory is called once; the backend it returns is closed at the end,
whatever happened. Every check runs even after one fails, so one report
lists everything that is wrong, and the ``AssertionError`` it raises names
each failed check and what it saw. The kit needs nothing from pytest, so it
also runs from a script against a live sandbox.

A check writes under one directory it creates in the workspace root and
removes at the end; nothing else in the workspace is touched. The shell
checks use only what any Linux image has (``sh``, ``cat``, ``sleep``,
``pwd``); a check that stops a runaway command takes a few seconds, and
:func:`assert_backend_conformance` can leave it out with ``timeouts=False``
when every second of a live sandbox costs.
"""

from __future__ import annotations

import shlex
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from locus.deepagent.backends.protocol import BackendError
from locus.harness.backend import ExecUnsupportedError, JobError, WorkspaceBackend
from locus.harness.evidence import ExecRecord
from locus.harness.sandbox import ResourceLimits, SandboxBackend


__all__ = [
    "CHECKS",
    "Check",
    "ConformanceError",
    "ConformanceReport",
    "assert_backend_conformance",
    "run_conformance",
]


class ConformanceError(AssertionError):
    """One check found the backend in breach of the contract."""


@dataclass
class Scratch:
    """The directory one conformance run works in, and how to name files there."""

    backend: WorkspaceBackend
    root: str
    written: list[str] = field(default_factory=list)

    def path(self, name: str) -> str:
        return f"{self.root}/{name}"

    def put(self, name: str, data: bytes | str) -> str:
        path = self.path(name)
        self.backend.write_bytes(path, data.encode() if isinstance(data, str) else data)
        self.written.append(path)
        return path


#: One conformance check: a name, which capability it needs, and the check.
Check = tuple[str, str, Callable[[Scratch], None]]


@dataclass
class ConformanceReport:
    """What a conformance run found.

    Attributes:
        label: The backend's label.
        passed: The names of the checks that held.
        failed: Each failed check's name and what it saw.
        skipped: Checks left out, with the reason.
    """

    label: str
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        lines = [
            f"{self.label}: {len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped"
        ]
        lines.extend(f"  FAIL {name}: {why}" for name, why in self.failed)
        lines.extend(f"  skip {name}: {why}" for name, why in self.skipped)
        return "\n".join(lines)


def _expect(condition: bool, message: str) -> None:  # noqa: FBT001 - a check's verdict
    if not condition:
        raise ConformanceError(message)


def _error_code(call: Callable[[], object]) -> str | None:
    try:
        call()
    except BackendError as exc:
        return exc.code
    return None


# ------------------------------------------------------------------ files --


def check_bytes_round_trip(s: Scratch) -> None:
    blob = bytes(range(256)) * 3
    path = s.put("bin/blob.dat", blob)
    _expect(s.backend.read_bytes(path) == blob, "read_bytes did not return the bytes written")
    _expect(s.backend.stat(f"{s.root}/bin").is_dir, "a parent directory was not created")


def check_missing_file_errors(s: Scratch) -> None:
    ghost = s.path("ghost.txt")
    for name, call in (
        ("read_bytes", lambda: s.backend.read_bytes(ghost)),
        ("stat", lambda: s.backend.stat(ghost)),
        ("remove", lambda: s.backend.remove(ghost)),
    ):
        code = _error_code(call)
        _expect(code == "file_not_found", f"{name} on a missing file raised {code!r}")
    _expect(
        _error_code(lambda: s.backend.read_lines(ghost)) is not None,
        "read_lines on a missing file did not raise BackendError",
    )


def check_directory_errors(s: Scratch) -> None:
    s.put("src/a.py", "x")
    directory = s.path("src")
    for name, call in (
        ("read_bytes", lambda: s.backend.read_bytes(directory)),
        ("remove", lambda: s.backend.remove(directory)),
        ("read_lines", lambda: s.backend.read_lines(directory)),
    ):
        code = _error_code(call)
        _expect(code == "is_directory", f"{name} on a directory raised {code!r}")
    _expect(
        _error_code(lambda: s.backend.write_bytes(directory, b"x")) is not None,
        "write_bytes over a directory did not raise BackendError",
    )


def _resolve_code(backend: WorkspaceBackend, path: str) -> str | None:
    return _error_code(lambda: backend.resolve(path))


def check_paths_stay_in_the_root(s: Scratch) -> None:
    root = s.backend.capabilities.root
    if root == "/":
        return  # nothing is outside a root that is everything
    for path in ("../escape.py", "src/../../../out.py", "/etc/passwd"):
        code = _resolve_code(s.backend, path)
        _expect(code == "invalid_path", f"resolve({path!r}) raised {code!r}, not invalid_path")


def check_absolute_and_relative_name_one_file(s: Scratch) -> None:
    path = s.put("same.txt", "same")
    absolute = s.backend.resolve(path)
    _expect(absolute.startswith("/"), f"resolve returned a relative path: {absolute!r}")
    _expect(s.backend.read_bytes(absolute) == b"same", "the absolute path reads a different file")


def check_stat_hashes_content(s: Scratch) -> None:
    path = s.put("hash.txt", "one")
    first = s.backend.stat(path)
    _expect(first.size == 3, f"stat size was {first.size}, not 3")
    _expect(len(first.sha256) == 64, "stat did not hash a small file")
    _expect(not first.is_dir, "stat called a file a directory")
    s.put("hash.txt", "two")
    _expect(s.backend.stat(path).sha256 != first.sha256, "the hash did not change with the content")


def check_remove_deletes(s: Scratch) -> None:
    path = s.put("gone.txt", "x")
    s.backend.remove(path)
    _expect(not s.backend.exists(path), "the file exists after remove")


def check_read_lines_windows(s: Scratch) -> None:
    path = s.put("big.txt", "\n".join(str(i) for i in range(500)))
    window = s.backend.read_lines(path, offset=10, limit=5)
    _expect(window.total == 500, f"total was {window.total}, not 500")
    _expect(len(window.lines) == 5, f"the window had {len(window.lines)} lines, not 5")
    _expect(window.lines[0] == "    11\t10", f"the first line was {window.lines[0]!r}")
    past = s.backend.read_lines(path, offset=1000)
    _expect(past.lines == () and past.total == 500, "a window past the end did not count")


def check_read_lines_cuts_long_lines(s: Scratch) -> None:
    path = s.put("bundle.js", "a\n" + "z" * 5000 + "\nb\n")
    window = s.backend.read_lines(path, max_line_chars=100)
    _expect(window.total == 3, f"total was {window.total}, not 3")
    _expect("[line is 5,000 characters]" in window.lines[1], "the long line was not cut")
    _expect(len(window.lines[1]) < 200, "the cut line is still long")


def check_crlf_is_kept(s: Scratch) -> None:
    path = s.put("win.txt", b"one\r\ntwo\r\n")
    _expect(s.backend.read_bytes(path) == b"one\r\ntwo\r\n", "CRLF bytes were changed")
    _expect(
        s.backend.read_lines(path).lines == ("     1\tone", "     2\ttwo"),
        "CRLF was not stripped from the window",
    )


def check_ls(s: Scratch) -> None:
    s.put("ls/a.py", "x")
    s.put("ls/src/b.py", "x")
    s.put("ls/src/pkg/c.py", "x")
    top = {PurePosixPath(i.path).name for i in s.backend.ls(s.path("ls"))}
    _expect({"a.py", "src"} <= top, f"ls of one level gave {sorted(top)}")
    two = {PurePosixPath(i.path).name for i in s.backend.ls(s.path("ls"), recursive=True, depth=2)}
    _expect({"a.py", "src", "b.py", "pkg"} <= two and "c.py" not in two, f"depth=2 gave {two}")
    deep = s.backend.ls(s.path("ls"), recursive=True)
    _expect(any(i.path.endswith("c.py") for i in deep), "a full walk missed a deep file")
    (only,) = s.backend.ls(s.path("ls/a.py"))
    _expect(only.path.endswith("a.py") and not only.is_dir, "ls of a file is not that file")
    _expect(
        _error_code(lambda: s.backend.ls(s.path("nowhere"))) is not None,
        "ls of nothing did not raise BackendError",
    )


def check_walks_skip_noise(s: Scratch) -> None:
    s.put("walk/src/app.py", "needle")
    s.put("walk/node_modules/junk.py", "needle")
    s.put("walk/.git/junk.py", "needle")
    walked = [i.path for i in s.backend.ls(s.path("walk"), recursive=True)]
    _expect(
        not any("/node_modules/" in p or "/.git/" in p for p in walked),
        "a recursive listing walked into node_modules or .git",
    )
    found = [PurePosixPath(i.path).name for i in s.backend.glob("**/*.py", path=s.path("walk"))]
    _expect(found == ["app.py"], f"glob found {found}")
    hits = [m.path for m in s.backend.grep("needle", path=s.path("walk"))]
    _expect(
        len(hits) == 1 and hits[0].endswith("src/app.py"),
        f"grep searched the noise: {hits}",
    )


def check_glob(s: Scratch) -> None:
    s.put("glob/setup.py", "x")
    s.put("glob/src/app.py", "x")
    s.put("glob/src/app.txt", "x")
    names = sorted(
        PurePosixPath(i.path).name for i in s.backend.glob("**/*.py", path=s.path("glob"))
    )
    _expect(names == ["app.py", "setup.py"], f"**/*.py matched {names}")
    txt = [PurePosixPath(i.path).name for i in s.backend.glob("*.txt", path=s.path("glob/src"))]
    _expect(txt == ["app.txt"], f"*.txt under src matched {txt}")


def check_grep(s: Scratch) -> None:
    s.put("grep/a.py", "one\nneedle here\nthree\n")
    s.put("grep/b.md", "needle too\n")
    s.put("grep/sub/deep.txt", "needle deep\n")
    hits = sorted(
        (PurePosixPath(m.path).name, m.line, m.text)
        for m in s.backend.grep("nee+dle", path=s.path("grep"))
    )
    _expect(
        hits
        == [("a.py", 2, "needle here"), ("b.md", 1, "needle too"), ("deep.txt", 1, "needle deep")],
        f"grep reported {hits}",
    )
    only_py = [
        PurePosixPath(m.path).name
        for m in s.backend.grep("needle", path=s.path("grep"), glob_filter="*.py")
    ]
    _expect(only_py == ["a.py"], f"glob_filter gave {only_py}")
    _expect(
        len(s.backend.grep("needle", path=s.path("grep"), limit=1)) == 1, "limit was not honoured"
    )
    flat = sorted(
        PurePosixPath(m.path).name
        for m in s.backend.grep("needle", path=s.path("grep"), recursive=False)
    )
    _expect(flat == ["a.py", "b.md"], f"recursive=False searched {flat}")
    single = [m.line for m in s.backend.grep("needle", path=s.path("grep/a.py"))]
    _expect(single == [2], f"grep of one file gave {single}")
    _expect(
        _error_code(lambda: s.backend.grep("x", path=s.path("nowhere"))) is not None,
        "grep of nothing did not raise BackendError",
    )


def check_text_operations(s: Scratch) -> None:
    path = s.path("notes/a.md")
    s.backend.write(path, "alpha\nbeta\n")
    _expect(s.backend.exists(path), "write did not create the file")
    s.backend.edit(path, "beta", "gamma")
    _expect("gamma" in s.backend.read(path), "edit did not change the file")
    _expect(not s.backend.exists(s.path("notes/missing.md")), "exists found a missing file")


def check_capabilities(s: Scratch) -> None:
    caps = s.backend.capabilities
    _expect(caps.root.startswith("/"), f"root is not absolute: {caps.root!r}")
    _expect(bool(caps.label), "the label is empty")


# ------------------------------------------------------------------ shell --


def check_exec_status_and_streams(s: Scratch) -> None:
    result = s.backend.exec("echo one; echo two >&2; echo three; exit 3", timeout=30)
    _expect(result.exit_code == 3, f"exit code was {result.exit_code}, not 3")
    out = result.output.decode(errors="replace")
    _expect("one" in out and "two" in out and "three" in out, f"output lost a stream: {out!r}")
    _expect(out.index("one") < out.index("three"), "stdout came back out of order")
    _expect(not result.timed_out, "a quick command was reported as timed out")


def check_exec_runs_in_the_root_with_stdin_closed(s: Scratch) -> None:
    result = s.backend.exec("pwd; cat; echo done", timeout=30)
    lines = result.output.decode(errors="replace").split()
    _expect(bool(lines) and lines[0] == s.backend.capabilities.root, f"pwd printed {lines[:1]}")
    _expect(lines[-1] == "done", "cat waited on stdin: the command did not finish on its own")


def check_exec_cwd_and_env(s: Scratch) -> None:
    s.put("cwd/x.txt", "x")
    result = s.backend.exec(
        'pwd; echo "$LOCUS_CONFORMANCE"',
        timeout=30,
        cwd=s.path("cwd"),
        env={"LOCUS_CONFORMANCE": "hi there"},
    )
    lines = result.output.decode(errors="replace").split("\n")[:2]
    _expect(lines == [s.backend.resolve(s.path("cwd")), "hi there"], f"cwd/env gave {lines}")


def check_exec_streams_output(s: Scratch) -> None:
    seen: list[bytes] = []
    s.backend.exec("echo streamed", timeout=30, on_output=seen.append)
    _expect(b"streamed" in b"".join(seen), "on_output never saw the output")


def check_exec_kills_at_the_deadline(s: Scratch) -> None:
    started = time.monotonic()
    result = s.backend.exec("echo started; sleep 30; echo late", timeout=2)
    elapsed = time.monotonic() - started
    _expect(elapsed < 20, f"the command ran {elapsed:.0f}s past a 2s deadline")
    _expect(result.timed_out, "a command past its deadline was not reported as timed out")
    _expect(result.exit_code is None, f"a timed-out command reported exit code {result.exit_code}")
    _expect(b"started" in result.output, "output before the deadline was lost")
    _expect(b"late" not in result.output, "the command ran on past the deadline")


def check_exit_124_is_not_a_timeout(s: Scratch) -> None:
    result = s.backend.exec("exit 124", timeout=30)
    _expect(result.exit_code == 124 and not result.timed_out, "exit 124 was taken for a timeout")


def check_background_takes_input_until_closed(s: Scratch) -> None:
    job = s.backend.start_background("cat; echo eof")
    _expect(job.running and job.has_stdin and bool(job.handle), f"the job started as {job}")
    s.backend.write_background(job.handle, b"hello\n")
    deadline = time.monotonic() + 15
    while b"hello" not in s.backend.read_background(job.handle).data:
        _expect(time.monotonic() < deadline, "input written to the job never reached its output")
        time.sleep(0.1)
    s.backend.write_background(job.handle, b"", close=True)
    status = s.backend.wait_background(job.handle, 15)
    _expect(not status.running and status.exit_code == 0, f"after close the job is {status}")
    read = s.backend.read_background(job.handle)
    _expect(read.data == b"hello\neof\n", f"the job's output was {read.data!r}")
    _expect(read.offset == len(b"hello\neof\n"), f"the offset was {read.offset}")
    _expect(s.backend.read_background(job.handle, since=6).data == b"eof\n", "since= was ignored")
    s.backend.release_background(job.handle)


def check_background_kill_list_release(s: Scratch) -> None:
    job = s.backend.start_background("sleep 30")
    listed = s.backend.list_background()
    _expect([j.handle for j in listed] == [job.handle], f"list_background gave {listed}")
    _expect(listed[0].running, "a sleeping job was listed as finished")
    status = s.backend.kill_background(job.handle)
    _expect(not status.running, "the job is running after kill")
    s.backend.release_background(job.handle)
    _expect(s.backend.list_background() == [], "a released job is still listed")
    s.backend.release_background(job.handle)  # releasing twice is quiet
    try:
        s.backend.read_background(job.handle)
    except JobError:
        pass
    else:
        raise ConformanceError("a released handle still reads")


def check_foreground_job_has_no_stdin(s: Scratch) -> None:
    job = s.backend.start_background("cat; echo done", foreground=True)
    _expect(not job.has_stdin, "a foreground job reports stdin")
    status = s.backend.wait_background(job.handle, 15)
    _expect(not status.running, "a foreground cat did not finish with stdin closed")
    _expect(s.backend.read_background(job.handle).data == b"done\n", "the output was wrong")
    s.backend.release_background(job.handle)


# -------------------------------------------------------------- protocol --


def check_lifecycle_transitions(s: Scratch) -> None:
    backend = s.backend
    if not isinstance(backend, SandboxBackend):
        raise ConformanceError("the backend does not implement SandboxBackend")
    _expect(backend.state in ("starting", "running"), f"a fresh backend is {backend.state!r}")
    backend.start()
    _expect(backend.state == "running", f"after start the state is {backend.state!r}")
    backend.start()  # idempotent
    _expect(backend.state == "running", "a second start changed the state")
    limits = backend.limits
    _expect(limits.bound(10) <= 10, "limits.bound raised a timeout")


def check_exec_hooks_see_commands(s: Scratch) -> None:
    backend = s.backend
    if not isinstance(backend, SandboxBackend):
        raise ConformanceError("the backend does not implement SandboxBackend")
    seen: list[ExecRecord] = []
    remove = backend.add_exec_hook(seen.append)
    try:
        backend.exec("echo hooked", timeout=30)
    finally:
        remove()
    _expect(len(seen) == 1, f"the hook saw {len(seen)} records for one command")
    record = seen[0]
    _expect(record.exit_code == 0, f"the record's exit code was {record.exit_code}")
    _expect(record.backend_label == backend.capabilities.label, "the record names another backend")
    _expect(len(record.command_sha256) == 64, "the record does not carry a command digest")
    backend.exec("echo unhooked", timeout=30)
    _expect(len(seen) == 1, "a removed hook still receives records")


def check_limits_bound_exec(s: Scratch) -> None:
    backend = s.backend
    if not isinstance(backend, SandboxBackend):
        raise ConformanceError("the backend does not implement SandboxBackend")
    before = backend.limits
    try:
        backend.limits = ResourceLimits(max_exec_s=1)  # type: ignore[misc]
    except AttributeError:
        return  # the backend fixes its limits; nothing to set here
    try:
        started = time.monotonic()
        result = backend.exec("sleep 30", timeout=600)
        _expect(time.monotonic() - started < 20, "a 1s command-time limit was not enforced")
        _expect(result.timed_out, "a command past the limit was not reported as timed out")
    finally:
        backend.limits = before  # type: ignore[misc]


#: Every check, in the order they run. The middle field says what the check
#: needs: ``files`` runs everywhere, ``shell`` and ``background`` need
#: ``can_exec``, ``timeouts`` is the slow deadline check, ``protocol`` needs
#: :class:`SandboxBackend`, ``hooks`` and ``limits`` need both.
CHECKS: tuple[Check, ...] = (
    ("bytes round-trip exactly", "files", check_bytes_round_trip),
    ("a missing file is file_not_found", "files", check_missing_file_errors),
    ("a directory is is_directory", "files", check_directory_errors),
    ("paths stay inside the root", "files", check_paths_stay_in_the_root),
    ("absolute and relative name one file", "files", check_absolute_and_relative_name_one_file),
    ("stat hashes the content", "files", check_stat_hashes_content),
    ("remove deletes", "files", check_remove_deletes),
    ("read_lines numbers and windows", "files", check_read_lines_windows),
    ("read_lines cuts long lines", "files", check_read_lines_cuts_long_lines),
    ("CRLF is kept on disk, stripped in windows", "files", check_crlf_is_kept),
    ("ls lists one level or a bounded tree", "files", check_ls),
    ("recursive walks skip the noise", "files", check_walks_skip_noise),
    ("glob matches the top level with **", "files", check_glob),
    ("grep reports file, line and text", "files", check_grep),
    ("the text operations work", "files", check_text_operations),
    ("capabilities describe the workspace", "files", check_capabilities),
    ("exec returns status and both streams", "shell", check_exec_status_and_streams),
    (
        "exec runs in the root with stdin closed",
        "shell",
        check_exec_runs_in_the_root_with_stdin_closed,
    ),
    ("exec takes a cwd and an environment", "shell", check_exec_cwd_and_env),
    ("exec streams output", "shell", check_exec_streams_output),
    ("exit 124 on its own is not a timeout", "shell", check_exit_124_is_not_a_timeout),
    ("exec kills a command at its deadline", "timeouts", check_exec_kills_at_the_deadline),
    (
        "a background command takes input until closed",
        "background",
        check_background_takes_input_until_closed,
    ),
    (
        "a background command is killed, listed, released",
        "background",
        check_background_kill_list_release,
    ),
    ("a foreground job has no stdin", "background", check_foreground_job_has_no_stdin),
    ("lifecycle transitions", "protocol", check_lifecycle_transitions),
    ("exec hooks see every command", "hooks", check_exec_hooks_see_commands),
    ("a command-time limit bounds exec", "limits", check_limits_bound_exec),
)


def _cleanup(scratch: Scratch) -> None:
    """Remove the scratch directory: with the shell when there is one, file by file otherwise."""
    backend = scratch.backend
    if backend.capabilities.can_exec:
        try:
            backend.exec(f"rm -rf {shlex.quote(scratch.root)}", timeout=60)
            return
        except (BackendError, ExecUnsupportedError, JobError):  # pragma: no cover - fall through
            pass
    for path in (*scratch.written, scratch.path("notes/a.md")):
        try:
            backend.remove(path)
        except BackendError:
            pass


def _selected(
    backend: WorkspaceBackend, *, shell: bool, background: bool, timeouts: bool, limits: bool
) -> Iterator[tuple[Check, str | None]]:
    """Each check with the reason it is skipped, or ``None`` to run it."""
    can_exec = backend.capabilities.can_exec
    is_sandbox = isinstance(backend, SandboxBackend)
    wanted = {
        "files": True,
        "shell": shell,
        "background": shell and background,
        "timeouts": shell and timeouts,
        "limits": shell and timeouts and limits,
        "protocol": True,
        "hooks": shell,
    }
    for check in CHECKS:
        _, needs, _ = check
        reason: str | None = None
        if needs not in ("files", "protocol") and not can_exec:
            reason = "the backend has no shell"
        elif not wanted[needs]:
            reason = f"{needs} checks were left out"
        elif needs in ("protocol", "limits", "hooks") and not is_sandbox:
            reason = "the backend does not implement SandboxBackend"
        yield check, reason


def run_conformance(
    backend: WorkspaceBackend,
    *,
    shell: bool = True,
    background: bool = True,
    timeouts: bool = True,
    limits: bool = True,
) -> ConformanceReport:
    """Run every applicable check on ``backend`` and report; close nothing.

    Args:
        backend: An open backend. Its workspace gets one scratch directory,
            removed at the end.
        shell: Run the command checks (on a backend with a shell).
        background: Run the background-command checks.
        timeouts: Run the slow check that a runaway command is stopped.
        limits: Run the check that a declared command-time limit is enforced.
    """
    label = backend.capabilities.label
    report = ConformanceReport(label=label)
    root = f"locus-conformance-{uuid.uuid4().hex[:8]}"
    scratch = Scratch(backend, root)
    try:
        for (name, _, check), reason in _selected(
            backend, shell=shell, background=background, timeouts=timeouts, limits=limits
        ):
            if reason is not None:
                report.skipped.append((name, reason))
                continue
            try:
                check(scratch)
            except ConformanceError as exc:
                report.failed.append((name, str(exc)))
            except Exception as exc:  # noqa: BLE001 - a backend may raise anything; it is the finding
                report.failed.append((name, f"{type(exc).__name__}: {exc}"))
            else:
                report.passed.append(name)
    finally:
        _cleanup(scratch)
    return report


def _close(backend: WorkspaceBackend) -> str | None:
    """Close ``backend`` twice; what went wrong, or ``None``."""
    close = getattr(backend, "close", None)
    if close is None:
        return "the backend has no close()"
    try:
        close()
        if isinstance(backend, SandboxBackend) and backend.state != "closed":
            return f"the state after close is {backend.state!r}"
        close()
    except Exception as exc:  # noqa: BLE001 - closing is itself under test
        return f"close raised {type(exc).__name__}: {exc}"
    return None


def assert_backend_conformance(
    factory: Callable[[], WorkspaceBackend],
    *,
    shell: bool = True,
    background: bool = True,
    timeouts: bool = True,
    limits: bool = True,
) -> ConformanceReport:
    """Build a backend with ``factory``, check it, close it; raise if it failed.

    The lifecycle is part of what is checked: after the checks, ``close()``
    must leave the backend ``closed`` and a second ``close()`` must be quiet.

    Raises:
        AssertionError: One or more checks failed; the message lists them.
    """
    backend = factory()
    try:
        report = run_conformance(
            backend, shell=shell, background=background, timeouts=timeouts, limits=limits
        )
    finally:
        problem = _close(backend)
    if problem is None:
        report.passed.append("close leaves the sandbox closed")
    else:
        report.failed.append(("close leaves the sandbox closed", problem))
    if not report.ok:
        raise AssertionError(report.summary())
    return report
