# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""One behavioural contract, every backend.

The tools are written once against :class:`WorkspaceBackend`; these tests
are what makes "once" true. Each runs on every backend that can do what it
tests — the file tests on all four, the shell tests on the three with a
shell — and asserts the same outcome.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.unit.harness.conftest import get, put
from tulip.harness import (
    BackendError,
    ExecUnsupportedError,
    JobError,
    MemoryBackend,
    WorkspaceBackend,
)
from tulip.harness.backend import SKIP_DIRS


# ------------------------------------------------------------------ files --


def test_bytes_round_trip_exactly(workspace: WorkspaceBackend) -> None:
    blob = bytes(range(256)) * 3
    put(workspace, "bin/blob.dat", blob)
    assert workspace.read_bytes("bin/blob.dat") == blob


def test_writing_creates_the_parent_directories(workspace: WorkspaceBackend) -> None:
    put(workspace, "deep/down/here.txt", "ok")
    assert get(workspace, "deep/down/here.txt") == "ok"
    assert workspace.stat("deep/down").is_dir


def test_a_missing_file_is_file_not_found(workspace: WorkspaceBackend) -> None:
    for op in (workspace.read_bytes, workspace.stat, workspace.remove):
        with pytest.raises(BackendError) as caught:
            op("ghost.txt")
        assert caught.value.code == "file_not_found"
    with pytest.raises(BackendError):
        workspace.read_lines("ghost.txt")


def test_a_directory_is_not_a_file(workspace: WorkspaceBackend) -> None:
    put(workspace, "src/a.py", "x")
    for op in (workspace.read_bytes, workspace.remove, workspace.read_lines):
        with pytest.raises(BackendError) as caught:
            op("src")
        assert caught.value.code == "is_directory"
    with pytest.raises(BackendError):
        workspace.write_bytes("src", b"x")


@pytest.mark.parametrize("path", ["../escape.py", "src/../../out.py"])
def test_nothing_resolves_outside_the_root(workspace: WorkspaceBackend, path: str) -> None:
    if isinstance(workspace, MemoryBackend):
        # ``..`` cannot climb above the root of a namespace that is all root.
        assert workspace.resolve(path).startswith("/")
        return
    with pytest.raises(BackendError) as caught:
        workspace.read_bytes(path)
    assert caught.value.code == "invalid_path"
    with pytest.raises(BackendError):
        workspace.write_bytes(path, b"x")


def test_an_absolute_path_outside_the_root_is_refused(workspace: WorkspaceBackend) -> None:
    if isinstance(workspace, MemoryBackend):
        pytest.skip("the memory namespace is rooted at /")
    with pytest.raises(BackendError):
        workspace.resolve("/etc/passwd")


def test_an_absolute_path_inside_the_root_is_the_same_file(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.txt", "same")
    absolute = workspace.resolve("a.txt")
    assert get(workspace, absolute) == "same"


def test_stat_hashes_the_content(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.txt", "one")
    first = workspace.stat("a.txt")
    assert first.size == 3
    assert len(first.sha256) == 64
    assert not first.is_dir
    put(workspace, "a.txt", "two")
    assert workspace.stat("a.txt").sha256 != first.sha256


def test_remove_deletes(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.txt", "x")
    workspace.remove("a.txt")
    assert not workspace.exists("a.txt")


def test_read_lines_numbers_and_windows(workspace: WorkspaceBackend) -> None:
    put(workspace, "big.txt", "\n".join(str(i) for i in range(500)))
    window = workspace.read_lines("big.txt", offset=10, limit=5)
    assert window.total == 500
    assert window.lines[0] == "    11\t10"
    assert len(window.lines) == 5


def test_read_lines_cuts_a_long_line_and_says_how_long(workspace: WorkspaceBackend) -> None:
    put(workspace, "bundle.js", "a\n" + "z" * 5000 + "\nb\n")
    window = workspace.read_lines("bundle.js", max_line_chars=100)
    assert window.total == 3
    assert "[line is 5,000 characters]" in window.lines[1]
    assert len(window.lines[1]) < 200


def test_read_lines_past_the_end_is_empty_but_counts(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.txt", "one\ntwo\n")
    window = workspace.read_lines("a.txt", offset=10)
    assert window.lines == ()
    assert window.total == 2


def test_read_lines_keeps_the_window_under_the_byte_cap(workspace: WorkspaceBackend) -> None:
    put(workspace, "wide.txt", "\n".join("y" * 1000 for _ in range(100)))
    window = workspace.read_lines("wide.txt", max_bytes=5_000)
    assert 1 <= len(window.lines) <= 5


def test_crlf_is_kept_on_disk_and_stripped_in_windows(workspace: WorkspaceBackend) -> None:
    put(workspace, "win.txt", b"one\r\ntwo\r\n")
    assert workspace.read_bytes("win.txt") == b"one\r\ntwo\r\n"
    assert workspace.read_lines("win.txt").lines == ("     1\tone", "     2\ttwo")


def test_ls_lists_one_level_or_a_bounded_tree(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x")
    put(workspace, "src/b.py", "x")
    put(workspace, "src/pkg/c.py", "x")
    top = {Path(i.path).name for i in workspace.ls(".")}
    assert {"a.py", "src"} <= top
    two = {Path(i.path).name for i in workspace.ls(".", recursive=True, depth=2)}
    assert {"a.py", "src", "b.py", "pkg"} <= two
    assert "c.py" not in two
    assert any(i.path.endswith("c.py") for i in workspace.ls(".", recursive=True))


def test_ls_of_a_file_is_that_file(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x")
    (only,) = workspace.ls("a.py")
    assert only.path.endswith("a.py")
    assert not only.is_dir


def test_ls_of_nothing_is_file_not_found(workspace: WorkspaceBackend) -> None:
    with pytest.raises(BackendError):
        workspace.ls("nowhere")


def test_a_recursive_walk_skips_the_noise(workspace: WorkspaceBackend) -> None:
    put(workspace, "src/app.py", "needle")
    for noise in sorted(SKIP_DIRS):
        put(workspace, f"{noise}/junk.py", "needle")
    walked = [i.path for i in workspace.ls(".", recursive=True)]
    assert not any(f"/{d}/" in p for p in walked for d in SKIP_DIRS)
    assert [Path(i.path).name for i in workspace.glob("**/*.py")] == ["app.py"]
    assert [m.path.rsplit("/", 2)[-2] for m in workspace.grep("needle")] == ["src"]


def test_glob_matches_the_top_level_with_double_star(workspace: WorkspaceBackend) -> None:
    put(workspace, "setup.py", "x")
    put(workspace, "src/app.py", "x")
    put(workspace, "src/app.txt", "x")
    names = sorted(Path(i.path).name for i in workspace.glob("**/*.py"))
    assert names == ["app.py", "setup.py"]
    assert [Path(i.path).name for i in workspace.glob("*.txt", path="src")] == ["app.txt"]


def test_grep_reports_file_line_and_text(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "one\nneedle here\nthree\n")
    put(workspace, "b.md", "needle too\n")
    hits = workspace.grep(r"needle\s\w+")
    assert sorted((Path(m.path).name, m.line, m.text) for m in hits) == [
        ("a.py", 2, "needle here"),
        ("b.md", 1, "needle too"),
    ]
    only_py = workspace.grep("needle", glob_filter="*.py")
    assert [Path(m.path).name for m in only_py] == ["a.py"]
    assert len(workspace.grep("needle", limit=1)) == 1


def test_grep_can_stay_on_one_level(workspace: WorkspaceBackend) -> None:
    put(workspace, "top.txt", "needle\n")
    put(workspace, "sub/deep.txt", "needle\n")
    names = [Path(m.path).name for m in workspace.grep("needle", recursive=False)]
    assert names == ["top.txt"]


def test_grep_of_a_single_file(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.txt", "x\nneedle\n")
    assert [m.line for m in workspace.grep("needle", path="a.txt")] == [2]


def test_grep_of_nothing_is_file_not_found(workspace: WorkspaceBackend) -> None:
    with pytest.raises(BackendError):
        workspace.grep("x", path="nowhere")


def test_the_deepagent_text_operations_work_on_every_backend(
    workspace: WorkspaceBackend,
) -> None:
    workspace.write("notes/a.md", "alpha\nbeta\n")
    assert workspace.exists("notes/a.md")
    workspace.edit("notes/a.md", "beta", "gamma")
    assert "gamma" in workspace.read("notes/a.md")
    assert not workspace.exists("notes/missing.md")


def test_capabilities_describe_the_workspace(workspace: WorkspaceBackend) -> None:
    caps = workspace.capabilities
    assert caps.root.startswith("/")
    assert caps.label
    assert caps.can_exec == (not isinstance(workspace, MemoryBackend))


# ------------------------------------------------------------------ shell --


def test_exec_returns_status_and_both_streams_in_order(shell: WorkspaceBackend) -> None:
    result = shell.exec("echo one; echo two >&2; echo three; exit 3", timeout=10)
    assert result.exit_code == 3
    out = result.output.decode()
    assert out.index("one") < out.index("three")
    assert "two" in out
    assert not result.timed_out


def test_exec_runs_in_the_root_with_stdin_closed(shell: WorkspaceBackend) -> None:
    result = shell.exec("pwd; cat; echo done", timeout=10)
    lines = result.output.decode().split()
    assert lines[0] == shell.capabilities.root
    assert lines[-1] == "done"


def test_exec_takes_a_working_directory_and_environment(shell: WorkspaceBackend) -> None:
    put(shell, "sub/x.txt", "x")
    result = shell.exec(
        'pwd; echo "$GREETING"', timeout=10, cwd="sub", env={"GREETING": "hi there"}
    )
    assert result.output.decode().split("\n")[:2] == [shell.resolve("sub"), "hi there"]


def test_exec_streams_what_it_prints(shell: WorkspaceBackend) -> None:
    seen: list[bytes] = []
    shell.exec("echo streamed", timeout=10, on_output=seen.append)
    assert b"streamed" in b"".join(seen)


def test_exec_kills_what_runs_past_the_timeout(shell: WorkspaceBackend, tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    result = shell.exec(f"echo started; sleep 30 & echo $! > {pidfile}; wait", timeout=1)
    assert time.monotonic() - started < 15
    assert result.timed_out
    assert result.exit_code is None
    assert b"started" in result.output
    time.sleep(0.2)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_command_that_exits_124_on_its_own_did_not_time_out(shell: WorkspaceBackend) -> None:
    result = shell.exec("exit 124", timeout=30)
    assert result.exit_code == 124
    assert not result.timed_out


def test_a_background_command_takes_input_until_it_is_closed(shell: WorkspaceBackend) -> None:
    job = shell.start_background("cat; echo eof")
    assert job.running
    assert job.has_stdin
    assert job.handle
    shell.write_background(job.handle, b"hello\n")
    deadline = time.monotonic() + 5
    while b"hello" not in shell.read_background(job.handle).data:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    shell.write_background(job.handle, b"", close=True)
    status = shell.wait_background(job.handle, 5)
    assert not status.running
    assert status.exit_code == 0
    read = shell.read_background(job.handle)
    assert read.data == b"hello\neof\n"
    assert read.offset == len(b"hello\neof\n")
    assert shell.read_background(job.handle, since=6).data == b"eof\n"
    assert "exited 0" in status.describe()


def test_a_background_command_can_be_killed_with_its_children(
    shell: WorkspaceBackend, tmp_path: Path
) -> None:
    pidfile = tmp_path / "child.pid"
    job = shell.start_background(f"sleep 30 & echo $! > {pidfile}; wait")
    deadline = time.monotonic() + 5
    while not pidfile.exists() or not pidfile.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    status = shell.kill_background(job.handle)
    assert not status.running
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert "running" not in shell.kill_background(job.handle).describe()


def test_a_foreground_job_has_no_stdin(shell: WorkspaceBackend) -> None:
    job = shell.start_background("cat; echo done", foreground=True)
    assert not job.has_stdin
    status = shell.wait_background(job.handle, 5)
    assert not status.running
    assert shell.read_background(job.handle).data == b"done\n"
    waiting = shell.start_background("sleep 30", foreground=True)
    with pytest.raises(JobError, match="has no stdin"):
        shell.write_background(waiting.handle, b"x")
    shell.kill_background(waiting.handle)


def test_writing_to_a_finished_command_says_it_cannot_take_input(
    shell: WorkspaceBackend,
) -> None:
    job = shell.start_background("true")
    shell.wait_background(job.handle, 5)
    with pytest.raises(JobError, match="cannot take input"):
        shell.write_background(job.handle, b"x")


def test_a_running_job_is_listed_then_released(shell: WorkspaceBackend) -> None:
    job = shell.start_background("sleep 30")
    listed = shell.list_background()
    assert [s.handle for s in listed] == [job.handle]
    assert listed[0].running
    assert "is running" in listed[0].describe()
    shell.kill_background(job.handle)
    shell.release_background(job.handle)
    assert shell.list_background() == []
    shell.release_background(job.handle)  # releasing twice is quiet


def test_an_unknown_handle_lists_the_ones_that_exist(shell: WorkspaceBackend) -> None:
    with pytest.raises(JobError, match="none have been started"):
        shell.read_background("sh99")
    job = shell.start_background("sleep 30")
    with pytest.raises(JobError, match=f"These exist:\n  {job.handle}"):
        shell.kill_background("sh99")
    shell.kill_background(job.handle)


def test_waiting_on_a_finished_job_returns_at_once(shell: WorkspaceBackend) -> None:
    job = shell.start_background("exit 7", foreground=True)
    assert shell.wait_background(job.handle, 5).exit_code == 7
    started = time.monotonic()
    assert shell.wait_background(job.handle, 5).exit_code == 7
    assert time.monotonic() - started < 2


def test_the_background_cap_is_a_message_not_a_hang(root: Path) -> None:
    from tests.unit.harness.conftest import make_backend

    for kind in ("local", "session"):
        backend = make_backend(kind, root)
        backend._max_background = 1  # type: ignore[attr-defined]
        first = backend.start_background("sleep 30")
        with pytest.raises(JobError, match="already running"):
            backend.start_background("sleep 30")
        # A command the caller waits on is not what the cap is for.
        waited = backend.start_background("true", foreground=True)
        backend.wait_background(waited.handle, 5)
        backend.kill_background(first.handle)
        backend.close()  # type: ignore[attr-defined]


def test_the_memory_backend_has_no_shell() -> None:
    backend = MemoryBackend()
    calls = [
        lambda: backend.exec("ls", timeout=1),
        lambda: backend.start_background("ls"),
        lambda: backend.wait_background("sh1", 1),
        lambda: backend.read_background("sh1"),
        lambda: backend.write_background("sh1", b""),
        lambda: backend.kill_background("sh1"),
        lambda: backend.release_background("sh1"),
    ]
    for call in calls:
        with pytest.raises(ExecUnsupportedError, match="has no shell"):
            call()
    assert backend.list_background() == []
