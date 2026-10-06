# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""LocalBackend: the output buffer and process groups. Ported from tulip-code."""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tulip.harness import BackendError, JobError, LocalBackend
from tulip.harness.local import Output, kill_group


def test_output_keeps_the_head_and_the_latest_tail() -> None:
    out = Output(head_keep=5, tail_keep=5)
    out.append(b"0123456789")
    out.append(b"abcdefghij")
    data, total, lost = out.since(0)
    assert total == 20
    assert lost == 10
    assert data.startswith(b"01234")
    assert data.endswith(b"fghij")
    assert b"10 bytes of output were dropped here" in data


def test_output_since_an_offset_returns_only_what_is_new() -> None:
    out = Output(head_keep=5, tail_keep=100)
    out.append(b"hello world")
    assert out.since(6) == (b"world", 11, 0)
    assert out.since(99) == (b"", 11, 0)


def test_a_group_that_is_already_gone_is_quiet() -> None:
    proc = subprocess.Popen(["true"], start_new_session=True)  # noqa: S607
    proc.wait()
    kill_group(proc)


def test_a_group_that_ignores_sigterm_gets_sigkill(tmp_path: Path) -> None:
    proc = subprocess.Popen(  # noqa: S603
        ["sh", "-c", "trap '' TERM; while :; do sleep 0.1; done"],  # noqa: S607
        start_new_session=True,
    )
    time.sleep(0.2)
    started = time.monotonic()
    kill_group(proc)
    assert proc.wait(timeout=5) == -signal.SIGKILL
    assert time.monotonic() - started < 10


def test_close_kills_what_is_still_running(tmp_path: Path) -> None:
    backend = LocalBackend(tmp_path)
    job = backend.start_background("sleep 30")
    backend.close()
    assert job.pid is not None
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(job.pid, 0)


def test_a_collected_backend_kills_what_it_started(tmp_path: Path) -> None:
    backend = LocalBackend(tmp_path)
    pid = backend.start_background("sleep 30").pid
    assert pid is not None
    del backend
    gc.collect()
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_finished_jobs_are_forgotten_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tulip.harness import local

    monkeypatch.setattr(local, "MAX_FINISHED", 2)
    backend = LocalBackend(tmp_path)
    handles = []
    for _ in range(4):
        job = backend.start_background("true")
        backend.wait_background(job.handle, 5)
        handles.append(job.handle)
    remaining = [s.handle for s in backend.list_background()]
    assert handles[0] not in remaining
    assert handles[-1] in remaining


def test_writing_to_a_command_that_closed_its_input_says_so(tmp_path: Path) -> None:
    backend = LocalBackend(tmp_path)
    job = backend.start_background("exec 0<&-; sleep 30")
    time.sleep(0.3)
    with pytest.raises(JobError, match="not reading its input"):
        backend.write_background(job.handle, b"x" * 200_000)
    backend.close()


def test_an_unreadable_file_is_permission_denied(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads everything")
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    secret.chmod(0)
    backend = LocalBackend(tmp_path)
    for op in (backend.read_bytes, backend.read_lines):
        with pytest.raises(BackendError) as caught:
            op("secret.txt")
        assert caught.value.code == "permission_denied"
    assert backend.grep("x") == []
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "f").write_text("x")
    locked.chmod(0o500)
    try:
        for op in (
            lambda: backend.write_bytes("locked/new", b"x"),
            lambda: backend.remove("locked/f"),
        ):
            with pytest.raises(BackendError) as caught:
                op()
            assert caught.value.code == "permission_denied"
        locked.chmod(0)
        with pytest.raises(BackendError):
            backend.stat("locked/f")
    finally:
        locked.chmod(0o700)
        secret.chmod(0o600)


def test_a_large_file_is_not_hashed_to_stat_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tulip.harness import local

    monkeypatch.setattr(local, "HASH_LIMIT", 3)
    (tmp_path / "big").write_text("0123456789")
    stat = LocalBackend(tmp_path).stat("big")
    assert stat.size == 10
    assert stat.sha256 == ""


def test_huge_files_are_skipped_by_the_grep_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tulip.harness import local

    monkeypatch.setattr(local, "GREP_MAX_FILE_BYTES", 3)
    (tmp_path / "big.txt").write_text("needle needle")
    assert LocalBackend(tmp_path).grep("needle") == []


def test_a_glob_out_of_time_returns_what_it_has(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    assert LocalBackend(tmp_path).glob("*.py", timeout_s=-1) == []
    with pytest.raises(BackendError):
        LocalBackend(tmp_path).glob("*.py", path="a.py")


def test_a_broken_output_callback_does_not_stop_the_command(tmp_path: Path) -> None:
    def broken(chunk: bytes) -> None:
        raise RuntimeError("listener fell over")

    result = LocalBackend(tmp_path).exec("echo fine", timeout=5, on_output=broken)
    assert result.output == b"fine\n"


def test_a_symlink_out_of_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(BackendError) as caught:
        LocalBackend(root).read_bytes("link.txt")
    assert caught.value.code == "invalid_path"


def test_a_dangling_entry_is_left_out_of_a_listing(tmp_path: Path) -> None:
    (tmp_path / "gone").symlink_to(tmp_path / "nowhere")
    (tmp_path / "here.txt").write_text("x")
    names = [Path(i.path).name for i in LocalBackend(tmp_path).ls(".")]
    assert names == ["here.txt"]
