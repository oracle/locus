# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""SessionBackend: what is particular to driving a sandbox with shell scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.harness.conftest import ShellSession, needs_posix
from tulip.harness import BackendError, JobError, SessionBackend, SessionLike
from tulip.harness.session import _cap, _parse_status


pytestmark = needs_posix


def backend(root: Path, session: ShellSession | None = None, **kwargs: object) -> SessionBackend:
    return SessionBackend(
        session or ShellSession(),
        root=str(root),
        state_dir=str(root.parent / "state"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_fake_is_a_session() -> None:
    assert isinstance(ShellSession(), SessionLike)


def test_the_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        SessionBackend(ShellSession(), root="relative")


def test_a_session_that_runs_on_the_host_must_say_so(root: Path) -> None:
    caps = backend(root, isolated=False, label="docker:dev").capabilities
    assert not caps.isolated
    assert caps.label == "docker:dev"


def test_ripgrep_is_probed_once(root: Path) -> None:
    session = ShellSession()
    b = backend(root, session)
    assert b.capabilities == b.capabilities
    assert sum("command -v rg" in c for c in session.commands) == 1
    assert b.session is session


def test_a_failed_download_is_classified(root: Path) -> None:
    session = ShellSession()
    b = backend(root, session)
    (root / "a.txt").write_text("x")
    session.fail_download = True
    with pytest.raises(BackendError) as caught:
        b.read_bytes("a.txt")
    assert caught.value.code == "permission_denied"


def test_a_failed_upload_is_permission_denied(root: Path) -> None:
    class Refusing(ShellSession):
        def upload_file(self, path: str, data: bytes) -> None:
            raise OSError("read-only")

    with pytest.raises(BackendError) as caught:
        backend(root, Refusing()).write_bytes("a.txt", b"x")
    assert caught.value.code == "permission_denied"


def test_null_bytes_in_a_path_are_invalid(root: Path) -> None:
    with pytest.raises(BackendError):
        backend(root).resolve("a\0b")


def test_a_root_of_slash_contains_everything() -> None:
    b = SessionBackend(ShellSession(), root="/")
    assert b.resolve("etc/hosts") == "/etc/hosts"


def test_a_broken_stat_reply_is_permission_denied(root: Path) -> None:
    class Broken(ShellSession):
        def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
            if "sha256sum" in command:
                return 0, b"weird reply"
            if "stat -c" in command and "find" not in command:
                return 1, b""
            return super().exec(command, timeout=timeout)

    b = backend(root, Broken())
    with pytest.raises(BackendError) as caught:
        b.stat("a.txt")
    assert caught.value.code == "file_not_found"


def test_long_output_keeps_its_head_and_tail() -> None:
    from tulip.harness.local import HEAD_KEEP, TAIL_KEEP

    data = b"h" * HEAD_KEEP + b"m" * 10 + b"t" * TAIL_KEEP
    capped, truncated = _cap(data)
    assert truncated
    assert b"10 bytes of output were dropped" in capped
    assert _cap(b"short") == (b"short", False)


def test_status_lines_parse() -> None:
    assert _parse_status("r -") == (True, None)
    assert _parse_status("x 3") == (False, 3)
    assert _parse_status("x ???") == (False, -1)


def test_a_huge_background_output_is_read_as_head_and_tail(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tulip.harness import session

    monkeypatch.setattr(session, "HEAD_KEEP", 10)
    monkeypatch.setattr(session, "TAIL_KEEP", 10)
    b = backend(root)
    job = b.start_background("seq 1 100", foreground=True)
    b.wait_background(job.handle, 5)
    read = b.read_background(job.handle)
    assert read.lost > 0
    assert b"bytes of output were dropped here" in read.data
    assert read.data.endswith(b"99\n100\n")


def test_a_job_that_cannot_start_says_so(root: Path) -> None:
    class Failing(ShellSession):
        def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
            if "setsid" in command:
                return 1, b""
            return super().exec(command, timeout=timeout)

    with pytest.raises(JobError, match="could not start"):
        backend(root, Failing()).start_background("true")


def test_a_job_killed_from_outside_has_ended(root: Path) -> None:
    import os
    import signal
    import time

    b = backend(root)
    job = b.start_background("sleep 30")
    assert job.pid is not None
    os.killpg(job.pid, signal.SIGKILL)
    time.sleep(0.2)
    (status,) = b.list_background()
    assert not status.running
    assert status.exit_code == -1


def test_writing_to_a_job_whose_input_hangs_says_it_is_not_reading(
    root: Path,
) -> None:
    class Stuck(ShellSession):
        def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
            if "base64 -d" in command:
                return 33, b""
            return super().exec(command, timeout=timeout)

    b = backend(root, Stuck())
    job = b.start_background("sleep 30")
    with pytest.raises(JobError, match="not reading its input"):
        b.write_background(job.handle, b"x")
    b.close()


def test_close_kills_and_cleans_up(root: Path) -> None:
    b = backend(root)
    job = b.start_background("sleep 30")
    b.close()
    assert b.list_background() == []
    import os

    assert job.pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(job.pid, 0)
    assert not any((root.parent / "state").iterdir())


def test_unreadable_replies_are_skipped_or_refused(root: Path) -> None:
    class Garbled(ShellSession):
        def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
            if "sha256sum" in command:
                return 1, b""
            if "find" in command:
                return 0, b"not a listing line\n"
            if "grep" in command and "-nIH" in command:
                return 0, b"no colon here\n"
            return super().exec(command, timeout=timeout)

    b = backend(root, Garbled())
    (root / "a.txt").write_text("x")
    with pytest.raises(BackendError) as caught:
        b.stat("a.txt")
    assert caught.value.code == "permission_denied"
    assert b.ls(".") == []
    assert b.grep("x") == []
