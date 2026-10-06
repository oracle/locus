# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The workspaces every harness test runs against.

The same behavioural tests run on four backends:

- ``memory`` — :class:`MemoryBackend`, file operations only;
- ``local`` — :class:`LocalBackend` on a temporary directory;
- ``session`` — :class:`SessionBackend` over :class:`ShellSession`, a session
  that runs each command with this machine's ``sh``. The backend's shell
  scripts run for real, so what is tested is what a sandbox would run;
- ``openshell`` — :class:`OpenShellBackend` over :class:`FakeOpenShellClient`,
  which implements the ``openshell.SandboxClient`` methods the backend calls
  (``exec_stream`` yielding chunks then a result, ``create``, ``wait_ready``,
  ``delete``) the same way: commands in ``sh``, stdin bytes on standard input,
  exit status 124 when ``timeout_seconds`` runs out.

The remote backends are rooted at a temporary directory instead of
``/sandbox``, which is the only difference from a real sandbox.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tulip.harness import (
    LocalBackend,
    MemoryBackend,
    OpenShellBackend,
    SessionBackend,
    WorkspaceBackend,
)


_REMOTE_TOOLS = ("setsid", "mkfifo", "base64", "timeout", "sha256sum", "tar", "awk", "find")
_HAS_TOOLS = all(shutil.which(t) for t in _REMOTE_TOOLS)
needs_posix = pytest.mark.skipif(not _HAS_TOOLS, reason="needs a Linux userland")


def _run(
    argv: Sequence[str], *, timeout: float | None, stdin: bytes | None, merge: bool
) -> tuple[int | None, bytes, bytes]:
    """Run ``argv`` in its own process group; kill the group at the timeout."""
    proc = subprocess.Popen(  # noqa: S603 - the test sandbox runs what the backend sends
        list(argv),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge else subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return None, out or b"", err or b""
    return proc.returncode, out or b"", err or b""


class ShellSession:
    """A :class:`SessionLike` that is this machine's shell."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.fail_download = False

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        self.commands.append(command)
        code, out, _ = _run(["sh", "-c", command], timeout=timeout, stdin=None, merge=True)
        return code, out

    def upload_file(self, path: str, data: bytes) -> None:
        Path(path).write_bytes(data)

    def download_file(self, path: str) -> bytes:
        if self.fail_download:
            raise RuntimeError("transport down")
        return Path(path).read_bytes()


@dataclass(frozen=True)
class Chunk:
    stream: str
    data: bytes


@dataclass(frozen=True)
class Result:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Ref:
    name: str
    workspace: str


@dataclass
class FakeOpenShellClient:
    """The ``SandboxClient`` surface the backend uses, run with ``sh``."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    raise_after: float | None = None

    def exec_stream(
        self,
        sandbox: str,
        command: Sequence[str],
        *,
        workspace: str,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> Iterator[Any]:
        self.calls.append(
            {"sandbox": sandbox, "command": list(command), "workspace": workspace, "stdin": stdin}
        )
        if self.raise_after is not None:
            import time

            time.sleep(self.raise_after)
            raise RuntimeError("deadline exceeded")
        code, out, err = _run(command, timeout=timeout_seconds, stdin=stdin, merge=False)
        if out:
            yield Chunk("stdout", out)
        if err:
            yield Chunk("stderr", err)
        # The gateway reports a command it stopped at its deadline as 124.
        yield Result(124 if code is None else code, out.decode(errors="replace"), "")

    def create(self, *, workspace: str, name: str | None, spec: Any, labels: Any) -> Ref:
        ref = Ref(name or "sbx-1", workspace)
        self.created.append(ref.name)
        return ref

    def wait_ready(self, name: str, *, workspace: str, timeout_seconds: float) -> Ref:
        return Ref(name, workspace)

    def delete(self, name: str, *, workspace: str, allow_missing: bool = False) -> None:
        self.deleted.append(name)


def make_backend(kind: str, root: Path) -> WorkspaceBackend:
    if kind == "memory":
        return MemoryBackend()
    if kind == "local":
        return LocalBackend(root)
    state = root.parent / f"{root.name}-state"
    if kind == "session":
        return SessionBackend(ShellSession(), root=str(root), state_dir=str(state))
    return OpenShellBackend(
        FakeOpenShellClient(), "sbx", workspace="team", root=str(root), state_dir=str(state)
    )


FILE_BACKENDS = [
    "memory",
    "local",
    pytest.param("session", marks=needs_posix),
    pytest.param("openshell", marks=needs_posix),
]
SHELL_BACKENDS = [
    "local",
    pytest.param("session", marks=needs_posix),
    pytest.param("openshell", marks=needs_posix),
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


def _closing(backend: WorkspaceBackend) -> Iterator[WorkspaceBackend]:
    yield backend
    close: Callable[[], None] | None = getattr(backend, "close", None)
    if close is not None:
        close()


@pytest.fixture(params=FILE_BACKENDS)
def workspace(request: pytest.FixtureRequest, root: Path) -> Iterator[WorkspaceBackend]:
    """Every backend, for file behaviour."""
    yield from _closing(make_backend(request.param, root))


@pytest.fixture(params=SHELL_BACKENDS)
def shell(request: pytest.FixtureRequest, root: Path) -> Iterator[WorkspaceBackend]:
    """Every backend with a shell."""
    yield from _closing(make_backend(request.param, root))


def put(backend: WorkspaceBackend, path: str, content: str | bytes) -> None:
    """Create a file behind the tools' back, the way a person or formatter would."""
    data = content.encode() if isinstance(content, str) else content
    backend.write_bytes(path, data)


def get(backend: WorkspaceBackend, path: str) -> str:
    return backend.read_bytes(path).decode()
