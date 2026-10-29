# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The workspaces every harness test runs against.

The same behavioural tests run on five backends:

- ``memory`` — :class:`MemoryBackend`, file operations only;
- ``local`` — :class:`LocalBackend` on a temporary directory;
- ``session`` — :class:`SessionBackend` over :class:`ShellSession`, a session
  that runs each command with this machine's ``sh``. The backend's shell
  scripts run for real, so what is tested is what a sandbox would run;
- ``openshell`` — :class:`OpenShellBackend` over :class:`FakeOpenShellClient`,
  which implements the ``openshell.SandboxClient`` methods the backend calls
  (``exec_stream`` yielding chunks then a result, ``create``, ``wait_ready``,
  ``delete``) the same way: commands in ``sh``, stdin bytes on standard input,
  exit status 124 when ``timeout_seconds`` runs out;
- ``oci_sandbox`` — :class:`OCISandboxBackend` over :class:`FakeOCISandboxClient`,
  which implements the ``GenerativeAiSandboxClient`` methods the backend calls
  the same way the service does: commands in ``sh`` streamed as NDJSON events
  with sequence numbers, ``exitCode -1`` when the service timeout runs out,
  files written and read whole, a 412 for a wrong checksum and a 404 for a
  missing file.

The remote backends are rooted at a temporary directory instead of
``/sandbox``, which is the only difference from a real sandbox.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.harness import (
    LocalBackend,
    MemoryBackend,
    OCISandboxBackend,
    OpenShellBackend,
    SessionBackend,
    WorkspaceBackend,
    oci_sandbox,
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
    specs: list[Any] = field(default_factory=list)
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
        self.specs.append(spec)
        return ref

    def wait_ready(self, name: str, *, workspace: str, timeout_seconds: float) -> Ref:
        return Ref(name, workspace)

    def delete(self, name: str, *, workspace: str, allow_missing: bool = False) -> None:
        self.deleted.append(name)


#: The backend's real model loader, for the tests that check its error when
#: the SDK is missing.
REAL_OCI_MODELS = oci_sandbox._models


def _model(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def _stand_in_oci_models() -> SimpleNamespace:
    """The request models the backend builds, for an ``oci`` without sandboxes."""
    return SimpleNamespace(
        RunSandboxCommandDetails=_model,
        CreateSandboxDetails=_model,
        StopSandboxDetails=_model,
        ExtendSandboxDetails=_model,
    )


@pytest.fixture(autouse=True)
def _oci_sandbox_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the SDK's request models when it has them, stand-ins otherwise.

    ``oci.generative_ai_sandbox`` ships from ``oci`` 2.189.0. An environment
    that resolves an older ``oci`` (CI's Python 3.14 job gets 2.188.0) still
    runs every backend test against the fake client.
    """
    try:
        import oci.generative_ai_sandbox.models  # noqa: F401
    except ImportError:
        monkeypatch.setattr(oci_sandbox, "_models", _stand_in_oci_models)


class FakeServiceError(Exception):
    """What ``oci.exceptions.ServiceError`` carries that the backend reads."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"{status} {code}")
        self.status = status
        self.code = code


def _seconds(duration: str) -> float:
    match = re.fullmatch(r"PT(\d+)S", duration)
    assert match, f"the backend sends whole seconds, got {duration!r}"
    return float(match.group(1))


@dataclass
class FakeOCISandboxClient:
    """The ``GenerativeAiSandboxClient`` surface the backend uses, run with ``sh``."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    states: list[str] = field(default_factory=lambda: ["RUNNING"])
    created: list[tuple[str, Any]] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    extended: list[str] = field(default_factory=list)

    def run_sandbox_command_and_stream(
        self, sandbox_id: str, details: Any, **kwargs: Any
    ) -> SimpleNamespace:
        self.calls.append(
            {"sandbox": sandbox_id, "command": details.command, "timeout": details.timeout}
        )
        code, out, err = _run(
            ["sh", "-c", details.command],
            timeout=_seconds(details.timeout),
            stdin=None,
            merge=False,
        )
        events: list[dict[str, Any]] = [{"event": "STARTED", "seq": 1}]
        for kind, data in (("STDOUT", out), ("STDERR", err)):
            if data:
                events.append(
                    {"event": kind, "seq": len(events) + 1, "data": data.decode(errors="replace")}
                )
        events.append(
            {"event": "FINISHED", "seq": len(events) + 1, "exitCode": -1 if code is None else code}
        )
        return SimpleNamespace(data=[json.dumps(e).encode() for e in events])

    def write_sandbox_file(
        self,
        sandbox_id: str,
        file_path: str,
        body: Any,
        opc_content_sha256: str | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        data = body.read()
        digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
        if opc_content_sha256 is not None and opc_content_sha256 != digest:
            raise FakeServiceError(412, "PreconditionFailed")
        self.writes.append({"sandbox": sandbox_id, "path": file_path, "size": len(data)})
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return SimpleNamespace(data=SimpleNamespace(content_sha256=digest))

    def read_sandbox_file(self, sandbox_id: str, file_path: str, **kwargs: Any) -> SimpleNamespace:
        target = Path(file_path)
        if not target.is_file():
            raise FakeServiceError(404, "not_found")
        return SimpleNamespace(data=SimpleNamespace(content=target.read_bytes()))

    def create_sandbox(self, project_id: str, details: Any, **kwargs: Any) -> SimpleNamespace:
        self.created.append((project_id, details))
        return SimpleNamespace(data=SimpleNamespace(id=f"sbx-{len(self.created)}"))

    def get_sandbox(self, sandbox_id: str, **kwargs: Any) -> SimpleNamespace:
        if sandbox_id in self.deleted:
            raise FakeServiceError(404, "not_found")
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(data=SimpleNamespace(lifecycle_state=state))

    def stop_sandbox(self, sandbox_id: str, details: Any, **kwargs: Any) -> SimpleNamespace:
        self.stopped.append(sandbox_id)
        self.states = ["STOPPED"]
        return SimpleNamespace(data=None)

    def delete_sandbox(self, sandbox_id: str, **kwargs: Any) -> None:
        self.deleted.append(sandbox_id)

    def extend_sandbox(self, sandbox_id: str, details: Any, **kwargs: Any) -> SimpleNamespace:
        self.extended.append(details.additional_duration)
        return SimpleNamespace(data=SimpleNamespace(time_expires="2026-10-25T10:00:00Z"))


def make_backend(kind: str, root: Path) -> WorkspaceBackend:
    if kind == "memory":
        return MemoryBackend()
    if kind == "local":
        return LocalBackend(root)
    state = root.parent / f"{root.name}-state"
    if kind == "session":
        return SessionBackend(ShellSession(), root=str(root), state_dir=str(state))
    if kind == "oci_sandbox":
        return OCISandboxBackend(
            FakeOCISandboxClient(),
            "sbx-1",
            region="eu-frankfurt-1",
            root=str(root),
            state_dir=str(state),
        )
    return OpenShellBackend(
        FakeOpenShellClient(), "sbx", workspace="team", root=str(root), state_dir=str(state)
    )


FILE_BACKENDS = [
    "memory",
    "local",
    pytest.param("session", marks=needs_posix),
    pytest.param("openshell", marks=needs_posix),
    pytest.param("oci_sandbox", marks=needs_posix),
]
SHELL_BACKENDS = [
    "local",
    pytest.param("session", marks=needs_posix),
    pytest.param("openshell", marks=needs_posix),
    pytest.param("oci_sandbox", marks=needs_posix),
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
