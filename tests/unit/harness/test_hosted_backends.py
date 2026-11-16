# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""E2BBackend and DaytonaBackend over stand-ins for the vendors' clients.

The stand-ins run commands with this machine's ``sh`` and keep files on
disk, the way the real services would, so the shared contract and the
conformance kit run on both adapters; the mapping of each client's results
and exceptions is pinned with scripted doubles. The live tests under
``tests/integration`` need an account and run only when its key is set.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from locus.harness import DaytonaBackend, E2BBackend, SandboxBackend
from locus.harness import daytona as daytona_mod
from locus.harness import e2b as e2b_mod
from locus.harness.testing import run_conformance
from tests.unit.harness.conftest import _run, needs_posix


# --------------------------------------------------------------------- E2B --


class TimeoutException(Exception):  # noqa: N818 - the client's own name
    pass


class NotFoundException(Exception):  # noqa: N818 - the client's own name
    pass


class CommandExitException(Exception):  # noqa: N818 - the client's own name
    def __init__(self, stderr: str, stdout: str, exit_code: int) -> None:
        super().__init__(f"exit {exit_code}")
        self.stderr, self.stdout, self.exit_code = stderr, stdout, exit_code


class FakeE2BCommands:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        cmd: str,
        *,
        timeout: float | None = 60,
        on_stdout: Any = None,
        on_stderr: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({"cmd": cmd, "timeout": timeout})
        code, out, err = _run(["sh", "-c", cmd], timeout=timeout, stdin=None, merge=False)
        if out and on_stdout is not None:
            on_stdout(out.decode(errors="replace"))
        if err and on_stderr is not None:
            on_stderr(err.decode(errors="replace"))
        if code is None:
            raise TimeoutException("command timed out")
        if code != 0:
            raise CommandExitException(
                err.decode(errors="replace"), out.decode(errors="replace"), code
            )
        return types.SimpleNamespace(
            stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"), exit_code=0
        )


class FakeE2BFiles:
    def read(self, path: str, format: str = "text") -> bytearray:  # noqa: A002 - the client's name
        target = Path(path)
        if not target.is_file():
            raise NotFoundException(path)
        return bytearray(target.read_bytes())

    def write(self, path: str, data: bytes) -> None:
        Path(path).write_bytes(data)


class FakeE2BSandbox:
    def __init__(self, sandbox_id: str = "sbx-e2b") -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeE2BCommands()
        self.files = FakeE2BFiles()
        self.killed = False

    def kill(self) -> bool:
        self.killed = True
        return True


def e2b_backend(root: Path, sandbox: FakeE2BSandbox | None = None, **options: Any) -> E2BBackend:
    return E2BBackend(
        sandbox or FakeE2BSandbox(), root=str(root), state_dir=str(root.parent / "s"), **options
    )


def test_e2b_is_an_isolated_sandbox_with_a_label(root: Path) -> None:
    backend = e2b_backend(root)
    assert isinstance(backend, SandboxBackend)
    assert backend.capabilities.isolated
    assert backend.capabilities.label == "e2b:sbx-e2b"
    assert E2BBackend(FakeE2BSandbox()).capabilities.root == "/home/user"


@needs_posix
def test_e2b_passes_the_conformance_kit(root: Path) -> None:
    report = run_conformance(e2b_backend(root), timeouts=False)
    assert report.ok, report.summary()


class ScriptedE2B:
    """An ``e2b`` sandbox whose ``commands.run`` does what the test says."""

    def __init__(self, outcome: Any) -> None:
        self.sandbox_id = "scripted"
        self.files = FakeE2BFiles()
        self.commands = types.SimpleNamespace(run=self._run)
        self.outcome = outcome

    def _run(self, cmd: str, **kwargs: Any) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_e2b_maps_results_and_exceptions() -> None:
    ok = e2b_mod.E2BSession(ScriptedE2B(types.SimpleNamespace(stdout="o", stderr="e", exit_code=0)))
    assert ok.run("x", timeout=5) == (0, b"oe")
    failed = e2b_mod.E2BSession(ScriptedE2B(CommandExitException("err", "out", 7)))
    assert failed.run("x", timeout=5) == (7, b"outerr")
    timed_out = e2b_mod.E2BSession(ScriptedE2B(TimeoutException("slow")))
    assert timed_out.run("x", timeout=5) == (None, b"")
    with pytest.raises(ValueError, match="other"):
        e2b_mod.E2BSession(ScriptedE2B(ValueError("other"))).run("x", timeout=5)


def test_e2b_a_missing_file_is_not_found(root: Path) -> None:
    session = e2b_mod.E2BSession(FakeE2BSandbox())
    with pytest.raises(FileNotFoundError):
        session.download_file(str(root / "missing"))


def test_e2b_close_kills_only_a_sandbox_it_made(root: Path) -> None:
    owned_sandbox = FakeE2BSandbox()
    e2b_backend(root, owned_sandbox, owns_sandbox=True).close()
    assert owned_sandbox.killed
    borrowed = FakeE2BSandbox()
    e2b_backend(root, borrowed).close()
    assert not borrowed.killed


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    made: dict[str, Any] = {}

    class Sandbox(FakeE2BSandbox):
        @classmethod
        def create(cls, **kwargs: Any) -> Sandbox:
            made["create"] = kwargs
            return cls("created")

        @classmethod
        def connect(cls, sandbox_id: str, **kwargs: Any) -> Sandbox:
            made["connect"] = (sandbox_id, kwargs)
            return cls(sandbox_id)

    module = types.ModuleType("e2b")
    module.Sandbox = Sandbox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "e2b", module)
    return made


def test_e2b_create_owns_and_connect_borrows(fake_e2b: dict[str, Any], root: Path) -> None:
    created = E2BBackend.create(
        template="base", lifetime_s=120, api_key="k", envs={"A": "1"}, root=str(root)
    )
    assert fake_e2b["create"] == {
        "template": "base",
        "timeout": 120,
        "envs": {"A": "1"},
        "api_key": "k",
    }
    assert created.capabilities.label == "e2b:created"
    created._jobs_dir = "/nonexistent"
    created.close()
    assert created.sandbox.killed
    connected = E2BBackend.connect("sbx-9", root=str(root))
    assert fake_e2b["connect"] == ("sbx-9", {})
    connected._jobs_dir = "/nonexistent"
    connected.close()
    assert not connected.sandbox.killed


def test_without_e2b_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "e2b", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[e2b\]"):
        E2BBackend.connect("sbx")


# ----------------------------------------------------------------- Daytona --


class DaytonaProcessExecutionTimeoutError(Exception):
    pass


class DaytonaFileNotFoundError(Exception):
    pass


class FakeDaytonaProcess:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        self.calls.append({"command": command, "timeout": timeout})
        code, out, _ = _run(["sh", "-c", command], timeout=timeout, stdin=None, merge=True)
        if code is None:
            raise DaytonaProcessExecutionTimeoutError("timed out")
        return types.SimpleNamespace(exit_code=code, result=out.decode(errors="replace"))


class FakeDaytonaFS:
    def upload_file(self, src: bytes, dst: str, timeout: int = 1800) -> None:
        Path(dst).write_bytes(src)

    def download_file(self, path: str) -> bytes | None:
        target = Path(path)
        if not target.is_file():
            raise DaytonaFileNotFoundError(path)
        return target.read_bytes()


class FakeDaytonaSandbox:
    def __init__(self, root: str = "/home/daytona", sandbox_id: str = "sbx-dt") -> None:
        self.id = sandbox_id
        self.root = root
        self.process = FakeDaytonaProcess()
        self.fs = FakeDaytonaFS()
        self.events: list[str] = []

    def get_work_dir(self) -> str:
        return self.root

    def start(self, timeout: float | None = 60) -> None:
        self.events.append("start")

    def stop(self, timeout: float | None = 60, force: bool = False) -> None:
        self.events.append("stop")

    def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        self.events.append("delete")


def daytona_backend(
    root: Path, sandbox: FakeDaytonaSandbox | None = None, **options: Any
) -> DaytonaBackend:
    return DaytonaBackend(
        sandbox or FakeDaytonaSandbox(str(root)), state_dir=str(root.parent / "s"), **options
    )


def test_daytona_is_an_isolated_sandbox_rooted_at_its_work_dir(root: Path) -> None:
    backend = daytona_backend(root)
    assert isinstance(backend, SandboxBackend)
    assert backend.capabilities.isolated
    assert backend.capabilities.label == "daytona:sbx-dt"
    assert backend.capabilities.root == str(root), "the work dir when no root is given"


@needs_posix
def test_daytona_passes_the_conformance_kit(root: Path) -> None:
    report = run_conformance(daytona_backend(root), timeouts=False)
    assert report.ok, report.summary()


def test_daytona_maps_results_and_timeouts() -> None:
    sandbox = FakeDaytonaSandbox()
    sandbox.process = types.SimpleNamespace(  # type: ignore[assignment]
        exec=lambda command, timeout=None, **kw: types.SimpleNamespace(exit_code=5, result="out")
    )
    assert daytona_mod.DaytonaSession(sandbox).run("x", timeout=2.2) == (5, b"out")

    def slow(command: str, timeout: int | None = None, **kw: Any) -> Any:
        raise DaytonaProcessExecutionTimeoutError("slow")

    sandbox.process = types.SimpleNamespace(exec=slow)  # type: ignore[assignment]
    assert daytona_mod.DaytonaSession(sandbox).run("x", timeout=2) == (None, b"")


def test_daytona_a_missing_file_is_not_found(root: Path) -> None:
    session = daytona_mod.DaytonaSession(FakeDaytonaSandbox(str(root)))
    with pytest.raises(FileNotFoundError):
        session.download_file(str(root / "missing"))
    sandbox = FakeDaytonaSandbox(str(root))
    sandbox.fs = types.SimpleNamespace(download_file=lambda path: None)  # type: ignore[assignment]
    with pytest.raises(FileNotFoundError):
        daytona_mod.DaytonaSession(sandbox).download_file("x")


def test_daytona_lifecycle_stops_starts_and_deletes_what_it_owns(root: Path) -> None:
    sandbox = FakeDaytonaSandbox(str(root))
    backend = daytona_backend(root, sandbox, owns_sandbox=True)
    backend._jobs_dir = "/nonexistent"
    backend.stop()
    backend.start()
    backend.close()
    assert sandbox.events == ["stop", "start", "stop", "delete"]
    borrowed = FakeDaytonaSandbox(str(root))
    b = daytona_backend(root, borrowed)
    b._jobs_dir = "/nonexistent"
    b.close()
    assert borrowed.events == ["stop"]


@pytest.fixture
def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    made: dict[str, Any] = {}

    class DaytonaConfig:
        def __init__(self, **fields: Any) -> None:
            made["config"] = fields

    class Daytona:
        def __init__(self, config: Any = None) -> None:
            made["configured"] = config is not None

        def create(self, params: Any) -> FakeDaytonaSandbox:
            made["create"] = params
            return FakeDaytonaSandbox("/w", "created")

        def get(self, sandbox_id: str) -> FakeDaytonaSandbox:
            made["get"] = sandbox_id
            return FakeDaytonaSandbox("/w", sandbox_id)

    class Params:
        def __init__(self, **fields: Any) -> None:
            self.fields = fields

    module = types.ModuleType("daytona")
    module.Daytona = Daytona  # type: ignore[attr-defined]
    module.DaytonaConfig = DaytonaConfig  # type: ignore[attr-defined]
    module.CreateSandboxFromSnapshotParams = type("Snapshot", (Params,), {})  # type: ignore[attr-defined]
    module.CreateSandboxFromImageParams = type("Image", (Params,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "daytona", module)
    return made


def test_daytona_create_owns_and_connect_borrows(fake_daytona: dict[str, Any]) -> None:
    created = DaytonaBackend.create(image="python:3.12", lifetime_minutes=30, api_key="k")
    assert fake_daytona["config"] == {"api_key": "k"}
    assert type(fake_daytona["create"]).__name__ == "Image"
    assert fake_daytona["create"].fields == {"image": "python:3.12", "ttl_minutes": 30}
    assert created.capabilities.label == "daytona:created"
    assert created.capabilities.root == "/w"
    created._jobs_dir = "/nonexistent"
    created.close()
    assert created.sandbox.events[-1] == "delete"

    connected = DaytonaBackend.connect("sbx-3")
    assert fake_daytona["get"] == "sbx-3"
    assert fake_daytona["configured"] is False, "no settings: the client reads its environment"
    connected._jobs_dir = "/nonexistent"
    connected.close()
    assert "delete" not in connected.sandbox.events


def test_daytona_create_from_a_snapshot_by_default(fake_daytona: dict[str, Any]) -> None:
    DaytonaBackend.create(snapshot="snap", language="python", lifetime_minutes=None)
    assert type(fake_daytona["create"]).__name__ == "Snapshot"
    assert fake_daytona["create"].fields == {"snapshot": "snap", "language": "python"}


def test_without_daytona_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "daytona", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[daytona\]"):
        DaytonaBackend.connect("sbx")
