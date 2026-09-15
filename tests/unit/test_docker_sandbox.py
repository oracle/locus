# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The Docker sandbox, checked without a daemon.

The container flags are the boundary, so these tests pin them exactly: no
network unless asked, read-only root, no capabilities, no privilege escalation,
resource limits, and nothing in the environment the manifest did not grant.
Live behaviour against a real daemon is in
``tests/integration/test_docker_sandbox_live.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.tools import sandbox as sandbox_module
from tulip.tools.sandbox import (
    TOOL_RESULT_MARKER,
    DockerSandbox,
    SandboxError,
    SandboxManifest,
    SandboxResult,
    SandboxSpec,
    resolve_sandbox,
)


class _Docker:
    """Stands in for ``subprocess.run`` and records what the container would get."""

    def __init__(
        self, *, returncode: int = 0, stdout: str = "", stderr: str = "", timeout: bool = False
    ) -> None:
        self.returncode, self.stdout, self.stderr, self.timeout = (
            returncode,
            stdout,
            stderr,
            timeout,
        )
        self.calls: list[list[str]] = []
        self.workspace: dict[str, str] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if argv[1] == "run":
            mount = argv[argv.index("--mount") + 1]
            root = Path(mount.split("source=", 1)[1].split(",", 1)[0])
            self.workspace = {p.name: p.read_text() for p in root.iterdir() if p.is_file()}
            if self.timeout:
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0), output="partial")
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)

    @property
    def run_argv(self) -> list[str]:
        return next(call for call in self.calls if call[1] == "run")


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> _Docker:
    fake = _Docker()
    monkeypatch.setattr(sandbox_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sandbox_module.subprocess, "run", fake)
    return fake


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_the_container_is_locked_down_by_default(docker: _Docker) -> None:
    result = DockerSandbox().run_code("print('hi')")

    argv = docker.run_argv
    assert result.ok
    assert _flag(argv, "--network") == "none"
    assert "--read-only" in argv
    assert "--rm" in argv
    assert _flag(argv, "--cap-drop") == "ALL"
    assert _flag(argv, "--security-opt") == "no-new-privileges"
    assert _flag(argv, "--memory") == "512m"
    assert _flag(argv, "--cpus") == "1.0"
    assert _flag(argv, "--pids-limit") == "256"
    assert _flag(argv, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert argv[-4:] == ["python:3.12-slim", "python", "-I", "/work/_program.py"]
    assert docker.workspace["_program.py"] == "print('hi')"


def test_network_and_limits_are_configurable(docker: _Docker) -> None:
    DockerSandbox(
        "python:3.13-slim", network=True, memory="1g", cpus=2.0, pids_limit=64, user="1000:1000"
    ).run_code("pass")

    argv = docker.run_argv
    assert _flag(argv, "--network") == "bridge"
    assert (_flag(argv, "--memory"), _flag(argv, "--cpus"), _flag(argv, "--pids-limit")) == (
        "1g",
        "2.0",
        "64",
    )
    assert _flag(argv, "--user") == "1000:1000"
    assert "python:3.13-slim" in argv


def test_only_granted_environment_reaches_the_box(docker: _Docker) -> None:
    DockerSandbox().run_code("pass", manifest=SandboxManifest(env={"API_TOKEN": "t0k"}))

    argv = docker.run_argv
    granted = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    assert sorted(granted) == ["API_TOKEN=t0k", "LANG=C.UTF-8"]


def test_a_tool_value_round_trips(docker: _Docker) -> None:
    docker.stdout = f"noise\n{TOOL_RESULT_MARKER}{json.dumps({'sum': 5})}\n"

    result = DockerSandbox().run_tool(
        "def add(a, b):\n    return {'sum': a + b}", "add", {"a": 2, "b": 3}
    )

    assert isinstance(result, SandboxResult)
    assert result.value == {"sum": 5}
    assert json.loads(docker.workspace["_tool_args.json"]) == {"a": 2, "b": 3}
    assert "add(**_args)" in docker.workspace["_program.py"]


def test_a_timeout_kills_the_container(docker: _Docker) -> None:
    docker.timeout = True

    result = DockerSandbox().run_code("import time; time.sleep(60)", timeout=0.5)

    assert (result.ok, result.timed_out, result.exit_code) == (False, True, 124)
    assert result.stdout == "partial"
    name = _flag(docker.run_argv, "--name")
    assert docker.calls[-1] == ["/usr/bin/docker", "kill", name]


def test_a_docker_error_is_a_failed_run_not_an_exception(docker: _Docker) -> None:
    docker.returncode, docker.stderr = 125, "Unable to find image 'nope:latest' locally"

    result = DockerSandbox("nope:latest").run_code("pass")

    assert (result.ok, result.exit_code) == (False, 125)
    assert "Unable to find image" in result.stderr


def test_deps_without_a_network_are_refused(docker: _Docker) -> None:
    with pytest.raises(SandboxError, match="network=True"):
        DockerSandbox().run_code("pass", manifest=SandboxManifest(deps=["requests"]))
    assert docker.calls == []


def test_a_missing_docker_cli_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_module.shutil, "which", lambda name: None)

    with pytest.raises(SandboxError, match="CLI"):
        DockerSandbox().run_code("pass")


def test_docker_is_a_built_in_provider_name() -> None:
    assert isinstance(resolve_sandbox(SandboxSpec(provider="docker")), DockerSandbox)


class _Firecracker:
    def run_tool(self, code: str, func: str, args: Any = None, **kwargs: Any) -> Any:
        return SandboxResult(ok=True)


def _entry_points(monkeypatch: pytest.MonkeyPatch, *entries: Any) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: list(entries))


@pytest.mark.parametrize(
    "target",
    [_Firecracker, lambda: _Firecracker(), _Firecracker()],
    ids=["class", "factory", "instance"],
)
def test_other_providers_resolve_through_entry_points(
    monkeypatch: pytest.MonkeyPatch, target: Any
) -> None:
    _entry_points(
        monkeypatch, SimpleNamespace(name="firecracker", value="pkg:Provider", load=lambda: target)
    )

    assert isinstance(resolve_sandbox(SandboxSpec(provider="Firecracker")), _Firecracker)


def test_an_entry_point_that_is_not_a_provider_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _entry_points(
        monkeypatch, SimpleNamespace(name="broken", value="pkg:nothing", load=lambda: lambda: 42)
    )

    with pytest.raises(SandboxError, match="run_tool"):
        resolve_sandbox(SandboxSpec(provider="broken"))


def test_an_unknown_provider_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _entry_points(
        monkeypatch, SimpleNamespace(name="other", value="pkg:X", load=lambda: _Firecracker)
    )

    with pytest.raises(SandboxError, match="unknown sandbox provider 'firecracker'"):
        resolve_sandbox(SandboxSpec(provider="firecracker"))
