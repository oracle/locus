# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The Docker sandbox against a real daemon: the boundary holds.

Skipped unless the ``docker`` CLI reaches a daemon and ``python:3.12-slim`` is
already pulled (a pull inside a test would count against its timeout).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tulip.tools.sandbox import DockerSandbox, SandboxManifest


_IMAGE = "python:3.12-slim"


def _ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    probe = subprocess.run(  # noqa: S603
        [docker, "image", "inspect", _IMAGE], capture_output=True, text=True, check=False
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _ready(), reason=f"needs a Docker daemon and {_IMAGE}")


def test_a_tool_runs_and_returns_its_value() -> None:
    result = DockerSandbox(_IMAGE).run_tool(
        "def add(a, b):\n    return a + b", "add", {"a": 2, "b": 3}, timeout=60
    )

    assert result.ok, result.stderr
    assert result.value == 5


def test_there_is_no_network() -> None:
    code = (
        "def reach():\n"
        "    import socket\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    except OSError:\n"
        "        return 'blocked'\n"
        "    return 'reached'\n"
    )

    result = DockerSandbox(_IMAGE).run_tool(code, "reach", timeout=60)

    assert result.ok, result.stderr
    assert result.value == "blocked"


def test_the_root_filesystem_is_read_only_and_the_workspace_is_not() -> None:
    code = (
        "def write():\n"
        "    outcome = {}\n"
        "    try:\n"
        "        open('/etc/tulip-probe', 'w').write('x')\n"
        "        outcome['root'] = 'written'\n"
        "    except OSError:\n"
        "        outcome['root'] = 'refused'\n"
        "    open('/work/out.txt', 'w').write('ok')\n"
        "    outcome['work'] = open('/work/out.txt').read()\n"
        "    return outcome\n"
    )

    result = DockerSandbox(_IMAGE).run_tool(code, "write", timeout=60)

    assert result.ok, result.stderr
    assert result.value == {"root": "refused", "work": "ok"}


def test_only_granted_environment_is_visible() -> None:
    code = "def env():\n    import os\n    return sorted(k for k in os.environ if k in ('API_TOKEN', 'HOME_SECRET'))\n"

    result = DockerSandbox(_IMAGE).run_tool(
        code, "env", manifest=SandboxManifest(env={"API_TOKEN": "t0k"}), timeout=60
    )

    assert result.ok, result.stderr
    assert result.value == ["API_TOKEN"]


def test_a_timeout_kills_the_container() -> None:
    result = DockerSandbox(_IMAGE).run_code("import time\ntime.sleep(60)", timeout=5)

    assert result.timed_out
    left = subprocess.run(  # noqa: S603
        [shutil.which("docker") or "docker", "ps", "-q", "--filter", "name=tulip-sandbox-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert left.stdout.strip() == "", "the timed-out container must not keep running"
