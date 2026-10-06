# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The coding harness against a real OpenShell gateway.

Skipped unless ``TULIP_OPENSHELL_ENDPOINT`` names a gateway (``host:port``)
and the ``openshell`` package is installed. ``TULIP_OPENSHELL_WORKSPACE``
picks the workspace (default ``default``); ``TULIP_OPENSHELL_TOKEN`` is sent
as the bearer token when set. A sandbox is created for the module and
deleted after it.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest


ENDPOINT = os.environ.get("TULIP_OPENSHELL_ENDPOINT", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENDPOINT, reason="TULIP_OPENSHELL_ENDPOINT is not set"),
]

pytest.importorskip("openshell")

from tulip.harness import OpenShellBackend, build_harness  # noqa: E402


@pytest.fixture(scope="module")
def sandbox() -> Iterator[OpenShellBackend]:
    options: dict[str, object] = {}
    token = os.environ.get("TULIP_OPENSHELL_TOKEN")
    if token:
        options["bearer_token"] = token
    backend = OpenShellBackend.create(
        workspace=os.environ.get("TULIP_OPENSHELL_WORKSPACE", "default"),
        name=f"tulip-harness-{uuid.uuid4().hex[:8]}",
        endpoint=ENDPOINT,
        **options,
    )
    try:
        yield backend
    finally:
        backend.close()


def test_files_round_trip_in_the_sandbox(sandbox: OpenShellBackend) -> None:
    blob = bytes(range(256)) * 10
    sandbox.write_bytes("live/blob.bin", blob)
    assert sandbox.read_bytes("live/blob.bin") == blob
    assert sandbox.stat("live/blob.bin").size == len(blob)


def test_commands_run_in_the_workspace_root(sandbox: OpenShellBackend) -> None:
    result = sandbox.exec("pwd", timeout=30)
    assert result.exit_code == 0
    assert result.output.decode().strip() == sandbox.capabilities.root


def test_the_gateway_stops_a_command_at_its_deadline(sandbox: OpenShellBackend) -> None:
    result = sandbox.exec("sleep 30", timeout=2)
    assert result.timed_out


def test_a_background_command_runs_between_calls(sandbox: OpenShellBackend) -> None:
    job = sandbox.start_background("cat")
    sandbox.write_background(job.handle, b"ping\n", close=True)
    status = sandbox.wait_background(job.handle, 10)
    assert not status.running
    assert sandbox.read_background(job.handle).data == b"ping\n"
    sandbox.release_background(job.handle)


async def test_the_harness_edits_and_runs_in_the_sandbox(sandbox: OpenShellBackend) -> None:
    h = build_harness(sandbox)
    await h.tool("write").execute(path="live/app.py", content="print(1 + 1)\n")
    await h.tool("read").execute(path="live/app.py")
    await h.tool("edit").execute(path="live/app.py", old="1 + 1", new="40 + 2")
    out = await h.tool("bash").execute(command="python3 live/app.py || echo 42")
    assert "42" in out
    started = time.monotonic()
    assert "live/app.py" in await h.tool("glob").execute(pattern="**/*.py")
    assert time.monotonic() - started < 60
