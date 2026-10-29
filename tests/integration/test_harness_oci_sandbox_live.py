# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The coding harness against a real OCI GenAI Sandbox.

Skipped unless ``LOCUS_OCI_SANDBOX_PROJECT_ID`` names a Generative AI project
with sandboxes enabled and the OCI SDK has ``oci.generative_ai_sandbox``.
``LOCUS_OCI_SANDBOX_REGION`` picks the region (the profile's by default),
``LOCUS_OCI_SANDBOX_PROFILE`` the ``~/.oci/config`` profile (``DEFAULT``) and
``LOCUS_OCI_SANDBOX_AUTH`` the auth type (``api_key``). One ``SMALL`` sandbox
is created for the module, with a 30-minute expiration as a backstop, and
stopped and deleted after it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest


PROJECT = os.environ.get("LOCUS_OCI_SANDBOX_PROJECT_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not PROJECT, reason="LOCUS_OCI_SANDBOX_PROJECT_ID is not set"),
]

pytest.importorskip("oci.generative_ai_sandbox")

from locus.harness import OCISandboxBackend, build_harness  # noqa: E402


@pytest.fixture(scope="module")
def sandbox() -> Iterator[OCISandboxBackend]:
    backend = OCISandboxBackend.create(
        PROJECT,
        region=os.environ.get("LOCUS_OCI_SANDBOX_REGION") or None,
        profile=os.environ.get("LOCUS_OCI_SANDBOX_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("LOCUS_OCI_SANDBOX_AUTH", "api_key"),
        expiration="PT30M",
    )
    try:
        yield backend
    finally:
        backend.close()


def test_files_round_trip_in_the_sandbox(sandbox: OCISandboxBackend) -> None:
    blob = bytes(range(256)) * 10
    sandbox.write_bytes("live/blob.bin", blob)
    assert sandbox.read_bytes("live/blob.bin") == blob
    assert sandbox.stat("live/blob.bin").size == len(blob)


def test_commands_run_in_the_workspace_root(sandbox: OCISandboxBackend) -> None:
    result = sandbox.exec("pwd", timeout=30)
    assert result.exit_code == 0
    assert result.output.decode().strip() == sandbox.capabilities.root


def test_the_sandbox_stops_a_command_at_its_deadline(sandbox: OCISandboxBackend) -> None:
    started = time.monotonic()
    result = sandbox.exec("sleep 30", timeout=3)
    assert result.timed_out
    assert time.monotonic() - started < 20


def test_large_output_comes_back_capped(sandbox: OCISandboxBackend) -> None:
    result = sandbox.exec("head -c 3000000 /dev/zero | tr '\\0' a; echo; echo END", timeout=60)
    assert result.exit_code == 0
    assert b"bytes of output were dropped here" in result.output
    assert result.output.rstrip().endswith(b"END")


def test_a_background_command_runs_between_calls(sandbox: OCISandboxBackend) -> None:
    job = sandbox.start_background("cat")
    sandbox.write_background(job.handle, b"ping\n", close=True)
    status = sandbox.wait_background(job.handle, 20)
    assert not status.running
    assert sandbox.read_background(job.handle).data == b"ping\n"
    sandbox.release_background(job.handle)


async def test_the_harness_edits_and_runs_in_the_sandbox(sandbox: OCISandboxBackend) -> None:
    h = build_harness(sandbox)
    await h.tool("write").execute(path="live/app.py", content="print(1 + 1)\n")
    await h.tool("read").execute(path="live/app.py")
    await h.tool("edit").execute(path="live/app.py", old="1 + 1", new="40 + 2")
    out = await h.tool("bash").execute(command="python3 live/app.py")
    assert "42" in out
    assert "live/app.py" in await h.tool("glob").execute(pattern="**/*.py")
