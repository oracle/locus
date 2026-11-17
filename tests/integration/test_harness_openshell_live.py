# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The coding harness against a real OpenShell gateway.

Skipped unless ``LOCUS_OPENSHELL_ENDPOINT`` names a gateway (``host:port``)
and the ``openshell`` package is installed. ``LOCUS_OPENSHELL_WORKSPACE``
picks the workspace (default ``default``); ``LOCUS_OPENSHELL_TOKEN`` is sent
as the bearer token when set. A sandbox is created for the module and
deleted after it.

The provider test needs more: ``LOCUS_OPENSHELL_PROVIDER`` names an OpenShell
provider of type ``oci-genai`` on that gateway (see
``examples/openshell-gateway/README.md``), ``LOCUS_OPENSHELL_IMAGE`` an image
with ``curl`` (the default sandbox image has none), and optionally
``LOCUS_OPENSHELL_OCI_REGION`` / ``LOCUS_OPENSHELL_OCI_MODEL``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator

import pytest


ENDPOINT = os.environ.get("LOCUS_OPENSHELL_ENDPOINT", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ENDPOINT, reason="LOCUS_OPENSHELL_ENDPOINT is not set"),
]

pytest.importorskip("openshell")

from locus.harness import OpenShellBackend, build_harness  # noqa: E402
from locus.harness.testing import run_conformance  # noqa: E402


WORKSPACE = os.environ.get("LOCUS_OPENSHELL_WORKSPACE", "default")
PROVIDER = os.environ.get("LOCUS_OPENSHELL_PROVIDER", "")


def _create(prefix: str, **extra: object) -> OpenShellBackend:
    options: dict[str, object] = {}
    token = os.environ.get("LOCUS_OPENSHELL_TOKEN")
    if token:
        options["bearer_token"] = token
    return OpenShellBackend.create(
        workspace=WORKSPACE,
        # OpenShell caps sandbox names at 19 characters.
        name=f"{prefix}-{uuid.uuid4().hex[:8]}",
        endpoint=ENDPOINT,
        **extra,
        **options,
    )


@pytest.fixture(scope="module")
def sandbox() -> Iterator[OpenShellBackend]:
    backend = _create("lh")
    try:
        yield backend
    finally:
        backend.close()


def test_the_sandbox_conforms(sandbox: OpenShellBackend) -> None:
    report = run_conformance(sandbox)
    assert report.ok, report.summary()


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


@pytest.mark.skipif(not PROVIDER, reason="LOCUS_OPENSHELL_PROVIDER is not set")
def test_a_provider_credential_reaches_oci_genai_from_the_sandbox() -> None:
    """The gateway holds the key; the sandbox sees a placeholder the proxy swaps at OCI."""
    region = os.environ.get("LOCUS_OPENSHELL_OCI_REGION", "us-chicago-1")
    model = os.environ.get("LOCUS_OPENSHELL_OCI_MODEL", "meta.llama-3.3-70b-instruct")
    backend = _create(
        "lp", image=os.environ.get("LOCUS_OPENSHELL_IMAGE") or None, providers=[PROVIDER]
    )
    try:
        placeholder = backend.exec('printf %s "$OCI_GENAI_API_KEY"', timeout=30)
        assert placeholder.exit_code == 0
        assert placeholder.output, "the provider's variable is not in the sandbox"
        assert not placeholder.output.startswith(b"sk-"), "the sandbox holds the real key"

        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 5,
        }
        backend.write_bytes("live/body.json", json.dumps(body).encode())
        result = backend.exec(
            "curl -sS --fail-with-body --max-time 90 "
            f"https://inference.generativeai.{region}.oci.oraclecloud.com/openai/v1/chat/completions"
            ' -H "Authorization: Bearer $OCI_GENAI_API_KEY" -H "Content-Type: application/json"'
            " --data @live/body.json",
            timeout=120,
        )
        assert result.exit_code == 0, result.output.decode(errors="replace")
        reply = json.loads(result.output)
        assert reply["choices"][0]["message"]["content"].strip()
    finally:
        backend.close()
