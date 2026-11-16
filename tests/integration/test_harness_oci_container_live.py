# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The coding harness against a real OCI Container Instance.

Skipped unless ``LOCUS_OCI_CI_COMPARTMENT_ID`` and ``LOCUS_OCI_CI_SUBNET_ID``
name where the instance goes. ``LOCUS_OCI_CI_BASTION_ID`` names an OCI
Bastion in the subnet's VCN to connect through (the shape for a private
subnet); without it, set ``LOCUS_OCI_CI_PUBLIC_IP=1`` for a public subnet.
``LOCUS_OCI_CI_PROFILE``, ``LOCUS_OCI_CI_AUTH`` and ``LOCUS_OCI_CI_REGION``
pick the OCI profile (``DEFAULT``), auth type (``api_key``) and region (the
profile's). One 1-OCPU, 2 GB instance is created for the module and deleted
after it; installing ``sshd`` on the default image takes about a minute.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pytest


COMPARTMENT = os.environ.get("LOCUS_OCI_CI_COMPARTMENT_ID", "")
SUBNET = os.environ.get("LOCUS_OCI_CI_SUBNET_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (COMPARTMENT and SUBNET),
        reason="LOCUS_OCI_CI_COMPARTMENT_ID and LOCUS_OCI_CI_SUBNET_ID are not set",
    ),
]

pytest.importorskip("oci.container_instances")
pytest.importorskip("paramiko")

from locus.harness import OCIContainerInstanceBackend, build_harness  # noqa: E402
from locus.harness.testing import run_conformance  # noqa: E402


@pytest.fixture(scope="module")
def sandbox() -> Iterator[OCIContainerInstanceBackend]:
    backend = OCIContainerInstanceBackend.create(
        COMPARTMENT,
        SUBNET,
        bastion_id=os.environ.get("LOCUS_OCI_CI_BASTION_ID") or None,
        public_ip=os.environ.get("LOCUS_OCI_CI_PUBLIC_IP", "") == "1",
        region=os.environ.get("LOCUS_OCI_CI_REGION") or None,
        profile=os.environ.get("LOCUS_OCI_CI_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("LOCUS_OCI_CI_AUTH", "api_key"),
        display_name="locus-harness-live-test",
        ready_timeout=900,
    )
    try:
        yield backend
    finally:
        backend.close()


def test_the_container_instance_conforms(sandbox: OCIContainerInstanceBackend) -> None:
    report = run_conformance(sandbox)
    assert report.ok, report.summary()


def test_commands_run_as_root_in_the_workspace(sandbox: OCIContainerInstanceBackend) -> None:
    result = sandbox.exec("id -u; pwd", timeout=30)
    assert result.exit_code == 0
    assert result.output.decode().split() == ["0", sandbox.capabilities.root]
    assert sandbox.limits.cpus == 1.0
    assert sandbox.limits.memory_mb == 2048


def test_the_instance_stops_a_command_at_its_deadline(sandbox: OCIContainerInstanceBackend) -> None:
    started = time.monotonic()
    result = sandbox.exec("sleep 30", timeout=3)
    assert result.timed_out
    assert time.monotonic() - started < 20


async def test_the_harness_edits_and_runs_in_the_container(
    sandbox: OCIContainerInstanceBackend,
) -> None:
    h = build_harness(sandbox)
    await h.tool("write").execute(path="live/hello.sh", content="echo $((1 + 1))\n")
    await h.tool("read").execute(path="live/hello.sh")
    await h.tool("edit").execute(path="live/hello.sh", old="1 + 1", new="40 + 2")
    out = await h.tool("bash").execute(command="sh live/hello.sh")
    assert "42" in out
    assert "live/hello.sh" in await h.tool("glob").execute(pattern="**/*.sh")


def test_close_deletes_the_instance(sandbox: OCIContainerInstanceBackend) -> None:
    client: Any = sandbox.client
    instance_id = sandbox.instance_id
    sandbox.close()
    assert sandbox.state == "closed"
    try:
        state = str(client.get_container_instance(instance_id).data.lifecycle_state).upper()
    except Exception as exc:  # noqa: BLE001 - a 404 is "gone", which is the point
        state = "DELETED" if getattr(exc, "status", None) == 404 else f"error: {exc}"
    assert state == "DELETED"
