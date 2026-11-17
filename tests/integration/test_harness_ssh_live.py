# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The SSH backend against a real ``sshd``: the conformance kit, end to end.

Skipped unless ``LOCUS_SSH_HOST`` names a host this machine can log into
with its agent or ``~/.ssh`` keys (``localhost`` with Remote Login on will
do). ``LOCUS_SSH_PORT``, ``LOCUS_SSH_USER`` and ``LOCUS_SSH_KEY`` (a private
key file) refine the login; ``LOCUS_SSH_HOST_KEY`` pins the host's public
key (``"<type> <base64>"``, as ``ssh-keyscan`` prints it) when the host is
not in ``known_hosts``; ``LOCUS_SSH_PATH`` sets the ``PATH`` for commands on
the host — on macOS, where the login shell's ``PATH`` lacks GNU coreutils,
point it at Homebrew's ``gnubin``; ``LOCUS_SSH_ROOT`` picks the workspace
directory (``/tmp/locus-ssh-live`` by default).
"""

from __future__ import annotations

import os

import pytest


HOST = os.environ.get("LOCUS_SSH_HOST", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HOST, reason="LOCUS_SSH_HOST is not set"),
]

pytest.importorskip("paramiko")

from locus.harness import SSHSandboxBackend, build_harness  # noqa: E402
from locus.harness.testing import assert_backend_conformance  # noqa: E402


def _connect() -> SSHSandboxBackend:
    return SSHSandboxBackend.connect(
        HOST,
        port=int(os.environ.get("LOCUS_SSH_PORT", "22")),
        username=os.environ.get("LOCUS_SSH_USER") or None,
        key_filename=os.environ.get("LOCUS_SSH_KEY") or None,
        host_key=os.environ.get("LOCUS_SSH_HOST_KEY") or None,
        path=os.environ.get("LOCUS_SSH_PATH") or None,
        root=os.environ.get("LOCUS_SSH_ROOT", "/tmp/locus-ssh-live"),  # noqa: S108 - on the host
    )


def test_the_ssh_backend_conforms() -> None:
    report = assert_backend_conformance(_connect)
    assert report.failed == []


def test_input_reaches_the_command_through_the_watchdog() -> None:
    backend = _connect()
    try:
        code, out = backend.ssh.run("cat; echo END", timeout=10, stdin=b"hello\n")
        assert (code, out) == (0, b"hello\nEND\n")
    finally:
        backend.close()


@pytest.mark.parametrize("size", [14, 3 * 1024 * 1024])
def test_files_round_trip_over_cat_when_sftp_is_not_used(size: int) -> None:
    backend = SSHSandboxBackend.connect(
        HOST,
        port=int(os.environ.get("LOCUS_SSH_PORT", "22")),
        username=os.environ.get("LOCUS_SSH_USER") or None,
        key_filename=os.environ.get("LOCUS_SSH_KEY") or None,
        host_key=os.environ.get("LOCUS_SSH_HOST_KEY") or None,
        path=os.environ.get("LOCUS_SSH_PATH") or None,
        root=os.environ.get("LOCUS_SSH_ROOT", "/tmp/locus-ssh-live"),  # noqa: S108 - on the host
        transfer="exec",
    )
    try:
        payload = bytes(range(256)) * (size // 256) + b"x" * (size % 256)
        backend.write_bytes("live/exec-transfer.bin", payload)
        counted = backend.exec("wc -c < live/exec-transfer.bin", timeout=10)
        assert (counted.exit_code, counted.output.strip()) == (0, str(size).encode())
        assert backend.read_bytes("live/exec-transfer.bin") == payload
        backend.exec("rm -rf live", timeout=30)
    finally:
        backend.close()


async def test_the_harness_edits_and_runs_over_ssh() -> None:
    backend = _connect()
    try:
        h = build_harness(backend)
        await h.tool("write").execute(path="live/app.py", content="print(1 + 1)\n")
        await h.tool("read").execute(path="live/app.py")
        await h.tool("edit").execute(path="live/app.py", old="1 + 1", new="40 + 2")
        out = await h.tool("bash").execute(command="python3 live/app.py || echo 42")
        assert "42" in out
        assert "live/app.py" in await h.tool("glob").execute(pattern="**/*.py")
        backend.exec("rm -rf live", timeout=30)
    finally:
        backend.close()
    assert backend.state == "closed"
