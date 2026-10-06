# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenShellBackend: what is particular to it beyond the shared contract.

The contract tests already run on it (see conftest). These pin the parts
only it has: file transfer over exec, the gateway's timeout status, the
workspace scoping of every call, and the constructors that build a client.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tests.unit.harness.conftest import FakeOpenShellClient, needs_posix
from tulip.harness import OpenShellBackend, OpenShellSession, build_harness


pytestmark = needs_posix


def backend(root: Path, client: FakeOpenShellClient | None = None) -> OpenShellBackend:
    return OpenShellBackend(
        client or FakeOpenShellClient(),
        "sbx-7",
        workspace="team-a",
        root=str(root),
        state_dir=str(root.parent / "state"),
    )


def test_it_is_isolated_and_says_where_it_runs(root: Path) -> None:
    caps = backend(root).capabilities
    assert caps.isolated
    assert caps.can_exec
    assert caps.label == "openshell:team-a/sbx-7"


def test_the_default_root_is_the_sandbox_directory() -> None:
    assert OpenShellBackend(FakeOpenShellClient(), "s").capabilities.root == "/sandbox"


def test_every_call_is_scoped_to_the_workspace_and_sandbox(root: Path) -> None:
    client = FakeOpenShellClient()
    b = backend(root, client)
    b.write_bytes("a.txt", b"x")
    b.read_bytes("a.txt")
    b.exec("true", timeout=5)
    assert client.calls
    assert {c["workspace"] for c in client.calls} == {"team-a"}
    assert {c["sandbox"] for c in client.calls} == {"sbx-7"}
    assert all(c["command"][:2] == ["sh", "-c"] for c in client.calls)


def test_files_move_over_exec_byte_for_byte(root: Path) -> None:
    client = FakeOpenShellClient()
    b = backend(root, client)
    blob = bytes(range(256)) * 50
    b.write_bytes("nested/blob.bin", blob)
    assert (root / "nested/blob.bin").read_bytes() == blob
    assert b.read_bytes("nested/blob.bin") == blob
    assert any(c["stdin"] == blob for c in client.calls), "upload is the bytes on stdin"


def test_an_upload_keeps_the_files_mode(root: Path) -> None:
    script = root / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    backend(root).write_bytes("run.sh", b"#!/bin/sh\necho bye\n")
    assert script.stat().st_mode & 0o777 == 0o755


def test_a_large_upload_goes_in_chunks(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tulip.harness import openshell

    monkeypatch.setattr(openshell, "UPLOAD_CHUNK", 1000)
    client = FakeOpenShellClient()
    data = b"0123456789" * 350
    backend(root, client).write_bytes("big.txt", data)
    assert (root / "big.txt").read_bytes() == data
    uploads = [c for c in client.calls if c["stdin"]]
    assert [len(c["stdin"]) for c in uploads] == [1000, 1000, 1000, 500]
    assert ">>" in uploads[1]["command"][2]


def test_a_failed_upload_is_an_error(root: Path) -> None:
    session = OpenShellSession(FakeOpenShellClient(), "s", workspace="w")
    (root / "file").write_text("x")
    with pytest.raises(OSError, match="upload of"):
        session.upload_file(str(root / "file" / "under-a-file"), b"x")


def test_downloads_tell_a_directory_from_a_missing_file(root: Path) -> None:
    session = OpenShellSession(FakeOpenShellClient(), "s", workspace="w")
    (root / "d").mkdir()
    with pytest.raises(IsADirectoryError):
        session.download_file(str(root / "d"))
    with pytest.raises(FileNotFoundError):
        session.download_file(str(root / "missing"))


def test_a_download_of_an_unarchivable_member_is_not_found(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "elsewhere"
        archive.addfile(info)
    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w"):
        pass
    session = OpenShellSession(FakeOpenShellClient(), "s", workspace="w")
    for payload in (buffer.getvalue(), empty.getvalue()):
        monkeypatch.setattr(session, "_stdout", lambda command, p=payload: (0, p))
        with pytest.raises(FileNotFoundError):
            session.download_file("/x/link")


def test_the_gateways_timeout_is_a_timeout(root: Path) -> None:
    result = backend(root).exec("sleep 5", timeout=1)
    assert result.timed_out
    assert result.exit_code is None


def test_a_transport_failure_past_the_deadline_is_a_timeout(root: Path) -> None:
    client = FakeOpenShellClient(raise_after=1.1)
    result = backend(root, client).exec("echo never", timeout=1)
    assert result.timed_out
    assert result.output == b""


def test_a_transport_failure_before_the_deadline_is_raised(root: Path) -> None:
    client = FakeOpenShellClient(raise_after=0.0)
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        backend(root, client).exec("echo never", timeout=30)


def test_output_streams_as_it_arrives(root: Path) -> None:
    seen: list[bytes] = []
    backend(root).exec("echo out; echo err >&2", timeout=5, on_output=seen.append)
    assert seen == [b"out\n", b"err\n"]


def test_close_deletes_a_sandbox_it_owns(root: Path) -> None:
    client = FakeOpenShellClient()
    owned = OpenShellBackend(
        client, "mine", root=str(root), owns_sandbox=True, state_dir=str(root.parent / "s")
    )
    owned.close()
    assert client.deleted == ["mine"]
    borrowed = backend(root, client)
    borrowed.close()
    assert client.deleted == ["mine"]


def test_close_without_a_delete_method_is_quiet(root: Path) -> None:
    class Minimal:
        def __init__(self) -> None:
            self.inner = FakeOpenShellClient()

        def exec_stream(self, *args: Any, **kwargs: Any) -> Any:
            return self.inner.exec_stream(*args, **kwargs)

    OpenShellBackend(
        Minimal(), "x", root=str(root), owns_sandbox=True, state_dir=str(root.parent / "s")
    ).close()


@pytest.fixture
def fake_openshell(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An ``openshell`` module whose SandboxClient is the fake."""
    made: dict[str, Any] = {}

    class SandboxClient(FakeOpenShellClient):
        def __init__(self, endpoint: str | None = None, **options: Any) -> None:
            super().__init__()
            made["endpoint"], made["options"], made["client"] = endpoint, options, self

        @classmethod
        def from_active_cluster(cls, **options: Any) -> SandboxClient:
            client = cls(None, **options)
            made["active"] = True
            return client

    module = types.ModuleType("openshell")
    module.SandboxClient = SandboxClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openshell", module)
    return made


def test_connect_builds_a_client_for_an_endpoint(fake_openshell: dict[str, Any]) -> None:
    b = OpenShellBackend.connect("sbx", workspace="w", endpoint="gw:443", timeout=5)
    assert fake_openshell["endpoint"] == "gw:443"
    assert fake_openshell["options"] == {"timeout": 5}
    assert b.client is fake_openshell["client"]
    assert b.capabilities.label == "openshell:w/sbx"


def test_connect_without_an_endpoint_uses_the_active_gateway(
    fake_openshell: dict[str, Any],
) -> None:
    OpenShellBackend.connect("sbx")
    assert fake_openshell["active"]


def test_create_makes_a_sandbox_and_owns_it(fake_openshell: dict[str, Any], root: Path) -> None:
    b = OpenShellBackend.create(workspace="w", name="fresh", root=str(root))
    client = fake_openshell["client"]
    assert client.created == ["fresh"]
    b._jobs_dir = str(root.parent / "state")
    b.close()
    assert client.deleted == ["fresh"]


def test_without_the_package_the_error_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openshell", None)
    with pytest.raises(ImportError, match=r"tulip-agents\[openshell\]"):
        OpenShellBackend.connect("sbx")


async def test_the_harness_runs_end_to_end_in_the_sandbox(root: Path) -> None:
    h = build_harness(backend(root))
    assert "openshell:team-a/sbx-7" in h.prompt_fragment
    assert "not in a sandbox" not in h.prompt_fragment
    await h.tool("write").execute(path="app.py", content="print(40 + 2)\n")
    out = await h.tool("bash").execute(command="python3 app.py")
    assert out == "exit 0\n42"
