# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""SandboxSpec: a sandbox named in configuration, opened by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locus.harness import (
    LocalBackend,
    MemoryBackend,
    ResourceLimits,
    SandboxBackend,
    SandboxSpec,
    WarmPoolHint,
    open_pool,
    open_sandbox,
    register_backend,
    registered_backends,
)
from locus.harness import spec as spec_mod


@pytest.fixture(autouse=True)
def _restore_registry() -> Any:
    saved = dict(spec_mod._REGISTRY)
    yield
    spec_mod._REGISTRY.clear()
    spec_mod._REGISTRY.update(saved)


def test_from_mapping_reads_nested_limits_and_pool() -> None:
    spec = SandboxSpec.from_mapping(
        {
            "backend": "ssh",
            "root": "/srv/ws",
            "env": {"CI": "1"},
            "limits": {"cpus": 2, "max_exec_s": 300},
            "warm_pool": {"ready": 2, "name": "fast"},
            "options": {"host": "box"},
        }
    )
    assert spec.backend == "ssh"
    assert spec.limits == ResourceLimits(cpus=2, max_exec_s=300)
    assert spec.warm_pool == WarmPoolHint(name="fast", ready=2)
    assert spec.options == {"host": "box"}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"backnd": "local"}, "unknown sandbox spec keys: backnd"),
        ({"limits": {"cpu": 1}}, "unknown limits keys: cpu"),
        ({"limits": [1, 2]}, "limits must be a mapping, got list"),
        ({"warm_pool": {"size": 1}}, "unknown warm_pool keys: size"),
        ({"warm_pool": "two"}, "warm_pool must be a mapping, got str"),
        ({"backend": ""}, "non-empty string"),
    ],
)
def test_from_mapping_refuses_what_it_does_not_know(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SandboxSpec.from_mapping(data)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("limits", 5, "limits must be a ResourceLimits, got int"),
        ("warm_pool", "two", "warm_pool must be a WarmPoolHint, got str"),
        ("backend", "", "backend must be a non-empty string"),
        ("backend", 3, "backend must be a non-empty string"),
        ("env", ["A=1"], "env must be a mapping, got list"),
        ("options", 3, "options must be a mapping, got int"),
    ],
)
def test_a_spec_field_of_the_wrong_type_is_refused_when_the_spec_is_built(
    name: str, value: Any, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        SandboxSpec(**{name: value})


def test_limits_that_are_not_limits_never_reach_a_backend(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ResourceLimits"):
        open_sandbox(SandboxSpec(backend="local", root=str(tmp_path), limits=5))  # type: ignore[arg-type]


def test_the_default_spec_is_the_local_backend(tmp_path: Path) -> None:
    backend = open_sandbox(SandboxSpec(root=str(tmp_path), env={"X": "1"}))
    assert isinstance(backend, LocalBackend)
    assert backend.capabilities.root == str(tmp_path.resolve())
    assert backend.exec('echo "$X"', timeout=5).output == b"1\n"
    backend.close()


def test_memory_takes_initial_files() -> None:
    backend = open_sandbox(SandboxSpec(backend="memory", options={"files": {"a.txt": "x"}}))
    assert isinstance(backend, MemoryBackend)
    assert backend.read_bytes("a.txt") == b"x"


def test_limits_are_applied_to_the_opened_backend(tmp_path: Path) -> None:
    backend = open_sandbox(SandboxSpec(root=str(tmp_path), limits=ResourceLimits(max_exec_s=0.5)))
    assert backend.limits.max_exec_s == 0.5
    assert backend.exec("sleep 5", timeout=30).timed_out
    backend.close()


def test_an_unknown_backend_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match=r"no sandbox backend named 'cloud9'; known: .*local.*ssh"):
        open_sandbox(SandboxSpec(backend="cloud9"))


def test_a_registered_backend_is_opened_by_name(tmp_path: Path) -> None:
    seen: list[SandboxSpec] = []

    def mine(spec: SandboxSpec) -> SandboxBackend:
        seen.append(spec)
        return LocalBackend(tmp_path, label="mine", isolated=True)

    register_backend("mine", mine)
    assert "mine" in registered_backends()
    spec = SandboxSpec(backend="mine", options={"size": "xl"})
    backend = open_sandbox(spec)
    assert seen == [spec]
    assert backend.capabilities.label == "mine"
    backend.close()
    with pytest.raises(ValueError, match="empty"):
        register_backend("", mine)


def test_open_pool_keeps_the_hinted_number_ready(tmp_path: Path) -> None:
    register_backend("mine", lambda spec: LocalBackend(tmp_path))
    pool = open_pool(SandboxSpec(backend="mine", warm_pool=WarmPoolHint(ready=1)))
    backend = pool.acquire(timeout=5)
    assert isinstance(backend, LocalBackend)
    pool.release(backend)
    pool.close()
    inline = open_pool(SandboxSpec(backend="mine"), ready=0)
    assert inline.ready == 0
    inline.close()


# ------------------------------------------------------- built-in dispatch --


class Recorder:
    """Records which constructor the factory called, with what."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def method(self, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return MemoryBackend()

        return call


def test_ssh_dispatches_to_connect_with_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.harness import ssh

    rec = Recorder()
    monkeypatch.setattr(ssh.SSHSandboxBackend, "connect", rec.method("connect"))
    open_sandbox(
        SandboxSpec(backend="ssh", root="/w", env={"A": "1"}, options={"host": "box", "port": 2222})
    )
    assert rec.calls == [("connect", ("box",), {"port": 2222, "root": "/w", "env": {"A": "1"}})]
    with pytest.raises(ValueError, match="'host'"):
        open_sandbox(SandboxSpec(backend="ssh"))


def test_oci_container_dispatches_and_passes_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.harness import oci_container

    rec = Recorder()
    monkeypatch.setattr(oci_container.OCIContainerInstanceBackend, "create", rec.method("create"))
    monkeypatch.setattr(oci_container.OCIContainerInstanceBackend, "connect", rec.method("connect"))
    open_sandbox(
        SandboxSpec(
            backend="oci-container",
            limits=ResourceLimits(cpus=2, memory_mb=4096),
            options={"compartment_id": "c", "subnet_id": "s", "bastion_id": "b"},
        )
    )
    open_sandbox(SandboxSpec(backend="oci-container", options={"instance_id": "i", "pkey": "k"}))
    assert rec.calls == [
        ("create", ("c", "s"), {"bastion_id": "b", "ocpus": 2, "memory_gb": 4.0}),
        ("connect", ("i",), {"pkey": "k"}),
    ]
    with pytest.raises(ValueError, match="compartment_id"):
        open_sandbox(SandboxSpec(backend="oci-container"))


def test_oci_sandbox_dispatches_and_passes_the_warm_pool_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.harness import oci_sandbox

    rec = Recorder()
    monkeypatch.setattr(oci_sandbox.OCISandboxBackend, "create", rec.method("create"))
    monkeypatch.setattr(oci_sandbox.OCISandboxBackend, "connect", rec.method("connect"))
    open_sandbox(
        SandboxSpec(
            backend="oci-sandbox", warm_pool=WarmPoolHint(name="pool"), options={"project_id": "p"}
        )
    )
    open_sandbox(SandboxSpec(backend="oci-sandbox", options={"sandbox_id": "sbx"}))
    assert rec.calls == [
        ("create", ("p",), {"warmpool_name": "pool"}),
        ("connect", ("sbx",), {}),
    ]
    with pytest.raises(ValueError, match="project_id"):
        open_sandbox(SandboxSpec(backend="oci-sandbox"))


def test_openshell_e2b_and_daytona_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.harness import daytona, e2b, openshell

    rec = Recorder()
    for module, cls in (
        (openshell, "OpenShellBackend"),
        (e2b, "E2BBackend"),
        (daytona, "DaytonaBackend"),
    ):
        monkeypatch.setattr(getattr(module, cls), "create", rec.method(f"{cls}.create"))
        monkeypatch.setattr(getattr(module, cls), "connect", rec.method(f"{cls}.connect"))
    open_sandbox(SandboxSpec(backend="openshell", options={"workspace": "team"}))
    open_sandbox(SandboxSpec(backend="openshell", options={"sandbox": "sbx", "workspace": "t"}))
    open_sandbox(SandboxSpec(backend="e2b", options={"template": "base"}))
    open_sandbox(SandboxSpec(backend="e2b", options={"sandbox_id": "e"}))
    open_sandbox(SandboxSpec(backend="daytona"))
    open_sandbox(SandboxSpec(backend="daytona", options={"sandbox_id": "d"}))
    assert [c[0] for c in rec.calls] == [
        "OpenShellBackend.create",
        "OpenShellBackend.connect",
        "E2BBackend.create",
        "E2BBackend.connect",
        "DaytonaBackend.create",
        "DaytonaBackend.connect",
    ]
    assert rec.calls[1][1:] == (("sbx",), {"workspace": "t"})
    assert rec.calls[3][1:] == (("e",), {})
    assert rec.calls[5][1:] == (("d",), {})
