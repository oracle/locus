# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The sandbox protocol: lifecycle, limits, evidence hooks and the warm pool."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from locus.harness import (
    ExecRecord,
    LocalBackend,
    MemoryBackend,
    ResourceLimits,
    SandboxBackend,
    SandboxMixin,
    SandboxPool,
    WarmPoolHint,
    WorkspaceBackend,
)
from locus.harness.backend import BackendCapabilities
from tests.unit.harness.conftest import FILE_BACKENDS, make_backend


@pytest.mark.parametrize("kind", FILE_BACKENDS)
def test_every_backend_implements_the_protocol(kind: str, root: Path) -> None:
    backend = make_backend(kind, root)
    assert isinstance(backend, SandboxBackend)
    assert backend.state == "running"
    assert backend.limits == ResourceLimits()
    backend.close()
    assert backend.state == "closed"


def test_the_lifecycle_moves_and_is_idempotent(root: Path) -> None:
    backend = LocalBackend(root)
    backend.start()  # already running
    assert backend.state == "running"
    job = backend.start_background("sleep 30")
    backend.stop()
    assert backend.state == "stopped"
    assert not backend.wait_background(job.handle, 5).running, "stop ends the commands"
    backend.stop()  # again: quiet
    backend.start()
    assert backend.state == "running"
    assert backend.exec("echo back", timeout=5).output == b"back\n"
    backend.close()
    backend.close()
    assert backend.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        backend.start()


def test_hooks_see_every_command_with_digests_only(root: Path) -> None:
    backend = LocalBackend(root)
    seen: list[ExecRecord] = []
    remove = backend.add_exec_hook(seen.append)
    backend.exec("echo secret-token", timeout=5)
    job = backend.start_background("cat", foreground=True)
    backend.wait_background(job.handle, 5)
    assert [r.background for r in seen] == [False, True]
    first, second = seen
    assert first.exit_code == 0
    assert first.output_bytes == len(b"secret-token\n")
    assert "secret-token" not in repr(first), "records carry digests, not text"
    assert first.backend_label == backend.capabilities.label
    assert second.exit_code is None
    remove()
    backend.exec("true", timeout=5)
    assert len(seen) == 2, "a removed hook sees nothing more"
    backend.close()


def test_a_command_time_limit_clamps_every_timeout(root: Path) -> None:
    backend = LocalBackend(root)
    backend.limits = ResourceLimits(max_exec_s=0.5)
    assert backend.limits.bound(30) == 0.5
    started = time.monotonic()
    result = backend.exec("sleep 10", timeout=30)
    assert time.monotonic() - started < 8
    assert result.timed_out
    backend.close()


def test_limits_bound_without_a_limit_is_identity() -> None:
    assert ResourceLimits().bound(12.5) == 12.5
    assert ResourceLimits(max_exec_s=20).bound(12.5) == 12.5
    assert WarmPoolHint() == WarmPoolHint(name=None, ready=0)


class Flaky(SandboxMixin):
    """A backend whose lifecycle hooks fail on request."""

    def __init__(self) -> None:
        self.stop_failures = 0
        self.destroy_failures = 0
        self.calls: list[str] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolated=True, label="flaky", has_rg=False, root="/")

    def _start(self) -> None:
        self.calls.append("start")

    def _stop(self) -> None:
        self.calls.append("stop")
        if self.stop_failures:
            self.stop_failures -= 1
            raise ConnectionError("transport down")

    def _destroy(self) -> None:
        self.calls.append("destroy")
        if self.destroy_failures:
            self.destroy_failures -= 1
            raise ConnectionError("service down")


def test_close_destroys_even_when_stop_fails() -> None:
    flaky = Flaky()
    flaky.stop_failures = 1
    with pytest.raises(ConnectionError, match="transport"):
        flaky.close()
    assert flaky.calls == ["stop", "destroy"]
    assert flaky.state == "closed", "the sandbox is gone; there is nothing left to retry"
    flaky.close()
    assert flaky.calls == ["stop", "destroy"]


def test_a_failed_destroy_is_retried_and_a_successful_stop_is_not() -> None:
    flaky = Flaky()
    flaky.destroy_failures = 1
    with pytest.raises(ConnectionError, match="service"):
        flaky.close()
    assert flaky.state == "stopped"
    flaky.close()
    assert flaky.calls == ["stop", "destroy", "destroy"]
    assert flaky.state == "closed"


def test_stop_then_close_does_not_stop_twice() -> None:
    flaky = Flaky()
    flaky.stop()
    flaky.close()
    assert flaky.calls == ["stop", "destroy"]


def test_start_after_stop_runs_the_hook() -> None:
    flaky = Flaky()
    flaky.stop()
    flaky.start()
    assert flaky.calls == ["stop", "start"]
    assert flaky.state == "running"


def test_the_memory_backend_has_a_lifecycle_too() -> None:
    backend = MemoryBackend({"a.txt": "x"})
    assert isinstance(backend, SandboxBackend)
    backend.stop()
    backend.start()
    backend.close()
    assert backend.state == "closed"


# ------------------------------------------------------------------- pool --


class Counted(SandboxMixin):
    """A stand-in sandbox that records when it was made and closed."""

    made = 0

    def __init__(self) -> None:
        Counted.made += 1
        self.number = Counted.made
        self.closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolated=True, label=f"c{self.number}", has_rg=False, root="/")

    def _destroy(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    Counted.made = 0


def _wait_ready(pool: SandboxPool, count: int) -> None:
    deadline = time.monotonic() + 5
    while pool.ready < count:
        assert time.monotonic() < deadline, f"the pool never had {count} ready"
        time.sleep(0.01)


def test_the_pool_keeps_sandboxes_ready_and_replaces_what_it_hands_out() -> None:
    pool = SandboxPool(Counted, ready=2)  # type: ignore[arg-type]
    _wait_ready(pool, 2)
    assert Counted.made == 2
    first = pool.acquire(timeout=5)
    assert isinstance(first, Counted)
    _wait_ready(pool, 2)
    assert Counted.made == 3, "the one handed out was replaced"
    pool.release(first)
    assert first.closed, "a used sandbox is closed, not reused"
    pool.close()
    assert pool.ready == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire()


def test_a_pool_of_zero_creates_on_demand() -> None:
    pool = SandboxPool(Counted, ready=0)  # type: ignore[arg-type]
    assert Counted.made == 0
    backend = pool.acquire(timeout=5)
    assert Counted.made == 1
    pool.release(backend)
    assert Counted.made == 1
    pool.close()


def test_a_factory_error_reaches_the_caller() -> None:
    def broken() -> WorkspaceBackend:
        raise ValueError("no capacity")

    pool = SandboxPool(broken, ready=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no capacity"):
        pool.acquire(timeout=5)
    pool.close()


def test_acquire_times_out_when_nothing_arrives() -> None:
    gate = threading.Event()

    def slow() -> WorkspaceBackend:
        gate.wait(5)
        return Counted()  # type: ignore[return-value]

    pool = SandboxPool(slow, ready=1)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="within"):
        pool.acquire(timeout=0.1)
    gate.set()
    pool.close()


def test_a_negative_pool_size_is_refused() -> None:
    with pytest.raises(ValueError, match="zero or more"):
        SandboxPool(Counted, ready=-1)  # type: ignore[arg-type]


def test_a_sandbox_finished_after_close_is_closed_not_kept() -> None:
    gate = threading.Event()
    made: list[Counted] = []

    def slow() -> WorkspaceBackend:
        gate.wait(5)
        backend = Counted()
        made.append(backend)
        return backend  # type: ignore[return-value]

    pool = SandboxPool(slow, ready=1)  # type: ignore[arg-type]
    pool.close()
    gate.set()
    deadline = time.monotonic() + 5
    while not made or not made[0].closed:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert pool.ready == 0
