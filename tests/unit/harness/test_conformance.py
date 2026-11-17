# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The conformance kit: green on every backend here, and loud on a broken one."""

from __future__ import annotations

from pathlib import Path

import pytest

from locus.harness import LocalBackend, MemoryBackend, WorkspaceBackend
from locus.harness.testing import (
    CHECKS,
    ConformanceReport,
    assert_backend_conformance,
    run_conformance,
)
from tests.unit.harness.conftest import SHELL_BACKENDS, make_backend


def test_the_local_backend_conforms(root: Path) -> None:
    report = assert_backend_conformance(lambda: LocalBackend(root))
    assert report.ok
    assert len(report.passed) == len(CHECKS) + 1, report.summary()
    assert report.skipped == []
    assert not list(root.iterdir()), "the kit cleans up after itself"


@pytest.mark.parametrize("kind", SHELL_BACKENDS)
def test_every_shell_backend_conforms(kind: str, root: Path) -> None:
    report = run_conformance(make_backend(kind, root), timeouts=False)
    assert report.ok, report.summary()


def test_a_file_only_backend_skips_the_shell_checks() -> None:
    report = assert_backend_conformance(MemoryBackend)
    assert report.ok
    assert all("no shell" in why for _, why in report.skipped)
    assert {name for name, _ in report.skipped} == {
        name for name, needs, _ in CHECKS if needs not in ("files", "protocol")
    }


def test_checks_can_be_left_out(root: Path) -> None:
    report = run_conformance(LocalBackend(root), shell=False)
    assert report.ok
    assert all("left out" in why for _, why in report.skipped)
    assert any(name == "exec kills a command at its deadline" for name, _ in report.skipped)


class Corrupting(LocalBackend):
    """A backend that returns the wrong bytes and lies about missing files."""

    def read_bytes(self, path: str) -> bytes:
        return super().read_bytes(path)[::-1]

    def exists(self, path: str) -> bool:
        return True


def test_a_broken_backend_fails_with_every_finding_listed(root: Path) -> None:
    with pytest.raises(AssertionError) as caught:
        assert_backend_conformance(lambda: Corrupting(root), timeouts=False)
    message = str(caught.value)
    assert "FAIL bytes round-trip exactly" in message
    assert "FAIL remove deletes" in message
    assert "passed" in message.splitlines()[0]


class NeverCloses(LocalBackend):
    def close(self) -> None:
        pass


def test_a_backend_that_does_not_close_is_a_finding(root: Path) -> None:
    with pytest.raises(AssertionError, match="close leaves the sandbox closed"):
        assert_backend_conformance(lambda: NeverCloses(root), timeouts=False)


def test_an_exception_from_a_check_is_a_finding_not_a_crash(root: Path) -> None:
    class Exploding(LocalBackend):
        def grep(self, *args: object, **kwargs: object) -> list[object]:  # type: ignore[override]
            raise RuntimeError("grep is on fire")

    report = run_conformance(Exploding(root), timeouts=False)
    assert not report.ok
    failed = dict(report.failed)
    assert "RuntimeError: grep is on fire" in failed["grep reports file, line and text"]


def test_the_report_summary_reads_as_one_block() -> None:
    report = ConformanceReport(
        label="x", passed=["a"], failed=[("b", "why")], skipped=[("c", "no")]
    )
    assert report.summary() == "x: 1 passed, 1 failed, 1 skipped\n  FAIL b: why\n  skip c: no"


def test_the_kit_runs_on_a_plain_workspace_backend(root: Path) -> None:
    class Plain:
        """Not a SandboxBackend: the protocol checks are skipped, the rest run."""

        def __init__(self) -> None:
            self.inner = LocalBackend(root)

        def __getattr__(self, name: str) -> object:
            if name in ("state", "limits", "start", "stop", "add_exec_hook"):
                raise AttributeError(name)
            return getattr(self.inner, name)

    backend: WorkspaceBackend = Plain()  # type: ignore[assignment]
    report = run_conformance(backend, timeouts=False)
    assert report.ok, report.summary()
    assert {name for name, _ in report.skipped} == {
        "lifecycle transitions",
        "exec hooks see every command",
        "a command-time limit bounds exec",
        "exec kills a command at its deadline",
    }
