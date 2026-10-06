# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""glob and grep, with ripgrep and with the walk, on every backend."""

from __future__ import annotations

import dataclasses
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from tests.unit.harness.conftest import put
from tulip.harness import BackendError, MemoryBackend, WorkspaceBackend, build_harness
from tulip.harness.backend import ExecResult
from tulip.harness.tools import search
from tulip.harness.toolset import Harness


@pytest.fixture(params=["rg", "walk"])
def engine(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def harness(backend: WorkspaceBackend, engine: str = "rg") -> Harness:
    if engine == "walk" or not backend.capabilities.can_exec:
        caps = dataclasses.replace(backend.capabilities, has_rg=False)
        backend.__dict__["_capabilities"] = caps
        if hasattr(backend, "_has_rg"):
            backend._has_rg = False  # type: ignore[attr-defined]
    return build_harness(backend)


async def call(h: Harness, name: str, **kwargs: object) -> str:
    result: str = await h.tool(name).execute(**kwargs)
    return result


async def test_glob_finds_by_pattern_and_skips_the_noise(
    workspace: WorkspaceBackend, engine: str
) -> None:
    put(workspace, "setup.py", "x")
    put(workspace, "src/app.py", "x")
    put(workspace, "src/notes.txt", "x")
    put(workspace, "node_modules/lib.py", "x")
    out = await call(harness(workspace, engine), "glob", pattern="**/*.py")
    assert sorted(out.splitlines()) == ["setup.py", "src/app.py"]


async def test_glob_lists_newest_first(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "new.py", "x")
    put(workspace, "old.py", "x")
    # Written second, but made older: a day back on disk, or in the dict.
    if isinstance(workspace, MemoryBackend):
        workspace._mtime["/old.py"] -= timedelta(days=1)
    else:
        old = workspace.resolve("old.py")
        day_ago = time.time() - 86_400
        os.utime(old, (day_ago, day_ago))
    out = await call(harness(workspace, engine), "glob", pattern="*.py")
    assert out.splitlines() == ["new.py", "old.py"]


async def test_glob_says_so_when_nothing_matches(workspace: WorkspaceBackend, engine: str) -> None:
    assert "no files matching" in await call(harness(workspace, engine), "glob", pattern="*.rs")


async def test_glob_pages_a_long_listing(
    workspace: WorkspaceBackend, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "GLOB_LIMIT", 3)
    for i in range(5):
        put(workspace, f"f{i}.txt", "x")
    h = harness(workspace, engine)
    first = await call(h, "glob", pattern="*.txt")
    assert "2 more — call again with offset=3" in first
    assert len((await call(h, "glob", pattern="*.txt", offset=3)).splitlines()) == 2
    assert "offset 9 is past the end" in await call(h, "glob", pattern="*.txt", offset=9)


async def test_glob_under_a_directory(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "src/a.py", "x")
    put(workspace, "b.py", "x")
    assert await call(harness(workspace, engine), "glob", pattern="*.py", path="src") == "a.py"


async def test_grep_reports_the_file_and_line(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "src/a.py", "one\nneedle = 1\n")
    out = await call(harness(workspace, engine), "grep", pattern="needle")
    assert out == "src/a.py:2: needle = 1"


async def test_grep_says_what_is_wrong_with_a_bad_regex(workspace: WorkspaceBackend) -> None:
    assert "bad regex" in await call(harness(workspace), "grep", pattern="(")


async def test_grep_can_be_narrowed_by_filename(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "a.py", "needle\n")
    put(workspace, "b.md", "needle\n")
    out = await call(harness(workspace, engine), "grep", pattern="needle", glob_filter="*.md")
    assert out == "b.md:1: needle"


async def test_grep_in_one_file_or_one_directory(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "src/a.py", "needle\n")
    put(workspace, "other/b.py", "needle\n")
    h = harness(workspace, engine)
    assert await call(h, "grep", pattern="needle", path="src") == "src/a.py:1: needle"
    assert await call(h, "grep", pattern="needle", path="src/a.py") == "src/a.py:1: needle"


async def test_grep_skips_the_noise(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "src/a.py", "needle\n")
    put(workspace, ".git/config", "needle\n")
    put(workspace, "node_modules/x.js", "needle\n")
    out = await call(harness(workspace, engine), "grep", pattern="needle")
    assert out.splitlines() == ["src/a.py:1: needle"]


async def test_grep_pages_a_long_result(
    workspace: WorkspaceBackend, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "GREP_LIMIT", 3)
    put(workspace, "a.txt", "\n".join("needle" for _ in range(10)))
    h = harness(workspace, engine)
    first = await call(h, "grep", pattern="needle")
    assert first.count(": needle") == 3
    assert "call again with offset=3" in first
    assert "past offset 50" in await call(h, "grep", pattern="needle", offset=50)
    last = await call(h, "grep", pattern="needle", offset=9)
    assert last == "a.txt:10: needle"


async def test_grep_says_so_when_nothing_matches(workspace: WorkspaceBackend, engine: str) -> None:
    put(workspace, "a.txt", "hay\n")
    assert "no matches for needle" in await call(
        harness(workspace, engine), "grep", pattern="needle"
    )


async def test_a_python_only_regex_falls_back_to_the_walk(shell: WorkspaceBackend) -> None:
    # Look-behind: Python reads it, ripgrep's default engine refuses it.
    put(shell, "a.py", "x = foo_bar\n")
    out = await call(harness(shell), "grep", pattern=r"(?<=foo_)bar")
    assert out == "a.py:1: x = foo_bar"


async def test_a_failing_ripgrep_falls_back(
    shell: WorkspaceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(shell, "a.py", "needle\n")
    h = harness(shell)
    real = shell.exec

    def broken(command: str, **kwargs: object) -> ExecResult:
        if "rg " in command:
            return ExecResult(exit_code=0, output=b"garbage without a status", timed_out=False)
        return real(command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shell, "exec", broken)
    assert await call(h, "glob", pattern="*.py") == "a.py"
    assert await call(h, "grep", pattern="needle") == "a.py:1: needle"

    def refusing(command: str, **kwargs: object) -> ExecResult:
        if "rg " in command:
            raise BackendError("permission_denied")
        return real(command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shell, "exec", refusing)
    assert await call(h, "glob", pattern="*.py") == "a.py"


async def test_a_search_that_the_backend_cannot_walk_finds_nothing(
    workspace: WorkspaceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness(workspace, "walk")

    def fail(*args: object, **kwargs: object) -> list[object]:
        raise BackendError("permission_denied")

    monkeypatch.setattr(workspace, "glob", fail)
    monkeypatch.setattr(workspace, "grep", fail)
    assert "no files matching" in await call(h, "glob", pattern="*")
    assert "no matches" in await call(h, "grep", pattern="x")


def test_relative_paths_outside_the_root_are_shown_whole() -> None:
    assert search._rel("/elsewhere/x", "/root") == "/elsewhere/x"
    assert Path(search._rel("/root/a/b", "/root")).name == "b"
