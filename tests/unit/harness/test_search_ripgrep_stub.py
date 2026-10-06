# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The ripgrep path of glob and grep, whether or not ripgrep is installed.

A stand-in ``rg`` on the workspace's ``PATH`` answers the way ripgrep does —
paths prefixed ``./``, ``path:line:text`` hits, exit 0 / 1 / 2 — so what is
tested is the harness's side: the arguments it sends, how it reads the
answer, and when it falls back to the walk. ``test_tools_search`` runs the
same tools against the real ripgrep where there is one.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from tulip.harness import BackendError, LocalBackend, build_harness
from tulip.harness.backend import ExecResult
from tulip.harness.toolset import Harness


_STUB = """#!/bin/sh
echo "$@" >> "$RG_LOG"
[ -n "$RG_EMPTY" ] && exit "${RG_EXIT:-0}"
case "$*" in
  *--files*) printf './new.py\\n./src/old.py\\n' ;;
  *) printf 'src/a.py:2:   needle here   \\n' ;;
esac
exit "${RG_EXIT:-0}"
"""


def stubbed(root: Path, tmp_path: Path, **env: str) -> tuple[Harness, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "rg"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    log = tmp_path / "rg.log"
    backend = LocalBackend(
        root, env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "RG_LOG": str(log), **env}
    )
    backend._capabilities = dataclasses.replace(backend.capabilities, has_rg=True)
    return build_harness(backend, tools=["glob", "grep"]), log


async def call(h: Harness, name: str, **kwargs: object) -> str:
    result: str = await h.tool(name).execute(**kwargs)
    return result


async def test_glob_reads_ripgreps_listing(root: Path, tmp_path: Path) -> None:
    h, log = stubbed(root, tmp_path)
    assert await call(h, "glob", pattern="*.py") == "new.py\nsrc/old.py"
    sent = log.read_text()
    assert "--files --sortr modified" in sent
    assert "--glob !node_modules" in sent
    assert "--glob *.py" in sent


async def test_grep_reads_ripgreps_hits_and_sends_the_filter(root: Path, tmp_path: Path) -> None:
    (root / "src").mkdir()
    h, log = stubbed(root, tmp_path)
    out = await call(h, "grep", pattern="needle", path="src", glob_filter="*.py")
    assert out == "src/a.py:2: needle here"
    sent = log.read_text()
    assert "--glob *.py" in sent
    assert "--regexp needle -- src" in sent


async def test_nothing_found_is_trusted_without_a_walk(root: Path, tmp_path: Path) -> None:
    (root / "a.py").write_text("needle\n")
    h, _ = stubbed(root, tmp_path, RG_EMPTY="1", RG_EXIT="1")
    assert await call(h, "grep", pattern="needle") == "no matches for needle"
    assert await call(h, "glob", pattern="*.py") == "no files matching *.py"


@pytest.mark.parametrize("code", ["2", "127"])
async def test_a_search_ripgrep_cannot_run_falls_back_to_the_walk(
    root: Path, tmp_path: Path, code: str
) -> None:
    (root / "a.py").write_text("needle\n")
    h, _ = stubbed(root, tmp_path, RG_EMPTY="1", RG_EXIT=code)
    assert await call(h, "grep", pattern="needle") == "a.py:1: needle"
    assert await call(h, "glob", pattern="*.py") == "a.py"


async def test_an_error_alongside_hits_keeps_the_hits(root: Path, tmp_path: Path) -> None:
    h, _ = stubbed(root, tmp_path, RG_EXIT="2")
    assert await call(h, "grep", pattern="needle") == "src/a.py:2: needle here"


async def test_a_ripgrep_the_workspace_cannot_run_falls_back(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (root / "a.py").write_text("needle\n")
    h, _ = stubbed(root, tmp_path)
    backend = h.backend
    real = backend.exec

    def refusing(command: str, **kwargs: object) -> ExecResult:
        if "rg " in command:
            raise BackendError("permission_denied")
        return real(command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend, "exec", refusing)
    assert await call(h, "glob", pattern="*.py") == "a.py"

    def garbled(command: str, **kwargs: object) -> ExecResult:
        return ExecResult(exit_code=0, output=b"no status line", timed_out=False)

    monkeypatch.setattr(backend, "exec", garbled)
    assert await call(h, "grep", pattern="needle") == "a.py:1: needle"
