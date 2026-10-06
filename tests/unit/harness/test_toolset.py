# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""build_harness: the tool set, the single gate, the preview, the prompt."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tulip.control import ControlPolicy, gate_tool
from tulip.harness import (
    ALL_TOOLS,
    DEFAULT_TOOLS,
    PATCH_TOOLS,
    HarnessConfig,
    LocalBackend,
    MemoryBackend,
    build_harness,
)
from tulip.harness.labels import KIND_EXEC
from tulip.tools.decorator import Tool


def test_the_default_set_is_everything_but_apply_patch(tmp_path: Path) -> None:
    h = build_harness(LocalBackend(tmp_path))
    assert [t.name for t in h.tools] == list(DEFAULT_TOOLS)
    assert "apply_patch" not in DEFAULT_TOOLS
    assert "apply_patch" in ALL_TOOLS
    assert "edit" not in PATCH_TOOLS
    assert "apply_patch" in PATCH_TOOLS
    assert set(h.specs) == set(DEFAULT_TOOLS)
    assert not h.gated
    assert h.backend.capabilities.root == str(tmp_path.resolve())


def test_a_backend_without_a_shell_gets_no_shell_tools() -> None:
    h = build_harness(MemoryBackend())
    names = {t.name for t in h.tools}
    assert "bash" not in names
    assert "read" in names
    assert "Commands run" not in h.prompt_fragment
    with pytest.raises(ValueError, match="has no shell for bash"):
        build_harness(MemoryBackend(), tools=["read", "bash"])


def test_unknown_tools_are_refused() -> None:
    with pytest.raises(ValueError, match="'teleport'"):
        build_harness(MemoryBackend(), tools=["read", "teleport"])
    with pytest.raises(KeyError):
        build_harness(MemoryBackend(), tools=["read"]).tool("write")


def test_ungated_shell_on_the_host_is_logged_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="tulip.harness.toolset"):
        build_harness(LocalBackend(tmp_path))
    assert "ungated shell tools" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tulip.harness.toolset"):
        build_harness(LocalBackend(tmp_path), wrap=lambda t, spec: t)
        build_harness(LocalBackend(tmp_path), tools=["read"])
    assert caplog.text == ""


def test_wrap_sees_every_tool_with_its_spec(tmp_path: Path) -> None:
    seen: list[tuple[str, object]] = []

    def wrap(t: Tool, spec: object) -> Tool:
        seen.append((t.name, spec))
        return t

    h = build_harness(LocalBackend(tmp_path), wrap=wrap, tools=["read", "bash"])
    assert [name for name, _ in seen] == ["read", "bash"]
    assert h.gated
    assert seen[1][1] is h.specs["bash"]


async def test_the_control_gate_wraps_the_tools_and_holds_exec(tmp_path: Path) -> None:
    """The local CLI's wiring: one gate, applied by wrap, deciding on the labels."""
    policy = ControlPolicy(require_human_for=frozenset({KIND_EXEC}), require_verification_score=0.0)
    h = build_harness(
        LocalBackend(tmp_path),
        wrap=lambda t, spec: gate_tool(t, policy=policy, action=spec),
        config=HarnessConfig(environment="dev"),
    )
    (tmp_path / "a.txt").write_text("hello\n")
    read = await h.tool("read").execute(path="a.txt")
    assert "hello" in read
    listed = await h.tool("bash").execute(command="ls")
    assert "a.txt" in listed, "a read-only line is a read, and reads are allowed"
    held = await h.tool("bash").execute(command="touch made.txt")
    assert json.loads(held)["outcome"] == "require_human"
    assert not (tmp_path / "made.txt").exists()


def test_preview_shows_the_diff_without_writing() -> None:
    backend = MemoryBackend({"a.py": "x = 1\n"})
    h = build_harness(backend, config=HarnessConfig(require_read=False))
    preview = h.preview("edit", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
    assert preview is not None
    assert "-x = 1" in preview
    assert "+x = 2" in preview
    assert backend.read_bytes("a.py") == b"x = 1\n"
    assert h.preview("read", {"path": "a.py"}) is None
    assert "no such file" in str(h.preview("edit", {"path": "nope.py", "old": "a", "new": "b"}))
    patch = "*** Begin Patch\n*** Delete File: a.py\n*** Add File: b.py\n+b\n*** End Patch"
    shown = h.preview("apply_patch", {"input": patch})
    assert shown is not None
    assert "a.py: deleted — the file is removed" in shown
    assert "+b" in shown
    assert backend.exists("a.py")
    assert not backend.exists("b.py")


def test_the_prompt_says_what_the_model_needs(tmp_path: Path) -> None:
    host = build_harness(LocalBackend(tmp_path), wrap=lambda t, s: t).prompt_fragment
    assert "UNISOLATED: host shell" in host
    assert str(tmp_path.resolve()) in host
    assert "not in a sandbox" in host
    assert "Read a file before" in host
    assert "background=true" in host
    patched = build_harness(MemoryBackend(), tools=PATCH_TOOLS[:3] + ("apply_patch",))
    assert "apply_patch" in patched.prompt_fragment
    relaxed = build_harness(MemoryBackend(), config=HarnessConfig(require_read=False))
    assert "Read a file before" not in relaxed.prompt_fragment
