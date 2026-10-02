# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent definitions from Markdown files, in the formats people already have.

The failure each test guards: a team's existing ``.claude/agents`` or opencode
agent file silently read wrong — a ``tools`` line ignored, so a "read-only"
reviewer gets the shell — or one malformed file taking every agent down with
it.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from tulip.agent.specs import (
    AgentSpec,
    _mini_yaml,
    load_agent_file,
    load_agent_specs,
    merge_specs,
    parse_agent_markdown,
)


class _T:
    def __init__(self, name: str) -> None:
        self.name = name


TOOLS = [_T(n) for n in ("read", "grep", "glob", "edit", "write", "bash", "web_fetch")]
TOOLS += [_T("mcp__github__create_issue"), _T("mcp__github__list_prs")]


def _names(tools: list[Any]) -> list[str]:
    return [t.name for t in tools]


@pytest.fixture(params=["pyyaml", "builtin"])
def yaml_impl(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Every parsing test runs with PyYAML and with the built-in subset parser."""
    if request.param == "builtin":
        real_import = builtins.__import__

        def no_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_yaml)
    return str(request.param)


# ------------------------------------------------------------------ formats --


CLAUDE_CODE = """---
name: code-reviewer
description: Reviews code for bugs. Use proactively after a change.
tools: Read, Grep, Glob, WebFetch
model: inherit
---
You are a senior reviewer.

Report bugs with file and line.
"""

OPENCODE = """---
description: Writes and maintains docs
mode: subagent
model: anthropic/claude-sonnet-4
temperature: 0.2
steps: 12
tools:
  write: false
  bash: false
permission:
  edit: deny
  bash:
    "git push": ask
---
You write documentation.
"""


def test_claude_code_file(yaml_impl: str) -> None:
    spec = parse_agent_markdown(CLAUDE_CODE, default_mode="subagent")
    assert spec.name == "code-reviewer"
    assert spec.description.startswith("Reviews code for bugs.")
    assert spec.model is None  # inherit
    assert spec.mode == "subagent"
    assert spec.prompt == "You are a senior reviewer.\n\nReport bugs with file and line."
    assert _names(spec.select_tools(TOOLS)) == ["read", "grep", "glob", "web_fetch"]


def test_opencode_file(yaml_impl: str) -> None:
    spec = parse_agent_markdown(OPENCODE, name="docs-writer")
    assert spec.name == "docs-writer"
    assert spec.mode == "subagent"
    assert spec.model == "anthropic/claude-sonnet-4"
    assert spec.temperature == 0.2
    assert spec.max_turns == 12
    assert spec.tools is None
    assert spec.disallowed_tools == ("write", "bash")
    assert "bash" not in _names(spec.select_tools(TOOLS))
    assert "edit" in _names(spec.select_tools(TOOLS))
    # Not modelled, kept for the harness.
    assert spec.metadata["permission"]["edit"] == "deny"
    assert spec.metadata["permission"]["bash"] == {"git push": "ask"}


def test_tulip_file_with_lists_and_deny(yaml_impl: str) -> None:
    text = """---
name: planner
description: >
  Plans a change
  without making it.
mode: primary
max_turns: 30
tools: [read, grep, "mcp__github__*"]
disallowed_tools:
  - mcp__github__create_issue
---
Plan only.
"""
    spec = parse_agent_markdown(text)
    assert spec.description == "Plans a change without making it."
    assert spec.is_primary
    assert not spec.is_subagent
    assert spec.max_turns == 30
    assert _names(spec.select_tools(TOOLS)) == ["read", "grep", "mcp__github__list_prs"]


def test_a_star_map_switches_everything_off_but_what_is_on(yaml_impl: str) -> None:
    text = '---\nname: narrow\ntools:\n  "*": false\n  read: true\n  bash: false\n---\n'
    spec = parse_agent_markdown(text)
    assert spec.tools == ("read",)
    assert _names(spec.select_tools(TOOLS)) == ["read"]


@pytest.mark.parametrize("value", ['"*"', "all"])
def test_all_tools_spelled_out_is_the_default(yaml_impl: str, value: str) -> None:
    spec = parse_agent_markdown(f"---\nname: x\ntools: {value}\n---\n")
    assert spec.tools is None


def test_block_text_keeps_its_lines(yaml_impl: str) -> None:
    text = "---\nname: x\ndescription: |\n  line one\n  line two\n---\nbody\n"
    assert parse_agent_markdown(text).description == "line one\nline two"


def test_no_frontmatter_is_a_prompt_with_a_given_name() -> None:
    spec = parse_agent_markdown("Just do the thing.", name="plain")
    assert spec.name == "plain"
    assert spec.prompt == "Just do the thing."
    assert spec.mode == "all"


def test_a_name_is_required() -> None:
    with pytest.raises(ValueError, match="no name"):
        parse_agent_markdown("---\ndescription: x\n---\nbody")


def test_an_unclosed_frontmatter_is_an_error() -> None:
    with pytest.raises(ValueError, match="never closes"):
        parse_agent_markdown("---\nname: x\nbody")


@pytest.mark.parametrize("name", ["bad name", "-dash", "x" * 65, ""])
def test_names_are_validated(name: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 — pydantic's or ours
        AgentSpec(name=name)


def test_a_bad_mode_or_turn_count_is_an_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        parse_agent_markdown("---\nname: x\nmode: boss\n---\n")
    with pytest.raises(ValueError):  # noqa: PT011
        parse_agent_markdown("---\nname: x\nmax_turns: 0\n---\n")


def test_tools_must_be_names() -> None:
    with pytest.raises(ValueError, match="list of tool names"):
        parse_agent_markdown("---\nname: x\ndisallowed_tools: 3\n---\n")


# ---------------------------------------------------------- the subset parser --


def test_mini_yaml_scalars() -> None:
    data = _mini_yaml(
        "a: 1\nb: 2.5\nc: true\nd: off\ne: null\nf: 'single'\n"
        'g: "dq \\"x\\""\nh: plain # comment\ni: {k: v, n: 1}\nj: []\nk:\n'
    )
    assert data == {
        "a": 1,
        "b": 2.5,
        "c": True,
        "d": False,
        "e": None,
        "f": "single",
        "g": 'dq "x"',
        "h": "plain",
        "i": {"k": "v", "n": 1},
        "j": [],
        "k": None,
    }


def test_mini_yaml_list_at_the_keys_indentation() -> None:
    assert _mini_yaml("tools:\n- read\n- grep\nname: x") == {"tools": ["read", "grep"], "name": "x"}


def test_mini_yaml_inline_list_with_quoted_commas() -> None:
    assert _mini_yaml('a: ["x, y", z, [1, 2]]') == {"a": ["x, y", "z", [1, 2]]}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("just text", "key: value"),
        ("- a\n- b", "mapping"),
        ("a: 1\n    b: 2", "indentation"),
    ],
)
def test_mini_yaml_rejects_what_it_cannot_read(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _mini_yaml(text)


def test_pyyaml_rejects_a_non_mapping() -> None:
    pytest.importorskip("yaml")
    with pytest.raises(ValueError, match="mapping"):
        parse_agent_markdown("---\n- a\n---\n", name="x")


# ------------------------------------------------------------------ tools --


def test_names_compare_without_case_or_separators() -> None:
    spec = AgentSpec(name="x", tools=("WebFetch", "multi-edit"))
    assert spec.allows("web_fetch")
    assert spec.allows("multi_edit")
    assert not spec.allows("read")


def test_disallowed_wins_over_allowed() -> None:
    spec = AgentSpec(name="x", tools=("*",), disallowed_tools=("bash", "mcp__*"))
    assert _names(spec.select_tools(TOOLS)) == [
        "read",
        "grep",
        "glob",
        "edit",
        "write",
        "web_fetch",
    ]


def test_unknown_tools_are_reported() -> None:
    spec = AgentSpec(name="x", tools=("read", "Task", "mcp__slack__*"))
    assert spec.unknown_tools(TOOLS) == ["Task", "mcp__slack__*"]
    assert AgentSpec(name="y").unknown_tools(TOOLS) == []


# ---------------------------------------------------------------- loading --


def test_load_directories_later_wins_and_disable_removes(tmp_path: Path) -> None:
    user = tmp_path / "user"
    project = tmp_path / "project"
    claude = tmp_path / "claude"
    for d in (user, project, claude):
        d.mkdir()
    (user / "reviewer.md").write_text("---\ndescription: user reviewer\n---\nuser prompt")
    (user / "helper.md").write_text("---\ndescription: helper\n---\nhelp")
    (project / "reviewer.md").write_text("---\ndescription: project reviewer\n---\nproject")
    (project / "helper.md").write_text("---\ndisable: true\n---\n")
    (project / "notes.txt").write_text("not an agent")
    (claude / "cc.md").write_text("---\nname: cc\n---\nfrom claude")

    specs = load_agent_specs([user, project, (claude, "subagent"), tmp_path / "missing"])

    assert list(specs) == ["reviewer", "cc"]
    assert specs["reviewer"].description == "project reviewer"
    assert specs["reviewer"].source == str(project / "reviewer.md")
    assert specs["cc"].mode == "subagent"


def test_a_broken_file_is_skipped_and_reported(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("fine")
    (tmp_path / "bad.md").write_text("---\nname: bad name\n---\n")
    (tmp_path / "worse.md").write_bytes(b"\xff\xfe\x00bad")
    errors: list[tuple[Path, Exception]] = []
    specs = load_agent_specs([tmp_path], on_error=lambda p, e: errors.append((p, e)))
    assert list(specs) == ["good"]
    assert sorted(p.name for p, _ in errors) == ["bad.md", "worse.md"]


def test_a_broken_file_logs_by_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "bad.md").write_text("---\nname: x\nunclosed")
    assert load_agent_specs([tmp_path]) == {}
    assert "skipping agent file" in caplog.text


def test_load_agent_file_uses_the_stem(tmp_path: Path) -> None:
    file = tmp_path / "test-writer.md"
    file.write_text("﻿Write tests.")
    spec = load_agent_file(file)
    assert spec.name == "test-writer"
    assert spec.prompt == "Write tests."


def test_merge_specs_later_layers_win() -> None:
    a = {"x": AgentSpec(name="x", description="a")}
    b = {"x": AgentSpec(name="x", description="b"), "y": AgentSpec(name="y")}
    merged = merge_specs(a, b)
    assert merged["x"].description == "b"
    assert list(merged) == ["x", "y"]
