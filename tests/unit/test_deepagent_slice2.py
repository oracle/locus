# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Slice 2 tests — TODO tracker + AGENTS.md memory loading."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from locus.deepagent import (
    Todo,
    TodoState,
    create_deepagent,
    load_agents_md,
    make_todo_tools,
)


# ---------------------------------------------------------------------------
# TodoState + write_todos / read_todos round-trips
# ---------------------------------------------------------------------------


class TestTodos:
    def test_todo_status_is_plain_string(self) -> None:
        """Memory rule: LLM-populated fields are plain str, not Literal —
        so the model can emit a status we haven't anticipated without
        validation crashes."""
        # Should accept arbitrary strings (project convention).
        t = Todo(content="x", status="something_unknown")
        assert t.status == "something_unknown"

    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(self) -> None:
        state = TodoState()
        write_todos, read_todos = make_todo_tools(state)
        payload = json.dumps(
            [
                {"content": "find pdb metric", "status": "in_progress"},
                {"content": "emit ground_truth edge", "status": "pending"},
            ]
        )
        result = await write_todos.execute(todos_json=payload)
        decoded = json.loads(result)
        assert len(decoded) == 2
        assert decoded[0]["content"] == "find pdb metric"
        assert decoded[0]["status"] == "in_progress"

        read_back = await read_todos.execute()
        assert json.loads(read_back) == decoded

    @pytest.mark.asyncio
    async def test_external_inspection(self) -> None:
        """Caller-supplied TodoState lets you inspect the list after
        the agent run."""
        state = TodoState()
        write_todos, _ = make_todo_tools(state)
        await write_todos.execute(
            todos_json=json.dumps([{"content": "do thing", "status": "completed"}])
        )
        snapshot = state.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0].content == "do thing"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self) -> None:
        write_todos, _ = make_todo_tools(TodoState())
        result = await write_todos.execute(todos_json="not json {")
        assert "invalid json" in result.lower()

    @pytest.mark.asyncio
    async def test_non_list_payload_rejected(self) -> None:
        write_todos, _ = make_todo_tools(TodoState())
        result = await write_todos.execute(
            todos_json=json.dumps({"content": "x", "status": "pending"})
        )
        assert "array" in result.lower()

    @pytest.mark.asyncio
    async def test_replace_semantics(self) -> None:
        """write_todos replaces, doesn't append."""
        state = TodoState()
        write_todos, _ = make_todo_tools(state)
        await write_todos.execute(
            todos_json=json.dumps([{"content": "first", "status": "pending"}])
        )
        await write_todos.execute(
            todos_json=json.dumps([{"content": "second", "status": "pending"}])
        )
        assert [t.content for t in state.snapshot()] == ["second"]


# ---------------------------------------------------------------------------
# load_agents_md
# ---------------------------------------------------------------------------


class TestLoadAgentsMd:
    def test_joins_two_files_with_separator(self, tmp_path: Path) -> None:
        a = tmp_path / "AGENTS.md"
        b = tmp_path / "PROJECT.md"
        a.write_text("base instructions")
        b.write_text("project-specific instructions")
        result = load_agents_md([str(a), str(b)])
        assert "Memory: AGENTS.md" in result
        assert "base instructions" in result
        assert "Memory: PROJECT.md" in result
        assert "project-specific instructions" in result
        assert "---" in result

    def test_skips_missing_paths_silently(self, tmp_path: Path) -> None:
        existing = tmp_path / "AGENTS.md"
        existing.write_text("hello")
        result = load_agents_md([str(existing), str(tmp_path / "ghost.md")])
        assert "hello" in result
        # Doesn't crash on the missing path.

    def test_empty_list_returns_empty_string(self) -> None:
        assert load_agents_md([]) == ""

    def test_all_missing_returns_empty_string(self, tmp_path: Path) -> None:
        assert load_agents_md([str(tmp_path / "ghost.md")]) == ""


# ---------------------------------------------------------------------------
# Factory wiring — enable_todos + memory_files flags
# ---------------------------------------------------------------------------


class TestCreateDeepagentSlice2:
    def _stub_oci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCI_PROFILE", "DEFAULT")

    def test_todos_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            reflexion=False,
            grounding=False,
        )
        names = {t.name for t in agent.config.tools}
        assert "write_todos" not in names
        assert "read_todos" not in names

    def test_enable_todos_attaches_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            enable_todos=True,
            reflexion=False,
            grounding=False,
        )
        names = {t.name for t in agent.config.tools}
        assert {"write_todos", "read_todos"} <= names

    def test_explicit_todo_state_observable_after_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_oci(monkeypatch)
        state = TodoState()
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            enable_todos=True,
            todo_state=state,
            reflexion=False,
            grounding=False,
        )
        write_todos = next(t for t in agent.config.tools if t.name == "write_todos")
        asyncio.run(
            write_todos.execute(
                todos_json=json.dumps([{"content": "external", "status": "pending"}])
            )
        )
        assert state.snapshot()[0].content == "external"

    def test_memory_files_prepended_to_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub_oci(monkeypatch)
        memo = tmp_path / "AGENTS.md"
        memo.write_text("PROJECT RULE: never delete tables.")
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            memory_files=[str(memo)],
            reflexion=False,
            grounding=False,
        )
        prompt = agent.config.system_prompt
        # Memory block is *prepended* — search for our marker BEFORE
        # the recipe-specific identity.
        assert prompt.index("PROJECT RULE") < prompt.index("be helpful")

    def test_memory_files_missing_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            memory_files=[str(tmp_path / "ghost.md")],
            reflexion=False,
            grounding=False,
        )
        # No memory block — just the original prompt.
        assert agent.config.system_prompt == "be helpful"
