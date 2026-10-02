# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Text tool calls: only unambiguous shapes, and only when asked for.

The failure these guard against: a final answer that says
``run bash(command="pytest") to verify`` was parsed into a real ``bash``
call and executed, and ``read(path)`` became ``read({})``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip import Agent
from tulip.agent.text_tool_calls import parse_text_tool_calls
from tulip.testing import ScriptedModel, text
from tulip.tools import tool
from tulip.tools.registry import ToolRegistry


ran: list[tuple[str, dict[str, Any]]] = []


@tool
def bash(command: str) -> str:
    """Run a shell command."""
    ran.append(("bash", {"command": command}))
    return "ok"


@tool
def read(path: str) -> str:
    """Read a file."""
    ran.append(("read", {"path": path}))
    return "contents"


@pytest.fixture(autouse=True)
def _reset_ran() -> None:
    ran.clear()


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(bash)
    reg.register(read)
    return reg


def _calls(body: str, registry: ToolRegistry) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, c.arguments) for c in parse_text_tool_calls(body, registry)]


class TestProseIsNotACall:
    def test_the_reported_final_answer(self, registry: ToolRegistry) -> None:
        assert _calls('run bash(command="pytest") to verify', registry) == []

    def test_positional_argument_is_not_guessed(self, registry: ToolRegistry) -> None:
        assert _calls("read(path)", registry) == []
        assert _calls('read("src/app.py")', registry) == []

    def test_call_in_a_markdown_answer(self, registry: ToolRegistry) -> None:
        body = 'All tests pass. To check yourself:\n\nbash(command="pytest -q")\n\nDone.'
        assert _calls(body, registry) == []

    def test_json_mentioned_in_prose(self, registry: ToolRegistry) -> None:
        body = 'The model would send {"name": "bash", "arguments": {"command": "ls"}} here.'
        assert _calls(body, registry) == []

    def test_untagged_fence_is_an_example_not_a_call(self, registry: ToolRegistry) -> None:
        assert _calls('Example:\n```\nbash(command="ls")\n```', registry) == []

    def test_python_fence_is_code_not_a_call(self, registry: ToolRegistry) -> None:
        assert _calls('```python\nbash(command="ls")\n```', registry) == []

    def test_non_literal_argument_is_rejected(self, registry: ToolRegistry) -> None:
        assert _calls("bash(command=os.environ)", registry) == []

    def test_unknown_tool_in_a_json_list_rejects_the_list(self, registry: ToolRegistry) -> None:
        body = '[{"name": "bash", "arguments": {"command": "ls"}}, {"name": "rm", "arguments": {}}]'
        assert _calls(body, registry) == []

    def test_no_registry(self) -> None:
        assert parse_text_tool_calls('bash(command="ls")', None) == []


class TestUnambiguousShapes:
    def test_whole_message_call_syntax(self, registry: ToolRegistry) -> None:
        assert _calls('bash(command="pytest")', registry) == [("bash", {"command": "pytest"})]

    def test_literal_arguments_keep_their_type(self, registry: ToolRegistry) -> None:
        assert _calls("bash(command='ls')\nread(path='a.py')", registry) == [
            ("bash", {"command": "ls"}),
            ("read", {"path": "a.py"}),
        ]

    def test_whole_message_json_list(self, registry: ToolRegistry) -> None:
        body = (
            '[{"name": "bash", "arguments": {"command": "ls"}},'
            ' {"name": "read", "arguments": {"path": "x"}}]'
        )
        assert _calls(body, registry) == [("bash", {"command": "ls"}), ("read", {"path": "x"})]

    def test_openai_wire_shape(self, registry: ToolRegistry) -> None:
        body = '{"type": "function", "function": {"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}}'
        assert _calls(body, registry) == [("bash", {"command": "ls"})]

    def test_tool_call_fence_with_call_syntax(self, registry: ToolRegistry) -> None:
        body = 'Running the suite.\n```tool_call\nbash(command="pytest")\n```'
        assert _calls(body, registry) == [("bash", {"command": "pytest"})]

    def test_hermes_tag(self, registry: ToolRegistry) -> None:
        body = 'Let me look.\n<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>'
        assert _calls(body, registry) == [("read", {"path": "a.py"})]

    def test_bad_arguments_string_rejects_the_call(self, registry: ToolRegistry) -> None:
        assert _calls('{"name": "bash", "arguments": "not json"}', registry) == []


class TestLoopGate:
    """Whether the loop parses at all depends on ``text_tool_calls``."""

    @pytest.mark.asyncio
    async def test_default_never_runs_a_call_written_as_text(self) -> None:
        model = ScriptedModel([text('bash(command="pytest")')])
        agent = Agent(model=model, tools=[bash])

        result = await agent.arun("verify")

        assert ran == []
        assert result.text == 'bash(command="pytest")'
        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_on_runs_an_unambiguous_call(self) -> None:
        model = ScriptedModel([text('bash(command="pytest")'), text("All green.")])
        agent = Agent(model=model, tools=[bash], text_tool_calls="on")

        result = await agent.arun("verify")

        assert ran == [("bash", {"command": "pytest"})]
        assert result.text == "All green."

    @pytest.mark.asyncio
    async def test_on_does_not_run_the_reported_final_answer(self) -> None:
        answer = 'Fixed the import. run bash(command="pytest") to verify'
        model = ScriptedModel([text(answer)])
        agent = Agent(model=model, tools=[bash], text_tool_calls="on")

        result = await agent.arun("fix it")

        assert ran == []
        assert result.text == answer

    @pytest.mark.asyncio
    async def test_auto_parses_for_a_model_without_native_tool_calls(self) -> None:
        model = ScriptedModel([text('bash(command="ls")'), text("Listed.")])
        model.supports_native_tool_calls = False  # type: ignore[attr-defined]
        agent = Agent(model=model, tools=[bash])

        await agent.arun("list")

        assert ran == [("bash", {"command": "ls"})]

    @pytest.mark.asyncio
    async def test_off_ignores_a_model_without_native_tool_calls(self) -> None:
        model = ScriptedModel([text('bash(command="ls")')])
        model.supports_native_tool_calls = False  # type: ignore[attr-defined]
        agent = Agent(model=model, tools=[bash], text_tool_calls="off")

        await agent.arun("list")

        assert ran == []
