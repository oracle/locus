# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A final answer delivered as a schema-valid tool call, and held to it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import TerminateEvent, ToolCompleteEvent
from tulip.core.messages import Message, ToolCall
from tulip.core.state import ToolExecution
from tulip.models.base import ModelResponse
from tulip.tools.output import ToolOutput
from tulip.tools.structured_output import (
    ACCEPTED,
    SchemaError,
    StructuredOutputTool,
    load_schema,
)


SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": ["pass", "fail"]}, "count": {"type": "integer"}},
    "required": ["verdict", "count"],
}


class _Model:
    def __init__(self, script: list[ModelResponse]) -> None:
        self.script = list(script)
        self.seen: list[list[Message]] = []

    async def complete(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.seen.append(list(messages))
        return self.script.pop(0)


def _call(arguments: dict[str, Any], call_id: str = "c1") -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=None,
            tool_calls=[ToolCall(id=call_id, name="structured_output", arguments=arguments)],
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _say(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(text), usage={"prompt_tokens": 1, "completion_tokens": 1}
    )


def _agent(model: _Model, out: StructuredOutputTool) -> Agent:
    return Agent(
        model=model,
        tools=[out.tool],
        max_iterations=10,
        reflexion=False,
        grounding=False,
        **out.agent_options(),
    )


async def _run(agent: Agent) -> list[Any]:
    return [ev async for ev in agent.run("judge it")]


# ------------------------------------------------------------------ the tool --


async def test_valid_arguments_are_accepted_and_read_back() -> None:
    out = StructuredOutputTool(SCHEMA)
    assert await out.tool.execute(verdict="pass", count=2) == ACCEPTED
    record = ToolExecution(
        tool_name="structured_output", tool_call_id="c", arguments={"verdict": "pass", "count": 2}
    )
    assert out.value_from([record]) == {"verdict": "pass", "count": 2}


async def test_invalid_arguments_are_an_error_naming_every_path_to_fix() -> None:
    out = StructuredOutputTool(SCHEMA)
    result = await out.tool.execute(verdict="maybe")
    assert isinstance(result, ToolOutput)
    assert result.is_error
    assert "$.verdict: must be one of" in result
    assert "missing required property 'count'" in result


def test_a_non_object_schema_is_wrapped_and_unwrapped() -> None:
    out = StructuredOutputTool(
        {"type": "array", "items": {"$ref": "#/$defs/n"}, "$defs": {"n": {"type": "integer"}}}
    )
    assert out.wrapped
    assert out.tool.parameters["required"] == ["value"]
    assert "$defs" in out.tool.parameters
    assert out.errors({"value": [1, 2]}) == []
    assert out.errors({"value": ["x"]}) == ["$[0]: expected integer, got string"]
    assert out.errors({}) == ["$: missing required property 'value'"]
    record = ToolExecution(
        tool_name="structured_output", tool_call_id="c", arguments={"value": [3]}
    )
    assert out.value_from([record]) == [3]


def test_value_from_skips_failed_and_other_calls() -> None:
    out = StructuredOutputTool(SCHEMA)
    good = {"verdict": "fail", "count": 0}
    records = [
        ToolExecution(tool_name="structured_output", tool_call_id="1", arguments=good),
        ToolExecution(tool_name="other", tool_call_id="2", arguments={}),
        ToolExecution(
            tool_name="structured_output", tool_call_id="3", arguments={"verdict": 1}, error="bad"
        ),
        # Recorded as a success by something else, but not schema-valid.
        ToolExecution(tool_name="structured_output", tool_call_id="4", arguments={"count": 1}),
    ]
    assert out.value_from(records) == good
    with pytest.raises(LookupError):
        out.value_from(records[1:])
    assert out.value_from([], default=None) is None


def test_a_bad_schema_is_refused_at_construction() -> None:
    with pytest.raises(SchemaError):
        StructuredOutputTool({"type": "nope"})


def test_instructions_name_the_tool() -> None:
    out = StructuredOutputTool(SCHEMA, name="deliver")
    assert "`deliver`" in out.instructions
    assert out.tool.name == "deliver"


# ------------------------------------------------------------- load_schema --


def test_load_schema_from_inline_json_or_a_file(tmp_path: Path) -> None:
    assert load_schema(json.dumps(SCHEMA)) == SCHEMA
    path = tmp_path / "s.json"
    path.write_text(json.dumps(SCHEMA))
    assert load_schema(str(path)) == SCHEMA


@pytest.mark.parametrize(
    ("source", "match"),
    [("{not json", "not valid JSON"), ("/no/such/schema.json", "cannot read")],
)
def test_load_schema_says_what_is_wrong(source: str, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        load_schema(source)


def test_load_schema_refuses_a_non_object(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text("[1]")
    with pytest.raises(SchemaError, match="JSON object"):
        load_schema(str(path))


# ------------------------------------------------------------- in the loop --


async def test_a_turn_that_ends_in_prose_is_sent_back_to_call_the_tool() -> None:
    out = StructuredOutputTool(SCHEMA)
    model = _Model(
        [
            _say("It passed, three of them."),
            _call({"verdict": "pass", "count": "three"}, "c1"),
            _call({"verdict": "pass", "count": 3}, "c2"),
            _say("Done."),
        ]
    )
    events = await _run(_agent(model, out))

    completes = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert completes[0].error is not None
    assert "$.count: expected integer" in completes[0].error
    assert completes[1].error is None
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"
    executions = [
        ToolExecution(
            tool_name=e.tool_name,
            tool_call_id=e.tool_call_id,
            arguments={"verdict": "pass", "count": 3} if e.error is None else {},
            error=e.error,
        )
        for e in completes
    ]
    assert out.value_from(executions) == {"verdict": "pass", "count": 3}
    reminder = model.seen[1][-1].content or ""
    assert "Call the structured_output tool" in reminder


async def test_reminders_run_out_and_the_turn_ends_without_a_value() -> None:
    out = StructuredOutputTool(SCHEMA, max_reminders=1)
    model = _Model([_say("no"), _say("still no")])
    agent = _agent(model, out)

    result = await agent.arun("judge it")

    assert out.value_from(result.state.tool_executions, None) is None
    assert len(model.seen) == 2
