# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Elicitation: an MCP server asks the person a question mid-call.

Inside an agent run the question pauses the run through the interrupt
mechanism — an ``InterruptEvent`` whose ``fields`` describe the answer — and
``agent.resume(answer, perform_dangling=True)`` re-issues the call with the
answer parked on the client. Outside a run the client's
``elicitation_handler`` answers, and without one the question is declined.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()

from locus.agent import Agent  # noqa: E402
from locus.core.events import InterruptEvent, TerminateEvent, ToolCompleteEvent  # noqa: E402
from locus.integrations.fastmcp import (  # noqa: E402
    ElicitationAnswer,
    MCPClient,
    MCPElicitation,
)
from locus.testing import ScriptedModel, text, tool_call  # noqa: E402
from tests.unit.test_mcp_fidelity import _LoopbackServer  # noqa: E402


@pytest.fixture(scope="module")
def server() -> Iterator[_LoopbackServer]:
    srv = _LoopbackServer().start()
    yield srv
    srv.stop()


def _client(server: _LoopbackServer, **kwargs: Any) -> MCPClient:
    return MCPClient(base_url=server.url, verify_url=False, **kwargs)


# ---------------------------------------------------------------------------
# Inside an agent run: pause, answer, resume
# ---------------------------------------------------------------------------


async def test_a_question_pauses_the_run_and_the_answer_resumes_it(
    server: _LoopbackServer,
) -> None:
    client = _client(server)
    model = ScriptedModel([tool_call("refund", call_id="r1", order_id="o-1"), text("done")])
    agent = Agent(model=model, mcp_servers=[client], reflexion=False, grounding=False)

    events = [event async for event in agent.run("refund o-1")]
    paused = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(paused) == 1
    [question] = paused
    assert question.question == "Refund order o-1?"
    assert question.interrupt_id == "r1"
    assert question.fields == [
        {"name": "approve", "label": "Approve", "type": "boolean", "required": True}
    ]
    assert question.metadata["mcp_elicitation"]["tool"] == "refund"
    assert question.metadata["mcp_elicitation"]["server"] == server.url
    assert not any(isinstance(e, TerminateEvent) for e in events)  # paused, not finished
    assert not any(isinstance(e, ToolCompleteEvent) for e in events)

    # The person says yes; the resume re-issues the call and the server,
    # asked again, gets the answer.
    client.answer_elicitation(question, "yes")
    resumed = [event async for event in agent.resume("yes", perform_dangling=True)]
    done = next(e for e in resumed if isinstance(e, ToolCompleteEvent))
    assert done.tool_call_id == "r1"
    assert done.result == "refunded o-1"
    assert any(isinstance(e, TerminateEvent) for e in resumed)
    # The model saw the real result, not the pause.
    tool_msg = next(m for m in model.received_messages[-1] if m.tool_call_id == "r1")
    assert tool_msg.content == "refunded o-1"
    await client.close()


async def test_a_two_field_question_is_answered_with_a_mapping(server: _LoopbackServer) -> None:
    client = _client(server)
    model = ScriptedModel([tool_call("escalate", call_id="e1", ticket="T-9"), text("ok")])
    agent = Agent(model=model, mcp_servers=[client], reflexion=False, grounding=False)

    events = [event async for event in agent.run("escalate")]
    [question] = [e for e in events if isinstance(e, InterruptEvent)]
    assert [f["name"] for f in question.fields or []] == ["reason", "priority"]
    assert [f["required"] for f in question.fields or []] == [True, False]
    assert question.options is None

    client.answer_elicitation(question.interrupt_id, {"reason": "outage", "priority": 2})
    resumed = [event async for event in agent.resume("", perform_dangling=True)]
    done = next(e for e in resumed if isinstance(e, ToolCompleteEvent))
    assert done.result == "escalated T-9: outage (p2)"
    await client.close()


async def test_declining_on_resume_lets_the_server_decide(server: _LoopbackServer) -> None:
    client = _client(server)
    model = ScriptedModel([tool_call("refund", call_id="r2", order_id="o-2"), text("ok")])
    agent = Agent(model=model, mcp_servers=[client], reflexion=False, grounding=False)
    events = [event async for event in agent.run("refund")]
    [question] = [e for e in events if isinstance(e, InterruptEvent)]

    client.answer_elicitation(question, None)  # the person declined
    resumed = [event async for event in agent.resume("no", perform_dangling=True)]
    done = next(e for e in resumed if isinstance(e, ToolCompleteEvent))
    assert done.result == "decline: o-2"
    await client.close()


async def test_a_handler_answers_inside_a_run_without_pausing(server: _LoopbackServer) -> None:
    asked: list[MCPElicitation] = []

    def auto_approve(question: MCPElicitation) -> dict[str, Any]:
        asked.append(question)
        return {"approve": True}

    client = _client(server, elicitation_handler=auto_approve)
    model = ScriptedModel([tool_call("refund", call_id="r3", order_id="o-3"), text("ok")])
    agent = Agent(model=model, mcp_servers=[client], reflexion=False, grounding=False)
    events = [event async for event in agent.run("refund")]
    assert not any(isinstance(e, InterruptEvent) for e in events)
    done = next(e for e in events if isinstance(e, ToolCompleteEvent))
    assert done.result == "refunded o-3"
    assert asked[0].tool_name == "refund"
    assert asked[0].tool_call_id == "r3"
    await client.close()


# ---------------------------------------------------------------------------
# Outside a run: the handler, or a decline
# ---------------------------------------------------------------------------


async def test_outside_a_run_the_handler_answers(server: _LoopbackServer) -> None:
    async def by_text(question: MCPElicitation) -> str:
        assert question.message == "Refund order o-4?"
        return "approve"  # coerced onto the single boolean field

    client = _client(server, elicitation_handler=by_text)
    await client.connect()
    try:
        assert await client.call_tool("refund", {"order_id": "o-4"}) == "refunded o-4"
    finally:
        await client.close()


async def test_outside_a_run_without_a_handler_the_question_is_declined(
    server: _LoopbackServer, caplog: pytest.LogCaptureFixture
) -> None:
    client = _client(server)
    await client.connect()
    try:
        with caplog.at_level("INFO"):
            assert await client.call_tool("refund", {"order_id": "o-5"}) == "decline: o-5"
    finally:
        await client.close()
    assert any("no elicitation_handler" in r.getMessage() for r in caplog.records)


async def test_a_failing_handler_declines_rather_than_hanging(server: _LoopbackServer) -> None:
    def broken(question: MCPElicitation) -> dict[str, Any]:
        raise RuntimeError("boom")

    client = _client(server, elicitation_handler=broken)
    await client.connect()
    try:
        assert await client.call_tool("refund", {"order_id": "o-6"}) == "decline: o-6"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Answer shapes
# ---------------------------------------------------------------------------

ONE_BOOL = {"type": "object", "properties": {"approve": {"type": "boolean"}}}
ONE_INT = {"type": "object", "properties": {"count": {"type": "integer"}}}
TWO = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ({"approve": True}, ONE_BOOL, ElicitationAnswer.accept({"approve": True})),
        (None, ONE_BOOL, ElicitationAnswer.decline()),
        ("yes", ONE_BOOL, ElicitationAnswer.accept({"approve": True})),
        ("Denied", ONE_BOOL, ElicitationAnswer.accept({"approve": False})),
        (True, ONE_BOOL, ElicitationAnswer.accept({"approve": True})),
        ("3", ONE_INT, ElicitationAnswer.accept({"count": 3})),
        (' {"a": "x", "b": "y"}', TWO, ElicitationAnswer.accept({"a": "x", "b": "y"})),
        ("just text", TWO, ElicitationAnswer.decline()),
        ("free", None, ElicitationAnswer.accept({"value": "free"})),
        (ElicitationAnswer.cancel(), ONE_BOOL, ElicitationAnswer.cancel()),
    ],
)
def test_answers_are_shaped_to_the_schema(
    value: Any, schema: dict[str, Any] | None, expected: ElicitationAnswer
) -> None:
    assert ElicitationAnswer.from_value(value, schema) == expected


def test_elicitation_fields_and_options() -> None:
    question = MCPElicitation(
        message="Which tier?",
        schema={
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["gold", "silver"], "title": "Tier"},
            },
            "required": ["tier"],
        },
    )
    assert question.options == ["gold", "silver"]
    assert question.fields[0]["label"] == "Tier"
    assert MCPElicitation(message="url mode", mode="url", url="https://x").fields == []


def test_answer_elicitation_needs_an_id() -> None:
    client = MCPClient(base_url="https://x.example")
    with pytest.raises(ValueError, match="interrupt_id"):
        client.answer_elicitation(object(), "yes")
