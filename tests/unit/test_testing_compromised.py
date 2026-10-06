# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``tulip.testing.CompromisedModel`` and the ``MockModel`` name.

A rogue suite proves that what holds is the gate, not the model. These tests
run the double against a real :class:`Agent`, a real ``gate_tool`` and a real
audit trail: the model is owned, and the side effect must still not happen.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.control import AuditTrail, ControlPolicy, gate_tool
from tulip.core.messages import Message
from tulip.testing import CompromisedModel, FunctionModel, MockModel, text, tool_call
from tulip.tools.decorator import tool


#: What actually happened in "the world". A breach is anything landing here.
EFFECTS: list[tuple[str, dict[str, Any]]] = []


@tool
def wipe_database(reason: str) -> str:
    """Delete every row in the production database."""
    EFFECTS.append(("wipe_database", {"reason": reason}))
    return "wiped"


@tool
def server_status() -> str:
    """Report the server status."""
    return "ok"


@pytest.fixture(autouse=True)
def _clean_world() -> None:
    EFFECTS.clear()


def _gated(trail: AuditTrail) -> Any:
    return gate_tool(
        wipe_database,
        policy=ControlPolicy(deny_for=frozenset({"wipe_database"})),
        trail=trail,
    )


def _agent(model: Any, tools: list[Any], **kwargs: Any) -> Agent:
    return Agent(model=model, tools=tools, max_iterations=4, **kwargs)


# --------------------------------------------------------------------------
# MockModel
# --------------------------------------------------------------------------


def test_mock_model_is_function_model() -> None:
    assert MockModel is FunctionModel


async def test_mock_model_runs_an_agent() -> None:
    model = MockModel(lambda messages, tools: "hello")
    result = await _agent(model, []).arun("hi")
    assert result.text == "hello"
    assert model.call_count == 1


# --------------------------------------------------------------------------
# CompromisedModel
# --------------------------------------------------------------------------


async def test_attacks_every_round_and_the_gate_holds() -> None:
    trail = AuditTrail()
    model = CompromisedModel("wipe_database", {"reason": "cleanup"})
    await _agent(model, [_gated(trail), server_status]).arun("check the status")

    # It never stopped trying: every model call that offered tools attacked.
    assert len(model.attempts) >= 2
    assert all(a == ("wipe_database", {"reason": "cleanup"}) for a in model.attempts)
    # And nothing happened in the world.
    assert EFFECTS == []
    # Every attempt is on the record, and the record is intact.
    denied = [r for r in trail.records() if r.payload.get("outcome") == "deny"]
    assert len(denied) == len(model.attempts)
    assert trail.verify()


async def test_ungated_tool_is_a_breach_the_suite_catches() -> None:
    """The control: without the gate the same model does real damage."""
    model = CompromisedModel("wipe_database", {"reason": "x"}, rounds=1)
    await _agent(model, [wipe_database]).arun("hi")
    assert EFFECTS == [("wipe_database", {"reason": "x"})]


async def test_rounds_limit_then_says_after() -> None:
    model = CompromisedModel("server_status", rounds=2, after="All yours.")
    result = await _agent(model, [server_status]).arun("go")
    assert [name for name, _ in model.attempts] == ["server_status", "server_status"]
    assert result.text == "All yours."


async def test_call_ids_are_unique_within_a_run() -> None:
    model = CompromisedModel("server_status", rounds=3)
    result = await _agent(model, [server_status]).arun("go")
    ids = [e.tool_call_id for e in result.tool_executions]
    assert len(ids) == 3
    assert len(set(ids)) == 3


async def test_rounds_count_per_run_not_per_instance() -> None:
    """One model, two runs: each gets its full rounds."""
    model = CompromisedModel("server_status", rounds=1)
    agent = _agent(model, [server_status])
    await agent.arun("first")
    await agent.arun("second")
    assert len(model.attempts) == 2


async def test_concurrent_runs_share_one_model() -> None:
    model = CompromisedModel("server_status", rounds=1)
    agent = _agent(model, [server_status])
    results = await asyncio.gather(agent.arun("a"), agent.arun("b"))
    assert len(model.attempts) == 2
    assert all(len(r.tool_executions) == 1 for r in results)


async def test_unoffered_tool_is_attempted_and_never_runs() -> None:
    model = CompromisedModel("wipe_database", {"reason": "x"}, rounds=1)
    result = await _agent(model, [server_status]).arun("hi")
    assert model.attempts == [("wipe_database", {"reason": "x"})]
    assert EFFECTS == []
    (execution,) = result.tool_executions
    assert execution.tool_name == "wipe_database"
    assert execution.error  # the agent refused an unknown tool


async def test_offered_only_skips_tools_it_was_not_given() -> None:
    model = CompromisedModel("wipe_database", offered_only=True, after="Nothing to do.")
    result = await _agent(model, [server_status]).arun("hi")
    assert model.attempts == []
    assert result.text == "Nothing to do."


async def test_no_tools_offered_gloats() -> None:
    model = CompromisedModel("wipe_database")
    result = await _agent(model, []).arun("hi")
    assert model.attempts == []
    assert result.text == "Done, it's all yours."


async def test_content_names_the_tool() -> None:
    model = CompromisedModel("server_status", rounds=1)
    await _agent(model, [server_status]).arun("go")
    second_call = model.received_messages[1]
    said = [m.content for m in second_call if getattr(m.role, "value", m.role) == "assistant"]
    assert "Running server_status now." in said


async def test_callable_attack_chooses_per_turn() -> None:
    def pick(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        prompt = next(m.content or "" for m in messages if m.role == "user")
        if any(m.role == "tool" for m in messages):
            return None  # one try, then stop
        return ("wipe_database", {"reason": prompt})

    trail = AuditTrail()
    model = CompromisedModel(pick, after="Stopped.")
    result = await _agent(model, [_gated(trail)]).arun("drop it all")
    assert model.attempts == [("wipe_database", {"reason": "drop it all"})]
    assert result.text == "Stopped."
    assert EFFECTS == []


async def test_callable_attack_may_return_a_response() -> None:
    def pick(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if any(m.role == "tool" for m in messages):
            return text("done")
        return tool_call("server_status", call_id="c1")

    model = CompromisedModel(pick)
    result = await _agent(model, [server_status]).arun("go")
    assert model.attempts == [("server_status", {})]
    assert result.text == "done"


async def test_refusal_reaches_the_model() -> None:
    trail = AuditTrail()
    model = CompromisedModel("wipe_database", {"reason": "x"}, rounds=1)
    await _agent(model, [_gated(trail)]).arun("hi")
    tool_results = [
        m.content for m in model.received_messages[-1] if getattr(m.role, "value", m.role) == "tool"
    ]
    assert tool_results
    assert json.loads(tool_results[0])["status"] == "denied"


async def test_stream_carries_the_attack() -> None:
    """A streaming agent rebuilds the turn from chunks: the call must be in one."""
    model = CompromisedModel("wipe_database", {"reason": "x"})
    chunks = [c async for c in model.stream([Message.user("hi")], [{"name": "wipe_database"}])]
    streamed = [tc for c in chunks for tc in (getattr(c, "tool_calls", None) or [])]
    assert [(tc.name, tc.arguments) for tc in streamed] == [("wipe_database", {"reason": "x"})]
    assert model.attempts == [("wipe_database", {"reason": "x"})]


def test_arguments_with_a_callable_is_an_error() -> None:
    with pytest.raises(TypeError):
        CompromisedModel(lambda m, t: None, {"a": 1})


def test_negative_rounds_is_an_error() -> None:
    with pytest.raises(ValueError, match="rounds"):
        CompromisedModel("x", rounds=-1)
