# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Plumbing tests for server-stateful model transports.

Server-stateful transports (e.g. :class:`OCIResponsesModel`) own the
conversation thread server-side. Locus sends only the input added since
the last call and threads a continuation token via
:attr:`ModelResponse.provider_state` /
:attr:`AgentState.provider_state`.

These tests lock in the plumbing without requiring a real OCI endpoint:

- ``AgentState.with_provider_state`` round-trips.
- ``ModelResponse.provider_state`` defaults to ``None`` for stateless
  transports (no behavior change for the default path).
- The runtime loop sends only the slice since the last assistant
  message and forwards ``provider_state`` when ``model.server_stateful``
  is truthy.
"""

from __future__ import annotations

from typing import Any, ClassVar

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.core.state import AgentState
from locus.models.base import ModelResponse


class _StatefulFakeModel:
    """Scripted server-stateful model — records each call for assertions."""

    server_stateful: ClassVar[bool] = True

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "provider_state": kwargs.get("provider_state"),
            }
        )
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _assistant(
    content: str | None,
    tool_calls: list[ToolCall] | None = None,
    provider_state: dict | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        provider_state=provider_state,
    )


def test_agent_state_provider_state_round_trips() -> None:
    """AgentState.with_provider_state replaces the field functionally."""
    s = AgentState(agent_id="t")
    assert s.provider_state is None

    s2 = s.with_provider_state({"previous_response_id": "resp_abc"})
    assert s2.provider_state == {"previous_response_id": "resp_abc"}
    # Original is untouched (frozen, functional update).
    assert s.provider_state is None

    s3 = s2.with_provider_state(None)
    assert s3.provider_state is None


def test_model_response_provider_state_defaults_none() -> None:
    """Stateless transports leave provider_state at None — no behavior change."""
    r = ModelResponse(message=Message.assistant("hi"))
    assert r.provider_state is None


def test_server_stateful_first_turn_sends_full_message_list() -> None:
    """First turn (no prior assistant message): send everything."""
    model = _StatefulFakeModel(
        [_assistant("Done.", provider_state={"previous_response_id": "resp_1"})]
    )
    agent = Agent(model=model, max_iterations=2)
    result = agent.run_sync("Hi.")

    assert result.success
    assert len(model.calls) == 1
    # Initial state has [system, user] (system from Agent default + user prompt).
    first_call = model.calls[0]
    roles = [m.role for m in first_call["messages"]]
    assert "user" in roles
    # No prior provider_state on the first call.
    assert first_call["provider_state"] is None


def test_server_stateful_threads_provider_state_across_turns() -> None:
    """Subsequent turns reference the server thread via provider_state."""
    tool_call = ToolCall(id="call_x", name="noop", arguments={})

    from locus.tools.decorator import tool

    @tool
    def noop() -> str:
        """No-op tool."""
        return "ok"

    model = _StatefulFakeModel(
        [
            # Turn 1: model emits a tool call, returns continuation id resp_1.
            _assistant(
                None,
                tool_calls=[tool_call],
                provider_state={"previous_response_id": "resp_1"},
            ),
            # Turn 2: model terminates with continuation id resp_2.
            _assistant("Done.", provider_state={"previous_response_id": "resp_2"}),
        ]
    )
    agent = Agent(model=model, tools=[noop], max_iterations=4)
    result = agent.run_sync("Run the tool.")

    assert result.success
    assert len(model.calls) == 2

    # Turn 1: no provider_state (server hasn't returned a continuation yet).
    assert model.calls[0]["provider_state"] is None

    # Turn 2: should reference the continuation id from turn 1.
    assert model.calls[1]["provider_state"] == {"previous_response_id": "resp_1"}

    # Turn 2 input: only messages added after the turn-1 assistant message
    # (i.e. the tool result), not the entire history.
    turn2_roles = [m.role for m in model.calls[1]["messages"]]
    assert "user" not in turn2_roles  # the original user prompt isn't re-sent
    assert "tool" in turn2_roles  # the tool result IS sent
