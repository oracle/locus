# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: ``on_after_tool_call`` receives ``tool_call_id`` + ``arguments``.

Drives the real ``Agent`` runtime against a scripted fake model so the
hook orchestrator runs over the actual tool-execution path. Asserts the
post-tool hook sees ``event.tool_call_id`` (matching the model's tool call
id) and ``event.arguments`` (matching the dict the tool was invoked with).

This is the wire-through test for the almariel-shaped use case in #168:
mirror each tool call into a host-side action queue keyed by id.
"""

from __future__ import annotations

from typing import Any

from tulip.agent import Agent
from tulip.core.messages import Message, ToolCall
from tulip.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


class _ScriptedModel:
    """Returns a scripted sequence of model responses, looping the last one."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


@tool
def set_block(x: int, y: int, z: int, block_type: str) -> str:
    """Place a block at the given coordinates."""
    return f"placed {block_type} at ({x},{y},{z})"


def _assistant(content: str | None, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 5, "completion_tokens": 5},
    )


class _ActionQueueHook(HookProvider):
    """Mirror every tool call into an in-memory action queue (almariel-shape)."""

    def __init__(self) -> None:
        self.queue: list[dict[str, Any]] = []
        self.before_seen: list[tuple[str, str]] = []

    @property
    def priority(self) -> int:
        return HookPriority.BUSINESS_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        self.before_seen.append((event.tool_name, event.tool_call_id))

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self.queue.append(
            {
                "id": event.tool_call_id,
                "tool": event.tool_name,
                "args": dict(event.arguments),
                "result": event.result,
                "error": event.error,
            }
        )


def test_after_tool_hook_sees_call_id_and_arguments() -> None:
    """The hook receives the model's tool_call_id and the actual arguments."""
    tool_call = ToolCall(
        id="call_abc123",
        name="set_block",
        arguments={"x": 10, "y": 64, "z": -3, "block_type": "minecraft:cobblestone"},
    )
    model = _ScriptedModel(
        [
            _assistant(None, tool_calls=[tool_call]),
            _assistant("Placed the block. Done."),
        ]
    )
    hook = _ActionQueueHook()
    agent = Agent(
        model=model,
        tools=[set_block],
        hooks=[hook],
        max_iterations=4,
    )

    result = agent.run_sync("Place a cobblestone block.")

    assert result.success
    assert len(hook.queue) == 1
    entry = hook.queue[0]
    assert entry["id"] == "call_abc123"
    assert entry["tool"] == "set_block"
    assert entry["args"] == {
        "x": 10,
        "y": 64,
        "z": -3,
        "block_type": "minecraft:cobblestone",
    }
    assert entry["error"] is None

    # The before-hook saw the same correlation id, so id-keyed lookups
    # between before/after events are reliable.
    assert hook.before_seen == [("set_block", "call_abc123")]


def test_after_tool_hook_sees_modified_arguments_from_before_hook() -> None:
    """When a before-hook mutates event.arguments, the after-event reflects it."""

    class _MutatingHook(HookProvider):
        @property
        def priority(self) -> int:
            return HookPriority.SECURITY_DEFAULT

        async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
            mutated = dict(event.arguments)
            mutated["block_type"] = "minecraft:bedrock"
            event.arguments = mutated

    capture = _ActionQueueHook()
    tool_call = ToolCall(
        id="call_xyz",
        name="set_block",
        arguments={"x": 0, "y": 0, "z": 0, "block_type": "minecraft:dirt"},
    )
    model = _ScriptedModel(
        [
            _assistant(None, tool_calls=[tool_call]),
            _assistant("Done."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[set_block],
        hooks=[_MutatingHook(), capture],
        max_iterations=4,
    )

    agent.run_sync("Place dirt.")

    assert capture.queue[0]["args"]["block_type"] == "minecraft:bedrock"


def test_existing_event_construction_still_works() -> None:
    """AfterToolCallEvent built without new kwargs has sane defaults."""
    event = AfterToolCallEvent(tool_name="t", result="r", error=None)
    assert event.tool_call_id == ""
    assert event.arguments == {}


class _ResultRewriteHook(HookProvider):
    """Replace the tool result with a slimmed form (reel-shaped use case)."""

    def __init__(self, replacement: Any) -> None:
        self.replacement = replacement
        self.saw: list[Any] = []

    @property
    def priority(self) -> int:
        return HookPriority.BUSINESS_DEFAULT

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self.saw.append(event.result)
        event.result = self.replacement


def _tool_call() -> ToolCall:
    return ToolCall(
        id="call_rw1",
        name="set_block",
        arguments={"x": 1, "y": 2, "z": 3, "block_type": "minecraft:stone"},
    )


def test_after_tool_hook_result_replacement_reaches_the_model() -> None:
    """``event.result`` replacement lands in the tool message the model
    reads, in ``state.tool_executions``, and in the run's final state — as
    the AfterToolCallEvent docs promise. It used to be silently discarded
    on the run() path: the hook ran after the fold, so only ``retry`` had
    any effect."""
    model = _ScriptedModel(
        [
            _assistant(None, tool_calls=[_tool_call()]),
            _assistant("Done."),
        ]
    )
    hook = _ResultRewriteHook("SLIMMED")
    agent = Agent(model=model, tools=[set_block], hooks=[hook], max_iterations=4)

    result = agent.run_sync("Place a stone block.")

    assert result.success
    # The hook saw the raw result, and its replacement is what persisted.
    assert hook.saw == ["placed minecraft:stone at (1,2,3)"]
    executions = [e for e in result.state.tool_executions if e.tool_name == "set_block"]
    assert executions
    assert executions[0].result == "SLIMMED"
    tool_msgs = [m for m in result.state.messages if getattr(m.role, "value", m.role) == "tool"]
    assert tool_msgs
    assert tool_msgs[0].content == "SLIMMED"


def test_after_tool_hook_result_replacement_serializes_a_dict() -> None:
    """A hook may hand back a dict; it is serialized like a tool return."""
    model = _ScriptedModel(
        [
            _assistant(None, tool_calls=[_tool_call()]),
            _assistant("Done."),
        ]
    )
    hook = _ResultRewriteHook({"found": True, "slim": True})
    agent = Agent(model=model, tools=[set_block], hooks=[hook], max_iterations=4)

    result = agent.run_sync("Place a stone block.")

    assert result.success
    tool_msgs = [m for m in result.state.messages if getattr(m.role, "value", m.role) == "tool"]
    assert tool_msgs
    assert tool_msgs[0].content == '{"found": true, "slim": true}'


def test_tool_complete_event_carries_the_replaced_result() -> None:
    """In the default (sequential) event order the after-hook now runs
    BEFORE the ToolCompleteEvent, so consumers see the hook's version."""
    import asyncio

    model = _ScriptedModel(
        [
            _assistant(None, tool_calls=[_tool_call()]),
            _assistant("Done."),
        ]
    )
    hook = _ResultRewriteHook("SLIMMED")
    agent = Agent(model=model, tools=[set_block], hooks=[hook], max_iterations=4)

    async def _events() -> list[Any]:
        return [ev async for ev in agent.run("Place a stone block.")]

    events = asyncio.run(_events())
    completes = [ev for ev in events if getattr(ev, "event_type", "") == "tool_complete"]
    assert completes
    assert completes[0].result == "SLIMMED"
