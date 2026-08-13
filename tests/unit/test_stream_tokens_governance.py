"""Streaming must not change what the gate does.

`stream_tokens=True` asks for one thing: that the answer surface while it is
being written. It is a presentation choice. It must have no bearing whatsoever
on whether a tool call is admitted, cancelled, or held for a person — and that
is not a style preference, it is the property every other guarantee in this
library rests on.

The tests below drive the SAME agent, with the SAME hook, twice: once
streaming, once not. Anything that differs between the two columns is a defect,
because the caller only asked about tokens.

Found the hard way: the gateway's own suite began failing `assert run.paused is
True` the moment streaming was switched on, with the flag as the only variable
(see FINDING-stream-tokens-loses-the-hold.md in the workspace).
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent, AgentConfig
from tulip.core.messages import Message, ToolCall
from tulip.hooks.provider import HookProvider
from tulip.models.base import ModelResponse
from tulip.tools import tool


@tool
def wire_money(amount: int) -> str:
    """Move money. The kind of action a person is supposed to approve."""
    return f"sent {amount}"


class _CallsTheTool:
    """A model that asks for the tool once, then answers.

    Implements BOTH surfaces: `complete` and `stream`. A stub that offers only
    one silently changes which code path the loop takes, which is exactly the
    confound these tests exist to remove.
    """

    def __init__(self) -> None:
        self.turns = 0

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.turns += 1
        if self.turns == 1:
            return ModelResponse(
                message=Message.assistant(
                    content="",
                    tool_calls=[ToolCall(id="1", name="wire_money", arguments={"amount": 9000})],
                ),
                stop_reason="tool_use",
            )
        return ModelResponse(message=Message.assistant(content="done"), stop_reason="end_turn")

    async def stream(self, **kwargs: Any) -> Any:
        # A streaming model yields CHUNKS ONLY. The loop assembles the
        # ModelResponse from them (_complete_streaming), which means the tool
        # calls have to arrive in a chunk — a stream that carries text but
        # forgets tool_calls produces a response with none, and the agent
        # simply never calls the tool. No error, no gate, nothing to hold.
        from tulip.core.events import ModelChunkEvent

        self.turns += 1
        if self.turns == 1:
            yield ModelChunkEvent(content="I will ")
            yield ModelChunkEvent(content="wire it.")
            yield ModelChunkEvent(
                content="",
                tool_calls=[ToolCall(id="1", name="wire_money", arguments={"amount": 9000})],
                stop_reason="tool_use",
                done=True,
            )
            return
        yield ModelChunkEvent(content="done")
        yield ModelChunkEvent(content="", stop_reason="end_turn", done=True)


class _Gate(HookProvider):
    """A hook that refuses the call, the way a hold does."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    @property
    def priority(self) -> int:
        return 50

    async def on_before_tool_call(self, event: Any) -> None:
        if event.tool_name == "wire_money":
            self.seen.append(event.tool_name)
            event.cancel = "HELD: wire_money needs a person"


def _agent(gate: _Gate) -> Agent:
    return Agent(
        config=AgentConfig(
            system_prompt="Move the money when asked.",
            max_iterations=4,
            model=_CallsTheTool(),
            tools=[wire_money],
            hooks=[gate],
        )
    )


async def _drive(*, stream_tokens: bool) -> tuple[_Gate, list[str]]:
    gate = _Gate()
    kinds: list[str] = []
    async for event in _agent(gate).run("wire 9000", stream_tokens=stream_tokens):
        kinds.append(getattr(event, "event_type", "?"))
    return gate, kinds


@pytest.mark.asyncio
async def test_the_gate_fires_whether_or_not_tokens_were_asked_for() -> None:
    """The hook runs in both modes. If it does not, the flag disabled the gate."""
    plain, _ = await _drive(stream_tokens=False)
    streamed, _ = await _drive(stream_tokens=True)

    assert plain.seen == ["wire_money"]
    assert streamed.seen == ["wire_money"], (
        "the before-tool-call hook never fired under stream_tokens — a caller "
        "who asked for a nicer chat turned off the gate"
    )


@pytest.mark.asyncio
async def test_streaming_only_adds_chunks_and_changes_nothing_else() -> None:
    """Same run, same events — plus the chunks that were the entire request."""
    _, plain = await _drive(stream_tokens=False)
    _, streamed = await _drive(stream_tokens=True)

    assert "model_chunk" in streamed, "stream_tokens produced no chunks at all"
    assert [k for k in streamed if k != "model_chunk"] == plain, (
        f"streaming altered the event sequence beyond adding chunks: {streamed} vs {plain}"
    )
