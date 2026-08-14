# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Test doubles for agents — script a model, then assert on what it saw.

Testing an agent means controlling the one part you do not own: the model.
Without a supported double, every project writes its own, and this one is no
exception — the SDK's own suite carried thirteen private ``_ScriptedModel``
classes before this module existed.

Two doubles, covering the two shapes a test needs:

:class:`ScriptedModel`
    A fixed sequence of turns. Deterministic, no network, no keys.

:class:`FunctionModel`
    A callable that decides each turn from the conversation so far, for
    behaviour that depends on what came back from a tool.

Both record what they were asked, so a test can assert on the *inputs* the
agent produced — which prompt, which tools were offered — and not only on the
final string.

    from tulip.testing import ScriptedModel, text, tool_call

    model = ScriptedModel([
        tool_call("get_weather", city="Lisbon"),
        text("It is 18C and raining."),
    ])
    agent = Agent(model=model, tools=[get_weather])
    result = await agent.arun("What is the weather in Lisbon?")

    assert result.text == "It is 18C and raining."
    assert [t.tool_name for t in result.tool_executions] == ["get_weather"]
    assert model.call_count == 2
    assert "get_weather" in model.offered_tools[0]

Neither double parses or validates arguments the way a provider would: they
return exactly what you scripted. That is the point — a test that fails should
be telling you about your agent, not about a mock's opinion of your JSON.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelResponse


__all__ = [
    "FunctionModel",
    "ScriptedModel",
    "text",
    "tool_call",
]

#: Token usage reported by both doubles. Fixed and obviously synthetic, so a
#: test asserting on cost is clearly asserting on a fixture.
_USAGE = {"prompt_tokens": 10, "completion_tokens": 20}


def text(content: str, *, stop_reason: str = "end_turn") -> ModelResponse:
    """A plain assistant turn.

    Args:
        content: What the model says.
        stop_reason: Reported on the response; the default ends the loop.
    """
    return ModelResponse(
        message=Message.assistant(content=content),
        usage=dict(_USAGE),
        stop_reason=stop_reason,
    )


def tool_call(
    name: str,
    *,
    content: str | None = None,
    call_id: str | None = None,
    **arguments: Any,
) -> ModelResponse:
    """A turn that calls one tool.

    Arguments are passed as keywords, so the call reads like the tool::

        tool_call("issue_refund", order_id="ord-4821", amount=49.0)

    Args:
        name: Tool to call.
        content: Optional assistant text alongside the call.
        call_id: Tool-call id. Defaults to a stable id derived from the name,
            so assertions do not depend on a random value.
        **arguments: Arguments for the tool.
    """
    return ModelResponse(
        message=Message.assistant(
            content=content,
            tool_calls=[ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments)],
        ),
        usage=dict(_USAGE),
        stop_reason="tool_calls",
    )


class _RecordingModel:
    """Shared recording surface for the doubles."""

    def __init__(self) -> None:
        #: Every ``messages`` list this model was called with, in order.
        self.received_messages: list[list[Message]] = []
        #: Tool names offered on each call — ``[]`` when the agent bound none.
        self.offered_tools: list[list[str]] = []

    @property
    def call_count(self) -> int:
        """How many times the agent called the model."""
        return len(self.received_messages)

    @property
    def last_prompt(self) -> str:
        """Content of the most recent user turn, or ``""``."""
        for messages in reversed(self.received_messages):
            for message in reversed(messages):
                if getattr(message.role, "value", message.role) == "user":
                    return message.content or ""
        return ""

    def _record(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> None:
        self.received_messages.append(list(messages))
        self.offered_tools.append([_tool_name(t) for t in (tools or [])])

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream the same response ``complete`` would return.

        Implemented rather than raising, so a streaming agent can be tested
        with the same double as a non-streaming one.
        """
        response = await self.complete(messages, tools, **kwargs)  # type: ignore[attr-defined]
        content = response.content or ""
        for start in range(0, len(content), 12):
            yield ModelChunkEvent(content=content[start : start + 12])
        yield ModelChunkEvent(done=True, usage=response.usage)


def _tool_name(schema: dict[str, Any]) -> str:
    """Tool name from either payload shape (OpenAI nests under ``function``)."""
    inner = schema.get("function")
    source = inner if isinstance(inner, dict) else schema
    return str(source.get("name", ""))


class ScriptedModel(_RecordingModel):
    """Return a fixed sequence of turns, one per call.

    Args:
        turns: Responses to return in order. Build them with :func:`text` and
            :func:`tool_call`, or pass ``ModelResponse`` objects directly.
        repeat_last: When the script runs out, keep returning the final turn
            instead of raising. Useful for a loop whose length you do not want
            the test to depend on.

    Raises:
        AssertionError: If the agent asks for more turns than were scripted
            and ``repeat_last`` is False. That is nearly always a real finding
            — the loop did not stop when the test author expected — so it
            fails loudly rather than improvising a reply.
    """

    def __init__(
        self,
        turns: Sequence[ModelResponse | str],
        *,
        repeat_last: bool = False,
    ) -> None:
        super().__init__()
        self._turns: list[ModelResponse] = [text(t) if isinstance(t, str) else t for t in turns]
        self._repeat_last = repeat_last

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self._record(messages, tools)
        index = self.call_count - 1
        if index < len(self._turns):
            return self._turns[index]
        if self._repeat_last and self._turns:
            return self._turns[-1]
        raise AssertionError(
            f"ScriptedModel ran out of turns: the agent made {self.call_count} "
            f"model call(s) but only {len(self._turns)} were scripted. Add "
            f"another turn, or pass repeat_last=True if the count is not the "
            f"point of the test."
        )


class FunctionModel(_RecordingModel):
    """Decide each turn from the conversation so far.

    For behaviour a fixed script cannot express — most often "call the tool,
    then answer using its result"::

        def handler(messages, tools):
            if any(m.role == "tool" for m in messages):
                return text("The order was refunded.")
            return tool_call("issue_refund", order_id="ord-4821")


        model = FunctionModel(handler)

    Args:
        handler: Called with ``(messages, tools)`` and returning a
            ``ModelResponse`` — or a plain string, wrapped with :func:`text`.
    """

    def __init__(
        self,
        handler: Callable[[list[Message], list[dict[str, Any]]], ModelResponse | str],
    ) -> None:
        super().__init__()
        self._handler = handler

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self._record(messages, tools)
        result = self._handler(list(messages), list(tools or []))
        return text(result) if isinstance(result, str) else result
