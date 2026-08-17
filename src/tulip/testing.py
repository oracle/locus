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
    "AgentTestClient",
    "AgentTrace",
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


# ---------------------------------------------------------------------------
# Trace assertions
#
# The doubles above answer "what did the model see?". These answer the other
# half — "what did the agent do?" — because that is the half a regression
# usually breaks, and asserting it by hand means reaching through
# ``result.tool_executions`` and rebuilding the same three comprehensions in
# every test file.
# ---------------------------------------------------------------------------


class AgentTrace:
    """One run, with the questions a test actually asks made cheap.

    Every assertion here reports what happened rather than only that
    something did not, because ``assert False`` tells you nothing at 3am::

        AssertionError: expected tool 'refund' to be called, but the agent
        called: ['lookup_order', 'check_balance']

    Attributes are plain data — assert on them directly when no helper fits.
    """

    def __init__(self, result: Any, model: Any) -> None:
        #: The :class:`~tulip.core.results.AgentResult` the run returned.
        self.result = result
        #: The double the agent ran against, for input-side assertions.
        self.model = model

    # -- data ------------------------------------------------------------

    @property
    def message(self) -> str:
        """The agent's final answer."""
        return str(getattr(self.result, "message", "") or "")

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """``(name, arguments)`` for each tool the agent actually ran, in order."""
        return [(e.tool_name, dict(e.arguments or {})) for e in self._executions]

    @property
    def tool_names(self) -> list[str]:
        """Just the names, in call order — the common case."""
        return [name for name, _ in self.tool_calls]

    @property
    def model_calls(self) -> int:
        """How many times the agent went back to the model."""
        return int(getattr(self.model, "call_count", 0))

    @property
    def failed_tools(self) -> list[tuple[str, str]]:
        """``(name, error)`` for tools that raised."""
        return [(e.tool_name, str(e.error)) for e in self._executions if e.error]

    @property
    def _executions(self) -> list[Any]:
        return list(getattr(self.result, "tool_executions", ()) or ())

    # -- assertions ------------------------------------------------------

    def assert_tool_called(self, name: str, **expected: Any) -> AgentTrace:
        """The agent ran ``name``, optionally with these argument values.

        Only the arguments you name are compared, so a test can pin the one
        that matters without restating the whole call.
        """
        matches = [args for called, args in self.tool_calls if called == name]
        if not matches:
            raise AssertionError(
                f"expected tool {name!r} to be called, but the agent called: "
                f"{self.tool_names or 'no tools at all'}"
            )
        if not expected:
            return self
        for args in matches:
            if all(args.get(k) == v for k, v in expected.items()):
                return self
        raise AssertionError(
            f"tool {name!r} was called, but never with {expected!r}.\n  actual call(s): {matches!r}"
        )

    def assert_tool_not_called(self, name: str) -> AgentTrace:
        """The agent never ran ``name`` — the assertion a gate test needs."""
        if name in self.tool_names:
            raise AssertionError(
                f"expected tool {name!r} NOT to be called, but it ran. "
                f"Full call order: {self.tool_names}"
            )
        return self

    def assert_tools_called(self, *names: str) -> AgentTrace:
        """Exactly these tools ran, in exactly this order."""
        if self.tool_names != list(names):
            raise AssertionError(
                f"tool call order mismatch.\n"
                f"  expected: {list(names)}\n"
                f"  actual  : {self.tool_names}"
            )
        return self

    def assert_model_calls(self, count: int) -> AgentTrace:
        """The agent took exactly ``count`` turns with the model.

        Guards the loop against silently growing an extra round trip, which
        costs money on every run and shows up nowhere else.
        """
        if self.model_calls != count:
            raise AssertionError(
                f"expected {count} model call(s), got {self.model_calls}. "
                f"Tools run: {self.tool_names}"
            )
        return self

    def assert_tool_offered(self, name: str) -> AgentTrace:
        """``name`` was advertised to the model on the first turn.

        A tool the model was never shown cannot be called, and that failure
        otherwise looks identical to a model that chose not to call it.
        """
        offered = list(getattr(self.model, "offered_tools", []) or [])
        first = offered[0] if offered else []
        if name not in first:
            raise AssertionError(
                f"tool {name!r} was never offered to the model. Offered on the "
                f"first turn: {first or 'none'}"
            )
        return self

    def assert_succeeded(self) -> AgentTrace:
        """The run finished without an error and without a failing tool."""
        error = getattr(self.result, "error", None)
        if error:
            raise AssertionError(f"agent run failed: {error}")
        if self.failed_tools:
            raise AssertionError(f"tools raised during the run: {self.failed_tools}")
        return self


class AgentTestClient:
    """Run an agent and get a :class:`AgentTrace` back.

    A thin wrapper, deliberately: it owns no configuration and changes no
    behaviour, so what a test exercises is the same object production runs::

        from tulip.testing import AgentTestClient, ScriptedModel, text, tool_call

        model = ScriptedModel([tool_call("add", a=2, b=2), text("4")])
        client = AgentTestClient(Agent(model=model, tools=[add]))

        trace = client.run("what is 2 plus 2?")
        trace.assert_tool_called("add", a=2, b=2).assert_model_calls(2)
        assert trace.message == "4"

    Assertions chain, so a single expression can state the whole expectation.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    @property
    def model(self) -> Any:
        """The double the agent was built with."""
        return getattr(self.agent, "_model", None) or getattr(self.agent, "model", None)

    def run(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Run to completion and return the trace. Blocking."""
        return AgentTrace(self.agent.run_sync(prompt, **kwargs), self.model)

    async def arun(self, prompt: str, **kwargs: Any) -> AgentTrace:
        """Async counterpart of :meth:`run`."""
        return AgentTrace(await self.agent.arun(prompt, **kwargs), self.model)
