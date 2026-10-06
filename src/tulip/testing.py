# Copyright 2026 The Tulip Authors
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
    behaviour that depends on what came back from a tool. Also exported as
    :data:`MockModel`, the name people coming from other SDKs look for: a
    ``MockModel`` is a ``FunctionModel``, and a fixed list of turns is a
    :class:`ScriptedModel`.

They record what they were asked, so a test can assert on the *inputs* the
agent produced — which prompt, which tools were offered — and not only on the
final string.

A third double plays the adversary. :class:`CompromisedModel` is a model an
attacker has already won: it reaches for the harmful call on every turn, so a
test asserts on what stops it — the gate and the audit trail — rather than on
the model's judgement. It is the offline mode of ``python -m tulip.rogue``,
packaged for your own rogue suite.

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

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelResponse


__all__ = [
    "CONFORMANCE_TOOL",
    "AgentTestClient",
    "AgentTrace",
    "CompromisedModel",
    "ConformanceReport",
    "FunctionModel",
    "MockModel",
    "ScriptedModel",
    "check_model_conformance",
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

        **Tool calls ride in a chunk of their own.** The agent loop rebuilds the
        turn from these events alone, so a stream carrying only text yields a
        turn with no tool calls — and this class is most often used to test
        exactly the behaviour that needs them. It failed silently in the worst
        possible way: the double claimed to return what ``complete`` returns,
        the agent made no tool call, no error was raised, and the test passed
        having exercised nothing. ``stop_reason`` is carried for the same
        reason — the loop reads it to decide whether the turn ended.
        """
        response = await self.complete(messages, tools, **kwargs)  # type: ignore[attr-defined]
        content = response.content or ""
        for start in range(0, len(content), 12):
            yield ModelChunkEvent(content=content[start : start + 12])
        tool_calls = getattr(response.message, "tool_calls", None)
        if tool_calls:
            yield ModelChunkEvent(tool_calls=list(tool_calls))
        yield ModelChunkEvent(
            done=True, usage=response.usage, stop_reason=getattr(response, "stop_reason", None)
        )


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


#: :class:`FunctionModel` under the name other SDKs use for it. One class, two
#: names: ``MockModel(handler)`` decides each turn from ``(messages, tools)``.
#: For a fixed list of turns, use :class:`ScriptedModel`.
MockModel = FunctionModel


#: What an attack callable may return for one turn: a ``(tool, arguments)``
#: pair to call, a ready :class:`ModelResponse`, or ``None`` to stop attacking.
AttackChoice = tuple[str, Mapping[str, Any]] | ModelResponse | None


class CompromisedModel(_RecordingModel):
    """A model an attacker has already won, for your own rogue suite.

    It never refuses and never hedges: on every turn that offers tools it
    calls the attack. Nothing below the model is mocked, so what a test then
    checks is the part that has to hold when the model does not — the gate in
    front of each tool (:func:`~tulip.control.gate_tool`), the tools the agent
    offers, the audit trail. This is the offline mode of
    ``python -m tulip.rogue`` (whose OpsBot model picks its target from the
    operator's words), made reusable against your own agent::

        from tulip.testing import CompromisedModel

        ATTACKS = [
            (
                "refund to an outside account",
                "issue_refund",
                {"order_id": "ord-1", "usd": 9_999, "to": "attacker"},
            ),
            ("a tool it was never given", "drop_table", {"table": "orders"}),
        ]


        @pytest.mark.parametrize(("why", "name", "args"), ATTACKS)
        async def test_the_gate_holds(why, name, args):
            model = CompromisedModel(name, args)
            agent = Agent(model=model, tools=[gated_refund, lookup_order])
            await agent.arun(why)
            assert model.attempts  # the model really tried
            assert payments.refunds == []  # and nothing happened
            assert trail.verify()  # every attempt is on the record

    Args:
        attack: The tool to call, by name, with ``arguments``; or a callable
            ``(messages, tools) -> (name, arguments) | ModelResponse | None``
            choosing each turn's call from the conversation so far (``None``
            stops attacking and answers with ``after``).
        arguments: Arguments for a named ``attack``.
        rounds: Attack calls per run before it gives up and answers with
            ``after``. ``None`` (the default) attacks on every model call that
            offers tools: an owned model never stops trying, and the run ends
            only when the agent's own limits end it.
        content: Assistant text sent alongside each call. ``{name}`` is
            replaced with the tool's name.
        after: What it says once it stops: when ``rounds`` are used up, when
            ``attack`` returns ``None``, or when the agent offers no tools.
        offered_only: Call the attack only when the agent offers that tool.
            Off by default, because an owned model will also try a tool it was
            never given, and a suite should prove the agent refuses it.

    Rounds are counted from the conversation (tool calls since the last user
    message), not on the instance, so one model serves concurrent runs. Each
    call the model makes is appended to :attr:`attempts`.
    """

    def __init__(
        self,
        attack: str | Callable[[list[Message], list[dict[str, Any]]], AttackChoice],
        arguments: Mapping[str, Any] | None = None,
        *,
        rounds: int | None = None,
        content: str | None = "Running {name} now.",
        after: str = "Done, it's all yours.",
        offered_only: bool = False,
    ) -> None:
        super().__init__()
        if rounds is not None and rounds < 0:
            raise ValueError(f"rounds must be >= 0 or None, got {rounds}")
        if callable(attack) and arguments is not None:
            raise TypeError("arguments= goes with a tool name; a callable attack returns its own")
        self._attack = attack
        self._arguments = dict(arguments or {})
        self._rounds = rounds
        self._content = content
        self._after = after
        self._offered_only = offered_only
        #: Every call this model made, as ``(tool name, arguments)``, in order.
        self.attempts: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _calls_this_run(messages: list[Message]) -> int:
        """Tool calls the assistant made since the last user message."""
        count = 0
        for message in reversed(messages):
            role = getattr(message.role, "value", message.role)
            if role == "user":
                break
            if role == "assistant":
                count += len(message.tool_calls or [])
        return count

    def _choose(self, messages: list[Message], tools: list[dict[str, Any]]) -> AttackChoice:
        if callable(self._attack):
            return self._attack(list(messages), list(tools))
        return self._attack, self._arguments

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self._record(messages, tools)
        offered = list(tools or [])
        made = self._calls_this_run(messages)
        if not offered or (self._rounds is not None and made >= self._rounds):
            return text(self._after)
        choice = self._choose(messages, offered)
        if choice is None:
            return text(self._after)
        if isinstance(choice, ModelResponse):
            for call in choice.message.tool_calls or []:
                self.attempts.append((call.name, dict(call.arguments)))
            return choice
        name, arguments = choice
        if self._offered_only and name not in {_tool_name(t) for t in offered}:
            return text(self._after)
        self.attempts.append((name, dict(arguments)))
        return tool_call(
            name,
            content=self._content.replace("{name}", name) if self._content else None,
            # Unique per call: a run that hammers one tool must not reuse an id.
            call_id=f"call_{name}_{made + 1}",
            **dict(arguments),
        )


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


# ---------------------------------------------------------------------------
# Model conformance
#
# A fallback chain is only as good as its worst tier: a backup provider that
# streams text but drops tool calls, or returns arguments that ignore the
# schema, fails over into a broken agent. ``check_model_conformance`` runs the
# handful of behaviours the agent loop depends on against any ModelProtocol,
# so each tier can be verified before it is trusted with traffic.
# ---------------------------------------------------------------------------

#: The tool the conformance check offers. Small and unambiguous on purpose.
CONFORMANCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Lisbon"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _schema_errors(arguments: Any, schema: dict[str, Any]) -> list[str]:
    """Shallow JSON-Schema check: object, required keys, property types, enums."""
    if not isinstance(arguments, dict):
        return [f"arguments are {type(arguments).__name__}, not an object"]
    errors = [
        f"missing required {key!r}" for key in schema.get("required", []) if key not in arguments
    ]
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            errors.append(f"unexpected argument {key!r}")
            continue
        expected = _JSON_TYPES.get(str(spec.get("type")))
        if expected is not None and not isinstance(value, expected):
            errors.append(f"{key!r} should be {spec.get('type')}, got {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key!r}={value!r} not in {spec['enum']}")
    return errors


class ConformanceReport:
    """Outcome of :func:`check_model_conformance`: one ``(check, ok, detail)`` per check."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, *, ok: bool, detail: str = "") -> None:
        """Record one check's outcome."""
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return all(ok for _, ok, _ in self.checks)

    @property
    def failures(self) -> list[str]:
        """``"check: detail"`` for each failed check."""
        return [f"{name}: {detail}" for name, ok, detail in self.checks if not ok]

    def __repr__(self) -> str:
        passed = sum(1 for _, ok, _ in self.checks if ok)
        return f"ConformanceReport({passed}/{len(self.checks)} passed)"


async def check_model_conformance(
    model: Any,
    *,
    raise_on_failure: bool = True,
    **model_kwargs: Any,
) -> ConformanceReport:
    """Check that ``model`` does what Tulip's agent loop relies on.

    Runs five checks, each a real call to the model:

    ``complete_text``
        A plain turn returns non-empty text.
    ``complete_tool_call``
        Offered :data:`CONFORMANCE_TOOL` and asked for the weather, the model
        calls it, with an id and arguments that satisfy the tool's schema.
    ``tool_result_round_trip``
        Given the tool's result, the model answers in text.
    ``stream_text``
        ``stream()`` yields :class:`~tulip.core.events.ModelChunkEvent` s whose
        content assembles into non-empty text.
    ``stream_tool_call``
        A streamed tool-calling turn delivers the tool call in a chunk — the
        loop rebuilds the turn from chunks alone, so a stream that carries only
        text silently loses every tool call.

    Meant for each tier of a :class:`~tulip.models.fallback.FallbackChain`
    (with real credentials, in an integration test or a deploy smoke)::

        for tier in chain.tiers:
            await check_model_conformance(tier.model)

    Args:
        model: Any ``ModelProtocol`` implementation.
        raise_on_failure: Raise ``AssertionError`` listing the failures
            (default); False returns the report either way.
        **model_kwargs: Forwarded to every call (``max_tokens``, …).

    Returns:
        The :class:`ConformanceReport`.
    """
    report = ConformanceReport()
    schema = CONFORMANCE_TOOL["function"]["parameters"]
    system = Message.system(
        "You are a terse assistant. Use tools when they are offered and relevant."
    )
    ask_weather = Message.user("What is the weather in Lisbon right now? Use the get_weather tool.")

    async def _check(name: str, coro: Any) -> Any:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 — every failure becomes a report line
            report.add(name, ok=False, detail=f"raised {type(exc).__name__}: {exc}"[:300])
            return None

    # 1. Plain completion.
    response = await _check(
        "complete_text",
        model.complete([system, Message.user("Reply with exactly the word: pong")], **model_kwargs),
    )
    if response is not None:
        content = getattr(response, "content", None)
        report.add(
            "complete_text",
            ok=isinstance(content, str) and bool(content.strip()),
            detail=f"content={content!r}"[:200],
        )

    # 2. Tool call, schema-conformant.
    call: ToolCall | None = None
    response = await _check(
        "complete_tool_call",
        model.complete([system, ask_weather], [CONFORMANCE_TOOL], **model_kwargs),
    )
    if response is not None:
        calls = list(getattr(response.message, "tool_calls", None) or [])
        match = [c for c in calls if c.name == "get_weather"]
        if not match:
            report.add(
                "complete_tool_call",
                ok=False,
                detail=f"no get_weather call; got {[c.name for c in calls]}",
            )
        else:
            call = match[0]
            problems = _schema_errors(call.arguments, schema)
            if not call.id:
                problems.append("tool call has no id")
            report.add("complete_tool_call", ok=not problems, detail="; ".join(problems) or "ok")

    # 3. Tool result round trip.
    if call is not None:
        from tulip.core.messages import ToolResult  # noqa: PLC0415

        history = [
            system,
            ask_weather,
            Message.assistant(content=None, tool_calls=[call]),
            Message.tool(
                ToolResult(tool_call_id=call.id, name=call.name, content="18C, light rain")
            ),
        ]
        response = await _check(
            "tool_result_round_trip",
            model.complete(history, [CONFORMANCE_TOOL], **model_kwargs),
        )
        if response is not None:
            content = getattr(response, "content", None)
            report.add(
                "tool_result_round_trip",
                ok=isinstance(content, str) and bool(content.strip()),
                detail=f"content={content!r}"[:200],
            )
    else:
        report.add("tool_result_round_trip", ok=False, detail="skipped: no tool call to answer")

    # 4 + 5. Streaming.
    async def _collect(messages: list[Message], tools: list[dict[str, Any]] | None) -> list[Any]:
        return [chunk async for chunk in model.stream(messages, tools, **model_kwargs)]

    chunks = await _check(
        "stream_text", _collect([system, Message.user("Reply with exactly the word: pong")], None)
    )
    if chunks is not None:
        wrong = [type(c).__name__ for c in chunks if not isinstance(c, ModelChunkEvent)]
        text_out = "".join(c.content or "" for c in chunks if isinstance(c, ModelChunkEvent))
        report.add(
            "stream_text",
            ok=not wrong and bool(text_out.strip()),
            detail=f"non-chunk items {wrong}" if wrong else f"text={text_out!r}"[:200],
        )

    chunks = await _check("stream_tool_call", _collect([system, ask_weather], [CONFORMANCE_TOOL]))
    if chunks is not None:
        streamed = [
            tc for c in chunks if isinstance(c, ModelChunkEvent) for tc in (c.tool_calls or [])
        ]
        match = [tc for tc in streamed if tc.name == "get_weather"]
        problems = (
            _schema_errors(match[0].arguments, schema)
            if match
            else ["no get_weather call in any chunk"]
        )
        report.add("stream_tool_call", ok=not problems, detail="; ".join(problems) or "ok")

    if raise_on_failure and not report.ok:
        failures = "\n  ".join(report.failures)
        raise AssertionError(f"{type(model).__name__} failed model conformance:\n  {failures}")
    return report
