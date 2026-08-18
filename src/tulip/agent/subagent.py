# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""First-class child agent runs — ``run_subagent`` and its plumbing.

A subagent is a fresh, isolated agent loop started from inside a running
one: its own conversation, its own system prompt, and an *explicit* tool
allowlist — never the parent's toolset by inheritance. What makes it
first-class rather than "construct an Agent inside a tool body yourself"
is the plumbing a hand-rolled child silently lacks:

- **Usage rolls up.** The child's token counters fold into the parent's
  :class:`~tulip.core.state.AgentState`, so the parent's ``token_budget``
  and its ``TerminateEvent.usage`` stay truthful when work is delegated.
- **Cancellation propagates.** ``parent.cancel()`` stops running children;
  a child finishing never *un*-cancels the parent.
- **Events are observable.** Every child event reaches the ``on_event``
  callback (and the SSE bus, when a run context is active), stamped with
  the child's ``agent_name`` so a front end can render nested activity.

Building a Claude-Code-style ``task`` tool from this is one function::

    from tulip import Agent, tool
    from tulip.agent.subagent import run_subagent


    @tool
    async def task(prompt: str) -> str:
        '''Delegate a focused subproblem to an isolated subagent.'''
        result = await run_subagent(
            prompt,
            model="openai:gpt-4o-mini",
            tools=[grep, read_file],  # explicit allowlist
            system_prompt="You are a focused research subagent.",
            max_iterations=8,
        )
        return result.text


    parent = Agent(model="openai:gpt-4o", tools=[task, edit_file])

Governance is not bypassed by delegation: a tool wrapped with
:func:`tulip.control.gate_tool` carries its gate *with it* into any
allowlist, and a process-global policy installed by a harness (the way
``tulip-code`` installs its ``Policy``) is consulted from inside the tool
bodies themselves, so the same checks fire no matter which loop calls the
tool. Per-agent :class:`~tulip.hooks.HookProvider` policies are attached
to the child via the ``hooks`` parameter.
"""

from __future__ import annotations

import inspect
import threading
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tulip.agent.result import StopReason
from tulip.core.events import TerminateEvent, TulipEvent


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tulip.core.state import AgentState


__all__ = ["SubagentResult", "run_subagent"]


class _LinkedCancelSignal(threading.Event):
    """A child's cancel signal, linked to — but distinct from — the parent's.

    Sharing the parent's :class:`threading.Event` outright would be wrong in
    one direction: the run loop *clears* its signal in its ``finally``, so a
    child winding down would silently un-cancel the parent that cancelled it.
    This subclass reads as set when either side is set, but ``clear()``
    touches only the child's own flag.
    """

    def __init__(self, parent: threading.Event) -> None:
        super().__init__()
        self._parent_signal = parent

    def is_set(self) -> bool:
        return super().is_set() or self._parent_signal.is_set()


class _ParentRunContext:
    """What a running loop exposes to the subagents spawned beneath it.

    Installed by the runtime loop for the duration of a run. Tool bodies
    execute in asyncio tasks created inside that run, so they see the
    context by contextvar propagation — no threading of handles through
    tool signatures.
    """

    __slots__ = ("cancel_signal", "usage_sink")

    def __init__(self, cancel_signal: threading.Event) -> None:
        self.cancel_signal = cancel_signal
        #: Child usage reports, drained into the parent's state by the loop.
        #: Each entry: (prompt, completion, cache_creation, cache_read).
        self.usage_sink: list[tuple[int, int, int, int]] = []


_PARENT_RUN: ContextVar[_ParentRunContext | None] = ContextVar(
    "tulip_parent_run_context", default=None
)


def enter_parent_run(agent: Any) -> Token[_ParentRunContext | None]:
    """Install ``agent``'s run as the parent context for subagents.

    Called by the runtime loop at run start. Ensures the agent has a cancel
    signal to link children to (``Agent.cancel()`` otherwise creates it
    lazily, which would be too late for a child spawned before the first
    ``cancel()``), then publishes the context. Returns a token for
    :func:`exit_parent_run`.
    """
    if agent._cancel_signal is None:  # noqa: SLF001 — loop-side plumbing on our own Agent
        agent._cancel_signal = threading.Event()  # noqa: SLF001
    return _PARENT_RUN.set(_ParentRunContext(agent._cancel_signal))  # noqa: SLF001


def exit_parent_run(token: Token[_ParentRunContext | None]) -> None:
    """Uninstall the context installed by :func:`enter_parent_run`."""
    try:
        _PARENT_RUN.reset(token)
    except ValueError:
        # The generator was finalized from a different context than the one
        # that drove it (a GC-triggered close). That context's copy of the
        # var dies with it, so there is nothing to restore.
        pass


def fold_subagent_usage(state: AgentState) -> AgentState:
    """Fold any pending child usage reports into ``state``'s counters.

    Called by the runtime loop once per iteration, before the budget and
    termination checks, so ``token_budget`` and every ``TerminateEvent``
    see delegated spend as spend. A no-op when no subagent ran.
    """
    ctx = _PARENT_RUN.get()
    if ctx is None or not ctx.usage_sink:
        return state
    pending, ctx.usage_sink[:] = list(ctx.usage_sink), []
    for prompt_toks, completion_toks, cache_creation, cache_read in pending:
        state = state.with_token_usage(prompt_toks, completion_toks, cache_creation, cache_read)
    return state


class SubagentResult(BaseModel):
    """What a finished subagent hands back to whoever spawned it."""

    model_config = {"frozen": True}

    #: The child's final assistant message ("" when it produced none).
    text: str
    #: Why the child stopped — same vocabulary as ``AgentResult.stop_reason``.
    stop_reason: StopReason
    #: Iterations the child used.
    iterations: int
    #: Tool calls the child made.
    tool_calls: int = 0
    #: The child's cumulative token usage (``prompt_tokens`` /
    #: ``completion_tokens`` / ``total_tokens``, plus cache counters when
    #: nonzero). ``None`` when nothing was metered — read absence as
    #: "unmetered", never as free.
    usage: dict[str, int] | None = None
    #: The name the child's events were stamped with.
    agent_name: str | None = None

    @property
    def success(self) -> bool:
        """Same convention as :class:`~tulip.agent.result.AgentResult`."""
        return self.stop_reason in ("complete", "terminal_tool", "confidence_met")


async def run_subagent(
    prompt: str,
    *,
    model: Any,
    tools: list[Any] | None = None,
    system_prompt: str = "You are a focused subagent. Complete the delegated task.",
    name: str = "subagent",
    max_iterations: int = 10,
    hooks: list[Any] | None = None,
    on_event: Callable[[TulipEvent], Awaitable[None] | None] | None = None,
    cancel_signal: threading.Event | None = None,
    **agent_kwargs: Any,
) -> SubagentResult:
    """Run an isolated child agent loop to completion and return its result.

    Callable from a tool body (where it inherits the running parent's
    cancellation and reports its usage into the parent's counters via the
    ambient run context) or from a harness directly (pass ``cancel_signal``
    to keep the linkage; usage then travels only on the returned result).
    Safe to fan out — ``asyncio.gather`` over several calls runs several
    children concurrently, and children spawned by parallel tool calls are
    already capped by the parent's ``max_concurrency`` executor bound.

    Args:
        prompt: The delegated task, as the child's user message.
        model: Model string or ``ModelProtocol`` instance for the child.
            Required — a child never implicitly runs on "whatever the
            parent had" unless the caller says so
            (:meth:`tulip.Agent.run_subagent` defaults it to the parent's).
        tools: Explicit allowlist for the child. ``None`` means *no tools*
            — the parent's toolset is never inherited. Gated tools
            (:func:`tulip.control.gate_tool`) stay gated here.
        system_prompt: The child's own system prompt.
        name: Attribution label stamped on every child event
            (``TulipEvent.agent_name``), so a consumer of a merged stream
            can tell the child's activity from the parent's.
        max_iterations: The child's iteration cap.
        hooks: Lifecycle hooks for the child — the place a harness attaches
            its per-agent policy (``on_before_tool_call`` + ``event.cancel``)
            so a subagent is not a gate bypass.
        on_event: Called with every event the child yields, sync or async.
            Exceptions propagate to the caller.
        cancel_signal: Explicit parent signal to link to, for callers
            outside a running loop. Inside a tool body the running parent's
            signal is picked up automatically; this parameter overrides it.
        **agent_kwargs: Any further :class:`~tulip.agent.config.AgentConfig`
            field for the child (``token_budget``, ``temperature``,
            ``termination``, ...).

    Returns:
        A :class:`SubagentResult` with the child's final text, usage,
        iterations, and stop reason.
    """
    from tulip.agent.agent import Agent  # noqa: PLC0415 — break the agent<->subagent import cycle

    # Capture the ambient parent BEFORE driving the child: while the child
    # runs it installs its own context (for grandchildren), and reporting
    # must go to the parent's sink, not the child's.
    parent_ctx = _PARENT_RUN.get()
    linked_to = cancel_signal or (parent_ctx.cancel_signal if parent_ctx is not None else None)

    child = Agent(
        model=model,
        tools=list(tools or []),
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        hooks=list(hooks or []),
        # Children are short-lived task runners; self-evaluation loops are
        # the parent's concern (mirrors the deepagent task tool's choice).
        reflexion=False,
        grounding=False,
        name=name,
        **agent_kwargs,
    )
    if linked_to is not None:
        child._cancel_signal = _LinkedCancelSignal(linked_to)  # noqa: SLF001 — deliberate linkage into our own Agent

    terminate: TerminateEvent | None = None
    events = child.run(prompt)
    try:
        async for event in events:
            if on_event is not None:
                maybe_awaitable = on_event(event)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            if isinstance(event, TerminateEvent):
                terminate = event
    finally:
        # Run the generator's finally in THIS context (not at GC), so the
        # child's own parent-run context is uninstalled before we report.
        # ``run()`` is typed as AsyncIterator (no ``aclose`` in the protocol)
        # but returns a generator; a test double overriding ``run`` with a
        # plain iterator simply has nothing to close.
        closer = getattr(events, "aclose", None)
        if closer is not None:
            await closer()

    # ``getattr`` rather than direct access: test doubles (and subclasses)
    # that override ``run`` without carrying the loop's bookkeeping simply
    # read as unmetered, the same as a provider that reports no usage.
    state = getattr(child, "_last_run_state", None)

    usage: dict[str, int] | None = None
    if state is not None and state.total_tokens_used > 0:
        usage = {
            "prompt_tokens": state.prompt_tokens_used,
            "completion_tokens": state.completion_tokens_used,
            "total_tokens": state.total_tokens_used,
        }
        if state.cache_creation_tokens_used or state.cache_read_tokens_used:
            usage["cache_creation_input_tokens"] = state.cache_creation_tokens_used
            usage["cache_read_input_tokens"] = state.cache_read_tokens_used
        if parent_ctx is not None:
            # One report per child, of its FINAL counters — grandchildren
            # already folded into them, so the parent counts them once.
            parent_ctx.usage_sink.append(
                (
                    state.prompt_tokens_used,
                    state.completion_tokens_used,
                    state.cache_creation_tokens_used,
                    state.cache_read_tokens_used,
                )
            )

    from tulip.agent.runtime_loop import _normalize_stop_reason  # noqa: PLC0415 — same cycle break

    # A stream that ends WITHOUT a TerminateEvent is a pause: the loop's
    # interrupt path (``ask_user`` and friends) yields InterruptEvent and
    # returns. A subagent has no one to resume it, so the honest reading is
    # "interrupted". (Errors do not take this path — the loop yields an
    # error TerminateEvent and re-raises, and the raise propagates to the
    # caller of this function.)
    return SubagentResult(
        text=(terminate.final_message if terminate else None) or "",
        stop_reason=_normalize_stop_reason(terminate.reason if terminate else "interrupted"),
        iterations=terminate.iterations_used if terminate else 0,
        tool_calls=terminate.total_tool_calls if terminate else 0,
        usage=usage,
        agent_name=name,
    )
