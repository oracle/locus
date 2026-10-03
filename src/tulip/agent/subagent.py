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
  Called from a tool declared with ``emits_progress=True``, each one is
  also yielded live on the parent's own stream, wrapped in a
  :class:`~tulip.core.events.SubagentEvent`.
- **Budgets are shared.** A child cannot outlive the parent's
  ``time_budget_seconds`` or spend past what is left of its
  ``token_budget`` (and of ``max_cost_usd``, when the child's model is
  priced): each child is started with the smaller of its own limit and
  what the parent has left, and one that starts with nothing left returns
  at once.
- **Conversations can be resumed.** A :class:`Subagent` keeps its
  conversation under a ``task_id``, so a later call continues it rather
  than starting cold.

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

import asyncio
import inspect
import threading
import time
import uuid
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

from tulip.agent.result import StopReason
from tulip.core.events import SubagentEvent, TerminateEvent, TulipEvent
from tulip.tools.context import current_tool_context, forward_event, forwarding_events


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tulip.core.state import AgentState


__all__ = ["Subagent", "SubagentResult", "parent_hooks", "run_subagent"]


class _LinkedCancelSignal(threading.Event):
    """A child's cancel signal, linked to — but distinct from — its parents'.

    Sharing a parent's :class:`threading.Event` outright would be wrong in
    one direction: the run loop *clears* its signal in its ``finally``, so a
    child winding down would silently un-cancel the parent that cancelled it.
    This subclass reads as set when it or any parent signal is set, but
    ``clear()`` touches only the child's own flag.
    """

    def __init__(self, *parents: threading.Event) -> None:
        super().__init__()
        self._parent_signals = parents

    def is_set(self) -> bool:
        return super().is_set() or any(p.is_set() for p in self._parent_signals)


class ChildSpend(NamedTuple):
    """One finished child turn's spend, as reported to its parent run."""

    prompt: int
    completion: int
    cache_creation: int
    cache_read: int
    #: At the child's metadata prices; ``None`` when its model is unpriced.
    cost: float | None
    cached: int = 0
    cache_write: int = 0
    #: What the provider reported; ``None`` when no call reported a cost.
    reported_cost: float | None = None


class _ParentRunContext:
    """What a running loop exposes to the subagents spawned beneath it.

    Installed by the runtime loop for the duration of a run. Tool bodies
    execute in asyncio tasks created inside that run, so they see the
    context by contextvar propagation — no threading of handles through
    tool signatures.
    """

    __slots__ = ("cancel_signal", "deadline", "hooks", "observation_pack", "usage_sink")

    def __init__(
        self,
        cancel_signal: threading.Event,
        *,
        deadline: float | None = None,
        hooks: tuple[Any, ...] = (),
        observation_pack: Any = None,
    ) -> None:
        self.cancel_signal = cancel_signal
        #: Child usage reports, drained into the parent's state by the loop.
        self.usage_sink: list[ChildSpend] = []
        #: ``time.monotonic()`` at which the parent's time budget runs out.
        self.deadline = deadline
        #: The parent's own lifecycle hooks, for a delegating tool that
        #: applies the parent's policy inside the child.
        self.hooks = hooks
        #: The ObservationPack config a child gets (the parent's, archived
        #: under the parent's session), or ``None`` when the parent has none.
        self.observation_pack = observation_pack


_PARENT_RUN: ContextVar[_ParentRunContext | None] = ContextVar(
    "tulip_parent_run_context", default=None
)


def enter_parent_run(
    cancel_signal: threading.Event,
    *,
    time_budget_seconds: float | None = None,
    hooks: list[Any] | tuple[Any, ...] = (),
    observation_pack: Any = None,
) -> Token[_ParentRunContext | None]:
    """Install a running loop as the parent context for subagents.

    Called by the runtime loop at run start with that run's own cancel
    signal (``RunContext.cancel``), which ``Agent.cancel(thread_id=...)`` and
    a no-argument ``Agent.cancel()`` both set, its time budget (so children
    stop when it would), its hooks, and the ObservationPack config its
    subagents get. Returns a token for :func:`exit_parent_run`.
    """
    deadline = None if time_budget_seconds is None else time.monotonic() + time_budget_seconds
    return _PARENT_RUN.set(
        _ParentRunContext(
            cancel_signal,
            deadline=deadline,
            hooks=tuple(hooks),
            observation_pack=observation_pack,
        )
    )


def exit_parent_run(token: Token[_ParentRunContext | None]) -> None:
    """Uninstall the context installed by :func:`enter_parent_run`."""
    try:
        _PARENT_RUN.reset(token)
    except ValueError:
        # The generator was finalized from a different context than the one
        # that drove it (a GC-triggered close). That context's copy of the
        # var dies with it, so there is nothing to restore.
        pass


def parent_hooks() -> tuple[Any, ...]:
    """The hooks of the run this code is executing under, if any.

    A delegating tool passes them to its child so a policy attached to the
    parent (``on_before_tool_call`` + ``event.cancel``) also gates the
    child's calls: delegation must never be a way around it.
    """
    ctx = _PARENT_RUN.get()
    return ctx.hooks if ctx is not None else ()


def fold_subagent_usage(state: AgentState) -> AgentState:
    """Fold any pending child usage reports into ``state``'s counters.

    Called by the runtime loop once per iteration, before the budget and
    termination checks, so ``token_budget`` and every ``TerminateEvent``
    see delegated spend as spend. A child's spend is counted at the child's
    own prices when they are known, and at the parent's otherwise.
    """
    ctx = _PARENT_RUN.get()
    if ctx is None or not ctx.usage_sink:
        return state
    pending, ctx.usage_sink[:] = list(ctx.usage_sink), []
    for spend in pending:
        spent_before = state.cost_usd_used
        state = state.with_token_usage(
            spend.prompt,
            spend.completion,
            spend.cache_creation,
            spend.cache_read,
            cached_tokens=spend.cached,
            cache_write_tokens=spend.cache_write,
            reported_cost_usd=spend.reported_cost,
        )
        if spend.cost is not None:
            state = state.model_copy(update={"cost_usd_used": spent_before + spend.cost})
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
    #: The id that continues this child's conversation (:class:`Subagent`);
    #: ``None`` for a one-shot child.
    task_id: str | None = None

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
            signal is picked up automatically; this one is linked as well.
        **agent_kwargs: Any further :class:`~tulip.agent.config.AgentConfig`
            field for the child (``token_budget``, ``temperature``,
            ``termination``, ...).

    Returns:
        A :class:`SubagentResult` with the child's final text, usage,
        iterations, and stop reason.
    """
    budgets = _child_budgets(model, agent_kwargs)
    if isinstance(budgets, SubagentResult):
        return budgets.model_copy(update={"agent_name": name})
    child = _build_child(model, tools, system_prompt, name, max_iterations, hooks, budgets)
    return await _drive(child, prompt, name=name, on_event=on_event, cancel_signal=cancel_signal)


def _build_child(  # noqa: PLR0913 — every knob of a child, passed through
    model: Any,
    tools: list[Any] | None,
    system_prompt: str,
    name: str,
    max_iterations: int,
    hooks: list[Any] | None,
    agent_kwargs: dict[str, Any],
) -> Any:
    from tulip.agent.agent import Agent  # noqa: PLC0415 — break the agent<->subagent import cycle

    # A child of a run with ObservationPack gets one too, unless the caller
    # chose: its large outputs leave its requests the same way, archived
    # under the parent's session, and obs_recall comes with it.
    parent = _PARENT_RUN.get()
    if (
        parent is not None
        and parent.observation_pack is not None
        and "observation_pack" not in agent_kwargs
    ):
        agent_kwargs = {**agent_kwargs, "observation_pack": parent.observation_pack}
    return Agent(
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


def _priced(model: Any) -> bool:
    """Whether spend on ``model`` can be measured, so a cost cap can hold."""
    from tulip.models.metadata import metadata_for, model_id_of  # noqa: PLC0415

    model_id = model_id_of(model)
    meta = metadata_for(model_id) if model_id is not None else None
    return (
        meta is not None
        and meta.input_price_per_mtok is not None
        and meta.output_price_per_mtok is not None
    )


def _child_budgets(model: Any, agent_kwargs: dict[str, Any]) -> dict[str, Any] | SubagentResult:
    """The child's budgets, capped by what its parent has left.

    Time comes from the parent run's deadline. Tokens and spend come from
    the state of the tool call spawning the child — the parent's counters
    as of that batch, its latest model call included. Children started in
    the same batch each see the same remainder; the parent counts their
    combined spend at its next step and stops there if it is over.

    Returns a finished result instead when the parent has nothing left: a
    child started past the parent's deadline would spend a model call the
    parent is about to refuse anyway.
    """
    ctx = _PARENT_RUN.get()
    kwargs = dict(agent_kwargs)
    if ctx is None:
        return kwargs

    def _exhausted(reason: StopReason) -> SubagentResult:
        return SubagentResult(text="", stop_reason=reason, iterations=0)

    if ctx.deadline is not None:
        left = ctx.deadline - time.monotonic()
        if left <= 0:
            return _exhausted("time_budget")
        own = kwargs.get("time_budget_seconds")
        kwargs["time_budget_seconds"] = left if own is None else min(own, left)

    call = current_tool_context()
    state: Any = call.state if call is not None else None
    token_budget = getattr(state, "token_budget", None)
    if token_budget is not None:
        tokens_left = int(token_budget) - int(state.total_tokens_used)
        if tokens_left <= 0:
            return _exhausted("token_budget")
        own = kwargs.get("token_budget")
        kwargs["token_budget"] = tokens_left if own is None else min(own, tokens_left)
    cost_budget = getattr(state, "cost_budget_usd", None)
    if cost_budget is not None:
        cost_left = float(cost_budget) - float(state.cost_usd_used)
        if cost_left <= 0:
            return _exhausted("cost_budget")
        # An unpriced child cannot hold a cost cap (the SDK refuses one it
        # cannot measure); its spend still folds into the parent's at the
        # parent's prices, so the parent stops at its next step instead.
        if _priced(model):
            own = kwargs.get("max_cost_usd")
            kwargs["max_cost_usd"] = cost_left if own is None else min(own, cost_left)
    return kwargs


async def _drive(
    child: Any,
    prompt: str,
    *,
    name: str,
    on_event: Callable[[TulipEvent], Awaitable[None] | None] | None,
    cancel_signal: threading.Event | None,
    thread_id: str | None = None,
    task_id: str | None = None,
) -> SubagentResult:
    """Run ``child`` on ``prompt`` to the end and account for it to its parent."""
    # Capture the ambient parent BEFORE driving the child: while the child
    # runs it installs its own context (for grandchildren), and reporting
    # must go to the parent's sink, not the child's.
    parent_ctx = _PARENT_RUN.get()
    linked_to = [
        signal
        for signal in (cancel_signal, parent_ctx.cancel_signal if parent_ctx else None)
        if signal is not None
    ]
    if linked_to:
        child._cancel_signal = _LinkedCancelSignal(*linked_to)  # noqa: SLF001 — deliberate linkage into our own Agent

    # The tool call this child serves, when it runs inside one whose stream
    # is listening: its events go out live, wrapped, as they happen.
    call = current_tool_context() if forwarding_events() else None

    terminate: TerminateEvent | None = None
    events = child.run(prompt, thread_id=thread_id) if thread_id else child.run(prompt)
    try:
        async for event in events:
            if on_event is not None:
                maybe_awaitable = on_event(event)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            if call is not None:
                forward_event(
                    SubagentEvent(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        task_id=task_id,
                        event=event,
                        agent_name=event.agent_name or name,
                    )
                )
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
            # One report per child turn, of its FINAL counters — grandchildren
            # already folded into them, so the parent counts them once. The
            # counters are per turn, so a resumed child reports only the new
            # turn's spend.
            parent_ctx.usage_sink.append(
                ChildSpend(
                    state.prompt_tokens_used,
                    state.completion_tokens_used,
                    state.cache_creation_tokens_used,
                    state.cache_read_tokens_used,
                    state.cost_usd_used if state.priced else None,
                    cached=state.cached_tokens_used,
                    cache_write=state.cache_write_tokens_used,
                    reported_cost=(state.reported_cost_usd if state.reported_cost_calls else None),
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
        task_id=task_id,
    )


class Subagent:
    """A child agent that keeps its conversation, so it can be resumed.

    :func:`run_subagent` is one-shot: the child's conversation is gone when
    it returns. A ``Subagent`` keeps it under :attr:`task_id`, and each
    :meth:`send` is a new turn on the same conversation — "now also check
    the tests", without re-reading everything the first turn read. Every
    turn gets the same accounting as :func:`run_subagent`: usage into the
    calling run, the calling run's cancellation and budgets, and its events
    on the calling run's stream.

    One turn at a time: a second :meth:`send` waits for the first. Turns run
    on a fresh :class:`~tulip.agent.agent.Agent` each time, sharing an
    in-memory checkpointer, so nothing about one turn's run (its cancel
    signal, its budgets) leaks into the next.

    Args:
        model: Model string or ``ModelProtocol`` instance for the child.
        tools: Explicit allowlist for the child (``None`` means no tools).
        system_prompt: The child's system prompt.
        name: Attribution label stamped on the child's events.
        max_iterations: The child's iteration cap, per turn.
        hooks: Lifecycle hooks for the child.
        task_id: The id to keep the conversation under; generated if omitted.
        **agent_kwargs: Further ``AgentConfig`` fields for the child.
    """

    def __init__(  # noqa: PLR0913 — mirrors run_subagent's surface
        self,
        *,
        model: Any,
        tools: list[Any] | None = None,
        system_prompt: str = "You are a focused subagent. Complete the delegated task.",
        name: str = "subagent",
        max_iterations: int = 10,
        hooks: list[Any] | None = None,
        task_id: str | None = None,
        **agent_kwargs: Any,
    ) -> None:
        from tulip.memory.backends.memory import MemoryCheckpointer  # noqa: PLC0415

        if "checkpointer" in agent_kwargs:
            msg = "a Subagent keeps its own conversation; do not pass a checkpointer"
            raise ValueError(msg)
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        self.name = name
        #: Turns this child has run.
        self.turns = 0
        self._model = model
        self._tools = list(tools or [])
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._hooks = list(hooks or [])
        self._agent_kwargs = agent_kwargs
        self._checkpointer = MemoryCheckpointer()
        self._lock = asyncio.Lock()

    async def send(
        self,
        prompt: str,
        *,
        on_event: Callable[[TulipEvent], Awaitable[None] | None] | None = None,
        cancel_signal: threading.Event | None = None,
    ) -> SubagentResult:
        """Run one turn of this child's conversation and return its result."""
        async with self._lock:
            budgets = _child_budgets(self._model, self._agent_kwargs)
            if isinstance(budgets, SubagentResult):
                return budgets.model_copy(update={"agent_name": self.name, "task_id": self.task_id})
            child = _build_child(
                self._model,
                self._tools,
                self._system_prompt,
                self.name,
                self._max_iterations,
                self._hooks,
                {**budgets, "checkpointer": self._checkpointer},
            )
            result = await _drive(
                child,
                prompt,
                name=self.name,
                on_event=on_event,
                cancel_signal=cancel_signal,
                thread_id=self.task_id,
                task_id=self.task_id,
            )
            self.turns += 1
            return result
