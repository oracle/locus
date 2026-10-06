# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""HookProvider that bridges every tulip agent-lifecycle event onto
the singleton :class:`EventBus`.

Wire it into any :class:`Agent` to get end-to-end SSE telemetry for
the agent loop. The hook itself is stateless; the only configuration
is the ``run_id`` it tags on every emitted :class:`StreamEvent` so
SSE consumers can filter by dispatch.

Usage::

    from tulip.observability import EventBusHook
    from tulip import Agent

    hook = EventBusHook(run_id="my-run-1")
    agent = Agent(model=..., tools=[...], hooks=[hook])
    result = agent.invoke("...")
    # ↳ on_before_invocation, on_iteration_*, on_before_tool_call,
    #   on_after_tool_call, on_after_model_call, on_after_invocation
    #   all flow through the bus as ``agent.<phase>`` events.

One Agent serving many runs (a chat backend builds it once) takes one hook
with no ``run_id``. Each event is then tagged with the id of the run that
produced it, read when the event fires: the id bound by
:func:`~tulip.observability.context.run_context` if the caller entered one,
else the run's own ``AgentState.run_id``::

    agent = Agent(model=..., hooks=[EventBusHook()])

    async with run_context(turn_id):
        await agent.arun(message)  # every event under ``turn_id``

Inside the router the bridge is wired automatically by
:class:`Router.dispatch` so the user doesn't have to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.hooks.provider import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
)
from tulip.observability.context import current_run_id
from tulip.observability.event_bus import StreamEvent, get_event_bus


if TYPE_CHECKING:
    from tulip.core.state import AgentState


# Tulip uses a numeric priority where lower fires earlier. The
# observability range in HookPriority is around 100; we sit at 110
# so we run after instrumentation hooks but before user-customised
# ones with higher priorities.
_BUS_HOOK_PRIORITY = 110


def _ev(run_id: str, event_type: str, **data: Any) -> StreamEvent:
    return StreamEvent(run_id=run_id, event_type=event_type, data=data)


class EventBusHook(HookProvider):
    """Republishes every agent-lifecycle hook onto the event bus.

    The hook never mutates events — it's pure-observation. Tools and
    model calls retain whatever cancellation / retry / replacement
    behaviour the surrounding code dictates.

    Args:
        run_id: The id every event is tagged with. Omit it to tag each event
            with the run that produced it, read per call: the id of the
            active :func:`~tulip.observability.context.run_context` when
            there is one, else the run's own ``AgentState.run_id``. That is
            what lets one hook on one shared Agent serve every run.

    Raises:
        ValueError: ``run_id`` is an empty string.
    """

    def __init__(self, run_id: str | None = None) -> None:
        if run_id is not None and not run_id:
            raise ValueError("EventBusHook run_id must be non-empty (or None to follow the run)")
        # Kept under this private name: subclasses written before ``run_id``
        # became optional override ``_run_id`` to follow the run themselves.
        self._run_id = run_id
        self._bus = get_event_bus()

    @property
    def priority(self) -> int:
        return _BUS_HOOK_PRIORITY

    @property
    def run_id(self) -> str | None:
        """The id an event fired now would carry, outside any agent run.

        The fixed ``run_id`` when one was given; otherwise the active
        ``run_context`` id, or ``None`` when there is none (an event fired
        inside a run then falls back to that run's own id).
        """
        return self._run_id or current_run_id()

    def _rid(self, run: Any = None, state: AgentState | None = None) -> str:
        """The id for one event: fixed, else the active run context, else the run's."""
        if self._run_id is not None:
            return self._run_id
        bound = current_run_id()
        if bound:
            return bound
        own = getattr(run, "run_id", None) or getattr(state, "run_id", None)
        return str(own) if own else "default"

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
        await self._bus.publish(
            _ev(
                self._rid(state=state),
                "agent.invocation.started",
                prompt_preview=prompt[:160],
                agent_id=state.agent_id,
                max_iterations=state.max_iterations,
            )
        )
        return state

    async def on_after_invocation(self, state: AgentState, success: bool) -> None:
        await self._bus.publish(
            _ev(
                self._rid(state=state),
                "agent.invocation.completed",
                agent_id=state.agent_id,
                iteration=state.iteration,
                success=success,
                tool_calls=len(state.tool_executions),
                total_tokens=state.total_tokens_used,
            )
        )

    async def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        await self._bus.publish(
            _ev(
                self._rid(state=state),
                "agent.iteration.started",
                iteration=iteration,
                agent_id=state.agent_id,
            )
        )

    async def on_iteration_end(self, iteration: int, state: AgentState) -> None:
        await self._bus.publish(
            _ev(
                self._rid(state=state),
                "agent.iteration.completed",
                iteration=iteration,
                agent_id=state.agent_id,
                tool_calls_so_far=len(state.tool_executions),
            )
        )

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._rid(event.run),
                "agent.tool.started",
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                argument_keys=sorted((event.arguments or {}).keys()),
            )
        )

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        # Result can be huge; we only surface its type + length so the
        # bus stays cheap. Subscribers that need the full payload can
        # bridge to a different sink.
        result = event.result
        result_preview = ""
        if isinstance(result, str):
            result_preview = result[:200]
        elif result is not None:
            result_preview = repr(result)[:200]
        await self._bus.publish(
            _ev(
                self._rid(event.run),
                "agent.tool.completed",
                tool_name=event.tool_name,
                error=str(event.error) if event.error else None,
                result_preview=result_preview,
            )
        )

    async def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._rid(event.run),
                "agent.model.started",
                message_count=len(event.messages),
                tool_count=len(event.tools or []),
            )
        )

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._rid(event.run),
                "agent.model.completed",
                stop_reason=str(getattr(event.response, "stop_reason", "")),
                content_length=len(getattr(event.response.message, "content", "") or ""),
            )
        )
