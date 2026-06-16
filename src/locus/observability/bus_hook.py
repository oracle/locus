# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""HookProvider that bridges every locus agent-lifecycle event onto
the singleton :class:`EventBus`.

Wire it into any :class:`Agent` to get end-to-end SSE telemetry for
the agent loop. The hook itself is stateless; the only configuration
is the ``run_id`` it tags on every emitted :class:`StreamEvent` so
SSE consumers can filter by dispatch.

Usage::

    from locus.observability import EventBusHook
    from locus import Agent

    hook = EventBusHook(run_id="my-run-1")
    agent = Agent(model=..., tools=[...], hooks=[hook])
    result = agent.invoke("...")
    # ↳ on_before_invocation, on_iteration_*, on_before_tool_call,
    #   on_after_tool_call, on_after_model_call, on_after_invocation
    #   all flow through the bus as ``agent.<phase>`` events.

Inside the router the bridge is wired automatically by
:class:`Router.dispatch` so the user doesn't have to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from locus.hooks.provider import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
)
from locus.observability.event_bus import StreamEvent, get_event_bus


if TYPE_CHECKING:
    from locus.core.state import AgentState


# Locus uses a numeric priority where lower fires earlier. The
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
    """

    def __init__(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("EventBusHook requires a non-empty run_id")
        self._run_id = run_id
        self._bus = get_event_bus()

    @property
    def priority(self) -> int:
        return _BUS_HOOK_PRIORITY

    @property
    def run_id(self) -> str:
        return self._run_id

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
        await self._bus.publish(
            _ev(
                self._run_id,
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
                self._run_id,
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
                self._run_id,
                "agent.iteration.started",
                iteration=iteration,
                agent_id=state.agent_id,
            )
        )

    async def on_iteration_end(self, iteration: int, state: AgentState) -> None:
        await self._bus.publish(
            _ev(
                self._run_id,
                "agent.iteration.completed",
                iteration=iteration,
                agent_id=state.agent_id,
                tool_calls_so_far=len(state.tool_executions),
            )
        )

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._run_id,
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
                self._run_id,
                "agent.tool.completed",
                tool_name=event.tool_name,
                error=str(event.error) if event.error else None,
                result_preview=result_preview,
            )
        )

    async def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._run_id,
                "agent.model.started",
                message_count=len(event.messages),
                tool_count=len(event.tools or []),
            )
        )

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        await self._bus.publish(
            _ev(
                self._run_id,
                "agent.model.completed",
                stop_reason=str(getattr(event.response, "stop_reason", "")),
                content_length=len(getattr(event.response.message, "content", "") or ""),
            )
        )
