# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hook lifecycle orchestration for :class:`~locus.agent.agent.Agent`.

Six lifecycle phases — ``before_invocation``, ``after_invocation``,
``before_model_call``, ``after_model_call``, ``before_tool_call``,
``after_tool_call`` — are dispatched through this class. Every
``after_*`` phase runs in reverse order of registration so a hook
registered last gets to clean up first (symmetrical with a
``before_*`` pair).

Extracted from ``Agent`` so the runtime isn't responsible for
both driving the ReAct loop and dispatching hooks. The behavior is
identical to the previous inline ``_run_*_hooks`` methods — same
method signatures, same ordering, same write-through of
``event.messages`` / ``event.arguments``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.core.state import AgentState


class HookOrchestrator:
    """Dispatch the six agent lifecycle events to a list of hooks.

    The orchestrator does not own the hook list — it only holds a
    reference. The agent is free to mutate the list at initialization
    (for plugins, skills, etc.); the orchestrator picks up the final
    shape at dispatch time.

    Args:
        hooks: The same list of hook providers the Agent built during
            `_initialize`. Any object in this list that exposes the
            matching ``on_<phase>`` coroutine method is dispatched to.
    """

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[Any]) -> None:
        self._hooks = hooks

    async def run_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Dispatch ``on_before_invocation`` to every hook in order.

        Hooks may return a modified state; each hook sees the state
        as shaped by the preceding hook.
        """
        for hook in self._hooks:
            if hasattr(hook, "on_before_invocation"):
                state = await hook.on_before_invocation(prompt, state)
        return state

    async def run_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """Dispatch ``on_after_invocation`` in reverse order.

        Reverse ordering so setup/teardown pair up symmetrically with
        ``run_before_invocation``.
        """
        for hook in reversed(self._hooks):
            if hasattr(hook, "on_after_invocation"):
                await hook.on_after_invocation(state, success)

    async def run_before_model(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None,
    ) -> list[Any]:
        """Dispatch ``on_before_model_call``; returns possibly-modified messages.

        Hooks mutate ``event.messages`` in place — the agent uses the
        returned list to call the model.
        """
        from locus.hooks.provider import BeforeModelCallEvent

        event = BeforeModelCallEvent(messages=messages, tools=tools)
        for hook in self._hooks:
            if hasattr(hook, "on_before_model_call"):
                await hook.on_before_model_call(event)
        messages_out: list[Any] = event.messages
        return messages_out

    async def run_after_model(
        self,
        response: Any,
        messages: list[Any],
    ) -> Any:
        """Dispatch ``on_after_model_call`` in reverse order.

        Returns the event object so the caller can inspect hook
        signals (e.g. ``event.retry``).
        """
        from locus.hooks.provider import AfterModelCallEvent

        event = AfterModelCallEvent(response=response, messages=messages)
        for hook in reversed(self._hooks):
            if hasattr(hook, "on_after_model_call"):
                await hook.on_after_model_call(event)
        return event

    async def run_before_tool(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Dispatch ``on_before_tool_call``; returns the event.

        Callers check ``event.cancel`` to decide whether to skip the
        tool; they read ``event.arguments`` for hook-modified args.
        """
        from locus.hooks.provider import BeforeToolCallEvent

        event = BeforeToolCallEvent(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        for hook in self._hooks:
            if hasattr(hook, "on_before_tool_call"):
                await hook.on_before_tool_call(event)
        return event

    async def run_after_tool(
        self,
        tool_name: str,
        result: Any,
        error: str | None,
        *,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch ``on_after_tool_call`` in reverse order; returns the event.

        ``tool_call_id`` and ``arguments`` are passed through to the event
        so hooks can correlate the after-event with the corresponding
        ``BeforeToolCallEvent`` and observe the exact arguments the tool
        ran with (post-hook mutation). Default-empty for backwards
        compatibility with older callers.
        """
        from locus.hooks.provider import AfterToolCallEvent

        event = AfterToolCallEvent(
            tool_name=tool_name,
            result=result,
            error=error,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        for hook in reversed(self._hooks):
            if hasattr(hook, "on_after_tool_call"):
                await hook.on_after_tool_call(event)
        return event
