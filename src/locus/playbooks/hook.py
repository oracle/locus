# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Hook adapter that wires :class:`PlaybookEnforcer` into the agent loop.

The enforcer in :mod:`locus.playbooks.enforcer` was complete and well-tested
but disconnected from ``Agent`` — there was no automatic step-tracking
during a run. ``PlaybookEnforcerHook`` is the missing glue:

- ``on_before_tool_call`` calls ``enforcer.validate_tool_call(tool_name)``;
  if blocked, it sets ``event.cancel`` to the violation message so the
  agent loop turns the call into a no-op with a useful explanation.
- ``on_after_tool_call`` records the tool call and, when a step's
  ``expected_tools`` are all satisfied (or its ``max_tool_calls`` is
  reached), advances the plan via ``complete_current_step``.

This is the integration the README's "PlaybookEnforcer validates tool
calls against step constraints" claim was always describing — it just
hadn't been built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from locus.hooks.provider import HookPriority, HookProvider
from locus.playbooks.enforcer import PlaybookEnforcer
from locus.playbooks.models import Playbook


if TYPE_CHECKING:
    from locus.hooks.provider import (
        AfterToolCallEvent,
        BeforeToolCallEvent,
    )


class PlaybookEnforcerHook(HookProvider):
    """Hook that enforces a :class:`Playbook` over an agent run.

    Holds a single :class:`PlaybookEnforcer` instance and dispatches
    ``before/after_tool_call`` events into it so step compliance is tracked
    automatically. Auto-advances to the next step when the current step's
    expected tool list is exhausted.

    Args:
        playbook: The playbook to enforce. A fresh ``PlaybookEnforcer`` is
            built from it on construction.
        block_violations: When True (default), a violating tool call is
            cancelled via ``event.cancel``. When False, violations are
            recorded but the call still runs.
        record_violations: When True (default), violations land on
            ``enforcer.violations`` for post-run inspection.
        priority: Hook priority. Defaults to a high value so the enforcer
            runs before observability / retry hooks; that way a blocked
            tool call doesn't get logged as if it had executed.

    Example:
        from locus import Agent
        from locus.playbooks.loader import load_playbook
        from locus.playbooks.hook import PlaybookEnforcerHook

        playbook = load_playbook("playbooks/triage.yaml")
        agent = Agent(
            model="oci:openai.gpt-5",
            tools=[search, classify, escalate],
            hooks=[PlaybookEnforcerHook(playbook)],
        )
        result = agent.run_sync("Triage this incident.")
    """

    def __init__(
        self,
        playbook: Playbook,
        *,
        block_violations: bool = True,
        record_violations: bool = True,
        priority: int = HookPriority.SECURITY_DEFAULT,
    ) -> None:
        self._enforcer = PlaybookEnforcer.from_playbook(
            playbook,
            block_violations=block_violations,
            record_violations=record_violations,
        )
        self._priority = priority

    @property
    def name(self) -> str:
        return "PlaybookEnforcerHook"

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def enforcer(self) -> PlaybookEnforcer:
        """Return the underlying enforcer for inspection (violations, progress)."""
        return self._enforcer

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Validate the call against the current step; cancel on violation."""
        # ``ProtectedEvent`` (the base class for hook events) sets fields
        # via ``self._init(name, value)`` rather than class-level annotations,
        # so mypy can't see ``.tool_name`` / ``.error`` statically. They
        # exist at runtime; the ignore is the standard pattern for this
        # protocol.
        result = self._enforcer.validate_tool_call(event.tool_name)
        if result.allowed:
            return
        # Build a useful cancel message that the agent loop will turn into
        # a tool result so the model can recover. The hint list is the
        # enforcer's machine-readable "what to do next" for the model.
        msg_parts = []
        if result.violation is not None:
            msg_parts.append(result.violation.message)
        if result.hints:
            msg_parts.append("Hints: " + " ".join(result.hints))
        event.cancel = "PlaybookEnforcer blocked: " + " | ".join(msg_parts)

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Record the call and auto-advance when the current step is satisfied.

        The agent loop short-circuits past ``on_after_tool_call`` when the
        before-hook cancelled the call, so anything reaching this method
        actually executed.
        """
        # ``ProtectedEvent`` sets ``.error`` / ``.tool_name`` via
        # ``self._init(...)`` not class-level fields — see note in
        # ``on_before_tool_call``.
        if event.error:
            # Failed calls don't advance the step (the model will likely
            # retry); they're still recorded for the violation log.
            self._enforcer.record_tool_call(event.tool_name)
            return

        self._enforcer.record_tool_call(event.tool_name)

        step = self._enforcer.current_step
        if step is None:
            return

        step_exec = self._enforcer.plan.step_executions.get(step.id)
        if step_exec is None:
            return

        # Auto-advance the plan when the step's expected tools have all
        # been seen, OR when max_tool_calls is reached. Without this, the
        # enforcer would block legitimate next-step calls because the plan
        # is still pointing at a satisfied step.
        if step.expected_tools:
            seen = set(step_exec.tool_calls)
            if set(step.expected_tools).issubset(seen):
                self._enforcer.complete_current_step()
                return

        if step.max_tool_calls is not None and step_exec.tool_call_count >= step.max_tool_calls:
            self._enforcer.complete_current_step()


__all__ = ["PlaybookEnforcerHook"]
