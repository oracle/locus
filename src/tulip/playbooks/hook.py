# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Hook adapter that wires :class:`PlaybookEnforcer` into the agent loop.

The enforcer in :mod:`tulip.playbooks.enforcer` was complete and well-tested
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

One Agent serves many runs, so the plan's progress belongs to a run, not to
the hook: each run gets its own enforcer, keyed by ``event.run.run_id``. A
host that wants a playbook for some runs only passes ``select`` instead of a
fixed playbook and picks one per run (from ``run.metadata``, say).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from tulip.hooks.provider import HookPriority, HookProvider
from tulip.playbooks.enforcer import PlaybookEnforcer
from tulip.playbooks.models import Playbook


if TYPE_CHECKING:
    from tulip.core.events import RunInfo
    from tulip.hooks.provider import (
        AfterToolCallEvent,
        BeforeToolCallEvent,
    )

logger = logging.getLogger(__name__)

#: Chooses a run's playbook; ``None`` leaves the run unenforced.
PlaybookSelector = Callable[["RunInfo"], "Playbook | None"]

#: The key used for events dispatched outside a run (``event.run is None``).
_NO_RUN = "__no_run__"

#: What :class:`PlaybookEnforcerHook` keeps a plan for.
PlaybookScope = Literal["run", "thread", "agent"]


class PlaybookEnforcerHook(HookProvider):
    """Hook that enforces a :class:`Playbook` over agent runs.

    Dispatches ``before/after_tool_call`` events into a
    :class:`PlaybookEnforcer` so step compliance is tracked automatically,
    and auto-advances to the next step when the current step's expected tool
    list is exhausted.

    Each run gets its own enforcer (``scope="run"``): one Agent serving many
    users never lets one run's progress, or its violations, count for
    another's. The enforcers live in a bounded most-recently-used map, so a
    run paused for approval and resumed in the same process (it keeps its run
    id) carries on from the step it reached.

    Args:
        playbook: The playbook every run follows. Pass this or ``select``.
        select: ``(run) -> Playbook | None``, asked once per run (per scope
            key) on its first tool call, to enable a playbook for that run or
            turn only — e.g. from ``run.metadata``. ``None`` leaves the run
            unenforced. A selector that raises does NOT leave the run
            unenforced: every tool call of that run is cancelled with a
            message saying the playbook could not be chosen, so a broken
            selector fails closed instead of quietly switching enforcement
            off.
        block_violations: When True (default), a violating tool call is
            cancelled via ``event.cancel``. When False, violations are
            recorded but the call still runs.
        record_violations: When True (default), violations land on
            ``enforcer.violations`` for post-run inspection.
        skills: Skills by name, so ``PlaybookStep.uses`` resolves.
        scope: What one plan spans. ``"run"`` (default) is one agent run;
            ``"thread"`` follows a conversation across turns (keyed by
            ``run.thread_id``, falling back to the run id when the run has no
            thread); ``"agent"`` is one plan for everything the agent does,
            the behaviour of earlier releases.
        max_runs: Enforcers kept at once; the least recently used is dropped
            beyond it.
        priority: Hook priority. Defaults to a high value so the enforcer
            runs before observability / retry hooks; that way a blocked
            tool call doesn't get logged as if it had executed.

    Example:
        from tulip import Agent
        from tulip.playbooks.loader import load_playbook
        from tulip.playbooks.hook import PlaybookEnforcerHook

        playbook = load_playbook("playbooks/triage.yaml")
        agent = Agent(
            model="openai:gpt-4o",
            tools=[search, classify, escalate],
            hooks=[PlaybookEnforcerHook(playbook)],
        )
        result = agent.run_sync("Triage this incident.")

    Per run, from the run's metadata::

        books = {"triage": load_playbook("playbooks/triage.yaml")}
        hook = PlaybookEnforcerHook(
            select=lambda run: books.get(run.metadata.get("playbook"))
        )
        await agent.arun("Triage this.", metadata={"playbook": "triage"})  # enforced
        await agent.arun("Just chat.")  # not enforced
    """

    def __init__(
        self,
        playbook: Playbook | None = None,
        *,
        select: PlaybookSelector | None = None,
        block_violations: bool = True,
        record_violations: bool = True,
        skills: Mapping[str, Any] | None = None,
        scope: PlaybookScope = "run",
        max_runs: int = 1024,
        priority: int = HookPriority.SECURITY_DEFAULT,
    ) -> None:
        if (playbook is None) == (select is None):
            raise ValueError("PlaybookEnforcerHook takes a playbook or select=, exactly one")
        if scope not in ("run", "thread", "agent"):
            raise ValueError(f"scope must be 'run', 'thread' or 'agent', got {scope!r}")
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        self._playbook = playbook
        self._select = select
        self._block = block_violations
        self._record = record_violations
        self._skills = skills
        self._scope: PlaybookScope = scope
        self._max_runs = max_runs
        self._priority = priority
        # key -> the run's enforcer, or None (no playbook for it), or a str
        # (the selector failed: why every call of the run is cancelled).
        self._runs: OrderedDict[str, PlaybookEnforcer | str | None] = OrderedDict()
        self._last: str | None = None

    @property
    def name(self) -> str:
        return "PlaybookEnforcerHook"

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def enforcer(self) -> PlaybookEnforcer:
        """The enforcer of the most recent run, for inspection (violations, progress).

        With one run at a time that is the run just finished, as before.
        With concurrent runs use :meth:`enforcer_for`. Before any run, and for
        a run with no playbook, a fresh enforcer of ``playbook`` (with
        ``select``, there is nothing to show: ``LookupError``).
        """
        found = self._runs.get(self._last) if self._last is not None else None
        if isinstance(found, PlaybookEnforcer):
            return found
        if self._playbook is None:
            raise LookupError("no run has been enforced yet")
        # Outside any run: the one enforcer events with no run share, kept so
        # repeated reads (and direct hook calls in a test) see the same plan.
        made = self._runs.get(_NO_RUN)
        if not isinstance(made, PlaybookEnforcer):
            made = self._new(self._playbook)
            self._runs[_NO_RUN] = made
        return made

    def enforcer_for(self, run_id: str) -> PlaybookEnforcer | None:
        """The enforcer of one run (its scope key), or None when it had none."""
        found = self._runs.get(run_id)
        return found if isinstance(found, PlaybookEnforcer) else None

    def _new(self, playbook: Playbook) -> PlaybookEnforcer:
        return PlaybookEnforcer.from_playbook(
            playbook,
            block_violations=self._block,
            record_violations=self._record,
            skills=self._skills,
        )

    def _key(self, run: RunInfo | None) -> str:
        if self._scope == "agent" or run is None:
            return _NO_RUN
        if self._scope == "thread" and run.thread_id:
            return f"thread:{run.thread_id}"
        return run.run_id

    def _for(self, event: Any) -> PlaybookEnforcer | str | None:
        """The enforcer for the event's run, built on its first event."""
        run: RunInfo | None = event.run
        key = self._key(run)
        self._last = key
        if key in self._runs:
            self._runs.move_to_end(key)
            return self._runs[key]
        found: PlaybookEnforcer | str | None
        if self._select is None:
            found = self._new(self._playbook) if self._playbook is not None else None
        else:
            from tulip.core.events import RunInfo  # noqa: PLC0415

            info = run or RunInfo.build(run_id=_NO_RUN, thread_id=None, metadata=None)
            try:
                chosen = self._select(info)
            except Exception as exc:  # noqa: BLE001 — reported to the model, fails closed
                logger.warning("playbook select failed for run %s", info.run_id, exc_info=True)
                found = (
                    "PlaybookEnforcer blocked: the playbook for this run could not be "
                    f"chosen ({type(exc).__name__}), so no tool may run."
                )
            else:
                found = self._new(chosen) if chosen is not None else None
        self._runs[key] = found
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)
        return found

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Validate the call against the current step; cancel on violation."""
        enforcer = self._for(event)
        if enforcer is None:
            return
        if isinstance(enforcer, str):
            event.cancel = enforcer
            return
        result = enforcer.validate_tool_call(event.tool_name)
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
        enforcer = self._for(event)
        if not isinstance(enforcer, PlaybookEnforcer):
            return
        if event.error:
            # Failed calls don't advance the step (the model will likely
            # retry); they're still recorded for the violation log. The
            # ARGUMENTS still count as evidence-seeking — the agent did look —
            # but the result does not, because there isn't one.
            enforcer.record_tool_call(event.tool_name, arguments=getattr(event, "arguments", None))
            return

        # Arguments AND result, because a step's `required_probes` may be
        # satisfied by either: what was asked for, or what came back. A tool
        # asked a general question can answer a specific one.
        enforcer.record_tool_call(
            event.tool_name,
            arguments=getattr(event, "arguments", None),
            result=getattr(event, "result", None),
        )

        step = enforcer.current_step
        if step is None:
            return

        step_exec = enforcer.plan.step_executions.get(step.id)
        if step_exec is None:
            return

        # Auto-advance the plan when the step's expected tools have all
        # been seen, OR when max_tool_calls is reached. Without this, the
        # enforcer would block legitimate next-step calls because the plan
        # is still pointing at a satisfied step.
        if step.expected_tools:
            seen = set(step_exec.tool_calls)
            if set(step.expected_tools).issubset(seen):
                enforcer.complete_current_step()
                return

        if step.max_tool_calls is not None and step_exec.tool_call_count >= step.max_tool_calls:
            enforcer.complete_current_step()


__all__ = ["PlaybookEnforcerHook", "PlaybookScope", "PlaybookSelector"]
