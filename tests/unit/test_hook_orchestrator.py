# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Contract tests for :class:`locus.agent.hook_orchestrator.HookOrchestrator`.

The orchestrator was extracted from ``Agent`` — these tests lock in
the invariants that ``Agent`` used to enforce inline:

- ``before_*`` phases dispatch in registration order.
- ``after_*`` phases dispatch in reverse order (symmetrical
  teardown).
- Hooks missing a given ``on_<phase>`` method are skipped.
- ``run_before_model`` writes through ``event.messages`` to its
  return value.
- Mutations to the underlying hook list after orchestrator
  construction are picked up.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.agent.hook_orchestrator import HookOrchestrator
from locus.core.state import AgentState


class _Recorder:
    """Hook provider that records every dispatched phase."""

    def __init__(self, name: str, log: list[str], *, phases: set[str] | None = None) -> None:
        self.name = name
        self.log = log
        self._phases = phases or {
            "before_invocation",
            "after_invocation",
            "before_model_call",
            "after_model_call",
            "before_tool_call",
            "after_tool_call",
        }

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:  # noqa: ARG002
        if "before_invocation" in self._phases:
            self.log.append(f"{self.name}.before_invocation")
        return state

    async def on_after_invocation(self, state: AgentState, success: bool) -> None:  # noqa: ARG002
        if "after_invocation" in self._phases:
            self.log.append(f"{self.name}.after_invocation")

    async def on_before_model_call(self, event: Any) -> None:
        if "before_model_call" in self._phases:
            self.log.append(f"{self.name}.before_model_call")
            event.messages = [*event.messages, f"{self.name}-injected"]

    async def on_after_model_call(self, event: Any) -> None:  # noqa: ARG002
        if "after_model_call" in self._phases:
            self.log.append(f"{self.name}.after_model_call")

    async def on_before_tool_call(self, event: Any) -> None:  # noqa: ARG002
        if "before_tool_call" in self._phases:
            self.log.append(f"{self.name}.before_tool_call")

    async def on_after_tool_call(self, event: Any) -> None:  # noqa: ARG002
        if "after_tool_call" in self._phases:
            self.log.append(f"{self.name}.after_tool_call")


class TestHookOrchestratorOrdering:
    @pytest.mark.asyncio
    async def test_before_invocation_runs_in_order(self) -> None:
        log: list[str] = []
        orch = HookOrchestrator([_Recorder("a", log), _Recorder("b", log), _Recorder("c", log)])
        state = AgentState(agent_id="t")

        await orch.run_before_invocation("p", state)

        assert log == ["a.before_invocation", "b.before_invocation", "c.before_invocation"]

    @pytest.mark.asyncio
    async def test_after_invocation_runs_in_reverse_order(self) -> None:
        log: list[str] = []
        orch = HookOrchestrator([_Recorder("a", log), _Recorder("b", log), _Recorder("c", log)])
        state = AgentState(agent_id="t")

        await orch.run_after_invocation(state, success=True)

        assert log == ["c.after_invocation", "b.after_invocation", "a.after_invocation"]

    @pytest.mark.asyncio
    async def test_before_tool_runs_in_order(self) -> None:
        log: list[str] = []
        orch = HookOrchestrator([_Recorder("a", log), _Recorder("b", log)])

        await orch.run_before_tool(tool_name="t", tool_call_id="tc", arguments={})

        assert log == ["a.before_tool_call", "b.before_tool_call"]

    @pytest.mark.asyncio
    async def test_after_tool_runs_in_reverse_order(self) -> None:
        log: list[str] = []
        orch = HookOrchestrator([_Recorder("a", log), _Recorder("b", log)])

        await orch.run_after_tool(tool_name="t", result="ok", error=None)

        assert log == ["b.after_tool_call", "a.after_tool_call"]

    @pytest.mark.asyncio
    async def test_after_tool_forwards_call_id_and_arguments(self) -> None:
        """run_after_tool passes tool_call_id + arguments through to the event."""
        captured: dict[str, Any] = {}

        class _Capturer:
            name = "cap"

            async def on_after_tool_call(self, event: Any) -> None:
                captured["tool_name"] = event.tool_name
                captured["tool_call_id"] = event.tool_call_id
                captured["arguments"] = event.arguments
                captured["result"] = event.result

        orch = HookOrchestrator([_Capturer()])
        await orch.run_after_tool(
            tool_name="search",
            result="ok",
            error=None,
            tool_call_id="tc-42",
            arguments={"query": "weather", "limit": 3},
        )

        assert captured == {
            "tool_name": "search",
            "tool_call_id": "tc-42",
            "arguments": {"query": "weather", "limit": 3},
            "result": "ok",
        }

    @pytest.mark.asyncio
    async def test_after_tool_back_compat_without_new_kwargs(self) -> None:
        """Callers that don't pass tool_call_id/arguments still get a valid event."""
        captured: dict[str, Any] = {}

        class _Capturer:
            name = "cap"

            async def on_after_tool_call(self, event: Any) -> None:
                captured["tool_call_id"] = event.tool_call_id
                captured["arguments"] = event.arguments

        orch = HookOrchestrator([_Capturer()])
        await orch.run_after_tool(tool_name="t", result="ok", error=None)

        assert captured == {"tool_call_id": "", "arguments": {}}


class TestHookOrchestratorDispatch:
    @pytest.mark.asyncio
    async def test_missing_method_is_skipped(self) -> None:
        """A hook without ``on_before_invocation`` does not crash dispatch."""

        class Bare:
            name = "bare"

        log: list[str] = []
        orch = HookOrchestrator([Bare(), _Recorder("a", log)])

        state = AgentState(agent_id="t")
        await orch.run_before_invocation("p", state)

        assert log == ["a.before_invocation"]

    @pytest.mark.asyncio
    async def test_before_model_writes_through_event_messages(self) -> None:
        """Hooks mutate ``event.messages``; the orchestrator returns
        the final list the caller should hand to the model."""
        log: list[str] = []
        orch = HookOrchestrator([_Recorder("h1", log), _Recorder("h2", log)])

        out = await orch.run_before_model(messages=["initial"], tools=None)

        # Each hook appended its own tag.
        assert out == ["initial", "h1-injected", "h2-injected"]
        assert log == ["h1.before_model_call", "h2.before_model_call"]

    @pytest.mark.asyncio
    async def test_empty_hook_list_is_a_no_op(self) -> None:
        orch = HookOrchestrator([])
        state = AgentState(agent_id="t")

        # None of these should raise.
        restored = await orch.run_before_invocation("p", state)
        assert restored is state
        await orch.run_after_invocation(state, success=True)
        out = await orch.run_before_model(messages=["x"], tools=None)
        assert out == ["x"]


class TestHookOrchestratorLiveness:
    """The orchestrator holds a reference to the hook list, so plugin
    hooks appended after construction are picked up at dispatch."""

    @pytest.mark.asyncio
    async def test_late_added_hook_fires(self) -> None:
        log: list[str] = []
        hooks: list[Any] = [_Recorder("a", log)]
        orch = HookOrchestrator(hooks)

        hooks.append(_Recorder("b", log))

        state = AgentState(agent_id="t")
        await orch.run_before_invocation("p", state)

        assert log == ["a.before_invocation", "b.before_invocation"]
