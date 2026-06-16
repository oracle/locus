# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``PlaybookEnforcerHook`` + ``Agent(playbook=...)``.

PlaybookEnforcer was a fully-working class with no path through Agent.
These tests guard the new wiring:

- ``Agent(playbook=Playbook(...))`` auto-installs the hook.
- A tool call that matches the current step is allowed.
- A tool call that doesn't match is cancelled with a useful message.
- Calling all of a step's expected tools auto-advances the plan.
- ``allow_extra_tools=True`` lets unmatched calls through but still
  records them.
"""

from __future__ import annotations

from typing import Any

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.core.termination import MaxIterations
from locus.models.base import ModelResponse
from locus.playbooks.hook import PlaybookEnforcerHook
from locus.playbooks.models import Playbook, PlaybookStep
from locus.tools.decorator import tool


@tool
def search(query: str) -> str:
    """Search."""
    return f"results for {query}"


@tool
def classify(severity: str) -> str:
    """Classify."""
    return f"classified {severity}"


@tool
def escalate(to: str) -> str:
    """Escalate."""
    return f"escalated to {to}"


@tool
def chitchat(msg: str) -> str:
    """Off-script tool not in any step."""
    return f"chat: {msg}"


def _playbook(*, allow_extra: bool = False) -> Playbook:
    return Playbook(
        id="triage",
        name="Incident triage",
        allow_extra_tools=allow_extra,
        steps=[
            PlaybookStep(
                id="step-1",
                description="Find context",
                expected_tools=["search"],
            ),
            PlaybookStep(
                id="step-2",
                description="Classify the incident",
                expected_tools=["classify"],
            ),
            PlaybookStep(
                id="step-3",
                description="Escalate to oncall",
                expected_tools=["escalate"],
            ),
        ],
    )


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if not self._responses:
            raise AssertionError("ScriptedModel exhausted")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _assist(content: str | None, *, calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=calls or []),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


# =============================================================================
# Auto-install via Agent(playbook=...)
# =============================================================================


class TestAutoInstall:
    def test_passing_playbook_kwarg_installs_hook(self):
        responses = [_assist("done")]
        agent = Agent(
            model=_ScriptedModel(responses),
            tools=[search],
            playbook=_playbook(),
        )
        # The hook is on the agent's hook list (private attr via Agent's
        # initializer). Existence + name is enough to guard the wiring.
        names = [getattr(h, "name", type(h).__name__) for h in agent._hooks]
        assert any(n == "PlaybookEnforcerHook" for n in names), names

    def test_no_playbook_means_no_enforcer_hook(self):
        agent = Agent(model=_ScriptedModel([_assist("hi")]), tools=[search])
        names = [getattr(h, "name", type(h).__name__) for h in agent._hooks]
        assert "PlaybookEnforcerHook" not in names


# =============================================================================
# Step compliance: in-sequence calls advance the plan
# =============================================================================


class TestStepCompliance:
    def test_in_sequence_calls_advance_plan(self):
        responses = [
            _assist(None, calls=[ToolCall(name="search", arguments={"query": "X"})]),
            _assist(None, calls=[ToolCall(name="classify", arguments={"severity": "P1"})]),
            _assist(None, calls=[ToolCall(name="escalate", arguments={"to": "oncall"})]),
            _assist("All done."),
        ]
        playbook = _playbook()
        hook = PlaybookEnforcerHook(playbook)
        agent = Agent(
            model=_ScriptedModel(responses),
            tools=[search, classify, escalate],
            hooks=[hook],
            termination=MaxIterations(6),
            max_iterations=10,
        )
        agent.run_sync("triage incident")

        # All three steps reached COMPLETED.
        completed = hook.enforcer.plan.completed_steps
        assert "step-1" in completed
        assert "step-2" in completed
        assert "step-3" in completed
        assert hook.enforcer.is_complete is True
        assert hook.enforcer.violations == []


# =============================================================================
# Out-of-sequence calls are blocked
# =============================================================================


class TestBlockOutOfSequence:
    def test_unexpected_tool_is_cancelled(self):
        # Model tries to escalate before searching; with strict_sequence=True
        # and allow_extra_tools=False (defaults), the call must be blocked.
        responses = [
            _assist(None, calls=[ToolCall(name="escalate", arguments={"to": "oncall"})]),
            _assist("oh no"),
        ]
        playbook = _playbook()
        hook = PlaybookEnforcerHook(playbook)
        agent = Agent(
            model=_ScriptedModel(responses),
            tools=[search, classify, escalate],
            hooks=[hook],
            termination=MaxIterations(2),
            max_iterations=4,
        )
        result = agent.run_sync("triage incident")

        # The tool call was blocked → ToolExecution recorded with a
        # "PlaybookEnforcer blocked" content; body never executed.
        executions = list(result.tool_executions)
        assert any("PlaybookEnforcer blocked" in (e.result or "") for e in executions), [
            e.result for e in executions
        ]
        # The plan stays on step-1 (search hasn't been called).
        assert hook.enforcer.current_step is not None
        assert hook.enforcer.current_step.id == "step-1"
        # And the violation is recorded.
        assert any(v.tool_name == "escalate" for v in hook.enforcer.violations)


# =============================================================================
# allow_extra_tools=True lets unrelated calls through
# =============================================================================


class TestAllowExtra:
    def test_extra_tools_allowed_when_flag_set(self):
        responses = [
            _assist(None, calls=[ToolCall(name="chitchat", arguments={"msg": "hi"})]),
            _assist(None, calls=[ToolCall(name="search", arguments={"query": "X"})]),
            _assist("ok"),
        ]
        playbook = _playbook(allow_extra=True)
        hook = PlaybookEnforcerHook(playbook)
        agent = Agent(
            model=_ScriptedModel(responses),
            tools=[search, classify, escalate, chitchat],
            hooks=[hook],
            termination=MaxIterations(4),
            max_iterations=6,
        )
        result = agent.run_sync("go")

        # chitchat ran (not blocked); search ran and step-1 completed.
        names = [te.tool_name for te in result.tool_executions]
        assert "chitchat" in names
        assert "search" in names
        # step-1 should be complete after search ran.
        assert "step-1" in hook.enforcer.plan.completed_steps
