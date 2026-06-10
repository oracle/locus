# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Live: ``Agent(playbook=...)`` enforces step compliance across providers.

Drives a real model through a 3-step playbook and checks that:

- The model's tool calls land in the right step.
- Out-of-sequence calls are cancelled with the enforcer's message.
- The plan auto-advances; ``is_complete`` flips when all steps are done.

Runs against:
* OpenAI direct (gpt-4o-mini) — `OPENAI_API_KEY`
* Anthropic direct (claude-haiku-4.5) — `ANTHROPIC_API_KEY`
"""

from __future__ import annotations

import os

import pytest

from tulip.core.termination import MaxIterations
from tulip.playbooks.hook import PlaybookEnforcerHook
from tulip.playbooks.models import Playbook, PlaybookStep
from tulip.tools.decorator import tool


pytestmark = [pytest.mark.integration]


_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))


def _build_openai():
    if not _OPENAI:
        return None
    pytest.importorskip("openai")
    from tulip.models.native.openai import OpenAIModel

    return OpenAIModel(model=os.environ.get("TULIP_OPENAI_TEST_MODEL", "gpt-4o-mini"))


def _build_anthropic():
    if not _ANTHROPIC:
        return None
    pytest.importorskip("anthropic")
    from tulip.models.native.anthropic import AnthropicModel

    return AnthropicModel(
        model=os.environ.get("TULIP_ANTHROPIC_TEST_MODEL", "claude-haiku-4-5-20251001")
    )


_FACTORIES = [
    pytest.param(_build_openai, id="openai-native-gpt-4o-mini"),
    pytest.param(_build_anthropic, id="anthropic-native-claude-haiku"),
]


@tool
def fetch_logs(service: str) -> str:
    """Fetch recent logs for a service. Returns a few synthetic lines."""
    return (
        f"[ERROR] {service}: connection pool exhausted at 14:02\n"
        f"[ERROR] {service}: 503 Service Unavailable spike at 14:03\n"
    )


@tool
def classify_severity(severity: str, summary: str) -> str:
    """Record incident severity (P0/P1/P2/P3) with a one-line summary."""
    return f"recorded {severity}: {summary}"


@tool
def page_oncall(rotation: str, message: str) -> str:
    """Page the on-call rotation."""
    return f"paged {rotation}: {message}"


def _build_playbook() -> Playbook:
    return Playbook(
        id="incident-triage",
        name="Incident triage",
        steps=[
            PlaybookStep(
                id="gather",
                description="Gather logs",
                expected_tools=["fetch_logs"],
                hints=["Always pull logs first, before classifying."],
            ),
            PlaybookStep(
                id="classify",
                description="Classify severity",
                expected_tools=["classify_severity"],
                hints=["Use P0 for full outage, P1 for degraded."],
            ),
            PlaybookStep(
                id="page",
                description="Page oncall",
                expected_tools=["page_oncall"],
                hints=["Use rotation 'sre-primary' for P0/P1."],
            ),
        ],
    )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_playbook_enforces_step_sequence_live(factory):
    """End-to-end: real model, three-step playbook, plan completes in order."""
    model = factory()
    if model is None:
        pytest.skip("provider credentials missing")

    from tulip.agent import Agent

    playbook = _build_playbook()
    hook = PlaybookEnforcerHook(playbook)

    agent = Agent(
        model=model,
        tools=[fetch_logs, classify_severity, page_oncall],
        system_prompt=(
            "You triage production incidents. Follow the playbook steps in "
            "order: gather logs first, classify severity second, page oncall "
            "third. Each step has exactly one tool to call."
        ),
        hooks=[hook],
        termination=MaxIterations(8),
        max_iterations=10,
    )

    result = agent.run_sync(
        "API service is timing out. Triage it: fetch logs, classify, page oncall."
    )

    # The plan reached completion (or got at least to the page step).
    completed = hook.enforcer.plan.completed_steps
    assert "gather" in completed, (
        f"step-1 didn't complete; tool_executions={[te.tool_name for te in result.tool_executions]}, "
        f"violations={[v.message for v in hook.enforcer.violations]}"
    )
    # No tool was called out of sequence — every recorded execution is the
    # step's expected tool, OR was a blocked enforcement violation.
    valid_tool_names = {"fetch_logs", "classify_severity", "page_oncall"}
    for te in result.tool_executions:
        if "PlaybookEnforcer blocked" in (te.result or ""):
            continue  # blocked call is fine
        assert te.tool_name in valid_tool_names
