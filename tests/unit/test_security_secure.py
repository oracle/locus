# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for secure-by-default agents (GovernanceProfile / governed_agent / AuditHook)."""

from __future__ import annotations

from tulip.agent.config import GroundingConfig
from tulip.control import (
    AuditHook,
    AuditTrail,
    GovernanceProfile,
    GovernedAgent,
    governed_agent,
)
from tulip.hooks.builtin.guardrails import GuardrailsHook


def test_security_profile_defaults_all_on() -> None:
    profile = GovernanceProfile()
    assert profile.grounding
    assert profile.guardrails
    assert profile.audit


def test_secure_agent_wires_grounding_guardrails_and_audit() -> None:
    secured = governed_agent(model="openai:gpt-4o", tools=[])
    assert isinstance(secured, GovernedAgent)
    # grounding on by default -> a GroundingConfig on the agent
    assert isinstance(secured.agent.config.grounding, GroundingConfig)
    hooks = secured.agent.config.hooks
    assert any(isinstance(h, GuardrailsHook) for h in hooks)
    assert any(isinstance(h, AuditHook) for h in hooks)
    assert isinstance(secured.audit_trail, AuditTrail)


def test_secure_agent_respects_disabled_profile() -> None:
    profile = GovernanceProfile(grounding=False, guardrails=False, audit=False)
    secured = governed_agent(model="openai:gpt-4o", tools=[], profile=profile)
    assert secured.agent.config.grounding is None
    hooks = secured.agent.config.hooks
    assert not any(isinstance(h, GuardrailsHook) for h in hooks)
    assert not any(isinstance(h, AuditHook) for h in hooks)


def test_secure_agent_reuses_provided_trail_and_extra_hooks() -> None:
    trail = AuditTrail()
    secured = governed_agent(model="openai:gpt-4o", audit_trail=trail)
    assert secured.audit_trail is trail


async def test_audit_hook_records_lifecycle_to_chain() -> None:
    trail = AuditTrail()
    hook = AuditHook(trail)

    sentinel = object()
    returned = await hook.on_before_invocation("red-team the bot", sentinel)  # type: ignore[arg-type]
    assert returned is sentinel  # passthrough of state
    await hook.on_after_invocation(sentinel, success=True)  # type: ignore[arg-type]

    assert len(trail) == 2
    assert trail.records()[0].event_type == "agent.invocation.start"
    assert trail.records()[1].payload == {"success": True}
    assert trail.verify()


async def test_audit_hook_records_tool_calls() -> None:
    trail = AuditTrail()
    hook = AuditHook(trail)

    class _Before:
        tool_name = "scan_endpoint"

    class _After:
        tool_name = "scan_endpoint"
        error = None

    await hook.on_before_tool_call(_Before())  # type: ignore[arg-type]
    await hook.on_after_tool_call(_After())  # type: ignore[arg-type]

    assert len(trail) == 2
    assert trail.records()[0].payload == {"tool": "scan_endpoint"}
    assert trail.records()[1].payload == {"tool": "scan_endpoint", "error": False}
    assert trail.verify()


async def test_governed_agent_arun_delegates_to_inner() -> None:
    """GovernedAgent.arun forwards to the wrapped agent's async arun
    (thread-free path for browser/WASM)."""

    class _FakeAgent:
        async def arun(self, prompt: str, **kwargs: object) -> str:
            return f"arun:{prompt}"

        def run_sync(self, prompt: str, **kwargs: object) -> str:
            return f"sync:{prompt}"

    ga = GovernedAgent(agent=_FakeAgent(), audit_trail=AuditTrail(), profile=GovernanceProfile())
    assert await ga.arun("hello") == "arun:hello"
    assert ga.run_sync("hello") == "sync:hello"
