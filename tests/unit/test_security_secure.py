# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for secure-by-default agents (SecurityProfile / secure_agent / AuditHook)."""

from __future__ import annotations

from tulip.agent.config import GroundingConfig
from tulip.hooks.builtin.guardrails import GuardrailsHook
from tulip.security import (
    AuditHook,
    AuditTrail,
    SecureAgent,
    SecurityProfile,
    secure_agent,
)


def test_security_profile_defaults_all_on() -> None:
    profile = SecurityProfile()
    assert profile.grounding
    assert profile.guardrails
    assert profile.audit


def test_secure_agent_wires_grounding_guardrails_and_audit() -> None:
    secured = secure_agent(model="openai:gpt-4o", tools=[])
    assert isinstance(secured, SecureAgent)
    # grounding on by default -> a GroundingConfig on the agent
    assert isinstance(secured.agent.config.grounding, GroundingConfig)
    hooks = secured.agent.config.hooks
    assert any(isinstance(h, GuardrailsHook) for h in hooks)
    assert any(isinstance(h, AuditHook) for h in hooks)
    assert isinstance(secured.audit_trail, AuditTrail)


def test_secure_agent_respects_disabled_profile() -> None:
    profile = SecurityProfile(grounding=False, guardrails=False, audit=False)
    secured = secure_agent(model="openai:gpt-4o", tools=[], profile=profile)
    assert secured.agent.config.grounding is None
    hooks = secured.agent.config.hooks
    assert not any(isinstance(h, GuardrailsHook) for h in hooks)
    assert not any(isinstance(h, AuditHook) for h in hooks)


def test_secure_agent_reuses_provided_trail_and_extra_hooks() -> None:
    trail = AuditTrail()
    secured = secure_agent(model="openai:gpt-4o", audit_trail=trail)
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
