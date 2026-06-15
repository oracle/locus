# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Secure-by-default agents — the floor a trustworthy security agent stands on.

An agent that red-teams or assures *other* AI must itself be trustworthy.
:func:`secure_agent` builds a :class:`tulip.Agent` with the security spine
turned on by default: GSAR grounding (abstain rather than fabricate),
input/output guardrails (PII redaction, injection checks, tool allowlist),
and a tamper-evident :class:`~tulip.security.audit.AuditTrail` of every action.

This is a *supporting property*, not a governance product: the controls make
the working agent's output trustworthy and its actions auditable — they do not
constitute a policy-enforcement plane (cf. Cisco DefenseClaw / Microsoft Agent
Governance Toolkit). Plain :class:`tulip.Agent` stays opt-in for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tulip.agent import Agent  # security already depends on agent (see soc.py) — no cycle
from tulip.hooks import HookPriority, HookProvider
from tulip.hooks.builtin.guardrails import GuardrailsHook
from tulip.security.audit import AuditTrail


if TYPE_CHECKING:
    from tulip.core.state import AgentState
    from tulip.hooks import AfterToolCallEvent, BeforeToolCallEvent


@dataclass(frozen=True)
class SecurityProfile:
    """Which secure-by-default controls a :func:`secure_agent` turns on.

    All on by default — that is what makes the agent secure out of the box.
    """

    grounding: bool = True
    guardrails: bool = True
    audit: bool = True


class AuditHook(HookProvider):
    """Records the agent's lifecycle into a tamper-evident :class:`AuditTrail`."""

    def __init__(
        self,
        trail: AuditTrail,
        *,
        priority: int = HookPriority.OBSERVABILITY_DEFAULT,
    ) -> None:
        self._trail = trail
        self._priority = priority

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def trail(self) -> AuditTrail:
        return self._trail

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
        self._trail.record("agent.invocation.start", {"prompt": prompt[:500]})
        return state

    async def on_after_invocation(self, state: AgentState, success: bool) -> None:
        self._trail.record("agent.invocation.end", {"success": success})

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        self._trail.record("agent.tool.before", {"tool": getattr(event, "tool_name", "")})

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self._trail.record(
            "agent.tool.after",
            {
                "tool": getattr(event, "tool_name", ""),
                "error": bool(getattr(event, "error", None)),
            },
        )

    def register_hooks(self) -> dict[str, bool]:
        return {
            "on_before_invocation": True,
            "on_after_invocation": True,
            "on_before_tool_call": True,
            "on_after_tool_call": True,
        }


@dataclass(frozen=True)
class SecureAgent:
    """A secure-by-default :class:`tulip.Agent` plus its audit trail.

    ``run`` / ``run_sync`` pass through to the wrapped agent; ``audit_trail``
    is the tamper-evident record of everything it did.
    """

    agent: Agent
    audit_trail: AuditTrail
    profile: SecurityProfile

    def run(self, prompt: str, **kwargs: Any) -> Any:
        return self.agent.run(prompt, **kwargs)

    def run_sync(self, prompt: str, **kwargs: Any) -> Any:
        return self.agent.run_sync(prompt, **kwargs)


def secure_agent(
    model: Any = None,
    tools: list[Any] | None = None,
    *,
    system_prompt: str | None = None,
    profile: SecurityProfile | None = None,
    audit_trail: AuditTrail | None = None,
    hooks: list[Any] | None = None,
    **kwargs: Any,
) -> SecureAgent:
    """Build a secure-by-default agent: grounded, guarded, and audited.

    Args:
        model: Model string or instance (as :class:`tulip.Agent`).
        tools: Tools available to the agent.
        system_prompt: System prompt.
        profile: Which controls to enable (default: all on).
        audit_trail: Reuse an existing trail; one is created if omitted.
        hooks: Extra hooks to add alongside the security hooks.
        **kwargs: Passed through to :class:`tulip.Agent`.

    Returns:
        A :class:`SecureAgent` wrapping the configured agent and its audit trail.
    """
    profile = profile or SecurityProfile()
    # NB: an empty AuditTrail is falsy (len 0), so check identity, not truthiness.
    trail = audit_trail if audit_trail is not None else AuditTrail()
    hook_list: list[Any] = list(hooks or [])
    if profile.guardrails:
        hook_list.append(GuardrailsHook())
    if profile.audit:
        hook_list.append(AuditHook(trail))
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        grounding=profile.grounding,
        hooks=hook_list,
        **kwargs,
    )
    return SecureAgent(agent=agent, audit_trail=trail, profile=profile)


__all__ = ["AuditHook", "SecureAgent", "SecurityProfile", "secure_agent"]
