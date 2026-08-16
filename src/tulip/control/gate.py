# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Put the admission gate around a Tulip tool.

`tulip-frameworks` has shipped `gate_langchain_tool`, `gate_crewai_tool` and
their siblings for a while: wrap one tool, and from then on the model's
decision to call it goes through :func:`~tulip.control.admit` before anything
happens. One line, no rebuild.

Building on Tulip itself, you hand-wrote it::

    async def safe_refund(order_id: str, usd: float):
        try:
            return await admit(
                Action(name="refund", asset=order_id, ...),
                lambda: payments.refund(order_id, usd),
                policy=policy, trail=trail,
            )
        except AdmissionError as e:
            notify_oncall(e.decision)

That is the README's own example, and it is correct — but it means a LangChain
user got better ergonomics from Tulip than a Tulip user did, on the one
feature the project is built around. Everything needed was already here:
:mod:`tulip.control.action` was promoted into core in 2.3.0 precisely so the
SDK and the bridges could share one derivation. Only the bridges ever used it.

    from tulip.control import ControlPolicy, gate_tool

    agent = Agent(model=model, tools=[
        lookup_order,                                    # read-only, ungated
        gate_tool(issue_refund, policy=ControlPolicy()), # gated
    ])

The returned tool keeps the original's name, description and parameter schema,
so the model sees no difference and nothing else in the agent changes.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from tulip.control.action import ActionSpec, resolve_action
from tulip.security.admit import AdmissionError, admit


if TYPE_CHECKING:
    from tulip.security.audit import AuditTrail
    from tulip.security.findings import Evidence
    from tulip.security.policy import ControlPolicy
    from tulip.security.verify import VerificationResult
    from tulip.tools.decorator import Tool


__all__ = ["ApprovalBridge", "gate_tool"]


@runtime_checkable
class ApprovalBridge(Protocol):
    """Submit a held action for out-of-band approval, and check its state.

    A structural Protocol, deliberately: it has no import-time dependency on
    anything, so the same object satisfies this and ``tulip-frameworks``'s
    bridge of the same name. A gateway approval broker matches it in shape.

    Without one, a held action tells the model it was held and stops there —
    true, and not actionable. With one, the refusal carries an id the agent
    can poll while a human decides on a channel the agent cannot reach.
    """

    def submit(self, principal: str, tool: str, args: Mapping[str, Any]) -> str:
        """Record a pending approval; return an id the agent can poll."""
        ...

    def state(self, approval_id: str) -> str | None:
        """Current state for an id — ``"pending"`` / ``"approved"`` / ``"denied"``."""
        ...


#: What the model is handed when the gate refuses. Deliberately identical to
#: what the ``tulip-frameworks`` bridges return, so the same policy reads the
#: same way whether the agent is built on Tulip or wrapped from LangChain.
_REFUSAL_KEYS = ("status", "outcome", "action", "asset", "reason")


def _refusal(
    error: AdmissionError,
    *,
    approval: ApprovalBridge | None = None,
    principal: str = "agent",
    kwargs: Mapping[str, Any] | None = None,
) -> str:
    denied = error.decision.outcome == "deny"
    payload: dict[str, Any] = {
        "status": "denied" if denied else "held_for_approval",
        "outcome": error.decision.outcome,
        "action": error.decision.action.name,
        "asset": error.decision.action.asset,
        "reason": error.decision.reason,
    }
    # A denial is final — there is nothing to poll, and offering an id would
    # invite the agent to wait for a decision that will never come.
    if denied or approval is None:
        return json.dumps(payload)

    payload["approval_id"] = approval.submit(
        principal, error.decision.action.name, dict(kwargs or {})
    )
    payload["next"] = "call approval_status(approval_id) once a human decides"
    return json.dumps(payload)


def gate_tool(
    tool: Tool,
    *,
    policy: ControlPolicy,
    action: ActionSpec | None = None,
    trail: AuditTrail | None = None,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    on_refusal: Literal["return", "raise"] = "return",
    approval: ApprovalBridge | None = None,
    principal: str = "agent",
) -> Tool:
    """Return a copy of ``tool`` whose call goes through :func:`admit` first.

    Args:
        tool: The tool to gate. Not modified — an ungated reference stays
            usable, which matters when the same function is called by trusted
            code elsewhere.
        policy: The :class:`ControlPolicy` to weigh the call against.
        action: How to turn a call into an :class:`Action`. A constant
            ``Action`` when risk does not vary, or ``(name, kwargs) -> Action``
            when it does — the usual case, since a $12 refund and a $4,000,000
            refund differ only in their arguments. ``None`` uses
            :func:`~tulip.control.default_action`, which tags the action with
            the tool's own name so a policy can still gate it by name.
        trail: Records every decision, allowed or not. Omit and decisions are
            weighed but not written down.
        finding: Grounded evidence supporting the action, when the policy
            requires one.
        verdict: A verification result, when the policy sets
            ``require_verification_score``.
        approval: Where to submit an action held for a human. Without one, a
            hold tells the model it was held and stops there — true, and not
            actionable. With one, the refusal carries an ``approval_id`` the
            agent can poll. A denial never gets an id: it is final, and
            offering one would invite the agent to wait for a decision that is
            not coming.
        principal: Who the held action is attributed to on the approval.
        on_refusal: ``"return"`` hands the model a JSON refusal naming the
            outcome and the reason, so it can explain itself to the user and
            the run continues. ``"raise"`` re-raises
            :class:`AdmissionError` for a caller that would rather stop.

    Returns:
        A new :class:`~tulip.tools.decorator.Tool`. Name, description and
        parameter schema are the original's, so it is a drop-in wherever the
        original was passed — the model cannot tell the difference, which is
        the point: the gate is not something the model can be talked around.

    ``"return"`` is the default because a refusal is information the agent can
    act on. An exception ends the run, and "the refund was held for a human" is
    something the user should hear rather than a stack trace.
    """
    from tulip.tools.decorator import Tool  # noqa: PLC0415 — avoids a cycle

    inner = tool.fn
    # A sandboxed tool must keep running in its sandbox. `Tool.execute` returns
    # early for `sandbox is not None` and never reaches `fn`, so the two cannot
    # simply be stacked: carrying the sandbox onto the wrapper would skip the
    # gate, and dropping it -- as the first version of this did -- silently
    # moves the body back onto the host. Both fail quietly, which for a
    # security feature is the worst available outcome.
    #
    # They compose in one order only: gate first, then hand the admitted call
    # to the ORIGINAL tool, whose own `execute` still does the sandboxing.
    sandboxed = tool.sandbox is not None

    async def gated(**kwargs: Any) -> Any:
        async def perform() -> Any:
            if sandboxed:
                return await tool.execute(**kwargs)
            result = inner(**kwargs)
            return await result if inspect.isawaitable(result) else result

        try:
            return await admit(
                resolve_action(action, tool.name, kwargs),
                perform,
                policy=policy,
                finding=finding,
                verdict=verdict,
                trail=trail,
            )
        except AdmissionError as error:
            if on_refusal == "raise":
                raise
            return _refusal(error, approval=approval, principal=principal, kwargs=kwargs)

    # `sandbox` is deliberately not set on the wrapper: it would short-circuit
    # `execute` and skip the gate. The sandbox is not lost — `perform` above
    # delegates to the original tool, which still has it.
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        fn=gated,
        idempotent=tool.idempotent,
        labels=tool.labels,
    )
