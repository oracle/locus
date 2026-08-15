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
from typing import TYPE_CHECKING, Any, Literal

from tulip.control.action import ActionSpec, resolve_action
from tulip.security.admit import AdmissionError, admit


if TYPE_CHECKING:
    from tulip.security.audit import AuditTrail
    from tulip.security.findings import Evidence
    from tulip.security.policy import ControlPolicy
    from tulip.security.verify import VerificationResult
    from tulip.tools.decorator import Tool


__all__ = ["gate_tool"]

#: What the model is handed when the gate refuses. Deliberately identical to
#: what the ``tulip-frameworks`` bridges return, so the same policy reads the
#: same way whether the agent is built on Tulip or wrapped from LangChain.
_REFUSAL_KEYS = ("status", "outcome", "action", "asset", "reason")


def _refusal(error: AdmissionError) -> str:
    decision = error.decision
    return json.dumps(
        {
            "status": "denied" if decision.outcome == "deny" else "held_for_approval",
            "outcome": decision.outcome,
            "action": decision.action.name,
            "asset": decision.action.asset,
            "reason": decision.reason,
        }
    )


def gate_tool(
    tool: Tool,
    *,
    policy: ControlPolicy,
    action: ActionSpec | None = None,
    trail: AuditTrail | None = None,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    on_refusal: Literal["return", "raise"] = "return",
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

    async def gated(**kwargs: Any) -> Any:
        async def perform() -> Any:
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
            return _refusal(error)

    # Carried over deliberately: the model is shown the same tool, and a
    # policy matching on `labels` must still see them. `sandbox` is not
    # carried — the gate wraps the call, and sandboxing is about where the
    # body runs, so a sandboxed tool keeps its own execution path by staying
    # unwrapped rather than being silently double-handled.
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        fn=gated,
        idempotent=tool.idempotent,
        labels=tool.labels,
    )
