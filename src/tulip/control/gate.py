# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Put the admission gate around a Tulip tool.

Wrap one tool, and from then on the model's decision to call it goes through
:func:`~tulip.control.admit` before anything happens. One line, no rebuild.

Without it, you hand-write the same thing around every tool::

    async def safe_refund(order_id: str, usd: float):
        try:
            return await admit(
                Action(name="refund", asset=order_id, ...),
                lambda: payments.refund(order_id, usd),
                policy=policy, trail=trail,
            )
        except AdmissionError as e:
            notify_oncall(e.decision)

That is correct, but every copy has to build the action the same way.
``gate_tool`` derives it once, through :mod:`tulip.control.action`::

    from tulip.control import ControlPolicy, gate_tool

    agent = Agent(
        model=model,
        tools=[
            lookup_order,  # read-only, ungated
            gate_tool(issue_refund, policy=ControlPolicy()),  # gated
        ],
    )

The returned tool keeps the original's name, description and parameter schema,
so the model sees no difference and nothing else in the agent changes.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from tulip.control.action import ActionSpec, resolve_action
from tulip.control.approvals import ApprovalStore
from tulip.security.admit import AdmissionError, admit
from tulip.security.policy import ApprovalOutcome


if TYPE_CHECKING:
    from tulip.control.spend import SpendLedger
    from tulip.security.audit import AuditTrail
    from tulip.security.findings import Evidence
    from tulip.security.policy import ApprovalDecision, ControlPolicy
    from tulip.security.verify import VerificationResult
    from tulip.tools.decorator import Tool


__all__ = ["ApprovalBridge", "gate_tool"]


@runtime_checkable
class ApprovalBridge(Protocol):
    """Submit a held action for out-of-band approval, and check its state.

    A structural Protocol, deliberately: it has no import-time dependency on
    anything, so any approval queue with these two methods satisfies it.

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


#: What the model is handed when the gate refuses. A public contract: callers
#: and tests read these keys, so keys are only ever added, never renamed.
_REFUSAL_KEYS = ("status", "outcome", "action", "asset", "reason")


def _reason_for(
    error: AdmissionError,
    reason: str | Callable[[ApprovalDecision], str] | None,
) -> str:
    """The refusal sentence: the caller's if they gave one, else the policy's.

    The policy's own reason is a join of the checks that fired -- "blast radius
    3 exceeds the maximum 1", "labels ['large_refund'] are denied by policy".
    That is the right level of detail for the audit trail and for a developer
    reading a log. It is control-plane vocabulary, and a model handed it will
    repeat it verbatim to whoever is on the other end. ``refusal_reason`` is
    how an integrator says what the *user* should hear instead; the policy
    detail is still written to the trail either way.
    """
    if reason is None:
        return error.decision.reason
    return reason(error.decision) if callable(reason) else reason


def _refusal(
    error: AdmissionError,
    *,
    approval: ApprovalBridge | None = None,
    principal: str = "agent",
    kwargs: Mapping[str, Any] | None = None,
    reason: str | Callable[[ApprovalDecision], str] | None = None,
) -> str:
    denied = error.decision.outcome == "deny"
    payload: dict[str, Any] = {
        "status": "denied" if denied else "held_for_approval",
        "outcome": error.decision.outcome,
        "action": error.decision.action.name,
        "asset": error.decision.action.asset,
        "reason": _reason_for(error, reason),
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


def _scope_for(
    spend_scope: str | Callable[[str, dict[str, Any]], str],
    tool_name: str,
    kwargs: Mapping[str, Any],
) -> str:
    """The spend scope for one call."""
    return spend_scope(tool_name, dict(kwargs)) if callable(spend_scope) else spend_scope


def _approval_context(
    policy: ControlPolicy,
    extra: Mapping[str, str] | Callable[[str, dict[str, Any]], Mapping[str, str]] | None,
    tool_name: str,
    kwargs: Mapping[str, Any],
) -> dict[str, str]:
    """The context an approval is bound to: the policy version plus the caller's."""
    context: dict[str, str] = {}
    if policy.version:
        context["policy_version"] = policy.version
    if extra is not None:
        values = extra(tool_name, dict(kwargs)) if callable(extra) else extra
        context.update({str(key): str(value) for key, value in values.items()})
    return context


def gate_tool(
    tool: Tool,
    *,
    policy: ControlPolicy,
    action: ActionSpec | None = None,
    trail: AuditTrail | None = None,
    finding: Evidence | None = None,
    verdict: VerificationResult | None = None,
    on_refusal: Literal["return", "raise", "interrupt"] = "return",
    approval: ApprovalBridge | None = None,
    principal: str = "agent",
    refusal_reason: str | Callable[[ApprovalDecision], str] | None = None,
    approval_context: (
        Mapping[str, str] | Callable[[str, dict[str, Any]], Mapping[str, str]] | None
    ) = None,
    ledger: SpendLedger | None = None,
    spend_scope: str | Callable[[str, dict[str, Any]], str] = "default",
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
        approval_context: What an approval is bound to besides the call itself,
            as a mapping or ``(tool_name, arguments) -> mapping``: a thread, a
            tenant, a case id. It is part of the approval id, as is
            ``policy.version`` when set, so a decision made in one context or
            under one policy version is never redeemed in another.
        ledger: A spend ledger. The gate reads the scope's cumulative spend
            before each decision, for ``policy.spend_limit_usd``, and records
            ``action.cost_usd`` after the call runs.
        spend_scope: The scope spend counts against, as a string or
            ``(tool_name, arguments) -> str``: a customer, a tenant, a month.
        on_refusal: ``"return"`` hands the model a JSON refusal naming the
            outcome and the reason, so it can explain itself to the user and
            the run continues. ``"raise"`` re-raises
            :class:`AdmissionError` for a caller that would rather stop.
            ``"interrupt"`` pauses the run on a ``require_human`` hold: the call
            returns the runtime's interrupt marker, the agent yields an
            ``InterruptEvent`` whose ``metadata`` carries the ``approval_id``, and
            ``agent.resume(..., perform_dangling=True)`` re-issues the call once a
            person has decided on ``approval`` (which must be an
            :class:`~tulip.control.ApprovalStore`). Approved runs the call exactly
            once; denied returns a refusal. A policy ``deny`` never pauses.
        refusal_reason: What the model is told when an action is refused. By
            default it is the policy's own reason, which names the checks that
            fired — accurate, and written in control-plane vocabulary the model
            will repeat verbatim to the end user ("blast radius 3 exceeds the
            maximum 1"). Pass a string, or ``(decision) -> str`` to vary by
            outcome, to say what the user should hear instead. The full policy
            reason is recorded on the audit trail regardless.

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

    if on_refusal == "interrupt" and not isinstance(approval, ApprovalStore):
        raise TypeError(
            'on_refusal="interrupt" needs approval= an ApprovalStore (InMemoryApprovals, '
            "FileApprovals, or your own): a paused run has to find its decision somewhere"
        )

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

    async def hold(
        error: AdmissionError,
        resolved: Any,
        kwargs: dict[str, Any],
        perform_with: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """A ``require_human`` hold in interrupt mode: pause, or act on a decision."""
        store = cast("ApprovalStore", approval)
        context = _approval_context(policy, approval_context, tool.name, kwargs)
        approval_id = store.submit(
            principal,
            tool.name,
            kwargs,
            reason=error.decision.reason,
            labels=sorted(resolved.labels()),
            context=context,
        )
        record = store.get(approval_id)
        where = {"approval_id": approval_id, "action": resolved.name, "asset": resolved.asset}

        if record is not None and record.status in ("approved", "denied"):
            if trail is not None:
                trail.record(
                    "approval-decision",
                    {
                        **where,
                        "verdict": record.status,
                        "decided_by": record.decided_by,
                        "approvers": record.approvers,
                    },
                )
            if record.status == "approved":
                edited = record.approved_arguments
                run_kwargs = dict(edited) if edited is not None else kwargs
                run_action = (
                    resolve_action(action, tool.name, run_kwargs)
                    if edited is not None
                    else resolved
                )
                if edited is not None and trail is not None:
                    trail.record(
                        "approval-edited",
                        {
                            **where,
                            "requested_arguments": dict(kwargs),
                            "approved_arguments": edited,
                        },
                    )

                async def perform_once() -> Any:
                    # Consumed before the side effect: a crash mid-call leaves
                    # the approval spent, never a second execution on replay.
                    store.consume(approval_id)
                    return await perform_with(run_kwargs)

                try:
                    # An edited call is weighed again: an approver cannot edit
                    # an action into one the policy denies.
                    return await admit(
                        run_action,
                        perform_once,
                        policy=policy,
                        finding=finding,
                        verdict=verdict,
                        trail=trail,
                        approved_by=record.decided_by,
                        ledger=ledger,
                        spend_scope=_scope_for(spend_scope, tool.name, run_kwargs),
                    )
                except AdmissionError as denial:
                    store.consume(approval_id)
                    return _refusal(denial, reason=refusal_reason)
            store.consume(approval_id)
            return json.dumps(
                {
                    "status": "denied",
                    "outcome": error.decision.outcome,
                    "action": resolved.name,
                    "asset": resolved.asset,
                    "reason": f"denied by {record.decided_by}",
                    "approval_id": approval_id,
                }
            )

        if trail is not None:
            trail.record("approval-requested", {**where, "principal": principal})
        return json.dumps(
            {
                "__interrupt__": True,
                "question": f"Approve {resolved.name} on {resolved.asset}? "
                + _reason_for(error, refusal_reason),
                "metadata": {
                    **where,
                    "principal": principal,
                    "reason": error.decision.reason,
                    "arguments": dict(kwargs),
                    "approvers": record.approvers if record is not None else [],
                    "context": context,
                },
            },
            default=str,
        )

    async def perform_with(call_kwargs: dict[str, Any]) -> Any:
        if sandboxed:
            return await tool.execute(**call_kwargs)
        result = inner(**call_kwargs)
        return await result if inspect.isawaitable(result) else result

    async def gated(**kwargs: Any) -> Any:
        async def perform() -> Any:
            return await perform_with(kwargs)

        resolved = resolve_action(action, tool.name, kwargs)
        try:
            return await admit(
                resolved,
                perform,
                policy=policy,
                finding=finding,
                verdict=verdict,
                trail=trail,
                ledger=ledger,
                spend_scope=_scope_for(spend_scope, tool.name, kwargs),
            )
        except AdmissionError as error:
            if on_refusal == "raise":
                raise
            if on_refusal == "interrupt" and error.decision.outcome != ApprovalOutcome.DENY:
                return await hold(error, resolved, kwargs, perform_with)
            return _refusal(
                error,
                approval=approval,
                principal=principal,
                kwargs=kwargs,
                reason=refusal_reason,
            )

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
