# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission for a box runner, decided by the gateway.

In a box the runner is not trusted to judge its own calls: the gateway is. So
:class:`RemoteGate` is the runner's admission hook — it sits on the agent's
``before_tool_call`` seam (the same seam playbook enforcers and ``admit()``
gates use) at the highest priority and asks the gateway about every call
before the tool body runs.

It sends **only** what the model asked for::

    POST /v1/admit
    {"run_id": ..., "call_id": ..., "tool": ..., "arguments": {...}, "approval_id": ...?}

Never labels, an asset, a blast radius or an environment. The gateway builds
the action from its own copy of the run's tool surface, so a runner that lies
about what a tool does changes nothing. The answer:

- ``allow`` — the call runs. The answer's one-shot ``decision_token`` is kept
  for the call (:meth:`RemoteGate.decision_token`) and, when the gate was given
  a ``token_argument``, passed to the tool body in
  ``BeforeToolCallEvent.secret_arguments`` — never checkpointed, never in the
  conversation. MCP and API calls present it to the box guard on the wire.
- ``require_human`` — the call is held. The gate waits up to ``hold_wait_s``
  for a person to decide (``GET /v1/admit/approval/{id}``); approved, it asks
  again with the ``approval_id`` (the gateway checks the approval was granted
  for these exact arguments) and runs on the new answer. Still undecided, the
  call is cancelled with the hold's reason and recorded in :attr:`~RemoteGate.held`,
  so the runner can park the run until the decision comes.
- ``deny`` — the call is cancelled with the gateway's reason.

The gate **fails closed**: if the gateway cannot be reached or refuses the
request, the call is cancelled. There is no local fallback policy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tulip.hooks.provider import HookPriority, HookProvider
from tulip.runner.client import GatewayClient, GatewayError, GatewayUnavailable


if TYPE_CHECKING:
    from tulip.hooks.provider import BeforeToolCallEvent


__all__ = ["AdmitResult", "Hold", "RemoteGate"]

Outcome = Literal["allow", "require_human", "deny"]

#: Approval states after which no decision can still come.
_SETTLED = frozenset({"approved", "denied", "consumed", "expired", "abandoned"})


@dataclass(frozen=True)
class AdmitResult:
    """The gateway's answer for one call. ``decision_token`` is never repr'd."""

    outcome: Outcome
    reason: str
    audit_id: str = ""
    approval_id: str | None = None
    decision_token: str | None = field(default=None, repr=False)
    labels: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    policy_fingerprint: str | None = None

    @property
    def allowed(self) -> bool:
        """Whether the call may run now."""
        return self.outcome == "allow"

    @classmethod
    def from_body(cls, body: Any) -> AdmitResult:
        """Read the gateway's JSON answer; an unknown outcome is a denial."""
        if not isinstance(body, dict):
            return cls(outcome="deny", reason="the gateway's answer was not an object")
        outcome = str(body.get("outcome") or "")
        if outcome not in ("allow", "require_human", "deny"):
            return cls(
                outcome="deny", reason=f"the gateway answered an unknown outcome {outcome!r}"
            )
        # ``allowed`` must agree: an "allow" the gateway marks not allowed is not one.
        if outcome == "allow" and body.get("allowed") is False:
            return cls(outcome="deny", reason="the gateway's answer contradicts itself")
        return cls(
            outcome=outcome,  # type: ignore[arg-type]  # checked just above
            reason=str(body.get("reason") or ""),
            audit_id=str(body.get("audit_id") or ""),
            approval_id=body.get("approval_id") or None,
            decision_token=body.get("decision_token") or None,
            labels=tuple(str(label) for label in body.get("labels") or ()),
            checks=tuple(str(check) for check in body.get("checks") or ()),
            policy_fingerprint=body.get("policy_fingerprint") or None,
        )


@dataclass(frozen=True)
class Hold:
    """A call the gateway held for a person, still undecided when the gate gave up waiting."""

    call_id: str
    tool: str
    approval_id: str
    reason: str


class RemoteGate(HookProvider):
    """The runner's admission hook: every tool call is decided by the gateway.

    Args:
        client: The runner's :class:`~tulip.runner.client.GatewayClient`.
        hold_wait_s: How long to wait for a person on a held call before
            cancelling it and recording a :class:`Hold`. 0 records it at once.
        poll_interval_s: Seconds between checks of a held call's approval.
        token_argument: When set, an admitted call's decision token is passed
            to the tool body under this argument name, through
            ``secret_arguments``.
    """

    def __init__(
        self,
        client: GatewayClient,
        *,
        hold_wait_s: float = 0.0,
        poll_interval_s: float = 2.0,
        token_argument: str | None = None,
    ) -> None:
        self._client = client
        self.hold_wait_s = max(0.0, hold_wait_s)
        self.poll_interval_s = max(0.01, poll_interval_s)
        self.token_argument = token_argument
        self._tokens: dict[str, str] = {}
        #: Calls held for a person that were still undecided; the runner parks on these.
        self.held: list[Hold] = []

    @property
    def priority(self) -> int:
        # First of all hooks: nothing else sees a call the gateway has not admitted.
        return HookPriority.SECURITY_MIN

    def register_hooks(self) -> dict[str, bool]:
        return {name: name == "on_before_tool_call" for name in super().register_hooks()}

    async def admit(
        self,
        *,
        call_id: str,
        tool: str,
        arguments: dict[str, Any],
        approval_id: str | None = None,
    ) -> AdmitResult:
        """Ask the gateway about one call. Raises the client's errors as they come."""
        body: dict[str, Any] = {
            "run_id": self._client.run_id,
            "call_id": call_id,
            "tool": tool,
            "arguments": arguments,
        }
        if approval_id is not None:
            body["approval_id"] = approval_id
        return AdmitResult.from_body(await self._client.request("POST", "/v1/admit", json=body))

    async def approval_state(self, approval_id: str) -> str:
        """The current state of a hold: ``pending``, ``approved``, ``denied``, …"""
        view = await self._client.request("GET", f"/v1/admit/approval/{approval_id}")
        return str((view or {}).get("state") or "pending")

    async def wait_for_approval(self, approval_id: str, *, timeout_s: float) -> str:
        """Poll a hold until it settles or ``timeout_s`` passes; return its last state."""
        deadline = time.monotonic() + timeout_s
        state = await self.approval_state(approval_id)
        while state not in _SETTLED and time.monotonic() < deadline:
            await asyncio.sleep(min(self.poll_interval_s, max(0.0, deadline - time.monotonic())))
            state = await self.approval_state(approval_id)
        return state

    def decision_token(self, call_id: str) -> str | None:
        """Take the decision token of an admitted call (each is given out once)."""
        return self._tokens.pop(call_id, None)

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        try:
            result = await self._decide(event.tool_call_id, event.tool_name, dict(event.arguments))
        except (GatewayUnavailable, GatewayError) as exc:
            event.cancel = f"not run: the gateway could not decide this call ({exc})"
            return
        if result.allowed:
            if result.decision_token is not None:
                self._tokens[event.tool_call_id] = result.decision_token
                if self.token_argument:
                    event.secret_arguments = {
                        **event.secret_arguments,
                        self.token_argument: result.decision_token,
                    }
            return
        if result.outcome == "require_human" and result.approval_id:
            self.held.append(
                Hold(
                    call_id=event.tool_call_id,
                    tool=event.tool_name,
                    approval_id=result.approval_id,
                    reason=result.reason,
                )
            )
            event.cancel = f"held for a person's approval ({result.approval_id}): {result.reason}"
            return
        event.cancel = f"denied by the gateway: {result.reason or result.outcome}"

    async def _decide(self, call_id: str, tool: str, arguments: dict[str, Any]) -> AdmitResult:
        result = await self.admit(call_id=call_id, tool=tool, arguments=arguments)
        if result.outcome != "require_human" or not result.approval_id or not self.hold_wait_s:
            return result
        state = await self.wait_for_approval(result.approval_id, timeout_s=self.hold_wait_s)
        if state == "approved":
            # Ask again, naming the approval: the gateway admits only the call it approved.
            return await self.admit(
                call_id=call_id, tool=tool, arguments=arguments, approval_id=result.approval_id
            )
        if state in _SETTLED:
            return AdmitResult(
                outcome="deny",
                reason=f"the hold was {state}",
                approval_id=result.approval_id,
                audit_id=result.audit_id,
            )
        return result
