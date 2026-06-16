# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Risk-and-approval gate that runs between protocol selection and compile.

The policy gate is intentionally tiny — it produces one of three
verdicts (allow / require_approval / deny) from the
:class:`~locus.router.goal_frame.GoalFrame` and the chosen
:class:`~locus.router.protocol.Protocol`. The compiler honours the
verdict; the gate itself doesn't touch any locus primitive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from locus.router.goal_frame import GoalFrame, Risk
from locus.router.protocol import Protocol


class PolicyDeniedError(RuntimeError):
    """Raised when the policy gate refuses to compile a runnable."""


class PolicyVerdict(BaseModel):
    """One of three outcomes from :meth:`PolicyGate.check`."""

    allow: bool = Field(..., description="False for an outright deny.")
    require_approval: bool = Field(
        default=False,
        description=(
            "True when the runnable must be wrapped with an approval interrupt before execution."
        ),
    )
    reason: str = Field(default="", description="Human-readable explanation.")

    model_config = {"frozen": True}


class PolicyGate:
    """Pre-flight risk check for a (frame, protocol) pair.

    Two thresholds:

    * ``max_risk`` — anything strictly above is denied.
    * ``require_approval_above`` — anything strictly above (and within
      ``max_risk``) is allowed but flagged for an approval interrupt.

    Defaults: ``max_risk=Risk.HIGH`` (nothing denied by default) and
    ``require_approval_above=Risk.MEDIUM`` (HIGH-risk frames need
    explicit approval).
    """

    def __init__(
        self,
        *,
        max_risk: Risk = Risk.HIGH,
        require_approval_above: Risk = Risk.MEDIUM,
    ) -> None:
        self._max_risk = max_risk
        self._approval_above = require_approval_above

    def check(self, frame: GoalFrame, protocol: Protocol) -> PolicyVerdict:
        # Hard ceiling — frame.risk above the gate's max_risk is denied.
        if frame.risk > self._max_risk:
            return PolicyVerdict(
                allow=False,
                require_approval=False,
                reason=(
                    f"frame.risk={frame.risk.value!r} exceeds gate max_risk="
                    f"{self._max_risk.value!r}."
                ),
            )

        # Protocol-level ceiling — protocol cannot accept this risk.
        if frame.risk > protocol.risk_max:
            return PolicyVerdict(
                allow=False,
                require_approval=False,
                reason=(
                    f"protocol {protocol.id!r} caps risk at "
                    f"{protocol.risk_max.value!r}; frame is {frame.risk.value!r}."
                ),
            )

        # Caller asked for approval, or risk is above the approval threshold.
        if frame.approval_required or frame.risk > self._approval_above:
            return PolicyVerdict(
                allow=True,
                require_approval=True,
                reason=(
                    "approval required: "
                    + (
                        "frame.approval_required=True"
                        if frame.approval_required
                        else f"frame.risk={frame.risk.value!r} above "
                        f"approval threshold {self._approval_above.value!r}"
                    )
                ),
            )

        return PolicyVerdict(allow=True, require_approval=False, reason="ok")
