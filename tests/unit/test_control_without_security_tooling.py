# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The control core stands on its own.

The admission gate, policy, audit trail and grounding layer live in
``tulip.control`` and must never pull in the security-domain tooling, which is
a separate, opt-in distribution (``tulip_security``). Each check runs in a
fresh interpreter so ``sys.modules`` shows exactly what the import chain
loaded, both with that distribution importable and with it blocked outright.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


_EXERCISE_THE_CORE = textwrap.dedent(
    """
    import asyncio
    import sys

    {preamble}

    import tulip
    from tulip.control import (
        Action,
        AdmissionError,
        AuditTrail,
        ControlPolicy,
        Severity,
        VerificationResult,
        admit,
        gate_tool,
        ground_finding,
        is_finding,
    )
    from tulip.reasoning.gsar import Claim, EvidenceType, Partition
    from tulip.tools.decorator import tool

    # Lazy top-level names resolve through tulip.control, not tulip.security.
    assert tulip.Evidence is tulip.control.Evidence
    assert tulip.ground_finding is ground_finding

    trail = AuditTrail()
    strong = VerificationResult(survives=True, confidence=0.95, evidence_quality=0.95)

    async def perform() -> str:
        return "refunded"

    async def main() -> None:
        ran = await admit(
            Action(name="refund", asset="order:1", environment="staging"),
            perform,
            policy=ControlPolicy(),
            verdict=strong,
            trail=trail,
        )
        assert ran == "refunded"
        try:
            await admit(
                Action(name="refund", asset="order:2", environment="production"),
                perform,
                policy=ControlPolicy(),
                verdict=strong,
                trail=trail,
            )
        except AdmissionError:
            pass
        else:
            raise AssertionError("production action was not held")

    asyncio.run(main())
    assert len(trail.records()) == 2 and trail.verify()

    @tool
    def refund(order_id: str) -> str:
        "Refund an order."
        return order_id

    assert gate_tool(refund, policy=ControlPolicy()).name == "refund"

    finding = ground_finding(
        title="t",
        description="d",
        severity=Severity.LOW,
        asset="a",
        remediation="r",
        partition=Partition(
            grounded=[Claim(text="x", type=EvidenceType.TOOL_MATCH, evidence_refs=["tool:x"])]
        ),
    )
    assert is_finding(finding)

    leaked = sorted(
        name for name, module in sys.modules.items()
        if module is not None
        and (name == "tulip_security" or name.startswith(("tulip_security.", "tulip.security")))
    )
    assert not leaked, leaked
    print("ok")
    """
)


@pytest.mark.parametrize(
    "preamble",
    [
        pytest.param("", id="security-package-importable"),
        pytest.param("sys.modules['tulip_security'] = None", id="security-package-blocked"),
    ],
)
def test_control_core_runs_without_loading_security_tooling(preamble: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _EXERCISE_THE_CORE.format(preamble=preamble)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
