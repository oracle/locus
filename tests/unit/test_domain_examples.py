# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The non-security domain examples run end-to-end.

The control runtime is domain-neutral: ``notebook_83``..``87`` show ``admit()``
gating payments, infrastructure, customer-support, data-privacy, and cloud
actions. They run offline (no network or credentials) and must exit 0 and hold
the risky action for a human — the same contract the security examples are held
to, so a future change that breaks one fails CI instead of silently shipping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]


def _run(notebook: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, f"examples/{notebook}"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    ("notebook", "domain"),
    [
        ("notebook_83_payment_refund_gate.py", "payments"),
        ("notebook_84_infra_deploy_gate.py", "infra/devops"),
        ("notebook_85_support_account_gate.py", "customer support"),
        ("notebook_86_data_deletion_gate.py", "data/privacy"),
        ("notebook_87_cloud_resource_gate.py", "cloud"),
    ],
)
def test_domain_example_runs(notebook: str, domain: str) -> None:
    """Each domain example runs offline and holds the risky action for a human."""
    proc = _run(notebook)
    assert proc.returncode == 0, f"{notebook} ({domain}) exited {proc.returncode}:\n{proc.stderr}"
    # The whole point of the gate: the high-risk action is held, not auto-run.
    assert "require_human" in proc.stdout, (
        f"{notebook} ({domain}) never held an action for a human:\n{proc.stdout[-500:]}"
    )
