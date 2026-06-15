# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests: the security example notebooks actually run (offline).

These execute the shipped examples end-to-end via a subprocess and assert a
clean exit plus expected output. They need no model or credentials — the
examples drive `Target.from_callable` — so they run in the default suite and
guard the examples against API drift.
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
    ("notebook", "expected"),
    [
        ("notebook_75_agent_red_team.py", "FINDING"),
        ("notebook_76_redteam_support_bot.py", "FINDING"),
        ("notebook_77_ci_security_gate.py", "coverage="),
        ("notebook_78_verify_findings.py", "SURVIVES"),
        ("notebook_79_soc_alert_triage.py", "Alert"),
        ("notebook_80_model_fingerprint.py", "fingerprint"),
        ("notebook_81_ir_audit_trail.py", "Incident Response"),
        ("notebook_82_investigate_with_ctx.py", "logs.search"),
    ],
)
def test_security_example_runs(notebook: str, expected: str) -> None:
    result = _run(notebook)
    assert result.returncode == 0, f"{notebook} exited {result.returncode}:\n{result.stderr}"
    assert expected in result.stdout, f"{notebook} stdout missing {expected!r}:\n{result.stdout}"


def test_redteam_examples_show_findings_and_abstentions() -> None:
    # The defining behaviour, end-to-end: a vulnerable target yields findings,
    # a hardened one abstains.
    out = _run("notebook_76_redteam_support_bot.py").stdout
    assert "FINDING" in out
    assert "ABSTAIN" in out
