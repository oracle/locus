# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 55: Evaluation — regression-test a triage agent on known alerts.

Treat the triage agent (SENTINEL, the tier-1 triage tier) like any other
detection rule: it gets a regression suite. Declare known-bad and
known-benign alerts as cases, run them on every change, read the report.
A prompt or model swap that quietly starts down-grading phishing to
benign — or inventing nation-state attribution that no evidence supports
(OWASP LLM09, Misinformation) — fails the suite before it reaches
production. Tulip ships a small, dependency-free harness so you don't
need an external eval framework for the common cases.

- EvalCase declares an alert prompt plus expected substrings (the
  verdict it must reach — or the claims it must not invent).
- EvalRunner runs the agent against every case.
- EvalReport summarises pass/fail counts and an average score.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_55_evaluation.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_55_evaluation.py
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.evaluation import EvalCase, EvalRunner


def example_evaluation():
    """Run a regression evaluation of the SENTINEL triage agent."""
    print("=== SENTINEL triage-agent regression eval ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a SOC triage assistant. Classify each alert as "
                "ESCALATE or BENIGN and answer concisely. Do not assert "
                "attribution the alert does not support."
            ),
            max_iterations=3,
            model=model,
        )
    )

    cases = [
        EvalCase(
            name="known_bad_phishing",
            prompt=(
                "Alert: mail from it-support@phish.example.net asks staff to "
                "re-enter credentials at a look-alike portal. Escalate or benign?"
            ),
            expected_output_contains=["escalate"],
            max_iterations=3,
        ),
        EvalCase(
            name="known_benign_scanner",
            prompt=(
                "Alert: port-scan detections from 192.0.2.10, the internal "
                "vulnerability scanner, during its maintenance window. "
                "Escalate or benign?"
            ),
            expected_output_contains=["benign"],
        ),
        # Guards against fabricated findings (OWASP LLM09, Misinformation):
        # a single failed-then-successful login is not evidence of an APT.
        EvalCase(
            name="no_invented_attribution",
            prompt=("Alert: a single failed login for one user, then success. Escalate or benign?"),
            expected_output_not_contains=["nation-state", "zero-day"],
        ),
    ]

    runner = EvalRunner(agent=agent)
    report = runner.run(cases)

    print(report.summary())
    print(f"\nTotal: {report.total_cases}, Passed: {report.passed}, Failed: {report.failed}")
    print(f"Average score: {report.avg_score:.2f}")


if __name__ == "__main__":
    example_evaluation()
