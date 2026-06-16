# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 50: Evaluation — score an agent against a test suite.

Treat the agent like any other piece of code. Declare cases, run them,
read the report. Locus ships a small, dependency-free harness so you
don't need an external eval framework for the common cases.

- EvalCase declares prompt plus expected substrings (positive or negative).
- EvalRunner runs the agent against every case.
- EvalReport summarises pass/fail counts and an average score.

Run it
    # Default: OCI Generative AI auto-detected from ~/.oci/config
    python examples/notebook_55_evaluation.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_55_evaluation.py
"""

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.evaluation import EvalCase, EvalRunner


def example_evaluation():
    """Run a systematic evaluation of an agent."""
    print("=== Agent Evaluation ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a helpful assistant. Answer concisely.",
            max_iterations=3,
            model=model,
        )
    )

    cases = [
        EvalCase(
            name="basic_knowledge",
            prompt="What is the capital of France?",
            expected_output_contains=["paris"],
            max_iterations=3,
        ),
        EvalCase(
            name="math",
            prompt="What is 15 * 7?",
            expected_output_contains=["105"],
        ),
        EvalCase(
            name="no_hallucination",
            prompt="What is the capital of France?",
            expected_output_not_contains=["berlin", "london"],
        ),
    ]

    runner = EvalRunner(agent=agent)
    report = runner.run(cases)

    print(report.summary())
    print(f"\nTotal: {report.total_cases}, Passed: {report.passed}, Failed: {report.failed}")
    print(f"Average score: {report.avg_score:.2f}")


if __name__ == "__main__":
    example_evaluation()
