# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration tests for ``Agent(gsar=GSARConfig(...))``.

Exercises the single-pass v1 wiring end-to-end:

- An ``Agent`` with one ``@tool`` produces a tool-grounded answer; the
  configured GSAR judge sees the answer + tool execution as evidence
  and surfaces a ``proceed`` decision on ``AgentResult``.
- An ``Agent`` whose model spits out an unsupported claim (no tool
  invoked) gets caught by the judge — the agent's result carries a
  non-``proceed`` decision and a non-empty ungrounded partition.

Activation: ``OPENAI_API_KEY`` (uses ``gpt-4o-mini`` for both the
agent and the judge).
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import skip_without_openai


@skip_without_openai
@pytest.mark.asyncio
async def test_agent_gsar_grounded_answer_proceeds() -> None:
    from locus.agent import Agent
    from locus.agent.config import GSARConfig
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar_judge import StructuredOutputGSARJudge
    from locus.tools.decorator import tool

    @tool(name="lookup_cpu_metric")
    def lookup_cpu_metric(host: str) -> str:
        """Return the current CPU utilization for the given host."""
        if host == "db-prod-1":
            return "host=db-prod-1 cpu_pct=97.2 measured_at=14:02:01 alert_id=A-9912 severity=high"
        return f"host={host} cpu_pct=unknown"

    judge_model = OpenAIModel(model="gpt-4o-mini", max_tokens=2048)
    agent = Agent(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=512),
        tools=[lookup_cpu_metric],
        system_prompt=(
            "You are a diagnostic agent. When asked about CPU on a host, "
            "call lookup_cpu_metric and report the metric verbatim."
        ),
        max_iterations=4,
        gsar=GSARConfig(judge=StructuredOutputGSARJudge(model=judge_model)),
    )
    result = agent.run_sync("What's the current CPU utilisation on db-prod-1?")

    # The judge ran and produced a verdict.
    assert result.gsar_judgment is not None, f"GSAR did not run. message={result.message[:200]!r}"
    assert result.gsar_score is not None
    # On a tool-grounded answer the judge should not send δ=replan.
    # (regenerate is acceptable when the judge over-flags one inference.)
    assert result.gsar_decision in ("proceed", "regenerate"), (
        f"unexpected δ={result.gsar_decision} on grounded answer; "
        f"score={result.gsar_score:.3f}, "
        f"message={result.message[:200]!r}"
    )


@skip_without_openai
@pytest.mark.asyncio
async def test_agent_gsar_ungrounded_answer_does_not_proceed() -> None:
    from locus.agent import Agent
    from locus.agent.config import GSARConfig
    from locus.models.native.openai import OpenAIModel
    from locus.reasoning.gsar_judge import StructuredOutputGSARJudge

    judge_model = OpenAIModel(model="gpt-4o-mini", max_tokens=2048)
    # Agent has no tools — any specific factual claim it makes is
    # un-evidenced. We force it to invent something the judge can flag.
    agent = Agent(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=512),
        system_prompt=(
            "You are a diagnostic agent. Answer with very specific numbers, "
            "host names, and timestamps even when you don't have evidence. "
            "Do not say 'I don't know' — produce a confident-sounding answer."
        ),
        max_iterations=2,
        gsar=GSARConfig(judge=StructuredOutputGSARJudge(model=judge_model)),
    )
    result = agent.run_sync(
        "What was the CPU utilisation on host db-prod-7 at 03:14:09 UTC last Tuesday?"
    )

    # The judge ran.
    assert result.gsar_judgment is not None
    # And it did NOT send a confident-but-unsupported answer to proceed.
    # We accept regenerate or replan or abstain — any of those means the
    # framework recognised the un-grounded claim. proceed would be a
    # real failure of the judge.
    assert result.gsar_decision != "proceed", (
        f"GSAR judge wrongly accepted an un-evidenced answer: "
        f"score={result.gsar_score:.3f}, "
        f"message={result.message[:200]!r}, "
        f"|G|={len(result.gsar_judgment.grounded_claims)}, "
        f"|U|={len(result.gsar_judgment.ungrounded_claims)}, "
        f"|X|={len(result.gsar_judgment.contradicted_claims)}"
    )
    # The judge should have surfaced at least one non-grounded claim
    # (ungrounded or contradicted) — that's the load-bearing claim of
    # the typed-partition framework.
    judgment = result.gsar_judgment
    non_grounded = len(judgment.ungrounded_claims) + len(judgment.contradicted_claims)
    assert non_grounded >= 1 or judgment.abstained, (
        f"judge produced no non-grounded claims and didn't abstain: "
        f"|G|={len(judgment.grounded_claims)}, "
        f"|U|={len(judgment.ungrounded_claims)}, "
        f"|X|={len(judgment.contradicted_claims)}"
    )
