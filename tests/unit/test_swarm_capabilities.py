# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the typed capability registry on ``SwarmAgent`` /
``SwarmTask``.

The audit caught the swarm's old ``can_handle`` doing substring
matching against the task description — fragile and impossible to
audit from outside. This change adds typed ``required_tags`` /
``preferred_tags`` on the task plus set-membership routing on the
agent. Backwards compat: pre-tag tasks still route via the legacy
substring fallback.
"""

from __future__ import annotations

import pytest

from locus.multiagent.swarm import SwarmAgent, SwarmTask


@pytest.fixture
def log_analyst() -> SwarmAgent:
    return SwarmAgent(name="log_analyst", capabilities=["logs", "search", "alerts"])


@pytest.fixture
def db_specialist() -> SwarmAgent:
    return SwarmAgent(name="db_specialist", capabilities=["sql", "indexing"])


@pytest.fixture
def generalist() -> SwarmAgent:
    return SwarmAgent(name="generalist", capabilities=[])


class TestRequiredTags:
    def test_agent_with_all_required_tags_can_handle(self, log_analyst: SwarmAgent) -> None:
        task = SwarmTask(description="anything", required_tags=["logs"])
        assert log_analyst.can_handle(task)

    def test_agent_missing_a_required_tag_cannot_handle(self, log_analyst: SwarmAgent) -> None:
        task = SwarmTask(description="anything", required_tags=["logs", "sql"])
        assert log_analyst.can_handle(task) is False

    def test_required_tags_are_case_insensitive(self) -> None:
        agent = SwarmAgent(name="x", capabilities=["LOGS"])
        task = SwarmTask(description="d", required_tags=["logs"])
        assert agent.can_handle(task)


class TestGeneralistFallback:
    def test_generalist_claims_any_task_with_no_required_tags(self, generalist: SwarmAgent) -> None:
        assert generalist.can_handle(SwarmTask(description="anything"))

    def test_generalist_cannot_claim_task_with_required_tags(self, generalist: SwarmAgent) -> None:
        # An agent with no capabilities can't possibly satisfy a
        # required_tag — the task is asking for something specific.
        task = SwarmTask(description="d", required_tags=["sql"])
        assert generalist.can_handle(task) is False


class TestSubstringFallback:
    def test_pre_tag_swarms_keep_substring_routing(
        self, log_analyst: SwarmAgent, db_specialist: SwarmAgent
    ) -> None:
        # No tags on the task → the old substring path runs.
        task = SwarmTask(description="grep timeouts in api logs and alert oncall")
        assert log_analyst.can_handle(task)
        assert db_specialist.can_handle(task) is False


class TestPriorityScoring:
    def test_full_required_match_scores_one(self, log_analyst: SwarmAgent) -> None:
        task = SwarmTask(description="d", required_tags=["logs", "alerts"])
        assert log_analyst.priority_for_task(task) == pytest.approx(1.0)

    def test_preferred_tags_boost_score_below_required(self, log_analyst: SwarmAgent) -> None:
        # Required: 1 hit (logs) — weight 1.0.
        # Preferred: 1 hit (search) — weight 0.5.
        # Max possible: 1.0 (req) + 0.5 (pref) = 1.5.
        task = SwarmTask(
            description="d",
            required_tags=["logs"],
            preferred_tags=["search"],
        )
        assert log_analyst.priority_for_task(task) == pytest.approx(1.0)

    def test_partial_preferred_match(self, log_analyst: SwarmAgent) -> None:
        # required=0 hits, preferred=1 hit (alerts).
        # weight = 0 + 0.5 = 0.5; max = 0 (no required) + 0.5 = 0.5 → 1.0.
        # …but score is "value / max"; with no required, this is 0.5 / 0.5 = 1.0.
        # Run the case where the agent has *some* preferred but not all.
        task = SwarmTask(description="d", preferred_tags=["alerts", "missing_tag"])
        # weight = 0.5 (alerts only); max = 1.0 → 0.5.
        assert log_analyst.priority_for_task(task) == pytest.approx(0.5)

    def test_no_capability_match_gives_zero(self) -> None:
        agent = SwarmAgent(name="x", capabilities=["python"])
        task = SwarmTask(description="d", required_tags=["sql"])
        assert agent.priority_for_task(task) == 0.0

    def test_generalist_neutral_score_on_tagless_task(self, generalist: SwarmAgent) -> None:
        assert generalist.priority_for_task(SwarmTask(description="anything")) == 0.5

    def test_substring_score_on_tagless_task(self, log_analyst: SwarmAgent) -> None:
        # Backwards-compat: tag-less task scores via substring count.
        task = SwarmTask(description="search the logs and check alerts")
        # 3 of 3 capabilities mentioned → 1.0.
        assert log_analyst.priority_for_task(task) == pytest.approx(1.0)
