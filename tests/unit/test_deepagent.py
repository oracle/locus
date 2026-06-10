# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tulip.deepagent`` — the research-shaped Agent factory
and provider protocol.

These tests don't touch a model provider — they verify the factory
returns a properly-configured Agent (typed termination, output_schema,
reflexion/grounding flags) without making model-provider calls.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tulip import (
    Grounding,
    ItemRef,
    KnowledgeProvider,
    KnowledgeRow,
    create_deepagent,
)
from tulip.core.termination import (
    AndCondition,
    ConfidenceMet,
    MaxIterations,
    OrCondition,
    TokenLimit,
    ToolCalled,
)
from tulip.tools.decorator import tool


class _Echo(BaseModel):
    text: str
    confidence: float = 0.0


@tool
def submit_research(text: str, confidence: float) -> str:
    """Final-answer tool the deepagent terminates on."""
    return f"submitted: {text}"


def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestCreateDeepagent:
    def test_returns_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_env(monkeypatch)
        from tulip import Agent

        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
        )
        assert isinstance(agent, Agent)

    def test_typed_termination_attached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``(submit & confidence) | tokens | iters`` shape must
        attach to ``agent.config.termination`` so the loop's exit logic
        actually consults it. ``TokenLimit`` is only present when the
        caller opts in with an explicit ``total_token_budget``."""
        _stub_env(monkeypatch)

        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
            min_confidence=0.7,
            total_token_budget=12_345,
            max_iterations=11,
            submit_tool="submit_research",
        )
        term = agent.config.termination
        assert isinstance(term, OrCondition)
        # Walk the algebra and assert every leaf condition is present.
        # The exact tree is `((Submit & Conf) | Tokens) | Iters`.
        leaves: list[type] = []

        def _walk(node):
            if isinstance(node, (OrCondition, AndCondition)):
                for child in node._conditions:
                    _walk(child)
            else:
                leaves.append(type(node))

        _walk(term)
        assert ToolCalled in leaves
        assert ConfidenceMet in leaves
        assert TokenLimit in leaves
        assert MaxIterations in leaves

    def test_token_limit_omitted_when_budget_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default ``total_token_budget=None`` must NOT add a
        ``TokenLimit`` term to the termination algebra. Previously the
        80K default silently killed any run whose long input prompt
        ate the cumulative budget before the model wrote output —
        verified empirically against gpt-5.5 + Tulip's deepagent
        kernel (see CHANGELOG)."""
        _stub_env(monkeypatch)
        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
        )
        term = agent.config.termination
        leaves: list[type] = []

        def _walk(node):
            if isinstance(node, (OrCondition, AndCondition)):
                for child in node._conditions:
                    _walk(child)
            else:
                leaves.append(type(node))

        _walk(term)
        assert TokenLimit not in leaves, (
            "Default deepagent termination must not include a "
            "TokenLimit term. The historical 80K default was a "
            "foot-gun — callers should opt in explicitly via "
            "total_token_budget."
        )
        # The base algebra (submit+confidence OR iterations) still applies.
        assert ToolCalled in leaves
        assert ConfidenceMet in leaves
        assert MaxIterations in leaves

    def test_legacy_max_tokens_kwarg_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Breaking change in 0.2.0b23 — the old ``max_tokens=`` kwarg
        was removed. Callers who still pass it should hit a loud
        TypeError from ``Agent()`` (via **agent_kwargs), not a silent
        wrong-behavior run."""
        _stub_env(monkeypatch)
        with pytest.raises(TypeError):
            create_deepagent(
                model="openai:gpt-4o-mini",
                tools=[submit_research],
                system_prompt="be helpful",
                output_schema=_Echo,
                reflexion=False,
                grounding=False,
                max_tokens=54_321,
            )

    def test_max_output_tokens_propagated_independently_of_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``max_output_tokens`` is the per-completion cap — it goes
        to AgentConfig.max_tokens, NOT into the termination algebra.
        Without ``total_token_budget`` set, the termination still
        has no TokenLimit term."""
        _stub_env(monkeypatch)
        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
            max_output_tokens=65_536,
        )
        # Per-completion cap landed on AgentConfig.
        assert agent.config.max_tokens == 65_536
        # No TokenLimit term in termination.
        term = agent.config.termination
        leaves: list[type] = []

        def _walk(node):
            if isinstance(node, (OrCondition, AndCondition)):
                for child in node._conditions:
                    _walk(child)
            else:
                leaves.append(type(node))

        _walk(term)
        assert TokenLimit not in leaves

    def test_output_schema_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_env(monkeypatch)
        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
        )
        assert agent.config.output_schema is _Echo

    def test_reflexion_grounding_default_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The factory's whole point is research-shaped defaults — reflexion
        and grounding must be on unless callers explicitly opt out."""
        _stub_env(monkeypatch)
        agent = create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
        )
        assert agent.config.reflexion is not None
        assert agent.config.grounding is not None


class TestProtocolTypes:
    def test_item_ref_auto_key(self) -> None:
        ref = ItemRef(name="V$PDBS", provider="database")
        assert ref.key == "database:V$PDBS"

    def test_item_ref_explicit_key_preserved(self) -> None:
        ref = ItemRef(name="x", provider="p", key="custom-id")
        assert ref.key == "custom-id"

    def test_grounding_defaults_empty(self) -> None:
        g = Grounding()
        assert g.summary == ""
        assert g.payload == {}

    def test_knowledge_row_round_trip(self) -> None:
        row = KnowledgeRow(
            name="V$PDBS",
            provider="database",
            short_description="Pluggable databases dynamic view.",
            domains=["database"],
            tags=["v$"],
            confidence=0.92,
        )
        as_dict = row.model_dump()
        rebuilt = KnowledgeRow(**as_dict)
        assert rebuilt.name == row.name
        assert rebuilt.confidence == row.confidence

    def test_knowledge_provider_runtime_checkable(self) -> None:
        """Bare object isn't a provider; one with all the methods is."""

        class _Bad:
            pass

        class _Good:
            async def open(self): ...
            async def close(self): ...
            async def discover(self, query=None):
                return []

            async def ground(self, item):
                return Grounding()

            def tools_for_agent(self):
                return []

            def output_schema(self):
                return KnowledgeRow

            def merge_to_row(self, item, grounding, research, *, model_id, prompt_hash):
                return KnowledgeRow(name=item.name, provider=item.provider)

        assert not isinstance(_Bad(), KnowledgeProvider)
        assert isinstance(_Good(), KnowledgeProvider)
