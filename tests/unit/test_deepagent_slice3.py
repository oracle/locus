# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Slice 3 tests — subagent dispatch + summarization wiring."""

from __future__ import annotations

import pytest

from locus.deepagent import SubAgentDef, create_deepagent, task_tool
from locus.tools.decorator import tool


# ---------------------------------------------------------------------------
# task_tool — schema + dispatch
# ---------------------------------------------------------------------------


@tool
def _noop_tool(x: str) -> str:
    """A trivial tool the subagent can carry."""
    return f"got: {x}"


class TestTaskTool:
    def test_returns_a_single_tool_named_task(self) -> None:
        sa = SubAgentDef(
            name="reviewer",
            description="reviews stuff",
            system_prompt="You review.",
        )
        t = task_tool([sa], parent_model="oci:fake-model")
        assert t.name == "task"

    def test_parameter_schema_has_flat_fields(self) -> None:
        sa = SubAgentDef(
            name="reviewer",
            description="reviews stuff",
            system_prompt="You review.",
        )
        t = task_tool([sa], parent_model="oci:fake-model")
        props = t.parameters.get("properties", {})
        assert {"subagent_type", "description"} <= set(props.keys())

    def test_empty_subagents_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            task_tool([], parent_model="oci:fake-model")

    def test_catalog_attached_for_prompt_injection(self) -> None:
        sa1 = SubAgentDef(name="a", description="A does A.", system_prompt="A.")
        sa2 = SubAgentDef(name="b", description="B does B.", system_prompt="B.")
        t = task_tool([sa1, sa2], parent_model="oci:fake-model")
        catalog = getattr(t, "_subagent_catalog", "")
        assert "a: A does A." in catalog
        assert "b: B does B." in catalog

    @pytest.mark.asyncio
    async def test_unknown_subagent_returns_error_string(self) -> None:
        sa = SubAgentDef(name="reviewer", description="x", system_prompt="x")
        t = task_tool([sa], parent_model="oci:fake-model")
        result = await t.execute(subagent_type="ghost", description="...")
        assert "unknown subagent_type" in result
        assert "reviewer" in result  # surfaces what is available

    @pytest.mark.asyncio
    async def test_spawn_returns_subagent_final_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tool runs the subagent and returns its
        TerminateEvent.final_message verbatim. We mock the Agent class
        so we don't need OCI auth.
        """

        # Stub locus.agent.agent.Agent + locus.core.events.TerminateEvent
        # so the subagent's run loop is deterministic.
        from locus.agent import agent as agent_mod
        from locus.core.events import TerminateEvent

        class _FakeAgent:
            def __init__(
                self,
                *,
                model,
                tools,
                system_prompt,
                max_iterations,
                reflexion,
                grounding,
                **_kwargs,
            ):  # noqa: ARG002
                self._prompt = system_prompt

            async def run(self, description, *, thread_id=None):  # noqa: ARG002
                yield TerminateEvent(
                    reason="complete",
                    iterations_used=1,
                    final_confidence=0.9,
                    total_tool_calls=0,
                    final_message=f"REVIEWED: {description}",
                )

        monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)

        sa = SubAgentDef(
            name="reviewer",
            description="reviews drafts",
            system_prompt="You review.",
        )
        t = task_tool([sa], parent_model="oci:fake-model")

        result = await t.execute(
            subagent_type="reviewer",
            description="check the V$PDBS edge",
        )
        assert result == "REVIEWED: check the V$PDBS edge"


# ---------------------------------------------------------------------------
# create_deepagent — subagents + summarize_after_messages
# ---------------------------------------------------------------------------


def _stub_oci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCI_PROFILE", "DEFAULT")


class TestCreateDeepagentSlice3:
    def test_subagents_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            reflexion=False,
            grounding=False,
        )
        assert "task" not in {t.name for t in agent.config.tools}

    def test_subagents_attaches_task_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_oci(monkeypatch)
        sa = SubAgentDef(name="reviewer", description="x", system_prompt="x")
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            subagents=[sa],
            reflexion=False,
            grounding=False,
        )
        names = {t.name for t in agent.config.tools}
        assert "task" in names

    def test_summarization_off_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            reflexion=False,
            grounding=False,
        )
        # No conversation manager set — agent uses locus's default.
        cm = getattr(agent.config, "conversation_manager", None)
        # Either None or a NullManager — both mean "no summarization".
        if cm is not None:
            from locus.memory.conversation import SummarizingManager

            # Must NOT be a SummarizingManager.
            assert not isinstance(cm, SummarizingManager)

    def test_summarize_after_messages_wires_summarizing_manager(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            summarize_after_messages=20,
            summarize_keep_recent=5,
            reflexion=False,
            grounding=False,
        )
        from locus.memory.conversation import SummarizingManager

        cm = agent.config.conversation_manager
        assert isinstance(cm, SummarizingManager)
        assert cm.threshold == 20
        assert cm.keep_recent == 5
