# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Skills a host routes in code: ``SkillsPlugin(active=..., enforce_allowed_tools=...)``.

Regression for three gaps a product hit when it chose skills itself:

* activating a skill needed the model to call the ``skills`` tool — a host
  that already knows the skill had no way to put its instructions in front of
  the model;
* ``allowed-tools`` was advisory: a tool outside the list still ran;
* the catalog and activation responses carried server filesystem paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locus import Agent, tool
from locus.core.messages import Message
from locus.memory.backends.file import FileCheckpointer
from locus.skills.models import Skill
from locus.skills.plugin import SkillsPlugin
from locus.testing import FunctionModel, text, tool_call


PRICING = Skill(
    name="pricing",
    description="Quote live rates.",
    instructions="Quote rates line by line from get_offers.",
    allowed_tools=["get_offers"],
)
HEBREW = Skill(
    name="hebrew",
    description="Hebrew replies.",
    instructions="Keep hotel names in Latin script.",
)
BOOKING = Skill(
    name="booking",
    description="Reserve a rate.",
    instructions="Call prepare_booking.",
    allowed_tools=["prepare_booking"],
)

ran: list[str] = []


@tool
def get_offers(hotel: str) -> str:
    """Live rates for a hotel."""
    ran.append("get_offers")
    return f"{hotel}: Flexible 840 EUR"


@tool
def prepare_booking(quote_id: str) -> str:
    """Hold a checkout for a quote."""
    ran.append("prepare_booking")
    return "checkout prepared"


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


def _system_texts(messages: list[Message]) -> list[str]:
    return [m.content or "" for m in messages if getattr(m.role, "value", m.role) == "system"]


class TestActiveSkills:
    async def test_active_instructions_reach_every_model_call_without_the_skills_tool(
        self,
    ) -> None:
        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
            if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
                return text("Flexible is 840 EUR.")
            return tool_call("get_offers", hotel="Aliee")

        model = FunctionModel(handler)
        plugin = SkillsPlugin([PRICING, HEBREW, BOOKING], active=["pricing", "hebrew"])
        agent = Agent(model=model, tools=[get_offers], plugins=[plugin], system_prompt="Base.")
        result = await agent.arun("rates?")

        assert result.message == "Flexible is 840 EUR."
        assert model.call_count == 2
        for sent in model.received_messages:
            systems = _system_texts(sent)
            assert systems[0] == "Base.", "the system prompt stays first"
            joined = "\n".join(systems)
            assert "Quote rates line by line" in joined
            assert "Keep hotel names in Latin script." in joined
            assert "Call prepare_booking." not in joined, "an inactive skill is not injected"
            assert "<available_skills>" not in joined, "no catalog when the host routes"
        assert all("skills" not in offered for offered in model.offered_tools)
        assert plugin.activated_skills == ["pricing", "hebrew"]

    async def test_injected_instructions_never_reach_the_saved_conversation(
        self, tmp_path: Path
    ) -> None:
        model = FunctionModel(lambda _m, _t: text("Hello."))
        checkpointer = FileCheckpointer(tmp_path)
        agent = Agent(
            model=model,
            plugins=[SkillsPlugin([PRICING], active=["pricing"])],
            checkpointer=checkpointer,
        )
        result = await agent.arun("hi", thread_id="t1")

        assert all(
            "Quote rates line by line" not in (m.content or "") for m in result.state.messages
        )
        saved = await checkpointer.load("t1")
        assert saved is not None
        assert all("Quote rates line by line" not in (m.content or "") for m in saved.messages)

    def test_an_unknown_active_skill_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown active skill"):
            SkillsPlugin([PRICING], active=["nope"])

    def test_the_catalog_can_be_kept_next_to_active_skills(self) -> None:
        plugin = SkillsPlugin([PRICING, BOOKING], active=["pricing"], catalog=True)
        assert [t.name for t in plugin.get_tools()] == ["skills"]
        assert SkillsPlugin([PRICING], active=["pricing"]).get_tools() == []
        assert [t.name for t in SkillsPlugin([PRICING]).get_tools()] == ["skills"], (
            "without active skills the catalog stays on by default"
        )


class TestEnforcedAllowedTools:
    async def test_a_tool_outside_the_active_skills_is_cancelled_before_it_runs(self) -> None:
        seen_results: list[str] = []

        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
            results = [m for m in messages if getattr(m.role, "value", m.role) == "tool"]
            if results:
                seen_results.append(results[-1].content or "")
                return text("I can quote rates, not book.")
            return tool_call("prepare_booking", quote_id="q1")

        plugin = SkillsPlugin(
            [PRICING, HEBREW, BOOKING], active=["pricing", "hebrew"], enforce_allowed_tools=True
        )
        agent = Agent(
            model=FunctionModel(handler), tools=[get_offers, prepare_booking], plugins=[plugin]
        )
        await agent.arun("book it")

        assert ran == [], "the disallowed tool never ran"
        assert "not available for this request" in seen_results[0]
        assert "get_offers" in seen_results[0], "the model is told what it may use"

    async def test_allowed_tools_still_run(self) -> None:
        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
            if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
                return text("done")
            return tool_call("get_offers", hotel="Aliee")

        plugin = SkillsPlugin([PRICING], active=["pricing"], enforce_allowed_tools=True)
        agent = Agent(model=FunctionModel(handler), tools=[get_offers], plugins=[plugin])
        await agent.arun("rates")
        assert ran == ["get_offers"]

    def test_the_allowlist_is_the_union_of_what_active_skills_declare(self) -> None:
        both = SkillsPlugin([PRICING, BOOKING, HEBREW], active=["pricing", "booking", "hebrew"])
        assert both.allowed_tools() == frozenset({"get_offers", "prepare_booking"})
        assert SkillsPlugin([HEBREW], active=["hebrew"]).allowed_tools() is None, (
            "no declared list, no limit"
        )
        assert SkillsPlugin([PRICING]).allowed_tools() is None, "nothing active, no limit"

    async def test_without_enforcement_allowed_tools_stay_advisory(self) -> None:
        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
            if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
                return text("done")
            return tool_call("prepare_booking", quote_id="q1")

        plugin = SkillsPlugin([PRICING], active=["pricing"])
        agent = Agent(model=FunctionModel(handler), tools=[prepare_booking], plugins=[plugin])
        await agent.arun("book")
        assert ran == ["prepare_booking"], "default behaviour is unchanged"


class TestPaths:
    def _on_disk(self, tmp_path: Path) -> Path:
        d = tmp_path / "secret-layout" / "pricing"
        (d / "references").mkdir(parents=True)
        (d / "references" / "rates.md").write_text("x")
        (d / "SKILL.md").write_text("---\nname: pricing\ndescription: Rates.\n---\nQuote rates.")
        return d

    async def test_show_paths_false_keeps_server_paths_out_of_everything_the_model_sees(
        self, tmp_path: Path
    ) -> None:
        d = self._on_disk(tmp_path)
        plugin = SkillsPlugin([d], show_paths=False)
        assert "secret-layout" not in plugin._generate_catalog_xml()
        response = plugin.get_activation_tool().fn(skill_name="pricing")
        assert "secret-layout" not in response
        assert "Quote rates." in response

        active = SkillsPlugin([d], active=["pricing"], show_paths=False)
        model = FunctionModel(lambda _m, _t: text("ok"))
        await Agent(model=model, plugins=[active]).arun("hi")
        assert all("secret-layout" not in s for s in _system_texts(model.received_messages[0]))

    def test_paths_are_shown_by_default(self, tmp_path: Path) -> None:
        plugin = SkillsPlugin([self._on_disk(tmp_path)])
        assert "secret-layout" in plugin._generate_catalog_xml()


class TestPlacement:
    """``placement="after_prompt"``: a provider's prompt cache serves only a
    byte-identical prefix, so skills routed per run must not sit between the
    system prompt and the history."""

    @staticmethod
    def _roles(messages: list[Message]) -> list[str]:
        return [str(getattr(m.role, "value", m.role)) for m in messages]

    async def test_after_prompt_puts_the_skills_after_the_latest_user_message(self) -> None:
        def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
            if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
                return text("Flexible is 840 EUR.")
            return tool_call("get_offers", hotel="Aliee")

        model = FunctionModel(handler)
        plugin = SkillsPlugin([PRICING], active=["pricing"], placement="after_prompt")
        agent = Agent(model=model, tools=[get_offers], plugins=[plugin], system_prompt="Base.")
        await agent.arun("rates?")

        first, second = model.received_messages
        assert self._roles(first) == ["system", "user", "system"]
        assert "Quote rates line by line" in (first[2].content or "")
        # The tool call and its result come after the skills: the second call
        # starts with the first call's whole request, a prefix the cache serves.
        assert self._roles(second)[:3] == ["system", "user", "system"]
        assert [str(m.content) for m in second[:3]] == [str(m.content) for m in first]

    async def test_the_next_run_starts_with_the_previous_runs_history(self, tmp_path: Path) -> None:
        def route(text_: str, _run: Any) -> list[str]:
            return ["pricing"] if "rates" in text_ else ["booking"]

        model = FunctionModel(lambda _m, _t: text("Noted."))
        agent = Agent(
            model=model,
            plugins=[SkillsPlugin([PRICING, BOOKING], router=route, placement="after_prompt")],
            system_prompt="Base.",
            checkpointer=FileCheckpointer(tmp_path),
        )
        await agent.arun("rates?", thread_id="t1")
        await agent.arun("book it", thread_id="t1")

        first, second = model.received_messages
        # Different skills each run, yet the second request opens with the
        # first one's system prompt, prompt and reply.
        assert [m.content for m in second[:2]] == [m.content for m in first[:2]]
        assert self._roles(second) == ["system", "user", "assistant", "user", "system"]
        assert "Call prepare_booking." in (second[-1].content or "")
        assert all("Quote rates" not in (m.content or "") for m in second)

    async def test_after_system_stays_the_default(self) -> None:
        model = FunctionModel(lambda _m, _t: text("Hello."))
        agent = Agent(
            model=model,
            plugins=[SkillsPlugin([PRICING], active=["pricing"])],
            system_prompt="Base.",
        )
        await agent.arun("hi")
        assert self._roles(model.received_messages[0]) == ["system", "system", "user"]

    def test_an_unknown_placement_is_refused(self) -> None:
        with pytest.raises(ValueError, match="placement"):
            SkillsPlugin([PRICING], placement="first")  # type: ignore[arg-type]
