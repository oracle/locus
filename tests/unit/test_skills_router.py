# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A host-side skills router on one shared Agent: ``SkillsPlugin(router=...)``.

Regression for a host that routes skills per message in code. With ``active=``
fixed per plugin instance it had to build (and cache) one Agent per routed
skill set; and a skill the model activated in one run widened every later
run's tool allowlist, because activation state was the plugin's, not the run's.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from tulip import Agent, tool
from tulip.core.events import RunInfo
from tulip.core.messages import Message
from tulip.skills.models import Skill
from tulip.skills.plugin import SkillsPlugin
from tulip.testing import FunctionModel, text, tool_call


BUILD = Skill(
    name="build",
    description="Build things.",
    instructions="Build only in the kid's own plot.",
    allowed_tools=["build"],
)
GREET = Skill(name="greet", description="Say hello.", instructions="Greet by name, briefly.")
GUIDE = Skill(
    name="guide",
    description="Lead the way.",
    instructions="Walk to the named place.",
    allowed_tools=["walk_to"],
    compatibility="minecraft",
)

ran: list[str] = []


@tool
def build(recipe: str) -> str:
    """Build a recipe."""
    ran.append("build")
    return "built"


@tool
def walk_to(place: str) -> str:
    """Walk to a place."""
    ran.append("walk_to")
    return "walking"


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


def _system_text(messages: list[Message]) -> str:
    return "\n".join(
        m.content or "" for m in messages if getattr(m.role, "value", m.role) == "system"
    )


def _keyword_router(text_: str, run: RunInfo | None) -> list[str]:
    picked = []
    if "build" in text_:
        picked.append("build")
    if "take me" in text_:
        picked.append("guide")
    return picked or ["greet"]


def _plugin(**kwargs: Any) -> SkillsPlugin:
    return SkillsPlugin([BUILD, GREET, GUIDE], router=_keyword_router, **kwargs)


class TestRouter:
    async def test_one_shared_agent_routes_each_run_by_its_message(self) -> None:
        model = FunctionModel(lambda _m, _t: text("ok"))
        agent = Agent(model=model, plugins=[_plugin()], system_prompt="Base.")

        await agent.arun("build me a castle")
        await agent.arun("hello there")

        first, second = (_system_text(sent) for sent in model.received_messages)
        assert "Build only in the kid's own plot." in first
        assert "Greet by name" not in first
        assert "Greet by name, briefly." in second
        assert "Build only" not in second
        assert all("skills" not in offered for offered in model.offered_tools), (
            "a router means no catalog and no skills tool by default"
        )

    async def test_concurrent_runs_keep_their_own_skills(self) -> None:
        gate = asyncio.Event()
        seen: dict[str, str] = {}

        async def router(text_: str, run: RunInfo | None) -> list[str]:
            # Both runs are routed before either one's model call happens.
            if "build" in text_:
                await gate.wait()
            else:
                gate.set()
            return _keyword_router(text_, run)

        def handler(messages: list[Message], _tools: list[dict[str, Any]]) -> Any:
            user = next(m.content for m in messages if getattr(m.role, "value", "") == "user")
            seen[user or ""] = _system_text(messages)
            return text("ok")

        agent = Agent(
            model=FunctionModel(handler),
            plugins=[SkillsPlugin([BUILD, GREET, GUIDE], router=router)],
        )
        await asyncio.gather(agent.arun("build a tower"), agent.arun("take me home"))

        assert "Build only" in seen["build a tower"]
        assert "Walk to the named place" not in seen["build a tower"]
        assert "Walk to the named place." in seen["take me home"]
        assert "Build only" not in seen["take me home"]

    async def test_the_router_sees_run_metadata_and_is_asked_once_per_run(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def router(text_: str, run: RunInfo | None) -> list[str]:
            assert run is not None
            calls.append((text_, dict(run.metadata)))
            return [run.metadata["skill"]]

        def handler(messages: list[Message], _tools: list[dict[str, Any]]) -> Any:
            if any(getattr(m.role, "value", "") == "tool" for m in messages):
                return text("done")
            return tool_call("walk_to", place="spawn")

        model = FunctionModel(handler)
        agent = Agent(
            model=model,
            tools=[walk_to],
            plugins=[SkillsPlugin([BUILD, GREET, GUIDE], router=router)],
        )
        await agent.arun("go", metadata={"skill": "guide"})

        assert model.call_count == 2
        assert calls == [("go", {"skill": "guide"})], "one routing per run, not per model call"
        assert all("Walk to the named place." in _system_text(s) for s in model.received_messages)

    async def test_unknown_names_are_ignored_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = FunctionModel(lambda _m, _t: text("ok"))
        plugin = SkillsPlugin([BUILD, GREET], router=lambda _t, _r: ["nope", "greet"])
        agent = Agent(model=model, plugins=[plugin])
        with caplog.at_level(logging.WARNING, logger="tulip.skills.plugin"):
            await agent.arun("hi")

        assert "Greet by name, briefly." in _system_text(model.received_messages[0])
        assert "nope" in caplog.text
        assert plugin.activated_skills == ["greet"]

    async def test_a_failing_router_falls_back_to_active(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(_t: str, _r: RunInfo | None) -> list[str]:
            raise RuntimeError("router down")

        model = FunctionModel(lambda _m, _t: text("ok"))
        plugin = SkillsPlugin([BUILD, GREET], active=["greet"], router=broken)
        with caplog.at_level(logging.WARNING, logger="tulip.skills.plugin"):
            result = await Agent(model=model, plugins=[plugin]).arun("hi")

        assert result.message == "ok", "the turn is not lost"
        assert "Greet by name, briefly." in _system_text(model.received_messages[0])
        assert "router failed" in caplog.text

    async def test_none_keeps_active_and_empty_means_no_skill(self) -> None:
        none_model = FunctionModel(lambda _m, _t: text("ok"))
        plugin = SkillsPlugin([BUILD, GREET], active=["greet"], router=lambda _t, _r: None)
        await Agent(model=none_model, plugins=[plugin]).arun("hi")
        assert "Greet by name" in _system_text(none_model.received_messages[0])

        empty_model = FunctionModel(lambda _m, _t: text("ok"))
        plugin = SkillsPlugin([BUILD, GREET], active=["greet"], router=lambda _t, _r: [])
        await Agent(model=empty_model, plugins=[plugin]).arun("hi")
        assert "Follow these skills" not in _system_text(empty_model.received_messages[0])


class TestPerRunEnforcement:
    async def test_routed_skills_bind_the_tool_list_per_run(self) -> None:
        results: list[str] = []

        def handler(messages: list[Message], _tools: list[dict[str, Any]]) -> Any:
            done = [m for m in messages if getattr(m.role, "value", "") == "tool"]
            if done:
                results.append(done[-1].content or "")
                return text("done")
            return tool_call("walk_to", place="spawn")

        agent = Agent(
            model=FunctionModel(handler),
            tools=[build, walk_to],
            plugins=[_plugin(enforce_allowed_tools=True)],
        )
        await agent.arun("build a house")  # routed to build: walk_to is refused
        await agent.arun("take me to spawn")  # routed to guide: walk_to runs

        assert "not available for this request" in results[0]
        assert ran == ["walk_to"]

    async def test_a_skill_the_model_activates_does_not_leak_into_the_next_run(self) -> None:
        results: list[str] = []
        state = {"run": 0}

        def handler(messages: list[Message], _tools: list[dict[str, Any]]) -> Any:
            done = [m for m in messages if getattr(m.role, "value", "") == "tool"]
            first = state["run"] == 0
            if first and len(done) == 0:
                return tool_call("skills", skill_name="guide")
            if first and len(done) == 1:
                return tool_call("walk_to", place="spawn")
            if first:
                state["run"] = 1
                return text("first done")
            if not done:
                return tool_call("walk_to", place="spawn")
            results.append(done[-1].content or "")
            return text("second done")

        plugin = SkillsPlugin(
            [BUILD, GREET, GUIDE], active=["build"], catalog=True, enforce_allowed_tools=True
        )
        agent = Agent(model=FunctionModel(handler), tools=[build, walk_to], plugins=[plugin])
        await agent.arun("lead me")
        assert ran == ["walk_to"], "activated in its own run, the skill's tool runs"

        await agent.arun("lead me again")
        assert ran == ["walk_to"], "a new run starts from active, not the last run's"
        assert "not available for this request" in results[0]

    def test_allowed_tools_outside_a_run_starts_from_active(self) -> None:
        plugin = SkillsPlugin([BUILD, GUIDE], active=["build"])
        assert plugin.allowed_tools() == frozenset({"build"})
        assert plugin.allowed_tools("unknown-run") == frozenset({"build"})


class TestWrapperText:
    async def test_custom_preambles_and_rendering_reach_the_model_verbatim(self) -> None:
        model = FunctionModel(lambda _m, _t: text("ok"))
        plugin = SkillsPlugin(
            [BUILD, GREET],
            active=["build"],
            catalog=True,
            active_preamble="Procedures:\n",
            catalog_preamble="More, on request:\n",
            render_skill=lambda s: f"## {s.name}\n{s.instructions}",
        )
        await Agent(model=model, plugins=[plugin]).arun("hi")

        sent = _system_text(model.received_messages[0])
        assert "Procedures:\n## build\nBuild only in the kid's own plot." in sent
        assert "More, on request:\n<available_skills>" in sent
        assert "Follow these skills" not in sent
        assert "<skill name=" not in sent

    async def test_the_footer_can_be_left_out(self) -> None:
        with_footer = FunctionModel(lambda _m, _t: text("ok"))
        await Agent(model=with_footer, plugins=[SkillsPlugin([GUIDE], active=["guide"])]).arun("hi")
        assert "Allowed tools: walk_to" in _system_text(with_footer.received_messages[0])

        without = FunctionModel(lambda _m, _t: text("ok"))
        await Agent(
            model=without,
            plugins=[SkillsPlugin([GUIDE], active=["guide"], skill_footer=False)],
        ).arun("hi")
        sent = _system_text(without.received_messages[0])
        assert '<skill name="guide">\nWalk to the named place.\n</skill>' in sent
        assert "Allowed tools" not in sent
        assert "Compatibility" not in sent


class TestBookkeeping:
    def test_runs_that_never_finish_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tulip.skills.plugin as plugin_module

        monkeypatch.setattr(plugin_module, "_MAX_TRACKED_RUNS", 2)
        plugin = _plugin(active=["greet"])
        for run_id in ("a", "b", "c"):
            plugin._run(run_id)
        assert list(plugin._runs) == ["b", "c"]

    async def test_a_run_the_plugin_never_saw_ends_quietly(self) -> None:
        plugin = _plugin(active=["greet"])

        class _State:
            run_id = "never-seen"

        await plugin.on_after_invocation(_State(), True)
        assert plugin.activated_skills == ["greet"]

    def test_failing_telemetry_never_breaks_an_activation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        # The package re-exports a function named `emit`, so take the module itself.
        emit_module = importlib.import_module("tulip.observability.emit")

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("telemetry down")

        monkeypatch.setattr(emit_module, "emit_sync", boom)
        plugin = _plugin()
        reply = plugin.get_activation_tool().fn(skill_name="build")
        assert "Build only in the kid's own plot." in reply

    def test_the_latest_user_text_skips_other_roles_and_reads_any_content(self) -> None:
        from tulip.skills.plugin import _latest_user_text

        class _Odd:
            role = "user"
            content = 42

        assert _latest_user_text([_Odd(), Message.assistant("hi")]) == "42"
        assert _latest_user_text([Message.assistant("only me")]) == ""
