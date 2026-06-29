# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``Agent`` public-facade branches in
``tulip.agent.agent`` and the wiring in ``tulip.agent.initializer``.

Targets the ``run_sync`` teardown/edge branches (callback, missing final
state, checkpointer-close failure, background-task drain) and the plugin /
skills / string-auxiliary-model initializer branches.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tulip.agent import Agent
from tulip.agent.agent import Agent as AgentClass
from tulip.core.events import TerminateEvent
from tulip.core.messages import Message
from tulip.hooks.plugin import Plugin
from tulip.models.base import ModelResponse
from tulip.skills.models import Skill
from tulip.tools.decorator import tool


# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class _OneShotModel:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(message=Message.assistant(self.content), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _TaskSpawningModel:
    """Spawns a short-lived background task so the run_sync drain has work."""

    def __init__(self) -> None:
        self._bg_task: asyncio.Task[None] | None = None

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        async def _bg() -> None:
            await asyncio.sleep(0.2)

        # Held on the instance so it stays referenced (and pending) until the
        # run_sync teardown drain awaits it.
        self._bg_task = asyncio.create_task(_bg())
        return ModelResponse(message=Message.assistant("done"), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _agent(model: Any, **kwargs: Any) -> Agent:
    return Agent(model=model, reflexion=False, grounding=False, **kwargs)


# ---------------------------------------------------------------------------
# name property (line 219)
# ---------------------------------------------------------------------------


def test_name_property_returns_config_name() -> None:
    agent = _agent(_OneShotModel("x"), name="my-agent")
    assert agent.name == "my-agent"
    assert agent.name == agent.config.name


# ---------------------------------------------------------------------------
# run_sync callback handler (line 251)
# ---------------------------------------------------------------------------


def test_run_sync_fires_callback_handler() -> None:
    seen: list[Any] = []
    agent = _agent(_OneShotModel("answer"), callback_handler=seen.append)
    result = agent.run_sync("go")
    assert seen  # callback fired for each event
    assert any(isinstance(e, TerminateEvent) for e in seen)
    assert result.success


# ---------------------------------------------------------------------------
# run_sync when run() leaves no final state (lines 263-265)
# ---------------------------------------------------------------------------


async def _fake_run(
    self: Agent,
    prompt: str,
    *,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    yield TerminateEvent(
        reason="complete",
        iterations_used=0,
        final_confidence=0.0,
        total_tool_calls=0,
        final_message="forced final",
    )


def test_run_sync_reconstructs_state_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(_OneShotModel("x"))
    agent._last_run_state = None
    monkeypatch.setattr(AgentClass, "run", _fake_run, raising=True)
    result = agent.run_sync("go")
    assert result.message == "forced final"


# ---------------------------------------------------------------------------
# run_sync checkpointer-close failure is swallowed (lines 345-346)
# ---------------------------------------------------------------------------


def test_run_sync_checkpointer_close_failure_swallowed() -> None:
    ckpt = MagicMock()
    ckpt.save = AsyncMock()
    ckpt.load = AsyncMock(return_value=None)
    ckpt.close = AsyncMock(side_effect=RuntimeError("ckpt close boom"))
    agent = _agent(_OneShotModel("x"), checkpointer=ckpt)
    result = agent.run_sync("go")
    assert result.success
    ckpt.close.assert_awaited()


# ---------------------------------------------------------------------------
# run_sync drains lingering background tasks (line 362)
# ---------------------------------------------------------------------------


def test_run_sync_drains_pending_background_tasks() -> None:
    agent = _agent(_TaskSpawningModel())
    result = agent.run_sync("go")
    assert result.success


# ---------------------------------------------------------------------------
# run_sync drain failure is swallowed (lines 362-364)
# ---------------------------------------------------------------------------


def test_run_sync_drain_failure_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_all_tasks = asyncio.all_tasks

    def _maybe_boom(loop: Any = None) -> Any:
        # The teardown drain calls ``all_tasks()`` with no loop; asyncio's
        # own runner cleanup always passes one. Only blow up for the former.
        if loop is None:
            raise RuntimeError("all_tasks boom")
        return real_all_tasks(loop)

    monkeypatch.setattr(asyncio, "all_tasks", _maybe_boom)
    agent = _agent(_OneShotModel("x"))
    result = agent.run_sync("go")
    assert result.success


# ---------------------------------------------------------------------------
# initializer: plugin tools registered (line 93)
# ---------------------------------------------------------------------------


@tool
def plugin_tool_fn() -> str:
    """A tool bundled by a plugin."""
    return "p"


class _DemoPlugin(Plugin):
    name = "demo"
    helper = plugin_tool_fn


def test_plugin_tools_are_registered() -> None:
    agent = _agent(_OneShotModel("x"), plugins=[_DemoPlugin()])
    assert "plugin_tool_fn" in agent.tools


# ---------------------------------------------------------------------------
# initializer: skills register the activation tool (lines 97-103)
# ---------------------------------------------------------------------------


def test_skills_register_activation_tool() -> None:
    skill = Skill(
        name="demo-skill",
        description="A demo skill used only for tests.",
        instructions="Do the demo thing.",
    )
    agent = _agent(_OneShotModel("x"), skills=[skill])
    assert "skills" in agent.tools


# ---------------------------------------------------------------------------
# initializer: string auxiliary_model is resolved via get_model (line 161)
# ---------------------------------------------------------------------------


def test_string_auxiliary_model_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def _fake_get_model(spec: str) -> Any:
        requested.append(spec)
        return _OneShotModel("x")

    monkeypatch.setattr("tulip.agent.agent.get_model", _fake_get_model)
    agent = Agent(
        model="openai:gpt-4o",
        auxiliary_model="openai:gpt-4o-mini",
        reflexion=False,
        grounding=False,
    )
    assert "openai:gpt-4o-mini" in requested
    assert agent._auxiliary_model is not None
