# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The agent must not override sampling the caller configured on the model.

``complete_kwargs`` is forwarded as *per-call* arguments, which beat a
provider's own config. An agent-level default therefore silently discards
whatever ``get_model(...)`` was given — the caller sets ``temperature=1.0``
because their model's card asks for it, and gets 0.7.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip import Agent
from tulip.core.messages import Message
from tulip.models.base import ModelResponse


class _RecordingModel:
    """Captures the kwargs the loop hands the provider."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.seen.append(kwargs)
        return ModelResponse(
            message=Message.assistant(content="done"),
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


async def _run(agent: Agent) -> None:
    async for _ in agent.run("hi"):
        pass


@pytest.mark.asyncio
async def test_agent_does_not_send_sampling_it_was_never_given() -> None:
    model = _RecordingModel()
    await _run(Agent(model=model, tools=[], system_prompt="p"))

    assert model.seen, "model was never called"
    kwargs = model.seen[0]
    assert "temperature" not in kwargs, (
        "agent injected a temperature the caller never set, which overrides "
        "the model's own configuration"
    )
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_explicit_agent_sampling_is_still_forwarded() -> None:
    model = _RecordingModel()
    await _run(Agent(model=model, tools=[], system_prompt="p", temperature=0.2, max_tokens=64))

    kwargs = model.seen[0]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64


@pytest.mark.asyncio
async def test_model_kwargs_reach_the_provider() -> None:
    """Per-call params must be expressible from Agent — tool_choice above all,
    since it has to vary per run and model config cannot."""
    model = _RecordingModel()
    agent = Agent(model=model, tools=[], system_prompt="p")

    choice = {"type": "function", "function": {"name": "search"}}
    async for _ in agent.run("hi", model_kwargs={"tool_choice": choice}):
        pass

    assert model.seen[0]["tool_choice"] == choice


@pytest.mark.asyncio
async def test_model_kwargs_win_over_agent_config() -> None:
    """The per-run value is the most specific statement of intent."""
    model = _RecordingModel()
    agent = Agent(model=model, tools=[], system_prompt="p", temperature=0.9)

    async for _ in agent.run("hi", model_kwargs={"temperature": 0.1}):
        pass

    assert model.seen[0]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_arun_and_run_sync_accept_model_kwargs() -> None:
    model = _RecordingModel()
    agent = Agent(model=model, tools=[], system_prompt="p")

    await agent.arun("hi", model_kwargs={"user": "u-1"})
    assert model.seen[-1]["user"] == "u-1"
