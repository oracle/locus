# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a run reports about prompt caching and what the provider billed.

The cost of a long run depends on how much of each request the provider served
from its cache far more than on the token count: in one benchmark 8.5M input
tokens cost $0.26 and 3.7M cost $1.12. A report that says only "3.7M tokens"
cannot tell those runs apart, so the counters the providers send are kept —
OpenAI-style ``cached_tokens`` (inside ``prompt_tokens``), Anthropic-style
``cache_read_input_tokens`` (outside it), and OpenRouter's own figure for what
a call cost — through the loop, across subagents, onto ``TerminateEvent``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from tulip.agent import Agent
from tulip.agent.specs import AgentSpec, parse_agent_markdown
from tulip.agent.subagent import ChildSpend, _ParentRunContext, fold_subagent_usage
from tulip.core.events import TerminateEvent
from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.models.base import ModelResponse
from tulip.models.metadata import metadata_for
from tulip.models.native.anthropic import (
    AnthropicModel,
    _reported_cost,
    native_model_id,
    openrouter_model_id,
)
from tulip.models.native.openai import _chat_usage
from tulip.testing import FunctionModel


def test_openai_usage_keeps_cached_and_written_tokens_apart_from_anthropics() -> None:
    usage = SimpleNamespace(
        prompt_tokens=1_000,
        completion_tokens=50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=900, cache_write_tokens=40),
        cost=0.0123,
    )
    parsed, cost = _chat_usage(usage)
    assert parsed == {
        "prompt_tokens": 1_000,
        "completion_tokens": 50,
        "cached_tokens": 900,
        "cache_write_tokens": 40,
    }
    assert cost == pytest.approx(0.0123)


def test_openai_usage_as_a_plain_dict_and_without_details() -> None:
    parsed, cost = _chat_usage({"prompt_tokens": 7, "completion_tokens": 3})
    assert parsed == {"prompt_tokens": 7, "completion_tokens": 3}
    assert cost is None


def test_openai_usage_without_counts_is_unmetered() -> None:
    assert _chat_usage(SimpleNamespace(prompt_tokens=None, completion_tokens=None)) == ({}, None)


def test_a_bool_is_never_a_cost() -> None:
    assert _chat_usage({"prompt_tokens": 1, "completion_tokens": 1, "cost": True})[1] is None
    assert _reported_cost(SimpleNamespace(cost=True)) is None
    assert _reported_cost(SimpleNamespace(cost=0.5)) == 0.5


def _model(responses: list[ModelResponse]) -> FunctionModel:
    queue = list(responses)

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        return queue.pop(0)

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model="tulip-test-usage-cache")  # type: ignore[attr-defined]
    return model


async def test_the_run_reports_cached_tokens_and_the_providers_own_cost() -> None:
    reply = ModelResponse(
        message=Message.assistant("done"),
        usage={"prompt_tokens": 1_000, "completion_tokens": 10, "cached_tokens": 800},
        cost_usd=0.002,
    )
    agent = Agent(model=_model([reply]), reflexion=False, grounding=False)
    events = [e async for e in agent.run("go")]
    done = next(e for e in events if isinstance(e, TerminateEvent))
    assert done.usage is not None
    assert done.usage["cached_tokens"] == 800
    assert "cache_read_input_tokens" not in done.usage
    assert done.reported_cost_usd == pytest.approx(0.002)


async def test_no_reported_cost_is_none_not_zero() -> None:
    reply = ModelResponse(
        message=Message.assistant("done"), usage={"prompt_tokens": 5, "completion_tokens": 1}
    )
    agent = Agent(model=_model([reply]), reflexion=False, grounding=False)
    events = [e async for e in agent.run("go")]
    done = next(e for e in events if isinstance(e, TerminateEvent))
    assert done.reported_cost_usd is None


def test_a_subagents_cache_counts_and_reported_cost_reach_the_parent() -> None:
    ctx = _ParentRunContext(cancel_signal=__import__("threading").Event())
    ctx.usage_sink.append(
        ChildSpend(100, 10, 0, 0, None, cached=80, cache_write=5, reported_cost=0.01)
    )
    from tulip.agent import subagent as subagent_mod

    token = subagent_mod._PARENT_RUN.set(ctx)
    try:
        state = fold_subagent_usage(AgentState())
    finally:
        subagent_mod._PARENT_RUN.reset(token)
    assert state.prompt_tokens_used == 100
    assert state.cached_tokens_used == 80
    assert state.cache_write_tokens_used == 5
    assert state.reported_cost_usd == pytest.approx(0.01)
    assert state.reported_cost_calls == 1


def test_response_usage_ignores_what_is_not_a_count() -> None:
    state = AgentState().with_response_usage(
        {"prompt_tokens": 3, "completion_tokens": True, "cached_tokens": "9"}, object()
    )
    assert state.prompt_tokens_used == 3
    assert state.completion_tokens_used == 0
    assert state.cached_tokens_used == 0
    assert state.reported_cost_calls == 0


# ----------------------------------------------- Claude through OpenRouter --


def test_claude_ids_and_openrouter_slugs_map_both_ways() -> None:
    assert openrouter_model_id("claude-sonnet-5-5") == "anthropic/claude-sonnet-5.5"
    assert openrouter_model_id("claude-haiku-4-5-20251001") == "anthropic/claude-haiku-4.5"
    assert openrouter_model_id("claude-opus-5") == "anthropic/claude-opus-5"
    assert openrouter_model_id("anthropic/claude-opus-5.5") == "anthropic/claude-opus-5.5"
    assert native_model_id("anthropic/claude-sonnet-5.5") == "claude-sonnet-5-5"
    assert native_model_id("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_an_openrouter_slug_finds_the_claude_metadata() -> None:
    meta = metadata_for("anthropic:anthropic/claude-sonnet-5.5")
    assert meta is not None
    assert meta.model_id == "claude-sonnet-5-5"
    # The slug without its vendor part, as an agent file's provider/model gives it.
    dotted = metadata_for("anthropic:claude-sonnet-5.5")
    assert dotted is not None
    assert dotted.model_id == "claude-sonnet-5-5"
    assert metadata_for("anthropic/claude-nonexistent-9.9") is None


def test_the_endpoint_comes_from_anthropic_base_url_and_names_the_model_its_way() -> None:
    env = {"ANTHROPIC_BASE_URL": "https://openrouter.ai/api"}
    with patch.dict(os.environ, env):
        model = AnthropicModel(model="claude-sonnet-5-5", api_key="unused")
        assert model.base_url == "https://openrouter.ai/api"
        assert model.wire_model == "anthropic/claude-sonnet-5.5"
        params, _ = model._request_params([Message.user("hi")], None, {})
    assert params["model"] == "anthropic/claude-sonnet-5.5"
    # Sonnet 5.5 rejects temperature under either name.
    assert "temperature" not in params


def test_anthropics_own_endpoint_keeps_the_id() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        model = AnthropicModel(model="claude-sonnet-5-5", api_key="unused")
        assert model.wire_model == "claude-sonnet-5-5"


async def test_a_bearer_token_never_sends_an_anthropic_key_from_the_environment() -> None:
    env = {
        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
        "ANTHROPIC_AUTH_TOKEN": "sk-or-test",
        "ANTHROPIC_API_KEY": "sk-ant-should-not-leave",
    }
    with patch.dict(os.environ, env):
        model = AnthropicModel(model="claude-sonnet-5-5")
        client = model.client
        headers = client.auth_headers
    assert headers.get("Authorization") == "Bearer sk-or-test"
    assert "X-Api-Key" not in headers
    assert str(client.base_url).startswith("https://openrouter.ai/api")
    await model.close()


async def test_an_explicit_api_key_is_still_sent_with_a_token() -> None:
    model = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-ant", auth_token="tok")  # noqa: S106
    headers = model.client.auth_headers
    assert headers.get("X-Api-Key") == "sk-ant"
    assert headers.get("Authorization") == "Bearer tok"
    await model.close()


# ------------------------------------------------------- subagent budgets --


def test_an_agent_file_can_cap_a_tasks_tokens() -> None:
    spec = parse_agent_markdown(
        "---\nname: explore\nmode: subagent\ntokenBudget: 250000\n---\nSearch.", name="x"
    )
    assert spec.token_budget == 250_000
    assert "tokenBudget" not in spec.metadata


async def test_the_task_tool_gives_a_subagent_its_types_token_budget() -> None:
    from tulip.agent import tasks as tasks_mod

    seen: dict[str, Any] = {}

    class _Fake:
        task_id = "t-1"

        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        async def send(self, prompt: str) -> Any:
            return SimpleNamespace(
                text="found it", task_id="t-1", iterations=1, success=True, stop_reason="complete"
            )

    spec = AgentSpec(name="explore", mode="subagent", token_budget=50_000)
    with patch.object(tasks_mod, "Subagent", _Fake):
        task = tasks_mod.task_tool(
            [spec], tools=[], model="m", agent_kwargs={"token_budget": 80_000}
        )
        out = await task.execute(description="d", prompt="p")
    assert seen["token_budget"] == 50_000
    assert "found it" in str(getattr(out, "content", out))


def test_anthropic_cache_traffic_is_priced_not_free() -> None:
    state = AgentState(input_price_per_mtok=2.0, output_price_per_mtok=10.0)
    after = state.with_response_usage(
        {
            "prompt_tokens": 1_000,
            "completion_tokens": 100,
            "cache_creation_input_tokens": 4_000,
            "cache_read_input_tokens": 100_000,
        }
    )
    # 1,000 + 4,000 x 1.25 + 100,000 x 0.1 = 16,000 input-priced tokens.
    assert after.cost_usd_used == pytest.approx((16_000 * 2.0 + 100 * 10.0) / 1_000_000)


def test_openai_style_cached_tokens_stay_at_the_full_input_price() -> None:
    state = AgentState(input_price_per_mtok=2.0, output_price_per_mtok=10.0)
    after = state.with_response_usage(
        {"prompt_tokens": 1_000, "completion_tokens": 0, "cached_tokens": 900}
    )
    assert after.cost_usd_used == pytest.approx(1_000 * 2.0 / 1_000_000)
