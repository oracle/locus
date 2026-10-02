# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Context management that counts tokens, on by default.

The old default was a 40-message window: a run whose tool outputs were large
still grew past the model's context and failed late, after spending money and
possibly after a person had approved something. These tests pin the default
that replaces it and the cut it makes: a conversation a provider accepts, with
the opening request and a paused run's held call intact.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import InterruptEvent, TerminateEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.memory.compactor import LLMCompactor
from tulip.memory.conversation import NullManager, SlidingWindowManager
from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.testing import FunctionModel, text, tool_call
from tulip.tools.decorator import tool


_MODEL = "tulip-test-small-window"
_WINDOW = 20_000  # tokens
_OUTPUT = "x" * 30_000  # ~7,500 tokens by the char/4 estimate

register_metadata(
    ModelMetadata(model_id=_MODEL, family="test", context_length=_WINDOW, max_output_tokens=1_024)
)


@tool
def fetch_log(part: int) -> str:
    """Return one large page of a log."""
    return _OUTPUT


@tool
def needs_approval() -> str:
    """Ask a person before continuing."""
    return json.dumps({"__interrupt__": True, "question": "approve?"})


def _tokens(messages: list[Message]) -> int:
    return sum(len(m.content or "") for m in messages) // 4


def _provider(turns: list[Any], seen: list[list[Message]]) -> FunctionModel:
    """A model with a hard context limit, like a real provider, on a script."""
    index = {"n": 0}

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        seen.append(list(messages))
        if _tokens(messages) > _WINDOW:
            raise RuntimeError(f"context_length_exceeded: {_tokens(messages)} > {_WINDOW}")
        turn = turns[min(index["n"], len(turns) - 1)]
        index["n"] += 1
        return turn

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model=_MODEL)  # type: ignore[attr-defined]
    return model


def _log_turns(count: int) -> list[Any]:
    return [tool_call("fetch_log", call_id=f"c{i}", part=i) for i in range(count)]


def _assert_valid(messages: list[Message]) -> None:
    calls = {tc.id for m in messages if m.role == Role.ASSISTANT for tc in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.role == Role.TOOL}
    assert all(m.tool_call_id in calls for m in messages if m.role == Role.TOOL), "orphan result"
    for m in messages[:-1]:
        if m.role == Role.ASSISTANT and m.tool_calls:
            assert all(tc.id in answered for tc in m.tool_calls), "unanswered call mid-conversation"
    assert any(m.role == Role.USER for m in messages), "the opening request was lost"


def test_a_known_window_gets_a_token_counting_default() -> None:
    agent = Agent(model=_provider([text("ok")], []), tools=[], reflexion=False, grounding=False)

    assert isinstance(agent._conversation_manager, LLMCompactor)
    assert agent._conversation_manager.context_length == _WINDOW
    assert agent._conversation_manager.summarize_fn is None, "no extra model calls by default"


def test_an_unknown_window_gets_a_message_window_at_any_iteration_count() -> None:
    agent = Agent(
        model=FunctionModel(lambda m, t: text("ok")),
        tools=[],
        max_iterations=5,
        reflexion=False,
        grounding=False,
    )

    assert isinstance(agent._conversation_manager, SlidingWindowManager)


@pytest.mark.asyncio
async def test_large_tool_outputs_complete_with_the_default_and_fail_without_it() -> None:
    turns = [*_log_turns(6), text("done")]

    seen: list[list[Message]] = []
    agent = Agent(model=_provider(turns, seen), tools=[fetch_log], reflexion=False, grounding=False)
    events = [event async for event in agent.run("collect the logs")]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.final_message == "done"
    assert max(_tokens(m) for m in seen) <= _WINDOW
    for messages in seen:
        _assert_valid(messages)

    unmanaged = Agent(
        model=_provider(turns, []),
        tools=[fetch_log],
        conversation_manager=NullManager(),
        reflexion=False,
        grounding=False,
    )
    try:
        events = [event async for event in unmanaged.run("collect the logs")]
    except RuntimeError:
        failed = True
    else:
        failed = not any(
            isinstance(e, TerminateEvent) and e.final_message == "done" for e in events
        )
    assert failed, "without management the same run must overflow"


@pytest.mark.asyncio
async def test_a_paused_call_survives_compaction_and_resumes() -> None:
    seen: list[list[Message]] = []
    turns = [*_log_turns(3), tool_call("needs_approval", call_id="hold"), text("approved, done")]
    agent = Agent(
        model=_provider(turns, seen),
        tools=[fetch_log, needs_approval],
        reflexion=False,
        grounding=False,
    )

    events = [event async for event in agent.run("collect, then ask")]
    assert any(isinstance(e, InterruptEvent) for e in events)

    resumed = [event async for event in agent.resume("approve")]

    assert next(e for e in resumed if isinstance(e, TerminateEvent)).final_message == (
        "approved, done"
    )
    last = seen[-1]
    _assert_valid(last)
    assert any(
        m.role == Role.ASSISTANT and any(tc.id == "hold" for tc in m.tool_calls) for m in last
    )
    assert any(m.role == Role.TOOL and m.tool_call_id == "hold" for m in last)


def _conversation(parts: int) -> list[Message]:
    messages = [Message.system("you are helpful"), Message.user("collect the logs")]
    for i in range(parts):
        messages.append(
            Message.assistant(
                content=None,
                tool_calls=[ToolCall(id=f"c{i}", name="fetch_log", arguments={"part": i})],
            )
        )
        messages.append(Message(role=Role.TOOL, content=_OUTPUT, tool_call_id=f"c{i}"))
    return messages


@pytest.mark.parametrize("head_turns", [0, 1, 2, 3])
def test_a_cut_never_splits_a_call_from_its_result(head_turns: int) -> None:
    compactor = LLMCompactor(context_length=_WINDOW, head_turns=head_turns, tool_output_ttl_turns=0)

    out = compactor.apply(_conversation(5))

    assert _tokens(out) <= _WINDOW
    _assert_valid(out)
    assert out[0].role == Role.SYSTEM


def test_the_final_held_call_is_kept_without_a_result() -> None:
    messages = _conversation(4)
    messages.append(
        Message.assistant(
            content=None, tool_calls=[ToolCall(id="hold", name="needs_approval", arguments={})]
        )
    )

    out = LLMCompactor(context_length=_WINDOW, tool_output_ttl_turns=0).apply(messages)

    assert out[-1].tool_calls[0].id == "hold"


# --- Naming the window for a model the seed table does not know -------------
#
# A self-hosted model behind an OpenAI-compatible gateway (vLLM, LiteLLM) has
# no metadata entry, so before ``context_window`` there was no way to get the
# token-counting default for it short of building a compactor by hand.


def _unknown_model(**reported: Any) -> FunctionModel:
    model = FunctionModel(lambda m, t: text("ok"))
    model.config = SimpleNamespace(model="vllm:tulip-test-unlisted", **reported)  # type: ignore[attr-defined]
    return model


def _agent(model: Any, **kwargs: Any) -> Agent:
    return Agent(model=model, tools=[], reflexion=False, grounding=False, **kwargs)


@pytest.fixture(autouse=True)
def _no_window_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TULIP_CONTEXT_WINDOW", raising=False)


def test_an_explicit_window_counts_tokens_for_an_unknown_model() -> None:
    agent = _agent(_unknown_model(), context_window=64_000)

    assert isinstance(agent._conversation_manager, LLMCompactor)
    assert agent._conversation_manager.context_length == 64_000


def test_an_explicit_window_overrides_the_metadata_window() -> None:
    agent = _agent(_provider([text("ok")], []), context_window=8_000)

    assert isinstance(agent._conversation_manager, LLMCompactor)
    assert agent._conversation_manager.context_length == 8_000


def test_the_environment_names_the_window_when_the_config_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TULIP_CONTEXT_WINDOW", "131072")

    agent = _agent(_unknown_model())

    assert isinstance(agent._conversation_manager, LLMCompactor)
    assert agent._conversation_manager.context_length == 131_072
    explicit = _agent(_unknown_model(), context_window=32_000)
    assert explicit._conversation_manager.context_length == 32_000


@pytest.mark.parametrize("raw", ["lots", "0", "-5"])
def test_an_invalid_environment_window_is_ignored_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("TULIP_CONTEXT_WINDOW", raw)

    with caplog.at_level("WARNING", logger="tulip.agent.initializer"):
        agent = _agent(_unknown_model())

    assert isinstance(agent._conversation_manager, SlidingWindowManager)
    assert any(r.getMessage().startswith("Ignoring TULIP_CONTEXT_WINDOW") for r in caplog.records)


def test_a_window_the_model_reports_is_used_when_nothing_else_names_one() -> None:
    agent = _agent(_unknown_model(context_window=40_960))

    assert isinstance(agent._conversation_manager, LLMCompactor)
    assert agent._conversation_manager.context_length == 40_960


def test_metadata_wins_over_a_window_the_model_reports() -> None:
    model = _provider([text("ok")], [])
    model.config.context_length = 999_999  # type: ignore[attr-defined]

    agent = _agent(model)

    assert agent._conversation_manager.context_length == _WINDOW


def test_an_explicit_manager_wins_over_every_window() -> None:
    manager = NullManager()

    agent = _agent(_unknown_model(), context_window=64_000, conversation_manager=manager)

    assert agent._conversation_manager is manager


def test_the_message_window_fallback_says_how_to_name_the_window_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tulip.agent import initializer

    initializer._warned_unknown.discard("vllm:tulip-test-unlisted")
    with caplog.at_level("WARNING", logger="tulip.agent.initializer"):
        first = _agent(_unknown_model())
        _agent(_unknown_model())

    assert isinstance(first._conversation_manager, SlidingWindowManager)
    warnings = [r.getMessage() for r in caplog.records if "context_window" in r.getMessage()]
    assert len(warnings) == 1
    assert "TULIP_CONTEXT_WINDOW" in warnings[0]


def test_an_explicit_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _agent(_unknown_model(), context_window=0)
