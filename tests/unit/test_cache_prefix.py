# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Each request starts with the whole of the one before it, except at a compaction.

Providers cache a request's prefix: OpenAI-compatible ones automatically, and
Anthropic up to a ``cache_control`` breakpoint. Either way a hit needs the
start of the request to be byte-identical to an earlier request, so a harness
that rewrites an early message between iterations pays full price for the
whole conversation on every call of a long run. These tests drive scripted
multi-step sessions through the real loop and check every request against the
previous one, in Tulip's messages and in what the OpenAI and Anthropic
adapters put on the wire.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import CompactionEvent, CustomEvent, TerminateEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.memory.conversation import SlidingWindowManager
from tulip.memory.manager import BaseMemoryManager, Memory, MemoryType
from tulip.models.base import ModelResponse
from tulip.models.native.anthropic import AnthropicModel
from tulip.models.native.openai import OpenAIModel
from tulip.testing import FunctionModel
from tulip.tools.decorator import tool


@tool(idempotent=False)
def read(path: str) -> str:
    """Read a file."""
    return f"# {path}\n" + "\n".join(f"line {n} of {path}: value = {n * 7}" for n in range(120))


def _scripted(steps: int, *, parallel: bool = False) -> tuple[FunctionModel, list[list[Message]]]:
    """A model that reads ``steps`` files, one per turn, then answers; and its requests."""
    requests: list[list[Message]] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        # Counted by calls, not by the assistant turns in view: a message
        # window hides the early ones.
        turn = len(requests)
        requests.append(list(messages))
        if turn >= steps:
            return ModelResponse(
                message=Message.assistant("Done."),
                usage={"prompt_tokens": 1_000, "completion_tokens": 10},
            )
        calls = [ToolCall(id=f"c{turn}a", name="read", arguments={"path": f"f{turn}.py"})]
        if parallel:
            calls.append(ToolCall(id=f"c{turn}b", name="read", arguments={"path": f"g{turn}.py"}))
        return ModelResponse(
            message=Message.assistant(f"Reading file {turn}.", tool_calls=calls),
            usage={"prompt_tokens": 1_000, "completion_tokens": 10},
        )

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model="tulip-test-cache-prefix")  # type: ignore[attr-defined]
    return model, requests


def _wire(messages: list[Message]) -> list[str]:
    return [
        json.dumps(
            {
                "role": m.role.value,
                "content": m.content,
                "tool_calls": [(c.id, c.name, c.arguments) for c in m.tool_calls],
                "tool_call_id": m.tool_call_id,
            },
            sort_keys=True,
        )
        for m in messages
    ]


def _extends(before: list[str], after: list[str]) -> bool:
    return len(after) >= len(before) and after[: len(before)] == before


def _openai_wire(messages: list[Message]) -> list[str]:
    model = OpenAIModel(model="gpt-test", api_key="sk-test")
    return [json.dumps(m, sort_keys=True) for m in model._convert_messages(messages)]


def _anthropic_blocks(messages: list[Message], *, prompt_cache: bool = False) -> list[str]:
    """The request's system prompt and conversation as a flat sequence of blocks.

    Anthropic's cache works on blocks: a note appended to the last user turn
    extends the prefix even though that turn's message object grows.
    """
    model = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-test", prompt_cache=prompt_cache)
    params, _ = model._request_params(messages, None, {})
    system = params.get("system") or []
    flat = [json.dumps(system if isinstance(system, str) else [b["text"] for b in system])]
    for turn in params["messages"]:
        content = turn["content"]
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            plain = {k: v for k, v in block.items() if k != "cache_control"}
            flat.append(json.dumps([turn["role"], plain], sort_keys=True, default=str))
    return flat


def _breaks(sequences: list[list[str]]) -> list[int]:
    """Indexes of the requests that did not start with the whole previous request."""
    return [i for i in range(1, len(sequences)) if not _extends(sequences[i - 1], sequences[i])]


async def _events(agent: Agent, prompt: str = "Fix the bug.", **kwargs: Any) -> list[Any]:
    return [e async for e in agent.run(prompt, **kwargs)]


async def test_a_long_run_only_ever_appends_to_its_requests() -> None:
    model, requests = _scripted(30)
    agent = Agent(
        model=model,
        tools=[read],
        max_iterations=60,
        context_window=1_000_000,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)

    assert isinstance(events[-1], TerminateEvent)
    assert len(requests) == 31
    assert _breaks([_wire(r) for r in requests]) == []
    assert _breaks([_openai_wire(r) for r in requests]) == []
    assert _breaks([_anthropic_blocks(r) for r in requests]) == []


async def test_a_run_rewrites_its_history_only_where_it_compacts() -> None:
    model, requests = _scripted(40)
    agent = Agent(
        model=model,
        tools=[read],
        max_iterations=80,
        # Small enough that forty file reads overflow it more than once.
        context_window=24_000,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)

    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert compactions, "the scripted run should have needed compacting"
    breaks = _breaks([_wire(r) for r in requests])
    # Every rewrite is a compaction: the request after each compaction event
    # is the only one allowed to differ from the request before it.
    assert len(breaks) <= len(compactions)
    # And compaction is batched: far fewer rewrites than requests.
    assert len(breaks) * 5 < len(requests)


async def test_the_message_window_slides_in_steps_not_one_message_per_request() -> None:
    model, requests = _scripted(30, parallel=True)
    # No context window known: the agent keeps a message window.
    agent = Agent(model=model, tools=[read], max_iterations=40, reflexion=False, grounding=False)
    assert isinstance(agent._conversation_manager, SlidingWindowManager)
    await _events(agent)

    # ``requests`` are what the model was sent, after the window.
    breaks = _breaks([_wire(r) for r in requests])
    assert breaks, "the window should have slid"
    # One message per request would break every request once the window is
    # full; a quarter-window step breaks about once per step.
    after_full = len(requests) - breaks[0]
    assert len(breaks) * 2 <= after_full


def test_a_mid_run_note_stays_where_it_was_written_in_the_window() -> None:
    manager = SlidingWindowManager(window_size=6, slide_step=2)
    history = [
        Message.system("instructions"),
        Message.user("task"),
        Message.assistant("a1"),
        Message.system("a note"),
        Message.assistant("a2"),
    ]
    out = manager.apply(history)
    assert [m.content for m in out] == ["instructions", "task", "a1", "a note", "a2"]


def test_the_window_holds_its_cut_until_a_whole_step_has_accumulated() -> None:
    manager = SlidingWindowManager(window_size=8, slide_step=4, preserve_first_user=False)
    history = [Message.system("instructions")]
    starts = []
    for n in range(20):
        history.append(Message.assistant(f"m{n}"))
        starts.append(manager.apply(history)[1].content)
    # The first kept message changes only when a whole step has piled up.
    assert len(set(starts)) == 4
    assert all(len(manager.apply(history)) - 1 <= 8 for _ in range(1))


def test_slide_step_must_fit_the_window() -> None:
    with pytest.raises(ValueError, match="slide_step"):
        SlidingWindowManager(window_size=4, slide_step=5)


class _TurnMemory(BaseMemoryManager):
    """Recalls a different memory every turn, as a relevance search does."""

    def __init__(self) -> None:
        self.turn = 0

    async def extract(self, messages: list[Message]) -> list[Memory]:
        return []

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        self.turn += 1
        return [Memory(type=MemoryType.PROJECT, key=f"k{self.turn}", content=f"fact {self.turn}")]

    async def save(self, memories: list[Memory]) -> None:
        return None


async def test_a_new_turn_reuses_the_last_turn_up_to_its_prompt_despite_new_memories() -> None:
    model, requests = _scripted(2)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        checkpointer=MemoryCheckpointer(),
        memory_manager=_TurnMemory(),
        reflexion=False,
        grounding=False,
    )
    await _events(agent, "first task", thread_id="t")
    turn_one_last = _openai_wire(requests[-1])
    first_of_turn_two = len(requests)
    await _events(agent, "second task", thread_id="t")
    turn_two_first = _openai_wire(requests[first_of_turn_two])

    shared = 0
    while shared < min(len(turn_one_last), len(turn_two_first)) and (
        turn_one_last[shared] == turn_two_first[shared]
    ):
        shared += 1
    # The system prompt and the first turn's prompt are reused; the memory
    # block that changed sits after them, not in front of them.
    assert shared >= 2
    assert "first task" in turn_two_first[1]
    assert "fact" not in "".join(turn_two_first[:2])


async def test_a_run_gets_one_note_to_converge_and_keeps_its_prefix() -> None:
    model, requests = _scripted(12)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        # 1,010 tokens a call: the note comes after the ninth, the stop after the eleventh.
        token_budget=11_000,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)

    nudges = [e for e in events if isinstance(e, CustomEvent) and e.name == "budget_nudge"]
    assert len(nudges) == 1
    assert nudges[0].data["budget"] == "token"
    notes = [m for m in requests[-1] if "[Budget note" in (m.content or "")]
    assert len(notes) == 1
    assert notes[0].role == Role.USER
    assert "do not stop with it unfinished" in (notes[0].content or "")
    assert _breaks([_wire(r) for r in requests]) == []


async def test_the_nudge_can_be_turned_off() -> None:
    model, requests = _scripted(12)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        token_budget=100_000,
        budget_nudge_at=None,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)
    assert not [e for e in events if isinstance(e, CustomEvent) and e.name == "budget_nudge"]
    assert not any("[Budget note" in (m.content or "") for m in requests[-1])


async def test_a_short_iteration_cap_is_not_paced_against() -> None:
    model, _ = _scripted(3)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        max_iterations=4,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)
    assert not [e for e in events if isinstance(e, CustomEvent) and e.name == "budget_nudge"]


async def test_a_long_iteration_cap_is() -> None:
    model, _ = _scripted(12)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        max_iterations=12,
        reflexion=False,
        grounding=False,
    )
    events = await _events(agent)
    nudges = [e for e in events if isinstance(e, CustomEvent) and e.name == "budget_nudge"]
    assert [n.data["budget"] for n in nudges] == ["iteration"]


def test_anthropic_caching_marks_the_end_of_the_conversation_within_four_breakpoints() -> None:
    model = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-test", prompt_cache=True)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    history = [
        Message.system("instructions"),
        Message.user("task"),
        Message.assistant("reading", tool_calls=[ToolCall(id="t1", name="read", arguments={})]),
        Message(role=Role.TOOL, tool_call_id="t1", name="read", content="contents"),
    ]
    params, _ = model._request_params(history, tools, {})
    text = json.dumps(params)
    assert text.count("cache_control") <= 4
    assert "cache_control" in json.dumps(params["tools"][-1])
    last_block = params["messages"][-1]["content"][-1]
    assert last_block["type"] == "tool_result"
    assert last_block["cache_control"] == {"type": "ephemeral"}
    # The earlier user turn carries the second rolling breakpoint.
    first_user = params["messages"][0]["content"]
    assert first_user[-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_never_exceeds_four_breakpoints_with_a_split_system_prompt() -> None:
    model = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-test", prompt_cache=True)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    history = [
        Message.system("instructions"),
        Message.system("memory"),
        Message.user("task"),
        Message.assistant("ok"),
        Message.user("more"),
    ]
    params, _ = model._request_params(history, tools, {})
    assert json.dumps(params).count("cache_control") == 4
    assert params["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(params["messages"][0])


def test_anthropic_caching_off_marks_nothing() -> None:
    model = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-test")
    params, _ = model._request_params([Message.user("task")], None, {})
    assert "cache_control" not in json.dumps(params)


def _strip_markers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_markers(v) for v in value]
    return value


async def test_anthropic_turns_keep_their_shape_when_the_breakpoint_moves_on() -> None:
    """A turn a breakpoint made into blocks is still blocks in the next request.

    Otherwise the request after it would send the same turn as a plain string,
    and the bytes the cache was written with would not match.
    """
    model, requests = _scripted(6)
    agent = Agent(
        model=model,
        tools=[read],
        context_window=1_000_000,
        reflexion=False,
        grounding=False,
    )
    await _events(agent)
    caching = AnthropicModel(model="claude-sonnet-5-5", api_key="sk-test", prompt_cache=True)
    sent = [caching._request_params(r, None, {})[0]["messages"] for r in requests]
    for before, after in itertools.pairwise(sent):
        settled = [_strip_markers(m) for m in before[:-1]]
        assert [_strip_markers(m) for m in after[: len(settled)]] == settled
    assert all(isinstance(m["content"], list) for turns in sent for m in turns)
