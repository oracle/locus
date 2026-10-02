# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarising compaction: a long run keeps working when its context fills.

The compactor is exercised on its own (stages, invariants, the incremental
summary, the thrash guard) and through the agent loop with a scripted model
(the run continues after a compaction and still finishes, a run that cannot fit
ends with ``context_exhausted``, the pre-compact hook steers or skips it). The
last test runs a 200-step session through a small window and checks that the
final answer still knows a decision made at the start.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tulip.agent import Agent, CompactionConfig
from tulip.core.events import CompactionEvent, TerminateEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.memory.compaction import (
    CLEARED_OUTPUT_KEY,
    SUMMARY_SYSTEM_PROMPT,
    CompactionTracker,
    ContextCompactor,
    is_summary_message,
)
from tulip.models.base import ModelResponse
from tulip.testing import FunctionModel, text, tool_call
from tulip.tools.decorator import tool


# --- helpers ----------------------------------------------------------------


class Summariser:
    """A summary model that records what it was asked."""

    def __init__(self, reply: Any = "SUMMARY") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        assert tools is None, "a summary call offers no tools"
        assert messages[0].content == SUMMARY_SYSTEM_PROMPT
        prompt = messages[-1].content or ""
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        if isinstance(reply, BaseException):
            raise reply
        return ModelResponse(
            message=Message.assistant(content=reply),
            usage={"prompt_tokens": 100, "completion_tokens": 40},
        )


def _turn(i: int, *, said: str = "", output: str = "") -> list[Message]:
    return [
        Message.assistant(
            content=said or f"step {i}",
            tool_calls=[ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"src/f{i}.py"})],
        ),
        Message(
            role=Role.TOOL, content=output or f"out{i}", tool_call_id=f"c{i}", name="read_file"
        ),
    ]


def _conversation(steps: int, *, said: int = 0, output: int = 0) -> list[Message]:
    messages = [Message.system("SYSTEM PROMPT"), Message.user("TASK: port the parser")]
    for i in range(steps):
        messages += _turn(i, said=f"step {i} " + "s" * said, output=f"out{i} " + "o" * output)
    return messages


def _assert_pairs(messages: list[Message]) -> None:
    calls = {tc.id for m in messages if m.role == Role.ASSISTANT for tc in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.role == Role.TOOL}
    assert all(m.tool_call_id in calls for m in messages if m.role == Role.TOOL), "orphan result"
    for m in messages[:-1]:
        if m.role == Role.ASSISTANT and m.tool_calls:
            assert all(tc.id in answered for tc in m.tool_calls), "call split from its result"


def _compactor(summary_model: Any = None, **kwargs: Any) -> ContextCompactor:
    # 10k window: 2k reserved, 8k usable, threshold 7,200, 2k of tool output kept.
    return ContextCompactor(context_length=10_000, summary_model=summary_model, **kwargs)


# --- thresholds ---------------------------------------------------------------


def test_defaults_scale_with_the_window() -> None:
    large = ContextCompactor(context_length=200_000)
    small = ContextCompactor(context_length=32_000)

    assert large.reserved_tokens == 20_000
    assert large.threshold == int(180_000 * 0.9)
    assert large.tool_output_keep_tokens == 40_000
    assert small.reserved_tokens == 6_400
    assert small.tool_output_keep_tokens == (32_000 - 6_400) // 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_length": 0},
        {"context_length": 1_000, "trigger_fraction": 0.0},
        {"context_length": 1_000, "tail_turns": 0},
        {"context_length": 1_000, "tail_token_fraction": 1.0},
        {"context_length": 1_000, "reserved_tokens": 1_000},
        {"context_length": 1_000, "min_iterations_between_summaries": -1},
    ],
)
def test_invalid_settings_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ContextCompactor(**kwargs)


@pytest.mark.asyncio
async def test_nothing_happens_under_the_threshold() -> None:
    compactor = _compactor(Summariser())

    assert (
        await compactor.compact(_conversation(3), iteration=3, tracker=CompactionTracker()) is None
    )


@pytest.mark.asyncio
async def test_requests_pass_through_because_the_loop_compacts_the_state() -> None:
    compactor = _compactor()
    messages = _conversation(30, output=4_000)

    assert compactor.apply(messages) == messages
    assert await compactor.async_apply(messages) == messages
    assert compactor.tool_tokens(None) == 0
    assert compactor.tool_tokens([{"name": "x" * 400}]) > 100
    assert "summarises=False" in repr(compactor)


def test_a_report_without_usage_is_ignored() -> None:
    tracker = CompactionTracker()

    tracker.observe({}, 5)

    assert tracker.reported_tokens is None


def test_the_providers_count_wins_over_a_lower_estimate() -> None:
    compactor = _compactor()
    messages = _conversation(2)
    tracker = CompactionTracker()
    tracker.observe({"prompt_tokens": 7_000, "cache_read_input_tokens": 500}, len(messages) - 1)

    measured = compactor.measure(messages, tracker=tracker)

    assert measured == 7_500 + compactor.tokens(messages[-1:])
    tracker.forget_report()
    assert compactor.measure(messages, tracker=tracker) == compactor.tokens(messages)


# --- stage 1: clearing tool output ------------------------------------------


@pytest.mark.asyncio
async def test_old_tool_output_is_cleared_first_and_named_in_a_stub() -> None:
    summariser = Summariser()
    compactor = _compactor(summariser)
    messages = _conversation(10, output=3_900)  # just under 1,000 tokens of output per step

    outcome = await compactor.compact(messages, iteration=10, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.stage == "prune"
    assert not outcome.exhausted
    assert outcome.tokens_after < outcome.threshold <= outcome.tokens_before
    assert summariser.prompts == [], "clearing alone was enough: no summary call"
    results = [m for m in outcome.messages if m.role == Role.TOOL]
    assert [bool(m.metadata.get(CLEARED_OUTPUT_KEY)) for m in results] == [True] * 8 + [False] * 2
    assert "read_file(path='src/f0.py')" in (results[0].content or "")
    assert results[-1].content == messages[-1].content, "the newest output is kept whole"
    assert len(outcome.messages) == len(messages)
    _assert_pairs(outcome.messages)


# --- stage 2: the summary -----------------------------------------------------


def _long_session() -> list[Message]:
    """Text-heavy turns (clearing cannot help), a later user turn and a memory block."""
    messages = _conversation(6, said=2_000, output=400)
    messages.insert(4, Message.user("FIRST: use pytest"))
    messages.insert(5, Message.system("[Budget Warning] 10 iterations left"))
    messages.append(Message.user("ALSO: keep the old API working"))
    messages.append(
        Message(
            role=Role.SYSTEM,
            content="[Long-term Memory] prefers uv",
            metadata={"tulip_memory_block": True},
        )
    )
    for i in range(6, 16):
        messages += _turn(i, said=f"step {i} " + "s" * 2_000, output=f"out{i} " + "o" * 400)
    return messages


@pytest.mark.asyncio
async def test_a_summary_keeps_the_head_the_latest_user_turn_and_a_verbatim_tail() -> None:
    summariser = Summariser("## Goal and constraints\nport the parser")
    compactor = _compactor(summariser)
    messages = _long_session()

    outcome = await compactor.compact(messages, iteration=16, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.stage == "summarize"
    assert not outcome.exhausted
    assert outcome.tokens_after < outcome.threshold
    out = outcome.messages
    assert out[0].content == "SYSTEM PROMPT"
    assert out[1].content == "TASK: port the parser"
    assert is_summary_message(out[2])
    assert "port the parser" in (out[2].content or "")
    assert any(m.content == "ALSO: keep the old API working" for m in out), "latest user turn"
    assert any(m.metadata.get("tulip_memory_block") for m in out), "memory block is pinned"
    tail = [m for m in out[3:] if m.role in (Role.ASSISTANT, Role.TOOL)]
    assert tail == messages[-len(tail) :], "the tail is the newest turns, verbatim"
    assert 1 <= len(tail) // 2 <= compactor.tail_turns
    _assert_pairs(out)
    # The summariser read the folded turns, with their tool calls.
    assert "-> calls read_file" in summariser.prompts[0]
    assert "step 0" in summariser.prompts[0]
    assert "[user]\nFIRST: use pytest" in summariser.prompts[0], "earlier user turns are folded"
    assert "[system note]\n[Budget Warning]" in summariser.prompts[0]
    assert summariser.kwargs[0]["max_tokens"] == compactor.summary_max_tokens
    calls = len(summariser.prompts)
    assert outcome.usage == {"prompt_tokens": 100 * calls, "completion_tokens": 40 * calls}


@pytest.mark.asyncio
async def test_the_next_summary_builds_on_the_previous_one() -> None:
    summariser = Summariser()
    summariser.reply = lambda prompt: f"SUMMARY #{len(summariser.prompts)}"
    compactor = _compactor(summariser)
    tracker = CompactionTracker()
    first = await compactor.compact(_long_session(), iteration=16, tracker=tracker)
    assert first is not None
    assert first.summary is not None
    calls_before = len(summariser.prompts)
    grown = list(first.messages)
    for i in range(100, 112):
        grown += _turn(i, said=f"step {i} " + "s" * 2_000, output="o" * 400)

    second = await compactor.compact(grown, iteration=30, tracker=tracker)

    assert second is not None
    assert second.stage == "summarize"
    opening = summariser.prompts[calls_before]
    assert f"<previous_summary>\n{first.summary}\n</previous_summary>" in opening, (
        "the previous summary is the starting point"
    )
    summaries = [m for m in second.messages if is_summary_message(m)]
    assert len(summaries) == 1
    assert second.summary is not None
    assert second.summary in (summaries[0].content or "")
    assert first.summary not in (summaries[0].content or ""), "replaced, not stacked"
    _assert_pairs(second.messages)


@pytest.mark.asyncio
async def test_history_larger_than_the_window_is_summarised_in_chunks() -> None:
    summariser = Summariser()
    summariser.reply = lambda prompt: f"S{len(summariser.prompts)}"
    compactor = _compactor(summariser)
    messages = _conversation(60, said=2_000, output=400)  # ~37k tokens, window 10k

    outcome = await compactor.compact(messages, iteration=60, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.stage == "summarize"
    assert len(summariser.prompts) > 1
    for _previous, prompt in zip(summariser.prompts, summariser.prompts[1:], strict=False):
        assert "<previous_summary>" in prompt, "each chunk folds into the summary so far"
    assert all(len(p) // 4 < compactor.usable_tokens for p in summariser.prompts)


@pytest.mark.asyncio
async def test_a_failing_summary_falls_back_to_a_marked_truncation() -> None:
    summariser = Summariser(RuntimeError("overloaded"))
    compactor = _compactor(summariser)

    outcome = await compactor.compact(_long_session(), iteration=16, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.stage == "truncate"
    assert not outcome.exhausted
    assert len(summariser.prompts) == 2, "retried once"
    note = next(m for m in outcome.messages if is_summary_message(m))
    assert "without a summary" in (note.content or "")
    assert "overloaded" in (note.content or "")
    _assert_pairs(outcome.messages)


@pytest.mark.asyncio
async def test_an_empty_summary_is_retried_then_truncated() -> None:
    summariser = Summariser("   ")

    outcome = await _compactor(summariser).compact(
        _long_session(), iteration=16, tracker=CompactionTracker()
    )

    assert outcome is not None
    assert outcome.stage == "truncate"
    assert outcome.detail == "the summary model returned no text"
    assert len(summariser.prompts) == 2


@pytest.mark.asyncio
async def test_without_a_summary_model_older_history_is_dropped_with_a_note() -> None:
    outcome = await _compactor(None).compact(
        _long_session(), iteration=16, tracker=CompactionTracker()
    )

    assert outcome is not None
    assert outcome.stage == "truncate"
    assert outcome.usage == {}


# --- the loop guards ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_summary_within_the_guard_window_is_thrashing() -> None:
    summariser = Summariser()
    compactor = _compactor(summariser, min_iterations_between_summaries=3)
    tracker = CompactionTracker(last_summary_iteration=14)

    outcome = await compactor.compact(_long_session(), iteration=16, tracker=tracker)

    assert outcome is not None
    assert outcome.exhausted
    assert "thrashing" in (outcome.detail or "")
    assert summariser.prompts == [], "no tokens spent on a summary that cannot help"


@pytest.mark.asyncio
async def test_a_fixed_part_bigger_than_the_window_is_exhausted_not_looped() -> None:
    summariser = Summariser()
    messages = [Message.system("x" * 40_000), Message.user("TASK")]

    outcome = await _compactor(summariser).compact(
        messages, iteration=1, tracker=CompactionTracker()
    )

    assert outcome is not None
    assert outcome.exhausted
    assert "nothing left to summarise" in (outcome.detail or "")
    assert summariser.prompts == []


# --- through the agent loop ---------------------------------------------------


@tool
def work(step: int) -> str:
    """Do one step of the task."""
    return f"result of step {step}: " + "r" * 1_200


def _is_summary_call(messages: list[Message]) -> bool:
    return bool(messages) and messages[0].content == SUMMARY_SYSTEM_PROMPT


def _agent(handler: Any, **kwargs: Any) -> Agent:
    kwargs.setdefault("max_iterations", 60)
    return Agent(
        model=FunctionModel(handler),
        tools=[work],
        system_prompt="You are a coding agent.",
        reflexion=False,
        grounding=False,
        context_window=8_000,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_run_continues_after_a_summary_and_finishes() -> None:
    agent_calls: list[list[Message]] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            return "## Next step\nkeep working"
        agent_calls.append(messages)
        step = len(agent_calls)
        if step > 20:
            return text("all done")
        return tool_call("work", call_id=f"w{step}", content="thinking " + "t" * 2_400, step=step)

    events = [e async for e in _agent(handler).run("do 20 steps")]

    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert any(e.stage == "summarize" for e in compactions)
    assert all(not e.exhausted and e.tokens_after < e.threshold for e in compactions)
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"
    assert terminate.final_message == "all done"
    for messages in agent_calls:
        assert messages[0].content == "You are a coding agent."
        assert messages[1].content == "do 20 steps"
        _assert_pairs(messages)
    after = agent_calls[-1]
    assert any(is_summary_message(m) for m in after)
    assert after[-1].role == Role.TOOL, "the model picks up from the latest result, unprompted"


@pytest.mark.asyncio
async def test_compaction_that_cannot_fit_ends_the_run_with_context_exhausted() -> None:
    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            return "summary"
        # Each turn alone is close to half the window: no summary can hold it.
        step = len(messages)
        return tool_call("work", call_id=f"w{step}", content="t" * 11_000, step=step)

    agent = _agent(handler)
    result = await agent.arun("go")

    assert result.stop_reason == "context_exhausted"
    assert result.message.startswith("[context exhausted]")


@pytest.mark.asyncio
async def test_a_system_prompt_bigger_than_the_window_stops_before_calling_the_model() -> None:
    calls: list[list[Message]] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        calls.append(messages)
        return text("never")

    agent = Agent(
        model=FunctionModel(handler),
        tools=[work],
        system_prompt="x" * 40_000,
        reflexion=False,
        grounding=False,
        context_window=8_000,
    )
    events = [e async for e in agent.run("go")]

    assert next(e for e in events if isinstance(e, TerminateEvent)).reason == "context_exhausted"
    assert calls == []


class _Hook:
    def __init__(self, *, cancel: bool = False) -> None:
        self.cancel = cancel
        self.seen: list[tuple[int, int, int]] = []

    async def on_before_compaction(self, event: Any) -> None:
        self.seen.append((event.tokens, event.threshold, len(event.messages)))
        event.instructions = "Keep every step number."
        if self.cancel:
            event.cancel = True


@pytest.mark.asyncio
async def test_the_pre_compact_hook_sees_the_full_history_and_steers_the_summary() -> None:
    prompts: list[str] = []
    steps: list[int] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            prompts.append(messages[-1].content or "")
            return "summary"
        steps.append(len(steps) + 1)
        if len(steps) > 12:
            return text("done")
        return tool_call("work", call_id=f"w{len(steps)}", content="t" * 2_400, step=len(steps))

    hook = _Hook()
    result = await _agent(handler, hooks=[hook]).arun("go")

    assert result.stop_reason == "complete"
    assert hook.seen
    assert all(tokens >= threshold for tokens, threshold, _ in hook.seen)
    assert prompts
    assert all("Keep every step number." in p for p in prompts)


@pytest.mark.asyncio
async def test_a_hook_can_skip_compaction() -> None:
    steps: list[int] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        assert not _is_summary_call(messages), "the hook cancelled every compaction"
        steps.append(len(steps) + 1)
        if len(steps) > 8:
            return text("done")
        return tool_call("work", call_id=f"w{len(steps)}", content="t" * 2_400, step=len(steps))

    hook = _Hook(cancel=True)
    events = [e async for e in _agent(handler, hooks=[hook]).run("go")]

    assert hook.seen
    assert not any(isinstance(e, CompactionEvent) for e in events)


def test_compaction_settings_reach_the_compactor() -> None:
    summary_model = Summariser()
    agent = _agent(
        lambda m, t: text("ok"),
        compaction=CompactionConfig(
            trigger_fraction=0.5,
            reserved_tokens=1_000,
            tail_turns=2,
            summary_model=summary_model,
            min_iterations_between_summaries=5,
        ),
    )

    compactor = agent._conversation_manager
    assert isinstance(compactor, ContextCompactor)
    assert compactor.threshold == int((8_000 - 1_000) * 0.5)
    assert compactor.tail_turns == 2
    assert compactor.summary_model is summary_model
    assert compactor.min_iterations_between_summaries == 5


# --- a long session -----------------------------------------------------------

_DECISION = re.compile(r"DECISION: [^\n]+")


def _faithful_summary(prompt: str) -> str:
    """A deterministic summariser that keeps every decision it is shown."""
    decisions = list(dict.fromkeys(_DECISION.findall(prompt)))
    steps = [int(n) for n in re.findall(r"result of step (\d+)", prompt)]
    done = f"steps up to {max(steps)}" if steps else "earlier steps"
    return "## Decisions made\n" + "\n".join(f"- {d}" for d in decisions) + f"\n## Verified\n{done}"


@pytest.mark.asyncio
async def test_a_200_step_session_in_a_small_window_still_remembers_its_first_decisions() -> None:
    window = 8_000
    agent_calls: list[list[Message]] = []
    summary_prompts: list[str] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            summary_prompts.append(messages[-1].content or "")
            return _faithful_summary(messages[-1].content or "")
        agent_calls.append(messages)
        step = len(agent_calls)
        if step > 200:
            # The final answer cites the decisions the model can still see.
            seen = "\n".join(m.content or "" for m in messages)
            return text("Final report. " + "; ".join(dict.fromkeys(_DECISION.findall(seen))))
        said = {
            2: "DECISION: store sessions in PostgreSQL, not Redis",
            5: "DECISION: keep the v1 API as a shim",
        }.get(step, f"working on step {step}: " + "n" * 400)
        return tool_call("work", call_id=f"w{step}", content=said, step=step)

    agent = _agent(handler, max_iterations=210)
    events = [e async for e in agent.run("migrate the session store")]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"
    assert "PostgreSQL" in (terminate.final_message or "")
    assert "v1 API as a shim" in (terminate.final_message or "")
    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert sum(e.stage == "summarize" for e in compactions) >= 2
    assert any("<previous_summary>" in p for p in summary_prompts), "summaries were incremental"
    # The decisions reached the end through the summary, not the raw turn.
    final_request = agent_calls[-1]
    assert not any(
        m.role == Role.ASSISTANT and "PostgreSQL" in (m.content or "") for m in final_request
    )
    for messages in agent_calls:
        assert sum(len(m.content or "") for m in messages) // 4 < window
        assert messages[1].content == "migrate the session store"
        _assert_pairs(messages)


@pytest.mark.asyncio
async def test_compaction_reaches_the_observability_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    from tulip.observability import agent_bridge

    published: list[tuple[str, dict[str, Any]]] = []

    async def record(name: str, **fields: Any) -> None:
        published.append((name, fields))

    monkeypatch.setattr(agent_bridge, "emit", record)
    await agent_bridge.bridge_tulip_event(
        CompactionEvent(
            iteration=4,
            stage="summarize",
            tokens_before=9_000,
            tokens_after=2_000,
            threshold=7_200,
            context_window=10_000,
            messages_before=40,
            messages_after=12,
        )
    )

    assert published == [
        (
            "agent.context.compacted",
            {
                "iteration": 4,
                "stage": "summarize",
                "tokens_before": 9_000,
                "tokens_after": 2_000,
                "threshold": 7_200,
                "exhausted": False,
            },
        )
    ]


@pytest.mark.parametrize(("flag", "enabled"), [(True, True), (False, False), (None, True)])
def test_compaction_accepts_a_plain_flag(flag: bool | None, enabled: bool) -> None:
    from tulip.agent import AgentConfig

    assert AgentConfig(model="openai:gpt-4o", compaction=flag).compaction.enabled is enabled


def test_a_summary_model_named_by_string_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    built = Summariser()
    names: list[str] = []

    def fake_get_model(name: str, **kwargs: Any) -> Any:
        names.append(name)
        return built

    monkeypatch.setattr("tulip.agent.agent.get_model", fake_get_model)
    agent = _agent(
        lambda m, t: text("ok"), compaction=CompactionConfig(summary_model="openai:gpt-4o-mini")
    )

    assert names == ["openai:gpt-4o-mini"]
    assert isinstance(agent._conversation_manager, ContextCompactor)
    assert agent._conversation_manager.summary_model is built
