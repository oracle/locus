# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tiered compaction: each tier at its own fraction of the window, pins kept.

Every tier is exercised on its own (clear, window, summarise, native,
relevance), then the pipeline (triggers, order, per-tier outcomes, the
thrash guard, exhaustion), pinning through every tier, the window source
(metadata, configured, bound by the agent), and the agent loop (one
``CompactionEvent`` per tier, the run completes, the bus payload).
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pytest

from locus.agent import Agent, CompactionConfig
from locus.core.events import CompactionEvent, TerminateEvent
from locus.core.messages import Message, Role, ToolCall
from locus.memory.compaction import (
    CLEARED_OUTPUT_KEY,
    SUMMARY_SYSTEM_PROMPT,
    CompactionTracker,
    ContextCompactor,
    is_pinned,
    is_summary_message,
    pin,
)
from locus.memory.tiers import (
    AGENT_MODEL,
    DROPPED_NOTE_KEY,
    ClearToolResults,
    CompactionTier,
    Native,
    Relevance,
    SlidingWindow,
    Summarize,
    TierContext,
    TieredCompaction,
    TierResult,
    default_tiers,
    resolve_context_length,
)
from locus.models.base import ModelResponse
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


# --- helpers ----------------------------------------------------------------


class Summariser:
    """A summary model that records what it was asked."""

    def __init__(self, reply: Any = "## Goal\nSUMMARY") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        assert messages[0].content == SUMMARY_SYSTEM_PROMPT
        prompt = messages[-1].content or ""
        self.prompts.append(prompt)
        reply = self.reply(prompt) if callable(self.reply) else self.reply
        return ModelResponse(
            message=Message.assistant(content=reply),
            usage={"prompt_tokens": 100, "completion_tokens": 40},
        )


class NativeModel:
    """A model whose provider compacts."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return ModelResponse(message=Message.assistant(content="ok"))

    async def compact_history(
        self, messages: list[Message], *, instructions: str | None = None
    ) -> tuple[list[Message], dict[str, int]]:
        self.calls.append(list(messages))
        folded = Message.assistant(content="[compacted by the provider]")
        return [folded, *messages[-2:]], {"prompt_tokens": 500, "completion_tokens": 20}


@dataclass
class Vector:
    embedding: list[float]


class Embedder:
    """Scores a text by the words it shares with a vocabulary."""

    vocabulary = ("parser", "database", "deploy", "tests", "docs")

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        self.batches.append(list(texts))
        return [
            Vector([1.0 if word in text.lower() else 0.0 for word in self.vocabulary])
            for text in texts
        ]


def _turn(i: int, *, said: str = "", output: str = "", tool: str = "read_file") -> list[Message]:
    return [
        Message.assistant(
            content=said or f"step {i}",
            tool_calls=[ToolCall(id=f"c{i}", name=tool, arguments={"path": f"src/f{i}.py"})],
        ),
        Message(role=Role.TOOL, content=output or f"out{i}", tool_call_id=f"c{i}", name=tool),
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


def _tiered(tiers: list[CompactionTier] | None = None, **kwargs: Any) -> TieredCompaction:
    # 10k window: 2k reserved, 8k usable. Triggers are fractions of the 8k.
    kwargs.setdefault("context_length", 10_000)
    kwargs.setdefault("summary_model", None)
    return TieredCompaction(tiers, **kwargs)


async def _compact(compactor: TieredCompaction, messages: list[Message], **kwargs: Any) -> Any:
    kwargs.setdefault("iteration", 20)
    kwargs.setdefault("tracker", CompactionTracker())
    return await compactor.compact(messages, **kwargs)


# --- the window -----------------------------------------------------------------


def test_the_window_comes_from_metadata_then_the_configured_length() -> None:
    assert resolve_context_length("openai:gpt-4o", None) == 128_000
    assert resolve_context_length("openai:gpt-4o", 8_000) == 8_000, "an explicit window wins"
    assert resolve_context_length("made-up-model", 8_000) == 8_000
    assert resolve_context_length(None, None) is None
    assert (
        resolve_context_length(FunctionModel(lambda m, t: "x", context_window=4_000), None) == 4_000
    )
    with pytest.raises(ValueError):
        resolve_context_length(None, 0)

    by_metadata = TieredCompaction(model="openai:gpt-4o", summary_model=None)
    assert by_metadata.bound
    assert by_metadata.context_length == 128_000
    assert by_metadata.threshold == int((128_000 - 20_000) * 0.5)


def test_an_unbound_pipeline_binds_later_or_refuses_to_run() -> None:
    compactor = TieredCompaction(summary_model=None)
    assert not compactor.bound
    assert "context_length=None" in repr(compactor)

    compactor.bind(context_length=10_000)
    assert compactor.bound
    assert compactor.context_length == 10_000
    assert compactor.threshold == 4_000

    with pytest.raises(ValueError, match="context window"):
        TieredCompaction(summary_model=None).bind()


async def test_an_unbound_pipeline_binds_to_the_model_it_compacts_for() -> None:
    model = FunctionModel(lambda m, t: "x", context_window=10_000)
    compactor = TieredCompaction(tiers=[ClearToolResults(0.5)])
    assert compactor.summary_model is None

    outcome = await _compact(compactor, _conversation(10, output=3_900), model=model)

    assert compactor.bound
    assert compactor.context_length == 10_000
    assert compactor.summary_model is model, "AGENT_MODEL resolves to the model the run uses"
    assert outcome is not None
    assert outcome.tiers[0].tier == "clear_tool_results"


def test_defaults_and_validation() -> None:
    compactor = _tiered()
    assert [t.name for t in compactor.tiers] == [
        "clear_tool_results",
        "sliding_window",
        "summarize",
        "relevance",
    ]
    assert [t.trigger for t in compactor.tiers] == [0.5, 0.7, 0.85, 0.95]
    assert compactor.threshold == int(8_000 * 0.5), "the pipeline starts at the lowest trigger"
    assert [t.name for t in default_tiers(native=True)] == [
        "clear_tool_results",
        "sliding_window",
        "native",
        "summarize",
        "relevance",
    ]
    with pytest.raises(ValueError):
        TieredCompaction([], context_length=1_000)
    for bad in (
        lambda: ClearToolResults(0.0),
        lambda: ClearToolResults(1.5),
        lambda: ClearToolResults(keep_tokens=-1),
        lambda: ClearToolResults(min_gain=1.0),
        lambda: SlidingWindow(keep_turns=0),
        lambda: Relevance(keep_turns=0),
    ):
        with pytest.raises(ValueError):
            bad()


async def test_nothing_happens_under_the_first_trigger() -> None:
    compactor = _tiered()
    assert await _compact(compactor, _conversation(3)) is None
    assert compactor.apply(_conversation(3)) == _conversation(3)


# --- tier: clear tool results -------------------------------------------------


async def test_clear_tool_results_stubs_old_outputs_and_keeps_the_newest() -> None:
    compactor = _tiered([ClearToolResults(0.5)])
    messages = _conversation(10, output=3_900)

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["clear_tool_results"]
    tier = outcome.tiers[0]
    assert tier.stage == "prune"
    assert tier.tokens_before == outcome.tokens_before
    assert tier.tokens_after == outcome.tokens_after < tier.tokens_before
    results = [m for m in outcome.messages if m.role == Role.TOOL]
    assert results[-1].content == messages[-1].content
    assert all(m.metadata.get(CLEARED_OUTPUT_KEY) for m in results[:-2])
    assert not outcome.exhausted
    _assert_pairs(outcome.messages)


async def test_clear_tool_results_honours_its_own_keep_budget() -> None:
    generous = _tiered([ClearToolResults(0.5, keep_tokens=6_000)])
    tight = _tiered([ClearToolResults(0.5, keep_tokens=0)])
    messages = _conversation(10, output=3_900)

    kept = await _compact(generous, messages)
    cleared = await _compact(tight, messages)

    assert kept is not None
    assert cleared is not None
    few = sum(bool(m.metadata.get(CLEARED_OUTPUT_KEY)) for m in kept.messages)
    assert 1 <= few < 9, "a larger budget keeps more of the newest output"
    assert sum(bool(m.metadata.get(CLEARED_OUTPUT_KEY)) for m in cleared.messages) == 9


async def test_a_small_clearing_is_skipped_unless_a_later_tier_runs_anyway() -> None:
    # Eight turns of text-heavy steps with one small clearable output each: the
    # clearing frees a few hundred tokens, well under a tenth of the window.
    messages = _conversation(8, said=2_400, output=300)
    alone = _tiered([ClearToolResults(0.5, keep_tokens=0)])
    with_window = _tiered([ClearToolResults(0.5, keep_tokens=0), SlidingWindow(0.6, keep_turns=4)])

    skipped = await _compact(alone, messages)
    assert skipped is not None, "over the last line: exhausted"
    assert skipped.tiers == ()
    assert skipped.exhausted
    both = await _compact(with_window, messages)
    assert both is not None
    assert [t.tier for t in both.tiers] == ["clear_tool_results", "sliding_window"]


# --- tier: sliding window -----------------------------------------------------


async def test_sliding_window_drops_the_oldest_turns_and_leaves_a_note() -> None:
    compactor = _tiered([SlidingWindow(0.5, keep_turns=3)])
    messages = _conversation(10, output=3_000)

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["sliding_window"]
    assert outcome.stage == "window"
    assert outcome.tiers[0].messages_before == len(messages)
    assert outcome.tiers[0].messages_after == len(outcome.messages)
    out = outcome.messages
    assert out[0].content == "SYSTEM PROMPT"
    assert out[1].content == "TASK: port the parser"
    note = out[2]
    assert note.role == Role.USER
    assert note.metadata.get(DROPPED_NOTE_KEY) == 7
    assert "7 earlier turns" in (note.content or "")
    assert out[3:] == messages[-6:]
    _assert_pairs(out)

    # A second pass folds its count into the existing note rather than adding one.
    more = [m for i in range(10, 13) for m in _turn(i, output="o" * 3_000)]
    again = await _compact(compactor, [*out, *more])
    assert again is not None
    notes = [m for m in again.messages if m.metadata.get(DROPPED_NOTE_KEY)]
    assert len(notes) == 1
    assert notes[0].metadata[DROPPED_NOTE_KEY] == 10


async def test_sliding_window_keeps_the_latest_request_and_summaries() -> None:
    compactor = _tiered([SlidingWindow(0.5, keep_turns=2)])
    messages = _conversation(4, output=3_000)
    messages.insert(6, Message.user("ALSO: keep the old API working"))
    messages += [*_turn(4, output="o" * 3_000), *_turn(5, output="o" * 3_000)]
    summary = Message(
        role=Role.USER, content="earlier summary", metadata={"locus_compaction_summary": True}
    )
    messages.insert(2, summary)

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    out = outcome.messages
    assert summary in out
    assert any(m.content == "ALSO: keep the old API working" for m in out)
    assert [m.tool_call_id for m in out if m.role == Role.TOOL] == ["c4", "c5"]
    _assert_pairs(out)


# --- tier: summarise ----------------------------------------------------------


async def test_summarize_reuses_the_summariser_and_reads_the_original_outputs() -> None:
    summariser = Summariser()
    compactor = _tiered([Summarize(0.5)], summary_model=summariser)
    messages = _conversation(12, said=800, output=800)

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["summarize"]
    assert outcome.stage == "summarize"
    assert outcome.summary == "## Goal\nSUMMARY"
    assert outcome.usage == {"prompt_tokens": 100, "completion_tokens": 40}
    assert len(summariser.prompts) == 1
    assert "out0 " in summariser.prompts[0], "the summariser saw the original tool output"
    out = outcome.messages
    assert out[0].content == "SYSTEM PROMPT"
    assert out[1].content == "TASK: port the parser"
    assert is_summary_message(out[2])
    assert out[-1] == messages[-1]
    _assert_pairs(out)


async def test_summarize_after_clearing_still_reads_what_was_cleared() -> None:
    summariser = Summariser()
    compactor = _tiered(
        [ClearToolResults(0.4, keep_tokens=0), Summarize(0.5)], summary_model=summariser
    )
    messages = _conversation(12, said=1_400, output=300)

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["clear_tool_results", "summarize"]
    assert "out1 ooo" in summariser.prompts[0], "cleared outputs are summarised from their text"
    assert "[output cleared" not in summariser.prompts[0]
    first, second = outcome.tiers
    assert first.tokens_after == second.tokens_before
    assert outcome.tokens_after == second.tokens_after


async def test_summarize_without_a_model_truncates_with_a_note() -> None:
    compactor = _tiered([Summarize(0.5)], summary_model=None)

    outcome = await _compact(compactor, _conversation(12, said=800, output=800))

    assert outcome is not None
    assert outcome.tiers[0].stage == "truncate"
    assert outcome.stage == "truncate"
    assert outcome.summary is None
    assert "removed without a summary" in (outcome.messages[2].content or "")


async def test_a_summary_needed_again_too_soon_ends_the_run() -> None:
    compactor = _tiered([Summarize(0.5)], summary_model=Summariser())
    tracker = CompactionTracker()
    messages = _conversation(12, said=800, output=800)

    first = await _compact(compactor, messages, iteration=12, tracker=tracker)
    assert first is not None
    assert not first.exhausted
    assert tracker.last_summary_iteration == 12

    second = await _compact(compactor, messages, iteration=13, tracker=tracker)
    assert second is not None
    assert second.exhausted
    assert second.tiers == ()
    assert "thrashing" in (second.detail or "")


async def test_a_held_back_summary_lets_the_cheaper_tiers_after_it_run() -> None:
    # Summarize first, the cheap tiers after it. A summary needed again within
    # the guard window used to end the run with most of the window free, the
    # tiers after it never consulted.
    summariser = Summariser()
    compactor = _tiered(
        [Summarize(0.45), ClearToolResults(0.7, keep_tokens=0), SlidingWindow(0.9, keep_turns=4)],
        summary_model=summariser,
    )
    tracker = CompactionTracker()
    messages = _conversation(12, said=800, output=1_200)  # about 6k tokens: over the 0.7 line

    first = await _compact(compactor, messages, iteration=12, tracker=tracker)
    assert first is not None
    assert [t.tier for t in first.tiers] == ["summarize"]
    assert tracker.last_summary_iteration == 12
    summary_calls = len(summariser.prompts)

    second = await _compact(compactor, messages, iteration=13, tracker=tracker)

    assert second is not None
    assert not second.exhausted
    assert [t.tier for t in second.tiers] == ["clear_tool_results"]
    assert second.tokens_after < int(8_000 * 0.7)
    assert len(summariser.prompts) == summary_calls, "no summary call within the guard window"


async def test_a_held_back_summary_under_the_other_lines_is_not_exhaustion() -> None:
    compactor = _tiered(
        [Summarize(0.45), ClearToolResults(0.7), SlidingWindow(0.9)], summary_model=Summariser()
    )
    tracker = CompactionTracker(last_summary_iteration=12)
    messages = _conversation(12, said=800, output=800)  # about 4.8k tokens: between the lines

    outcome = await _compact(compactor, messages, iteration=13, tracker=tracker)

    assert outcome is None, "the context fits the window; the model is called with it"


async def test_exhaustion_after_a_held_back_summary_names_both_causes() -> None:
    compactor = _tiered(
        [Summarize(0.5), ClearToolResults(0.6, keep_tokens=0)], summary_model=Summariser()
    )
    tracker = CompactionTracker(last_summary_iteration=12)
    messages = _conversation(12, said=2_400, output=100)  # text the clearing cannot touch

    outcome = await _compact(compactor, messages, iteration=13, tracker=tracker)

    assert outcome is not None
    assert outcome.exhausted
    assert outcome.tiers == ()
    assert "thrashing" in (outcome.detail or "")
    assert "summarize tier was needed again" in (outcome.detail or "")
    assert "no longer fit the window" in (outcome.detail or "")


# --- tier: native -------------------------------------------------------------


async def test_native_runs_through_the_models_endpoint_and_keeps_pins() -> None:
    model = NativeModel()
    summariser = Summariser()
    compactor = _tiered([Native(0.5), Summarize(0.9)], summary_model=summariser)
    messages = _conversation(12, said=800, output=800)
    pinned = pin(Message.user("ACCEPTANCE: all tests green"))
    messages.insert(6, pinned)

    outcome = await _compact(compactor, messages, model=model)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["native"]
    assert outcome.stage == "native"
    assert outcome.usage == {"prompt_tokens": 500, "completion_tokens": 20}
    assert summariser.prompts == [], "the endpoint took the stage; no client-side summary"
    out = outcome.messages
    assert out[:2] == messages[:2]
    assert out[2] is pinned, "pinned messages stay client-side"
    assert out[3].content == "[compacted by the provider]"
    handed = model.calls[0]
    assert handed[0].content == "SYSTEM PROMPT"
    assert pinned not in handed


async def test_native_is_skipped_for_a_model_without_the_endpoint() -> None:
    summariser = Summariser()
    compactor = _tiered([Native(0.5), Summarize(0.5)], summary_model=summariser)

    outcome = await _compact(compactor, _conversation(12, said=800, output=800), model=object())

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["summarize"]
    assert summariser.prompts


# --- tier: relevance ----------------------------------------------------------


def _topics() -> list[Message]:
    messages = [Message.system("SYSTEM PROMPT"), Message.user("TASK: fix the parser")]
    topics = ["parser", "database", "deploy", "docs", "database", "deploy", "parser", "docs"]
    for i, topic in enumerate(topics):
        messages += _turn(i, said=f"working on the {topic} " + "x" * 1_500, output="o" * 500)
    messages.append(Message.user("Now finish the parser work."))
    messages += _turn(8, said="checking the parser tests " + "x" * 1_500, output="o" * 500)
    return messages


async def test_relevance_keeps_the_turns_about_the_current_task() -> None:
    embedder = Embedder()
    compactor = _tiered([Relevance(0.5, keep_turns=3, embedder=embedder)])
    messages = _topics()

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["relevance"]
    assert outcome.stage == "relevance"
    said = [m.content or "" for m in outcome.messages if m.role == Role.ASSISTANT]
    assert len(said) == 3
    assert all("parser" in s for s in said), said
    assert "checking the parser tests" in said[-1], "the newest turn is always kept"
    assert any(m.metadata.get(DROPPED_NOTE_KEY) == 6 for m in outcome.messages)
    assert any(m.content == "Now finish the parser work." for m in outcome.messages)
    assert "finish the parser work" in embedder.batches[0][0], "the query is the current task"
    _assert_pairs(outcome.messages)


async def test_relevance_falls_back_to_recency_without_an_embedder_or_when_it_fails() -> None:
    class Broken:
        async def embed_batch(self, texts: list[str]) -> list[Vector]:
            raise RuntimeError("embedding service down")

    for embedder in (None, Broken()):
        compactor = _tiered([Relevance(0.5, keep_turns=3, embedder=embedder)])
        outcome = await _compact(compactor, _topics())
        assert outcome is not None
        said = [m.content or "" for m in outcome.messages if m.role == Role.ASSISTANT]
        assert len(said) == 3
        assert "working on the parser" in said[0]
        assert "working on the docs" in said[1]
        assert "checking the parser tests" in said[2]


# --- pins ---------------------------------------------------------------------


def test_pin_marks_a_copy_and_metadata_pins_are_recognised() -> None:
    original = Message.user("keep me")
    pinned = pin(original)
    assert is_pinned(pinned)
    assert not is_pinned(original)
    assert pinned.content == original.content
    assert is_pinned(Message(role=Role.USER, content="x", metadata={"pin": True}))
    assert not is_pinned(Message(role=Role.USER, content="x", metadata={"pin": "yes"}))


@pytest.mark.parametrize(
    "tier",
    [
        ClearToolResults(0.3, keep_tokens=0),
        SlidingWindow(0.3, keep_turns=2),
        Summarize(0.3),
        Native(0.3),
        Relevance(0.3, keep_turns=1),
    ],
    ids=lambda tier: tier.name,
)
async def test_pinned_messages_survive_every_tier(tier: CompactionTier) -> None:
    compactor = _tiered(
        [tier], summary_model=Summariser(), pins=lambda m: "RULE:" in (m.content or "")
    )
    messages = _conversation(14, said=800, output=800)
    by_pin = pin(Message.user("ACCEPTANCE: every test green"))
    by_predicate = Message.user("RULE: never touch the v1 API")
    decision = "DECISION: use uv " + "d" * 1_000
    pinned_call, pinned_result = _turn(99, said="recording the decision", output=decision)
    messages.insert(4, by_pin)
    messages.insert(9, by_predicate)
    messages[12:12] = [pinned_call, pin(pinned_result)]

    outcome = await _compact(compactor, messages, model=NativeModel())

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == [tier.name]
    out = outcome.messages
    assert by_pin in out
    assert by_predicate in out
    kept_result = next(m for m in out if m.tool_call_id == "c99")
    assert kept_result.content == decision, "a pinned tool result is never cleared"
    assert pinned_call in out, "and keeps the call that produced it"
    _assert_pairs(out)
    assert outcome.tiers[0].tokens_after < outcome.tiers[0].tokens_before


async def test_a_pinned_result_is_not_cleared_by_the_plain_compactor_either() -> None:
    compactor = ContextCompactor(context_length=10_000)
    messages = _conversation(10, output=3_900)
    messages[3] = pin(messages[3])

    outcome = await compactor.compact(messages, iteration=10, tracker=CompactionTracker())

    assert outcome is not None
    assert outcome.messages[3] is messages[3]
    assert outcome.messages[5].metadata.get(CLEARED_OUTPUT_KEY)


# --- the pipeline -------------------------------------------------------------


async def test_tiers_run_in_order_and_stop_once_under_the_next_trigger() -> None:
    summariser = Summariser()
    compactor = _tiered(default_tiers(), summary_model=summariser)
    messages = _conversation(10, output=3_900)  # tool output is the bulk: clearing suffices

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["clear_tool_results"]
    assert outcome.tokens_after < int(8_000 * 0.7)
    assert summariser.prompts == []
    assert not outcome.exhausted


async def test_each_tier_runs_only_at_its_own_trigger() -> None:
    ran: list[str] = []

    class Probe(CompactionTier):
        def __init__(self, name: str, trigger: float) -> None:
            super().__init__(trigger)
            self.name = name

        async def run(self, ctx: TierContext) -> TierResult | None:
            ran.append(f"{self.name}@{ctx.tokens}")
            return None

    compactor = _tiered([Probe("low", 0.3), Probe("mid", 0.6), Probe("high", 0.99)])
    messages = _conversation(6, said=2_000)  # about 3k tokens: between the low and mid lines

    outcome = await _compact(compactor, messages)

    assert outcome is None, "nothing changed and the context is under the highest line"
    assert [r.split("@")[0] for r in ran] == ["low"]


async def test_the_pipeline_reports_exhaustion_when_still_over_the_last_line() -> None:
    compactor = _tiered([ClearToolResults(0.5, keep_tokens=0)])
    messages = _conversation(8, said=2_400, output=100)  # text the tier cannot touch

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert outcome.exhausted
    assert outcome.tiers == ()
    assert "no longer fit the window" in (outcome.detail or "")


async def test_a_custom_tier_slots_into_the_pipeline() -> None:
    class DropSystemNotes(CompactionTier):
        name = "drop_notes"

        async def run(self, ctx: TierContext) -> TierResult | None:
            kept = [m for m in ctx.messages if not (m.content or "").startswith("[note]")]
            return TierResult(messages=kept) if len(kept) < len(ctx.messages) else None

    compactor = _tiered([DropSystemNotes(0.5), ClearToolResults(0.5)])
    messages = _conversation(10, output=3_900)
    messages.insert(4, Message.system("[note] 10 iterations left"))

    outcome = await _compact(compactor, messages)

    assert outcome is not None
    assert [t.tier for t in outcome.tiers] == ["drop_notes", "clear_tool_results"]
    assert not any((m.content or "").startswith("[note]") for m in outcome.messages)


async def test_a_sync_pass_never_compacts_and_the_report_is_forgotten_after_a_change() -> None:
    compactor = _tiered([ClearToolResults(0.5)])
    tracker = CompactionTracker()
    messages = _conversation(10, output=3_900)
    tracker.observe({"prompt_tokens": 12_000}, len(messages))

    outcome = await _compact(compactor, messages, tracker=tracker)

    assert outcome is not None
    assert outcome.tokens_before == 12_000, "the provider's count wins"
    assert tracker.reported_tokens is None, "the messages it measured were rewritten"
    assert await compactor.async_apply(messages) == messages


# --- through the agent --------------------------------------------------------


@tool
def work(step: int) -> str:
    """Do one step of the task."""
    return f"result of step {step}: " + "r" * 1_200


def _is_summary_call(messages: list[Message]) -> bool:
    return bool(messages) and messages[0].content == SUMMARY_SYSTEM_PROMPT


def _handler(total: int) -> Any:
    calls: list[int] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            return "## Decisions made\n- DECISION: use PostgreSQL"
        calls.append(1)
        step = len(calls)
        if step > total:
            seen = "\n".join(m.content or "" for m in messages)
            return text("Final: " + ("PostgreSQL" if "PostgreSQL" in seen else "unknown"))
        said = "DECISION: use PostgreSQL" if step == 2 else f"working on step {step}"
        return tool_call("work", call_id=f"w{step}", content=said, step=step)

    return handler


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


async def test_the_agent_yields_one_event_per_tier_and_the_run_completes() -> None:
    agent = _agent(
        _handler(40),
        conversation_manager=TieredCompaction(
            [ClearToolResults(0.5, keep_tokens=0), SlidingWindow(0.6, keep_turns=4), Summarize(0.7)]
        ),
    )
    compactor = agent._conversation_manager
    assert isinstance(compactor, TieredCompaction)
    assert compactor.bound
    assert compactor.context_length == 8_000
    assert compactor.summary_model is agent._model, "AGENT_MODEL bound to the agent's model"

    events = [e async for e in agent.run("migrate the session store")]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"
    assert "PostgreSQL" in (terminate.final_message or "")
    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert compactions, "the window was reached"
    tiers = {e.tier for e in compactions}
    assert "clear_tool_results" in tiers
    assert all(e.tier is not None for e in compactions)
    assert all(e.tokens_after <= e.tokens_before for e in compactions)
    assert all(e.context_window == 8_000 for e in compactions)
    # Consecutive tiers of one compaction chain their token counts.
    by_iteration: dict[int, list[CompactionEvent]] = {}
    for e in compactions:
        by_iteration.setdefault(e.iteration, []).append(e)
    for chain in by_iteration.values():
        for earlier, later in pairwise(chain):
            assert earlier.tokens_after == later.tokens_before
            assert earlier.messages_after == later.messages_before
            assert not earlier.exhausted
        assert chain[-1].messages_after > 0
    for request in agent._model.received_messages:  # type: ignore[attr-defined]
        assert sum(len(m.content or "") for m in request) // 4 < 8_000
        _assert_pairs(request)


@tool
def page(offset: int) -> str:
    """Read one large page of a document."""
    return f"page at {offset}: " + "p" * 6_000


def _paging_handler(total: int) -> Any:
    calls: list[int] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if _is_summary_call(messages):
            return "## Decisions made\n- DECISION: use PostgreSQL"
        calls.append(1)
        step = len(calls)
        if step > total:
            seen = "\n".join(m.content or "" for m in messages)
            return text("Final: " + ("PostgreSQL" if "PostgreSQL" in seen else "unknown"))
        said = "DECISION: use PostgreSQL" if step == 1 else f"reading page {step}"
        return tool_call("page", call_id=f"p{step}", content=said, offset=step * 8_000)

    return handler


async def test_the_run_completes_when_a_summary_is_held_back_and_a_cheaper_tier_runs() -> None:
    # The live shape: large tool pages arrive faster than the guard allows a
    # second summary, and the cheap tiers sit after the summarising one.
    agent = Agent(
        model=FunctionModel(_paging_handler(10)),
        tools=[page],
        system_prompt="You are a document assistant.",
        reflexion=False,
        grounding=False,
        context_window=8_000,
        max_iterations=40,
        conversation_manager=TieredCompaction(
            [
                Summarize(0.45),
                ClearToolResults(0.7, keep_tokens=0),
                SlidingWindow(0.9, keep_turns=4),
            ]
        ),
    )

    events = [e async for e in agent.run("find the access code in the document")]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete", terminate.final_message
    assert "PostgreSQL" in (terminate.final_message or "")
    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert not any(e.exhausted for e in compactions)
    summaries = [e.iteration for e in compactions if e.tier == "summarize"]
    assert summaries, "the summarising tier ran"
    cheap = [e for e in compactions if e.tier in ("clear_tool_results", "sliding_window")]
    guard = agent._conversation_manager.min_iterations_between_summaries  # type: ignore[union-attr]
    assert any(0 < e.iteration - s < guard for e in cheap for s in summaries), (
        "a cheaper tier ran in a pass where the summary was held back"
    )
    assert all(s2 - s1 >= guard for s1, s2 in pairwise(summaries)), "the guard still holds"
    for request in agent._model.received_messages:  # type: ignore[attr-defined]
        assert sum(len(m.content or "") for m in request) // 4 < 8_000
        _assert_pairs(request)


def test_compaction_config_tiers_build_the_pipeline() -> None:
    agent = _agent(_handler(1), compaction=CompactionConfig(tiers=True, native=True))
    compactor = agent._conversation_manager
    assert isinstance(compactor, TieredCompaction)
    assert [t.name for t in compactor.tiers] == [
        "clear_tool_results",
        "sliding_window",
        "native",
        "summarize",
        "relevance",
    ]
    assert compactor.context_length == 8_000
    assert compactor.summary_model is agent._model

    custom = _agent(_handler(1), compaction=CompactionConfig(tiers=[ClearToolResults(0.4)]))
    assert isinstance(custom._conversation_manager, TieredCompaction)
    assert [t.trigger for t in custom._conversation_manager.tiers] == [0.4]

    plain = _agent(_handler(1), compaction=CompactionConfig(tiers=False))
    assert type(plain._conversation_manager) is ContextCompactor


def test_an_unbound_pipeline_without_any_window_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="context window"):
        Agent(
            model=FunctionModel(lambda m, t: "x", context_window=None),
            conversation_manager=TieredCompaction(summary_model=None),
        )


async def test_tier_events_reach_the_observability_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.observability import agent_bridge

    published: list[tuple[str, dict[str, Any]]] = []

    async def record(name: str, **fields: Any) -> None:
        published.append((name, fields))

    monkeypatch.setattr(agent_bridge, "emit", record)
    await agent_bridge.bridge_locus_event(
        CompactionEvent(
            iteration=4,
            stage="window",
            tokens_before=9_000,
            tokens_after=5_000,
            threshold=4_000,
            context_window=10_000,
            messages_before=40,
            messages_after=-1,
            tier="sliding_window",
        )
    )

    assert published == [
        (
            "agent.context.compacted",
            {
                "iteration": 4,
                "stage": "window",
                "tokens_before": 9_000,
                "tokens_after": 5_000,
                "threshold": 4_000,
                "exhausted": False,
                "tier": "sliding_window",
            },
        )
    ]


def test_the_catalogue_names_every_stage_and_the_tier_field() -> None:
    import locus.observability.emit  # noqa: F401 - make sure the module is loaded

    emit_mod = sys.modules["locus.observability.emit"]
    page = pathlib.Path(__file__).resolve().parents[2] / "docs" / "concepts" / "sse-events.md"
    if not page.is_file():
        pytest.skip("docs are not part of this checkout")
    catalogue = page.read_text()
    assert f"`{emit_mod.EV_AGENT_COMPACTION}`" in catalogue
    for stage in CompactionEvent.model_fields["stage"].annotation.__args__:  # type: ignore[union-attr]
        assert f'"{stage}"' in catalogue, f"stage {stage!r} is not in the catalogue"
    assert "`tier`" in catalogue


def test_agent_model_sentinel_is_a_string_users_can_spell() -> None:
    assert AGENT_MODEL == "agent"
    compactor = TieredCompaction(context_length=10_000, summary_model="agent")
    assert compactor.summary_model is None, "resolved when the pipeline is bound to a model"
