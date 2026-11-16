# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tiered context compaction: cheap tiers first, each with its own trigger.

:class:`~locus.memory.compaction.ContextCompactor` clears old tool output and
then summarises. A long coding or research run does better with a pipeline
whose stages get progressively more aggressive, and more expensive, as the
context fills::

    TieredCompaction(
        [
            ClearToolResults(
                trigger=0.5
            ),  # stubs for tool outputs the run is done with
            SlidingWindow(trigger=0.7),  # drop the oldest turns past a turn window
            Summarize(trigger=0.85),  # fold older history into a summary
            Relevance(trigger=0.95),  # keep the turns relevant to the current task
        ]
    )

Each tier has a **trigger**: a fraction of the usable window (the model's
context window minus the room reserved for the reply). Before each model
call the context is measured; the tiers run in order, and a tier runs only
while the context is at or over its trigger. A tier that changes nothing is
skipped. After each tier the context is measured again, so a cheap tier that
bought enough room stops the pipeline before an expensive one runs. One
:class:`~locus.core.events.CompactionEvent` is yielded per tier that ran,
with the tokens before and after it.

The window comes from the model's metadata entry
(:func:`locus.models.metadata.metadata_for`); ``context_length=`` names it
when the model has none, and wins when both are given.

**Pinned messages survive every tier.** Pin a message with
:func:`~locus.memory.compaction.pin` (or ``metadata={"pin": True}``), or
pass ``pins=`` — a predicate over messages — to the compactor. A pinned
tool result keeps the call that produced it, so the conversation stays
valid for the provider.

**Native compaction** (``OCIResponsesModel`` with ``store=False``, or any
model with ``compact_history``) slots in as the :class:`Native` tier:
``TieredCompaction.default_tiers(native=True)`` puts it before
:class:`Summarize`, which then runs only when the model has no endpoint.

An agent uses the pipeline through ``conversation_manager=`` or
``CompactionConfig(tiers=[...])``::

    agent = Agent(
        model="oci:openai.gpt-4.1",
        tools=[...],
        conversation_manager=TieredCompaction(
            pins=lambda m: "ACCEPTANCE:" in (m.content or "")
        ),
    )

The agent binds the compactor to its model and window on initialisation.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from locus.core.messages import Message, Role
from locus.memory.compaction import (
    _AUTOMATED_NOTE_KEY,
    CompactionOutcome,
    CompactionStage,
    CompactionTracker,
    ContextCompactor,
    OutputArchive,
    TierOutcome,
    _clip,
    _head_end,
    _is_automated,
    _is_pinned,
    _turns,
    is_summary_message,
)
from locus.memory.compactor import _repair_boundaries


logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_MODEL",
    "ClearToolResults",
    "CompactionTier",
    "Native",
    "Relevance",
    "SlidingWindow",
    "Summarize",
    "TierContext",
    "TierResult",
    "TieredCompaction",
    "default_tiers",
    "resolve_context_length",
]

#: ``summary_model=AGENT_MODEL``: the agent's own model writes the summaries.
AGENT_MODEL: Final = "agent"

#: ``Message.metadata`` key on the note a window or relevance tier leaves
#: where it dropped turns, so the next pass updates the note instead of
#: adding another.
DROPPED_NOTE_KEY = "locus_compaction_dropped"

# A window the compactor carries until it is bound to a model; never used
# for a decision (``compact`` binds first or raises).
_UNBOUND_WINDOW = 1 << 20

# How much of a turn the relevance tier embeds.
_RELEVANCE_TEXT_CHARS = 4_000


# ---------------------------------------------------------------------------
# Tier protocol
# ---------------------------------------------------------------------------


@dataclass
class TierContext:
    """What a tier sees when it runs."""

    compactor: TieredCompaction
    #: The conversation as the tiers before this one left it.
    messages: list[Message]
    #: The conversation as it was handed to ``compact``; what the summariser reads.
    original: list[Message]
    #: The measured context, in tokens, before this tier.
    tokens: int
    tool_tokens: int
    #: Provider count over estimate; every estimate is scaled by it.
    scale: float
    iteration: int
    instructions: str | None = None
    archive: OutputArchive | None = None
    model: Any = None
    #: A later tier is due to run at this size, so the prompt cache is
    #: being broken anyway and a small gain is still worth taking.
    continuing: bool = False

    @property
    def usable(self) -> int:
        return self.compactor.usable_tokens

    def measure(self, messages: Sequence[Message]) -> int:
        """``messages`` in the same scaled tokens as :attr:`tokens`."""
        return int((self.compactor.tokens(messages) + self.tool_tokens) * self.scale)


@dataclass(frozen=True)
class TierResult:
    """What a tier did."""

    messages: list[Message]
    #: Overrides the tier's declared stage (``summarize`` vs ``truncate``).
    stage: CompactionStage | None = None
    summary: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    detail: str | None = None


class CompactionTier(ABC):
    """One stage of a :class:`TieredCompaction`.

    Args:
        trigger: Fraction of the usable window at which this tier runs.
    """

    #: The tier's name, as the event reports it.
    name: str = "tier"
    #: The stage the tier's event carries.
    stage: CompactionStage = "prune"
    #: Whether the tier calls a model (the thrash guard applies to it).
    calls_model: bool = False

    def __init__(self, trigger: float) -> None:
        if not 0.0 < trigger <= 1.0:
            raise ValueError(f"{type(self).__name__}: trigger must be in (0, 1]")
        self.trigger = trigger

    @abstractmethod
    async def run(self, ctx: TierContext) -> TierResult | None:
        """Compact ``ctx.messages``; ``None`` when the tier has nothing to do."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(trigger={self.trigger})"


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


class ClearToolResults(CompactionTier):
    """Replace tool outputs older than the newest ``keep_tokens`` with stubs.

    Free, and on a tool-heavy run usually enough. The newest output is always
    kept whole; a pinned one is never cleared; an offloaded one keeps its
    handle in the stub, and with an ObservationPack archive the stub names
    the archive id. Below ``min_gain`` of the usable window the clearing is
    skipped unless a later tier is about to run anyway: changing one old
    message per request defeats the provider's prompt cache for a few
    hundred tokens.

    Args:
        trigger: Fraction of the usable window at which the tier runs.
        keep_tokens: Newest tool output kept whole. Default: the compactor's
            ``tool_output_keep_tokens`` (``min(40_000, usable // 4)``).
        min_gain: Least share of the usable window a clearing must free to
            count on its own.
    """

    name = "clear_tool_results"
    stage: CompactionStage = "prune"

    def __init__(
        self, trigger: float = 0.5, *, keep_tokens: int | None = None, min_gain: float = 0.1
    ) -> None:
        super().__init__(trigger)
        if keep_tokens is not None and keep_tokens < 0:
            raise ValueError("keep_tokens must be non-negative")
        if not 0.0 <= min_gain < 1.0:
            raise ValueError("min_gain must be in [0, 1)")
        self.keep_tokens = keep_tokens
        self.min_gain = min_gain

    async def run(self, ctx: TierContext) -> TierResult | None:
        cleared = ctx.compactor.clear_tool_outputs(
            ctx.messages, ctx.archive, keep_tokens=self.keep_tokens
        )
        if cleared == ctx.messages:
            return None
        gain = ctx.tokens - ctx.measure(cleared)
        if not ctx.continuing and gain < int(ctx.usable * self.min_gain):
            return None
        return TierResult(messages=cleared)


class SlidingWindow(CompactionTier):
    """Drop the oldest turns beyond a window of ``keep_turns`` turns.

    A turn is one assistant message with its tool results, or one user
    message. The head (system prompt and task), pinned messages, the user's
    latest request and earlier summaries are not in the window and never
    leave. Where turns were dropped a short note says how many, so the model
    knows the history is not complete.

    Args:
        trigger: Fraction of the usable window at which the tier runs.
        keep_turns: Newest turns kept.
    """

    name = "sliding_window"
    stage: CompactionStage = "window"

    def __init__(self, trigger: float = 0.7, *, keep_turns: int = 30) -> None:
        super().__init__(trigger)
        if keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        self.keep_turns = keep_turns

    async def run(self, ctx: TierContext) -> TierResult | None:
        parts = _split(ctx.compactor, ctx.messages)
        if len(parts.candidates) <= self.keep_turns:
            return None
        drop = parts.candidates[: len(parts.candidates) - self.keep_turns]
        return TierResult(messages=_without(ctx, parts, drop, "to free the context window"))


class Summarize(CompactionTier):
    """Fold the older history into a model-written summary.

    The existing summariser: the compactor's ``summary_model`` (the agent's
    own model by default) writes a summary built for carrying on the task
    and updates the previous one instead of starting over; the head, pinned
    messages, the latest user request and the newest ``tail_turns`` turns
    stay verbatim. Without a summary model the older history is dropped with
    a note (stage ``truncate``). Counts against the thrash guard.

    Args:
        trigger: Fraction of the usable window at which the tier runs.
    """

    name = "summarize"
    stage: CompactionStage = "summarize"
    calls_model = True

    def __init__(self, trigger: float = 0.85) -> None:
        super().__init__(trigger)

    async def run(self, ctx: TierContext) -> TierResult | None:
        compactor = ctx.compactor
        aligned = _aligned_original(ctx.messages, ctx.original)
        outcome = await compactor.summarise_history(
            aligned,
            ctx.messages,
            ctx.tool_tokens,
            ctx.tokens,
            ctx.scale,
            ctx.instructions,
            ctx.archive,
            threshold=int(ctx.usable * self.trigger),
        )
        if outcome.messages == ctx.messages:
            return None
        return TierResult(
            messages=outcome.messages,
            stage=outcome.stage,
            summary=outcome.summary,
            usage=outcome.usage,
            detail=outcome.detail if outcome.stage == "truncate" else None,
        )


class Native(CompactionTier):
    """Hand the older history to the model's own compaction endpoint.

    Runs only when the model offers ``compact_history`` (``OCIResponsesModel``
    with ``store=False``); otherwise the tier is skipped and the next one
    runs. The head and pinned messages stay verbatim client-side; the
    endpoint folds the rest into an item the model replays itself. A failed
    call skips the tier. Counts against the thrash guard.

    Args:
        trigger: Fraction of the usable window at which the tier runs.
    """

    name = "native"
    stage: CompactionStage = "native"
    calls_model = True

    def __init__(self, trigger: float = 0.85) -> None:
        super().__init__(trigger)

    async def run(self, ctx: TierContext) -> TierResult | None:
        model = ctx.model
        if not callable(getattr(model, "compact_history", None)):
            return None
        compactor = ctx.compactor
        messages = ctx.messages
        head_end = _head_end(messages)
        head = messages[:head_end]
        body = messages[head_end:]
        pinned = [m for m in body if compactor.is_pinned(m)]
        handed = [m for m in body if not compactor.is_pinned(m)]
        if not handed:
            return None
        system = [m for m in head if m.role == Role.SYSTEM]
        try:
            compacted, usage = await model.compact_history(
                [*system, *handed], instructions=ctx.instructions
            )
        except Exception as exc:  # noqa: BLE001 — a failed native compaction skips the tier
            logger.warning(
                "Native compaction tier failed, continuing with the next tier: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None
        result = [*head, *pinned, *compacted]
        if result == messages:
            return None
        return TierResult(messages=result, usage=dict(usage or {}))


class Relevance(CompactionTier):
    """Keep the ``keep_turns`` turns most relevant to the current task.

    The last resort before the run ends for lack of context. With an
    ``embedder`` (anything with ``embed_batch(texts)`` or ``embed(text)``
    returning results with an ``embedding`` vector, such as the SDK's
    embedding providers) each turn is scored by cosine similarity to the
    current task — the user's latest request and the model's latest message
    — and the best-scoring turns stay, in their original order. Without one,
    the newest turns stay. The newest turn is always kept, as are the head,
    pinned messages, the latest request and summaries. A note marks where
    turns were dropped.

    Args:
        trigger: Fraction of the usable window at which the tier runs.
        keep_turns: Turns kept.
        embedder: Scores relevance; ``None`` keeps by recency.
    """

    name = "relevance"
    stage: CompactionStage = "relevance"

    def __init__(self, trigger: float = 0.95, *, keep_turns: int = 8, embedder: Any = None) -> None:
        super().__init__(trigger)
        if keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        self.keep_turns = keep_turns
        self.embedder = embedder

    async def run(self, ctx: TierContext) -> TierResult | None:
        parts = _split(ctx.compactor, ctx.messages)
        candidates = parts.candidates
        if len(candidates) <= self.keep_turns:
            return None
        keep = await self._select(ctx.messages, parts, candidates)
        drop = [turn for turn in candidates if id(turn) not in keep]
        if not drop:
            return None
        return TierResult(messages=_without(ctx, parts, drop, "as unrelated to the current task"))

    async def _select(
        self, messages: list[Message], parts: _Parts, candidates: list[list[int]]
    ) -> set[int]:
        """Identities of the candidate turns to keep."""
        newest = candidates[-1]
        by_recency = {id(turn) for turn in candidates[-self.keep_turns :]}
        if self.embedder is None:
            return by_recency
        query = _query_text(messages, parts)
        texts = [query, *(_turn_text(messages, turn) for turn in candidates)]
        try:
            vectors = await _embed(self.embedder, texts)
        except Exception as exc:  # noqa: BLE001 — scoring is best effort; recency is the fallback
            logger.warning(
                "Relevance tier could not embed the turns, keeping by recency: %s: %s",
                type(exc).__name__,
                exc,
            )
            return by_recency
        if len(vectors) != len(texts):
            return by_recency
        query_vector = vectors[0]
        scored = sorted(
            range(len(candidates)),
            key=lambda i: _cosine(query_vector, vectors[i + 1]),
            reverse=True,
        )
        keep = {id(newest)}
        for i in scored:
            if len(keep) >= self.keep_turns:
                break
            keep.add(id(candidates[i]))
        return keep


def default_tiers(*, native: bool = False) -> list[CompactionTier]:
    """The default pipeline: clear, window, (native,) summarise, relevance."""
    tiers: list[CompactionTier] = [ClearToolResults(0.5), SlidingWindow(0.7)]
    if native:
        tiers.append(Native(0.85))
    tiers.extend([Summarize(0.85), Relevance(0.95)])
    return tiers


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class TieredCompaction(ContextCompactor):
    """Run compaction tiers in order, each at its own fraction of the window.

    A :class:`~locus.memory.compaction.ContextCompactor`, so the agent loop
    drives it the same way; see the module docstring for the pipeline.

    Args:
        tiers: The tiers, in the order they run. Default: :func:`default_tiers`.
        context_length: The model's input window in tokens. Wins over the
            model's metadata when both are given.
        model: The model, or its id, whose metadata names the window; when
            it is a model object it also writes the summaries under
            ``summary_model=AGENT_MODEL``.
        summary_model: Who writes the :class:`Summarize` tier's summaries.
            :data:`AGENT_MODEL` (the default) is the agent's own model, bound
            on initialisation; ``None`` drops older history with a note
            instead of summarising it; a model instance uses that model.
        pins: A predicate over messages; those it accepts survive every tier,
            as do messages pinned with :func:`~locus.memory.compaction.pin`.
        reserved_tokens: As for :class:`~locus.memory.compaction.ContextCompactor`.
        tail_turns: As for :class:`~locus.memory.compaction.ContextCompactor`.
        tail_token_fraction: As for :class:`~locus.memory.compaction.ContextCompactor`.
        tool_output_keep_tokens: As for :class:`~locus.memory.compaction.ContextCompactor`.
        summary_max_tokens: As for :class:`~locus.memory.compaction.ContextCompactor`.
        min_iterations_between_summaries: As for
            :class:`~locus.memory.compaction.ContextCompactor`.
        token_counter: As for :class:`~locus.memory.compaction.ContextCompactor`.
    """

    def __init__(
        self,
        tiers: Sequence[CompactionTier] | None = None,
        *,
        context_length: int | None = None,
        model: Any = None,
        summary_model: Any = AGENT_MODEL,
        pins: Callable[[Message], bool] | None = None,
        reserved_tokens: int | None = None,
        tail_turns: int = 6,
        tail_token_fraction: float = 0.25,
        tool_output_keep_tokens: int | None = None,
        summary_max_tokens: int | None = None,
        min_iterations_between_summaries: int = 3,
        token_counter: Callable[[Message], int] | None = None,
    ) -> None:
        self.tiers: list[CompactionTier] = list(tiers) if tiers is not None else default_tiers()
        if not self.tiers:
            raise ValueError("TieredCompaction needs at least one tier")
        self.pins = pins
        self._summary_setting = summary_model
        self._options: dict[str, Any] = {
            "reserved_tokens": reserved_tokens,
            "tail_turns": tail_turns,
            "tail_token_fraction": tail_token_fraction,
            "tool_output_keep_tokens": tool_output_keep_tokens,
            "summary_max_tokens": summary_max_tokens,
            "min_iterations_between_summaries": min_iterations_between_summaries,
            "token_counter": token_counter,
        }
        self._closure: set[int] = set()
        self.bound = False
        window = resolve_context_length(model, context_length)
        self._configure(window if window is not None else _UNBOUND_WINDOW)
        self.bound = window is not None
        self._adopt_summary_model(model)

    def _configure(self, window: int) -> None:
        summary = getattr(self, "summary_model", None)
        if summary is None and not _is_agent_sentinel(self._summary_setting):
            summary = self._summary_setting
        ContextCompactor.__init__(
            self,
            context_length=window,
            summary_model=summary,
            trigger_fraction=min(tier.trigger for tier in self.tiers),
            **self._options,
        )

    def _adopt_summary_model(self, model: Any) -> None:
        if (
            _is_agent_sentinel(self._summary_setting)
            and self.summary_model is None
            and callable(getattr(model, "complete", None))
        ):
            self.summary_model = model

    def bind(self, *, model: Any = None, context_length: int | None = None) -> TieredCompaction:
        """Bind the window (and, under ``AGENT_MODEL``, the summariser) to a model.

        The agent calls this on initialisation with its model and the window
        it resolved (``context_window=``, ``LOCUS_CONTEXT_WINDOW``, metadata,
        the model itself). Returns ``self``.

        Raises:
            ValueError: no window can be found.
        """
        window = resolve_context_length(model, context_length)
        if window is None and self.bound:
            window = self.context_length
        if window is None:
            raise ValueError(
                "TieredCompaction needs the model's context window: pass context_length=, "
                "or a model with a metadata entry (locus.models.metadata.register_metadata)."
            )
        self._configure(window)
        self.bound = True
        self._adopt_summary_model(model)
        return self

    # ------------------------------------------------------------------
    # Pins
    # ------------------------------------------------------------------

    def is_pinned(self, message: Message) -> bool:
        """Pinned by metadata, by the ``pins`` predicate, or by sharing a turn with one."""
        if id(message) in self._closure or _is_pinned(message):
            return True
        return bool(self.pins is not None and self.pins(message))

    def _pin_closure(self, messages: list[Message]) -> set[int]:
        """Pinned messages plus whatever keeps them valid: a result's call, a call's results."""
        pinned = [m for m in messages if _is_pinned(m) or (self.pins is not None and self.pins(m))]
        if not pinned:
            return set()
        calls = {tc.id: m for m in messages if m.role == Role.ASSISTANT for tc in m.tool_calls}
        closure = {id(m) for m in pinned}
        for m in pinned:
            if m.role == Role.TOOL and m.tool_call_id in calls:
                closure.add(id(calls[m.tool_call_id]))
        call_ids = {
            tc.id
            for m in messages
            if m.role == Role.ASSISTANT and id(m) in closure
            for tc in m.tool_calls
        }
        for m in messages:
            if m.role == Role.TOOL and m.tool_call_id in call_ids:
                closure.add(id(m))
        return closure

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: Sequence[Message],
        *,
        iteration: int,
        tracker: CompactionTracker,
        tool_tokens: int = 0,
        tokens_before: int | None = None,
        instructions: str | None = None,
        archive: OutputArchive | None = None,
        model: Any = None,
    ) -> CompactionOutcome | None:
        """Run the tiers that the context has reached; ``None`` when none has.

        See :meth:`ContextCompactor.compact` for the arguments. The outcome's
        ``tiers`` lists what each tier did; ``exhausted`` is set when the
        context is still at or over the highest trigger after the last tier,
        or when a model-calling tier is needed again within
        ``min_iterations_between_summaries`` iterations of the last one.
        """
        if not self.bound:
            self.bind(model=model)
        self._adopt_summary_model(model)
        original = list(messages)
        before = (
            tokens_before
            if tokens_before is not None
            else self.measure(original, tool_tokens=tool_tokens, tracker=tracker)
        )
        if before < self.threshold:
            return None
        raw_before = self.tokens(original) + tool_tokens
        scale = max(1.0, before / raw_before) if raw_before else 1.0
        usable = self.usable_tokens
        self._closure = self._pin_closure(original)
        try:
            return await self._run_tiers(
                original,
                before=before,
                scale=scale,
                usable=usable,
                iteration=iteration,
                tracker=tracker,
                tool_tokens=tool_tokens,
                instructions=instructions,
                archive=archive,
                model=model,
            )
        finally:
            self._closure = set()

    async def _run_tiers(
        self,
        original: list[Message],
        *,
        before: int,
        scale: float,
        usable: int,
        iteration: int,
        tracker: CompactionTracker,
        tool_tokens: int,
        instructions: str | None,
        archive: OutputArchive | None,
        model: Any,
    ) -> CompactionOutcome | None:
        current = original
        tokens = before
        outcomes: list[TierOutcome] = []
        usage: dict[str, int] = {}
        summary: str | None = None
        exhausted = False
        detail: str | None = None
        guard = self.min_iterations_between_summaries
        lines = [int(usable * tier.trigger) for tier in self.tiers]
        for index, tier in enumerate(self.tiers):
            if tokens < lines[index]:
                continue
            if tier.calls_model:
                last = tracker.last_summary_iteration
                if guard and last is not None and iteration - last < guard:
                    exhausted = True
                    detail = (
                        f"compaction is thrashing: the {tier.name} tier was needed at "
                        f"iteration {iteration}, {iteration - last} after it last ran "
                        f"(minimum {guard}). The latest turns and fixed context "
                        f"({tokens} tokens against a {lines[index]}-token line) no longer "
                        "fit the window."
                    )
                    break
            continuing = any(tokens >= line for line in lines[index + 1 :])
            result = await tier.run(
                TierContext(
                    compactor=self,
                    messages=current,
                    original=original,
                    tokens=tokens,
                    tool_tokens=tool_tokens,
                    scale=scale,
                    iteration=iteration,
                    instructions=instructions,
                    archive=archive,
                    model=model,
                    continuing=continuing,
                )
            )
            if result is None or result.messages == current:
                continue
            after = int((self.tokens(result.messages) + tool_tokens) * scale)
            outcomes.append(
                TierOutcome(
                    tier=tier.name,
                    stage=result.stage or tier.stage,
                    tokens_before=tokens,
                    tokens_after=after,
                    messages_before=len(current),
                    messages_after=len(result.messages),
                    summary=result.summary,
                    detail=result.detail,
                )
            )
            for key, value in result.usage.items():
                usage[key] = usage.get(key, 0) + value
            if result.summary:
                summary = result.summary
            if tier.calls_model:
                tracker.last_summary_iteration = iteration
            current = result.messages
            tokens = after

        highest = max(lines)
        if not exhausted and tokens >= highest:
            exhausted = True
            detail = (
                f"compaction left {tokens} tokens against a {highest}-token line: the "
                "system prompt, task, tool definitions, pinned messages and latest turn "
                "no longer fit the window."
            )
        if not outcomes and not exhausted:
            return None
        if outcomes:
            tracker.forget_report()
        return CompactionOutcome(
            messages=current,
            stage=outcomes[-1].stage if outcomes else "prune",
            tokens_before=before,
            tokens_after=tokens,
            threshold=self.threshold,
            summary=summary,
            exhausted=exhausted,
            detail=detail,
            usage=usage,
            tiers=tuple(outcomes),
        )

    def __repr__(self) -> str:
        window = self.context_length if self.bound else None
        return (
            f"TieredCompaction(tiers={self.tiers!r}, context_length={window}, "
            f"summarises={self.summary_model is not None})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_agent_sentinel(value: Any) -> bool:
    return isinstance(value, str) and value == AGENT_MODEL


def resolve_context_length(model: Any, configured: int | None) -> int | None:
    """The window to count against: ``configured``, else the model's metadata, else the model.

    ``model`` is a model object or an id string. The metadata entry is the
    SDK's table (:func:`~locus.models.metadata.metadata_for`); after it, a
    ``context_window`` or ``context_length`` the model object reports itself.
    """
    if configured is not None:
        if configured < 1:
            raise ValueError("context_length must be positive")
        return configured
    if model is None:
        return None
    from locus.models.metadata import metadata_for, model_id_of  # noqa: PLC0415

    model_id = model_id_of(model)
    meta = metadata_for(model_id) if model_id else None
    if meta is not None:
        return meta.context_length
    for holder in (model, getattr(model, "config", None)):
        for attr in ("context_window", "context_length"):
            reported = getattr(holder, attr, None)
            if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
                return reported
    return None


@dataclass
class _Parts:
    """A conversation cut for a windowing tier."""

    head_end: int
    #: Every body turn, as message indices.
    turns: list[list[int]]
    #: Turns the tier may drop: no pinned message, not the latest request, no summary.
    candidates: list[list[int]]
    latest_user: int | None


def _split(compactor: ContextCompactor, messages: list[Message]) -> _Parts:
    head_end = _head_end(messages)
    body = list(range(head_end, len(messages)))
    latest_user = max(
        (i for i in body if messages[i].role == Role.USER and not _is_automated(messages[i])),
        default=None,
    )
    turns = _turns(body, messages)

    def protected(turn: list[int]) -> bool:
        return any(
            compactor.is_pinned(messages[i])
            or i == latest_user
            or is_summary_message(messages[i])
            or messages[i].metadata.get(DROPPED_NOTE_KEY)
            for i in turn
        )

    return _Parts(
        head_end=head_end,
        turns=turns,
        candidates=[turn for turn in turns if not protected(turn)],
        latest_user=latest_user,
    )


def _without(ctx: TierContext, parts: _Parts, drop: list[list[int]], why: str) -> list[Message]:
    """``ctx.messages`` without the turns in ``drop``, with a note where they were."""
    messages = ctx.messages
    dropped = {i for turn in drop for i in turn}
    first = min(dropped)
    # An earlier note is folded into this one: its count carries over.
    earlier = 0
    kept: list[Message] = []
    for i, message in enumerate(messages):
        if i in dropped:
            continue
        if message.metadata.get(DROPPED_NOTE_KEY):
            earlier += int(message.metadata.get(DROPPED_NOTE_KEY) or 0)
            first = min(first, i)
            continue
        kept.append(message)
    turns = len(drop) + earlier
    note = Message(
        role=Role.USER,
        content=(
            f"[Context compacted: an automated note, not a message from the user] "
            f"{turns} earlier turn{'s' if turns != 1 else ''} of this task were removed {why}. "
            "The messages above and any summary are what remains of them; re-read a file "
            "before you rely on what it contained."
        ),
        metadata={_AUTOMATED_NOTE_KEY: True, DROPPED_NOTE_KEY: turns},
    )
    at = sum(
        1
        for i in range(first)
        if i not in dropped and not messages[i].metadata.get(DROPPED_NOTE_KEY)
    )
    kept.insert(at, note)
    return _repair_boundaries(kept, ctx.original)


def _aligned_original(current: list[Message], original: list[Message]) -> list[Message]:
    """``original`` lined up with ``current``, so index ``i`` names the same message in both.

    The summariser reads the original text of a cleared tool output by index;
    after a tier has dropped messages the indices no longer match, so each
    current tool result is paired with the original that answered the same call.
    """
    by_call = {m.tool_call_id: m for m in original if m.role == Role.TOOL and m.tool_call_id}
    return [by_call.get(m.tool_call_id or "", m) if m.role == Role.TOOL else m for m in current]


def _query_text(messages: list[Message], parts: _Parts) -> str:
    """The current task: the latest request and the model's latest message."""
    pieces: list[str] = []
    if parts.latest_user is not None:
        pieces.append(messages[parts.latest_user].content or "")
    last_assistant = next(
        (m.content for m in reversed(messages) if m.role == Role.ASSISTANT and m.content), None
    )
    if last_assistant:
        pieces.append(last_assistant)
    return _clip("\n".join(p for p in pieces if p), _RELEVANCE_TEXT_CHARS) or "the task"


def _turn_text(messages: list[Message], turn: list[int]) -> str:
    parts: list[str] = []
    for i in turn:
        message = messages[i]
        if message.content:
            parts.append(message.content)
        parts.extend(f"{tc.name} {tc.arguments}" for tc in message.tool_calls)
    return _clip("\n".join(parts), _RELEVANCE_TEXT_CHARS) or "(empty)"


async def _embed(embedder: Any, texts: list[str]) -> list[list[float]]:
    batch = getattr(embedder, "embed_batch", None)
    if callable(batch):
        results = await batch(texts)
    else:
        results = [await embedder.embed(text) for text in texts]
    return [list(getattr(r, "embedding", r)) for r in results]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
