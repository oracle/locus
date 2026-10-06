# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarising context compaction for long autonomous runs.

A coding agent that works for hours makes hundreds of tool calls, and every
one of them stays in the conversation. Cutting the oldest turns away keeps the
request under the window but leaves the model without the goal's fine print,
the decisions it already made and the files it already changed, so it repeats
work or contradicts itself. :class:`ContextCompactor` keeps a long run coherent
the way established coding-agent harnesses (opencode, Codex) do:

1. **Clear old tool output.** Once the context passes the threshold, tool
   results older than the newest ``tool_output_keep_tokens`` become a one-line
   stub naming the call. That is free, and on a tool-heavy run it is usually
   enough. It counts only when it frees a tenth of the usable window, so the
   prompt cache is not broken on every turn for a few hundred tokens.
2. **Summarise the older history.** When clearing is not enough, the model
   writes a summary built for carrying on the task (goal, decisions, files,
   what was verified, what is left, open errors, the next step). It updates the
   previous summary instead of starting over, so nothing compounds into drift.
   The system prompt, the task message, the latest user message and the last
   ``tail_turns`` turns stay verbatim.
3. **Stop instead of looping.** When a compaction cannot bring the context
   under the threshold, or summaries are needed again within
   ``min_iterations_between_summaries`` iterations, the run ends with
   ``context_exhausted``. Compacting again would only burn the same tokens to
   reach the same place.

With an :class:`~tulip.memory.observation_pack.ObservationPack` (``archive``),
clearing is lossless: each cleared output is archived and its stub names the
id ``obs_recall`` reads it back by, and a summary ends with the ids of the
outputs it folded, so the model can still recall them after the summary.

Unlike the other conversation managers this one rewrites the run's state
rather than the request: the summary replaces the history in ``state.messages``
and in the checkpoint, so the next turn and the next compaction start from it.
The agent loop drives it through :meth:`ContextCompactor.compact`, which is why
:meth:`apply` returns its input unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tulip.core.messages import Message, Role
from tulip.memory.compactor import _char_count_tokens, _repair_boundaries
from tulip.memory.conversation import ConversationManager
from tulip.memory.observation_pack import OBSERVATION_ID_KEY


logger = logging.getLogger(__name__)

__all__ = [
    "CLEARED_OUTPUT_KEY",
    "SUMMARY_MESSAGE_KEY",
    "CompactionOutcome",
    "CompactionStage",
    "CompactionTracker",
    "ContextCompactor",
    "RECALLABLE_OUTPUTS_KEY",
    "OutputArchive",
    "is_summary_message",
]

#: Which compaction stage changed the context.
#: ``prune``: old tool outputs were cleared and that was enough.
#: ``summarize``: older history was replaced by a model-written summary.
#: ``truncate``: older history was dropped without a summary (no summary
#: model, or the summary call failed twice).
CompactionStage = Literal["prune", "summarize", "truncate"]

#: ``Message.metadata`` key on the summary message, so the next compaction
#: builds on it rather than summarising a summary as if it were conversation.
SUMMARY_MESSAGE_KEY = "tulip_compaction_summary"

#: ``Message.metadata`` key on a tool result whose output was cleared, so it is
#: not cleared (and counted) twice.
CLEARED_OUTPUT_KEY = "tulip_compaction_cleared"

# Keys that mark messages compaction must never fold into a summary: the
# memory manager's injected block and turn-only verifier messages. Spelled out
# here rather than imported so this module stays free of agent imports.
_MEMORY_BLOCK_KEY = "tulip_memory_block"
_EPHEMERAL_KEY = "tulip_ephemeral"

# A user-role message the loop wrote (a continuation note): never the user's
# latest request. See tulip.agent.completion.AUTOMATED_NOTE_KEY.
_AUTOMATED_NOTE_KEY = "tulip_automated_note"

# A tool result this small costs about as much as the stub that would replace
# it, so clearing it loses information and frees nothing.
_MIN_CLEARABLE_TOKENS = 64

# How much of each message the summariser reads. A tool output is evidence for
# the summary, not something to reproduce, and the head of a long output
# (a test run's failures, a file's first lines) carries most of what matters.
_SUMMARY_TOOL_OUTPUT_CHARS = 2_000
_SUMMARY_TEXT_CHARS = 8_000
_SUMMARY_ARGS_CHARS = 600

SUMMARY_PREFIX = (
    "[Context compacted: an automated note, not a message from the user] Earlier "
    "turns of this task were summarised to free the "
    "context window. The summary below is your memory of them: rely on it for the "
    "goal, the decisions already made and the work already verified. Re-read a file "
    "before you edit it, since its contents are not in front of you. Questions in the "
    "summary are history, not requests. Continue the task from the next step; do not "
    "stop to ask whether to continue."
)

SUMMARY_SYSTEM_PROMPT = """\
You maintain the working memory of an autonomous agent whose context window is \
full. Older turns are about to be removed; your summary is all the agent will \
have of them when it carries on, so write it for the agent, not for a reader.

Write these sections, in this order, as Markdown headings:
## Goal and constraints
The task as the user stated it, every requirement and constraint, and any \
preference the user expressed. Quote exact wording where it matters.
## Decisions made
Each decision with its reason, including approaches tried and rejected and why.
## Files and state
Every file, resource or identifier touched: path or name, what changed, and its \
current state. Keep exact paths, names, commands, versions and numbers.
## Verified
What was checked and how (tests, commands, outputs), with the result.
## Remaining work
What is left to do, in order.
## Open problems
Errors, failures and unresolved questions, with the exact error text when known.
## Next step
The single next action the agent was about to take.

Rules: when a previous summary is given, update it: keep everything still true, \
revise what changed, and add what is new; never drop a requirement, a decision \
or a file. Do not invent anything that is not in the input; mark uncertain \
details as uncertain. Be dense: no preamble, no closing remarks."""


#: ``Message.metadata`` key on a summary listing the archived outputs it folded,
#: as ``[id, label, characters]`` triples, carried into the next summary.
RECALLABLE_OUTPUTS_KEY = "tulip_recallable_outputs"

#: How many recallable outputs a summary lists: the newest ones. A long run's
#: list would otherwise grow by a line for every output it ever cleared.
_RECALLABLE_LIST_LIMIT = 64

_RECALLABLE_HEADING = "## Recallable tool outputs"


class OutputArchive(Protocol):
    """Where compaction archives the outputs it clears, so they stay recallable."""

    def clear(self, message: Message, label: str) -> Message | None:
        """``message`` as a stub naming its archive id; ``None`` when it cannot be archived."""
        ...

    def recall_id(self, message: Message) -> str | None:
        """The archive id of a tool output, archiving it first when needed."""
        ...


def is_summary_message(message: Message) -> bool:
    """Whether ``message`` is a summary a previous compaction wrote."""
    return bool(message.metadata.get(SUMMARY_MESSAGE_KEY))


def _is_pinned(message: Message) -> bool:
    """Messages that are never folded into a summary."""
    metadata = message.metadata
    return bool(metadata.get(_MEMORY_BLOCK_KEY) or metadata.get(_EPHEMERAL_KEY))


def _is_automated(message: Message) -> bool:
    """A user-role message the loop wrote, not the user."""
    metadata = message.metadata
    return bool(metadata.get(_AUTOMATED_NOTE_KEY) or metadata.get(_EPHEMERAL_KEY))


@dataclass
class CompactionTracker:
    """What one run remembers between compactions.

    Lives on the run, not on the compactor: one agent serves concurrent runs,
    and one run's compaction must not trip another run's thrash guard.
    """

    #: Iteration of the last summary (or truncation), for the thrash guard.
    last_summary_iteration: int | None = None
    #: Tokens the provider reported for the last request plus its reply: the
    #: most accurate measure of the context there is, when the provider gives it.
    reported_tokens: int | None = None
    #: How many state messages that report covers. Messages after it are new
    #: and are estimated.
    reported_length: int = 0

    def observe(self, usage: Mapping[str, int], length: int) -> None:
        """Record a model call's reported usage over the first ``length`` messages."""
        # Anthropic and Bedrock report cached prompt tokens apart from
        # ``prompt_tokens``; OpenAI-style providers include them in it and do not
        # send the cache keys, so the sum is the whole context either way.
        total = sum(
            int(usage.get(key, 0) or 0)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        if total > 0:
            self.reported_tokens = total
            self.reported_length = length

    def forget_report(self) -> None:
        """Drop the report: the messages it measured have been rewritten."""
        self.reported_tokens = None
        self.reported_length = 0


@dataclass(frozen=True)
class CompactionOutcome:
    """What one compaction did."""

    #: The compacted conversation. Valid for a provider: no tool call is
    #: separated from its result.
    messages: list[Message]
    stage: CompactionStage
    tokens_before: int
    tokens_after: int
    threshold: int
    #: The summary text, for ``summarize``.
    summary: str | None = None
    #: The context cannot be brought under the threshold; the run should end.
    exhausted: bool = False
    #: Why the run is exhausted, or why the summary was skipped.
    detail: str | None = None
    #: Token usage of the summary calls, to count against the run's budgets.
    usage: dict[str, int] = field(default_factory=dict)


class ContextCompactor(ConversationManager):
    """Clear old tool output, then summarise older history, under a token threshold.

    Args:
        context_length: The model's input window in tokens.
        summary_model: Model that writes the summary (anything with an async
            ``complete(messages=..., tools=..., max_tokens=...)``). ``None``
            drops older history with a note instead of summarising it, which
            costs no model calls and loses the history's content.
        trigger_fraction: Compact once the context reaches this fraction of
            the usable window (``context_length - reserved_tokens``).
        reserved_tokens: Room kept for the model's reply. Default:
            ``min(20_000, context_length // 5)``, opencode's 20k on large windows
            and proportionally less on small ones.
        tail_turns: Most recent turns kept verbatim by a summary. A turn is one
            assistant message with its tool results (or one user message).
        tail_token_fraction: Most of the usable window the verbatim tail may
            take; fewer turns are kept when they are larger (never fewer than one).
        tool_output_keep_tokens: Newest tool output kept when clearing. Default:
            ``min(40_000, usable // 4)``.
        summary_max_tokens: Output cap for each summary call. Default:
            ``min(4_096, context_length // 10)``.
        min_iterations_between_summaries: A summary needed again within this
            many iterations of the last one means the context cannot hold the
            run any more: the run ends rather than looping. ``0`` disables it.
        token_counter: Maps a message to a token count; char/4 by default.
    """

    def __init__(
        self,
        *,
        context_length: int,
        summary_model: Any = None,
        trigger_fraction: float = 0.9,
        reserved_tokens: int | None = None,
        tail_turns: int = 6,
        tail_token_fraction: float = 0.25,
        tool_output_keep_tokens: int | None = None,
        summary_max_tokens: int | None = None,
        min_iterations_between_summaries: int = 3,
        token_counter: Callable[[Message], int] | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0, 1]")
        if tail_turns < 1:
            raise ValueError("tail_turns must be at least 1")
        if not 0.0 < tail_token_fraction < 1.0:
            raise ValueError("tail_token_fraction must be in (0, 1)")
        if min_iterations_between_summaries < 0:
            raise ValueError("min_iterations_between_summaries must be non-negative")
        reserved = (
            reserved_tokens if reserved_tokens is not None else min(20_000, context_length // 5)
        )
        if not 0 <= reserved < context_length:
            raise ValueError("reserved_tokens must be in [0, context_length)")

        self.context_length = context_length
        self.summary_model = summary_model
        self.trigger_fraction = trigger_fraction
        self.reserved_tokens = reserved
        self.tail_turns = tail_turns
        self.tail_token_fraction = tail_token_fraction
        usable = context_length - reserved
        self.tool_output_keep_tokens = (
            tool_output_keep_tokens
            if tool_output_keep_tokens is not None
            else min(40_000, usable // 4)
        )
        self.summary_max_tokens = (
            summary_max_tokens
            if summary_max_tokens is not None
            else max(256, min(4_096, context_length // 10))
        )
        self.min_iterations_between_summaries = min_iterations_between_summaries
        self._count = token_counter or _char_count_tokens

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    @property
    def usable_tokens(self) -> int:
        """The window minus the room reserved for the reply."""
        return self.context_length - self.reserved_tokens

    @property
    def threshold(self) -> int:
        """Context size, in tokens, at which compaction starts."""
        return int(self.usable_tokens * self.trigger_fraction)

    def tokens(self, messages: Sequence[Message]) -> int:
        """Estimated tokens of ``messages``."""
        return sum(self._count(m) for m in messages)

    @staticmethod
    def tool_tokens(schemas: Sequence[Mapping[str, Any]] | None) -> int:
        """Estimated tokens the tool definitions add to every request.

        Part of the fixed cost compaction cannot reduce; leaving it out is how a
        harness ends up compacting on every turn without ever getting under.
        """
        if not schemas:
            return 0
        return len(json.dumps(list(schemas), default=str)) // 4

    def measure(
        self,
        messages: Sequence[Message],
        *,
        tool_tokens: int = 0,
        tracker: CompactionTracker | None = None,
    ) -> int:
        """The context the next request will carry, in tokens.

        The larger of the char/4 estimate and the provider's own count for the
        last request plus an estimate of what was added since. The estimate
        undercounts code and non-English text; the report is exact but stale.
        """
        estimate = self.tokens(messages) + tool_tokens
        if (
            tracker is not None
            and tracker.reported_tokens is not None
            and tracker.reported_length <= len(messages)
        ):
            since = self.tokens(messages[tracker.reported_length :])
            estimate = max(estimate, tracker.reported_tokens + since)
        return estimate

    # ------------------------------------------------------------------
    # ConversationManager
    # ------------------------------------------------------------------

    def apply(self, messages: list[Message]) -> list[Message]:
        """Return ``messages`` unchanged: the loop already compacted the state."""
        return list(messages)

    async def async_apply(self, messages: list[Message]) -> list[Message]:
        """Return ``messages`` unchanged: the loop already compacted the state."""
        return list(messages)

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
    ) -> CompactionOutcome | None:
        """Compact ``messages`` when they reach the threshold; ``None`` when not.

        Args:
            messages: The run's whole conversation.
            iteration: The run's current iteration, for the thrash guard.
            tracker: The run's compaction memory; updated in place.
            tool_tokens: Tokens the tool definitions add (see :meth:`tool_tokens`).
            tokens_before: The measured context, when the caller already has it.
            instructions: Extra guidance for the summary (from a pre-compact hook).
            archive: Archives cleared and folded outputs so ``obs_recall`` can
                read them back (see :mod:`tulip.memory.observation_pack`).
                ``None`` clears to a lossy stub.
        """
        original = list(messages)
        before = (
            tokens_before
            if tokens_before is not None
            else self.measure(original, tool_tokens=tool_tokens, tracker=tracker)
        )
        threshold = self.threshold
        if before < threshold:
            return None
        # When the provider's count is above the char/4 estimate (code and
        # non-English text tokenise denser), every estimate below is scaled by
        # the same ratio, so "under the threshold" means under it for real.
        raw_before = self.tokens(original) + tool_tokens
        scale = max(1.0, before / raw_before) if raw_before else 1.0

        # Stage 1: clear old tool output. Free, and enough on its own whenever
        # tool results are what filled the window. It has to buy real room,
        # though: clearing one output per turn would change an old message on
        # every request, which defeats the provider's prompt cache each time,
        # and only postpones the summary by a turn.
        cleared = self._clear_tool_outputs(original, archive)
        after_clearing = int((self.tokens(cleared) + tool_tokens) * scale)
        if after_clearing < threshold - self.usable_tokens // 10:
            tracker.forget_report()
            return CompactionOutcome(
                messages=cleared,
                stage="prune",
                tokens_before=before,
                tokens_after=after_clearing,
                threshold=threshold,
            )

        # Stage 2. A summary needed again this soon means the last one did not
        # buy room: the fixed part of the context is too big, and summarising
        # again would spend tokens to land in the same place.
        last = tracker.last_summary_iteration
        guard = self.min_iterations_between_summaries
        if guard and last is not None and iteration - last < guard:
            return CompactionOutcome(
                messages=cleared,
                stage="summarize",
                tokens_before=before,
                tokens_after=after_clearing,
                threshold=threshold,
                exhausted=True,
                detail=(
                    f"compaction is thrashing: the context needed summarising at "
                    f"iteration {iteration}, {iteration - last} after the last summary "
                    f"(minimum {guard}). The latest turns and fixed context "
                    f"({after_clearing} tokens against a {threshold}-token threshold) "
                    "no longer fit the window."
                ),
            )

        outcome = await self._summarise_history(
            original, cleared, tool_tokens, before, scale, instructions, archive
        )
        tracker.forget_report()
        if outcome.stage in ("summarize", "truncate") and not outcome.exhausted:
            tracker.last_summary_iteration = iteration
        return outcome

    def _clear_tool_outputs(
        self, messages: list[Message], archive: OutputArchive | None = None
    ) -> list[Message]:
        """Replace tool outputs older than the newest ``tool_output_keep_tokens``."""
        calls = {tc.id: tc for m in messages if m.role == Role.ASSISTANT for tc in m.tool_calls}
        out = list(messages)
        kept = 0
        full = False
        newest = True
        for index in range(len(out) - 1, -1, -1):
            message = out[index]
            if message.role != Role.TOOL or message.metadata.get(CLEARED_OUTPUT_KEY):
                continue
            size = self._count(message)
            if size <= _MIN_CLEARABLE_TOKENS:
                continue
            # The newest output is always kept whole: the model is about to act
            # on it, and a stub there would make it call the tool again forever.
            if newest or (not full and kept + size <= self.tool_output_keep_tokens):
                kept += size
                newest = False
                continue
            full = True
            call = calls.get(message.tool_call_id or "")
            recallable = (
                archive.clear(message, _call_label(call, message.name)) if archive else None
            )
            out[index] = (
                recallable.model_copy(
                    update={"metadata": {**recallable.metadata, CLEARED_OUTPUT_KEY: True}}
                )
                if recallable is not None
                else _stub(message, call)
            )
        return out

    async def _summarise_history(
        self,
        original: list[Message],
        cleared: list[Message],
        tool_tokens: int,
        before: int,
        scale: float,
        instructions: str | None,
        archive: OutputArchive | None = None,
    ) -> CompactionOutcome:
        threshold = self.threshold
        # Budgets below are in estimated tokens; the threshold, scaled down to
        # match them.
        limit = int(threshold / scale)
        head_end = _head_end(cleared)
        head = cleared[:head_end]
        previous = None
        recallable: list[list[Any]] = []
        body: list[int] = []
        for index in range(head_end, len(cleared)):
            message = cleared[index]
            if is_summary_message(message):
                previous = _summary_text(message)
                recallable = list(message.metadata.get(RECALLABLE_OUTPUTS_KEY) or [])
            else:
                body.append(index)

        # The user's latest request is kept verbatim. A note the loop added in
        # the user's role (a continuation, a verifier's feedback) is not that
        # request: taking it for one would fold the real task into the summary.
        latest_user = max(
            (i for i in body if cleared[i].role == Role.USER and not _is_automated(cleared[i])),
            default=None,
        )
        turns = _turns(body, cleared)

        # The verbatim tail gets what the fixed part of the next request leaves,
        # capped by ``tail_token_fraction`` and at least the newest turn.
        fixed = self.tokens(head) + tool_tokens + self.summary_max_tokens
        room = max(0, limit - fixed)
        tail_budget = min(int(self.usable_tokens * self.tail_token_fraction), room)
        tail: list[list[int]] = []
        tail_size = 0
        for turn in reversed(turns):
            size = sum(self._count(cleared[i]) for i in turn)
            if tail and (len(tail) >= self.tail_turns or tail_size + size > tail_budget):
                break
            tail.insert(0, turn)
            tail_size += size
        tail_indices = {i for turn in tail for i in turn}
        older = [i for i in body if i not in tail_indices]
        pinned = [i for i in older if _is_pinned(cleared[i]) or i == latest_user]
        folded = [i for i in older if i not in pinned]

        if not folded:
            size = int((self.tokens(cleared) + tool_tokens) * scale)
            return CompactionOutcome(
                messages=cleared,
                stage="summarize",
                tokens_before=before,
                tokens_after=size,
                threshold=threshold,
                exhausted=True,
                detail=(
                    "nothing left to summarise: the system prompt, task, tool "
                    f"definitions and latest turn alone are {size} tokens against "
                    f"a {threshold}-token threshold."
                ),
            )

        usage: dict[str, int] = {}
        summary: str | None = None
        error: str | None = None
        # The outputs being folded stay recallable: their ids go on the summary,
        # which is the only thing the model will have of them.
        ids: dict[int, str] = {}
        if archive is not None:
            for i in folded:
                if cleared[i].role != Role.TOOL:
                    continue
                stored = cleared[i].metadata.get(OBSERVATION_ID_KEY)
                found = stored if isinstance(stored, str) else archive.recall_id(original[i])
                if found:
                    ids[i] = found
                    label = _call_label(_call_for(cleared, i), cleared[i].name)
                    recallable.append([found, label, len(original[i].content or "")])
        if self.summary_model is not None:
            # The summariser reads the original tool output (clipped), not the
            # stubs: clearing was for the agent's window, not for its memory.
            rendered = [_render(original[i], cleared, i, ids.get(i)) for i in folded]
            for attempt in range(2):
                # A retry sends shorter input: a summary that failed on size would fail the same way.
                sent = rendered if attempt == 0 else [_shorten_for_retry(text) for text in rendered]
                try:
                    summary = await self._summarise(sent, previous, instructions, usage)
                except Exception as exc:  # noqa: BLE001 — a failed summary falls back below
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Context summary failed (attempt %d): %s", attempt + 1, error)
                    continue
                if summary:
                    break
                error = "the summary model returned no text"
        stage: CompactionStage = "summarize" if summary else "truncate"
        if not summary:
            summary = _truncation_note(previous, len(folded), error)

        # A user-role message: several adapters (the native Anthropic one
        # among them) send a single system prompt and take the last system
        # message for it, so a system-role summary would replace the agent's
        # instructions on those providers.
        metadata: dict[str, Any] = {SUMMARY_MESSAGE_KEY: True}
        content = f"{SUMMARY_PREFIX}\n\n{summary}"
        if recallable:
            recallable = _dedupe_recallable(recallable)[-_RECALLABLE_LIST_LIMIT:]
            metadata[RECALLABLE_OUTPUTS_KEY] = recallable
            content = f"{content}\n\n{_recallable_section(recallable)}"
        summary_message = Message(role=Role.USER, content=content, metadata=metadata)
        compacted = [
            *head,
            summary_message,
            *(cleared[i] for i in pinned),
            *(cleared[i] for turn in tail for i in turn),
        ]
        compacted = _repair_boundaries(compacted, original)
        after = int((self.tokens(compacted) + tool_tokens) * scale)
        exhausted = after >= threshold
        return CompactionOutcome(
            messages=compacted,
            stage=stage,
            tokens_before=before,
            tokens_after=after,
            threshold=threshold,
            summary=summary if stage == "summarize" else None,
            exhausted=exhausted,
            detail=(
                (
                    f"compaction left {after} tokens against a {threshold}-token "
                    "threshold: the system prompt, task, tool definitions, summary "
                    "and latest turn no longer fit the window."
                )
                if exhausted
                else error
            ),
            usage=usage,
        )

    async def _summarise(
        self,
        rendered: list[str],
        previous: str | None,
        instructions: str | None,
        usage: dict[str, int],
    ) -> str:
        """Fold ``rendered`` into ``previous``, a chunk at a time.

        Chunks keep each call inside the window when the history being folded
        is larger than the window itself, which is the normal case on a run
        whose earlier summary failed or that resumed from a large checkpoint.
        """
        budget = max(1_000, self.usable_tokens - 2 * self.summary_max_tokens - 1_500)
        chunks: list[list[str]] = [[]]
        size = 0
        for text in rendered:
            clipped = _clip(text, budget * 4)
            cost = len(clipped) // 4
            if chunks[-1] and size + cost > budget:
                chunks.append([])
                size = 0
            chunks[-1].append(clipped)
            size += cost
        summary = previous
        for chunk in chunks:
            summary = await self._summarise_chunk("\n\n".join(chunk), summary, instructions, usage)
            if not summary:
                return ""
        return summary or ""

    async def _summarise_chunk(
        self,
        transcript: str,
        previous: str | None,
        instructions: str | None,
        usage: dict[str, int],
    ) -> str:
        parts = []
        if previous:
            parts.append(f"<previous_summary>\n{previous}\n</previous_summary>")
        parts.append(f"<conversation>\n{transcript}\n</conversation>")
        if instructions:
            parts.append(f"Additional instructions for this summary:\n{instructions}")
        parts.append(
            "Write the updated summary now."
            if previous
            else "Write the summary of the conversation above now."
        )
        response = await self.summary_model.complete(
            messages=[Message.system(SUMMARY_SYSTEM_PROMPT), Message.user("\n\n".join(parts))],
            tools=None,
            max_tokens=self.summary_max_tokens,
        )
        for key, value in (getattr(response, "usage", None) or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        content = getattr(response.message, "content", None)
        return content.strip() if isinstance(content, str) else ""

    def __repr__(self) -> str:
        return (
            f"ContextCompactor(context_length={self.context_length}, "
            f"threshold={self.threshold}, tail_turns={self.tail_turns}, "
            f"summarises={self.summary_model is not None})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _head_end(messages: list[Message]) -> int:
    """End of the protected head: leading system messages and the task message."""
    index = 0
    while (
        index < len(messages)
        and messages[index].role == Role.SYSTEM
        and not is_summary_message(messages[index])
    ):
        index += 1
    if (
        index < len(messages)
        and messages[index].role == Role.USER
        and not is_summary_message(messages[index])
    ):
        index += 1
    return index


def _turns(indices: list[int], messages: list[Message]) -> list[list[int]]:
    """Group message indices into turns.

    A turn opens at each assistant or user message; tool results and system
    notes join the turn before them. A tool result therefore always shares a
    turn with the call it answers, so a cut between turns never separates them.
    """
    turns: list[list[int]] = []
    for index in indices:
        if not turns or messages[index].role in (Role.ASSISTANT, Role.USER):
            turns.append([index])
        else:
            turns[-1].append(index)
    return turns


def _summary_text(message: Message) -> str:
    content = message.content or ""
    content = content.removeprefix(SUMMARY_PREFIX)
    # The recallable list is rebuilt from metadata, not summarised.
    content = content.split(f"\n\n{_RECALLABLE_HEADING}\n", 1)[0]
    return content.strip()


def _call_for(messages: list[Message], index: int) -> Any:
    """The call a tool result at ``index`` answers, if it is still in ``messages``."""
    call_id = messages[index].tool_call_id
    for prior in reversed(messages[:index]):
        match = next((tc for tc in prior.tool_calls if tc.id == call_id), None)
        if match is not None:
            return match
    return None


def _dedupe_recallable(entries: list[list[Any]]) -> list[list[Any]]:
    """Each id once, at its latest position."""
    latest: dict[str, list[Any]] = {}
    for entry in entries:
        latest.pop(str(entry[0]), None)
        latest[str(entry[0])] = entry
    return list(latest.values())


def _recallable_section(entries: list[list[Any]]) -> str:
    lines = [
        _RECALLABLE_HEADING,
        "The exact text of these earlier tool outputs is archived. Call obs_recall "
        'with {"id": "<id>", "offset": 0} to read one again instead of re-running the tool.',
    ]
    lines.extend(f"- {entry[0]}: {entry[1]}, {entry[2]} characters" for entry in entries)
    return "\n".join(lines)


def _call_label(call: Any, name: str | None) -> str:
    """``tool(arg=value, …)``, short enough for a stub."""
    tool = getattr(call, "name", None) or name or "tool"
    arguments = getattr(call, "arguments", None) or {}
    rendered = ", ".join(f"{k}={_clip(repr(v), 80)}" for k, v in arguments.items())
    return f"{tool}({_clip(rendered, 200)})"


def _stub(message: Message, call: Any) -> Message:
    size = len(message.content or "")
    return Message(
        role=Role.TOOL,
        tool_call_id=message.tool_call_id,
        name=message.name,
        content=(
            f"[output cleared to free context: {_call_label(call, message.name)} "
            f"returned {size} characters. Call the tool again if you need it.]"
        ),
        metadata={CLEARED_OUTPUT_KEY: True},
    )


_RETRY_KEEP_CHARS = 2_000


def _shorten_for_retry(text: str) -> str:
    if len(text) <= 2 * _RETRY_KEEP_CHARS:
        return text
    return (
        f"{text[:_RETRY_KEEP_CHARS]}\n[... shortened for the summary retry ...]\n"
        f"{text[-_RETRY_KEEP_CHARS:]}"
    )


def _render(
    message: Message, cleared: list[Message], index: int, recall_id: str | None = None
) -> str:
    """One message as summariser input."""
    role = message.role
    if role == Role.TOOL:
        call = _call_for(cleared, index)
        label = call.name if call is not None else (message.name or "tool")
        archived = f", archived as {recall_id}" if recall_id else ""
        return (
            f"[tool result: {label}{archived}]\n"
            f"{_clip(message.content or '', _SUMMARY_TOOL_OUTPUT_CHARS)}"
        )
    text = _clip(message.content or "", _SUMMARY_TEXT_CHARS)
    if role == Role.ASSISTANT:
        calls = "\n".join(
            f"-> calls {tc.name}({_clip(json.dumps(tc.arguments, default=str), _SUMMARY_ARGS_CHARS)})"
            for tc in message.tool_calls
        )
        return "\n".join(
            part for part in (f"[assistant]\n{text}" if text else "[assistant]", calls) if part
        )
    if role == Role.USER:
        return f"[user]\n{text}"
    return f"[system note]\n{text}"


def _clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, keeping the head and the tail."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - 40)
    head = keep * 3 // 4
    tail = keep - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters omitted ...]\n{text[-tail:] if tail else ''}"


def _truncation_note(previous: str | None, dropped: int, error: str | None) -> str:
    reason = f" because the summary failed ({error})" if error else ""
    note = (
        f"[{dropped} earlier messages were removed without a summary{reason}. "
        "Re-check the workspace before relying on results from earlier in the task.]"
    )
    return f"{previous}\n\n{note}" if previous else note
