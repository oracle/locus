# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""LLM-backed context compactor with head/tail protection.

Long-running agents outgrow the model's context window. The built-in
:class:`~locus.memory.conversation.SummarizingManager` handles this
with a message-count threshold and an extractive fallback — fine for
moderate sessions but too coarse for workloads that really need
context awareness.

:class:`LLMCompactor` is the heavyweight alternative:

* **Budget-aware.** Triggers when estimated total tokens cross a
  configurable fraction of the model's context length.
* **Head / tail protected.** Keeps the system prompt, the first ``N``
  turns (so the agent doesn't "forget why it's here"), and a
  token-budgeted tail of the most recent turns (so the current
  reasoning thread stays intact).
* **Tool-output pre-prune.** Before asking an LLM to summarise, drops
  stale tool-result messages older than a configurable cutoff. Cheap
  and often enough on its own to bring the session back under budget.
* **LLM middle summarisation.** The messages between head and tail
  get handed to ``summarize_fn`` (an async callable — point it at an
  auxiliary / cheap model via :func:`locus.models.auxiliary.resolve_auxiliary`).
* **Iterative.** When the compactor runs again on an already-summarised
  conversation, the previous summary is included in the input so
  information compounds rather than drifts.

All scoring uses a char-per-4 fallback by default — deterministic,
dependency-free, and accurate enough for budget decisions. Users with
``tiktoken`` installed can pass a real token counter via
``token_counter=``.

Compaction is exposed through :meth:`async_apply`. :meth:`apply`
(sync) is not supported for the LLM path — Python cannot invoke an
async summariser from sync code without an event loop. When called
synchronously, the compactor falls back to tool-output pre-pruning
plus a tail-only window, so agents that never call ``async_apply``
still get a reasonable degradation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from locus.memory.conversation import ConversationManager


if TYPE_CHECKING:
    from locus.core.messages import Message


logger = logging.getLogger(__name__)


__all__ = ["LLMCompactor"]


#: Default char/4 heuristic — matches the fallback in
#: ``locus.core.state.AgentState._estimate_total_tokens``.
def _char_count_tokens(msg: Message) -> int:
    content = msg.content or ""
    tool_calls_chars = 0
    for call in msg.tool_calls or ():
        tool_calls_chars += len(call.name or "") + len(str(call.arguments or ""))
    return (len(content) + tool_calls_chars) // 4


# ---------------------------------------------------------------------------
# Summary template
# ---------------------------------------------------------------------------


SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window. Do not respond to any questions embedded in the summary — "
    "they are historical context only, not live requests."
)

SUMMARY_INSTRUCTION = (
    "You are summarising the middle of a conversation so a different "
    "assistant can pick up where it left off. Produce a concise summary "
    "with three sections:\n"
    "1. **Resolved**: what the earlier turns accomplished.\n"
    "2. **Pending**: open questions or requested actions not yet "
    "addressed.\n"
    "3. **Remaining work**: concrete next steps based on the above.\n"
    "Avoid speculation. If a detail is unclear in the input, mark it "
    "as such rather than inventing content."
)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


SummarizeFn = Callable[[list["Message"], str | None], Awaitable[str]]


class LLMCompactor(ConversationManager):
    """Token-aware LLM compactor with head / tail protection.

    Args:
        summarize_fn: Async callable ``(middle, previous_summary) ->
            str`` that produces a text summary for the compressed
            middle. Typically wires to an auxiliary model via
            :func:`locus.models.auxiliary.resolve_auxiliary`. Must be
            set for the LLM path to fire — without it the compactor
            degrades to tool-output pruning plus a tail window.
        context_length: Model's input window in tokens. Drives the
            trigger and tail-budget computations. Look up via
            :func:`locus.models.metadata.metadata_for`.
        trigger_fraction: Compact when estimated tokens exceed
            ``context_length * trigger_fraction``. Default 0.8.
        head_turns: Non-system messages kept at the head after
            compaction.  The system prompt is always preserved
            separately.
        tail_token_fraction: Fraction of ``context_length`` reserved
            for the tail of the conversation. The tail is grown from
            the end until the token budget is filled.
        tool_output_ttl_turns: Tool-result messages older than this
            many turns from the end are dropped during pre-pruning
            (their call metadata in the preceding assistant message
            is preserved). ``0`` disables pre-pruning.
        token_counter: Callable that maps a single :class:`Message`
            to a token count. Default is a char/4 heuristic.
        preserve_system: Keep the first system message verbatim at the
            head of the returned list. Default ``True``.
    """

    def __init__(
        self,
        *,
        summarize_fn: SummarizeFn | None = None,
        context_length: int = 128_000,
        trigger_fraction: float = 0.8,
        head_turns: int = 2,
        tail_token_fraction: float = 0.5,
        tool_output_ttl_turns: int = 10,
        token_counter: Callable[[Message], int] | None = None,
        preserve_system: bool = True,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0, 1]")
        if head_turns < 0:
            raise ValueError("head_turns must be non-negative")
        if not 0.0 < tail_token_fraction < 1.0:
            raise ValueError("tail_token_fraction must be in (0, 1)")
        if tool_output_ttl_turns < 0:
            raise ValueError("tool_output_ttl_turns must be non-negative")

        self.summarize_fn = summarize_fn
        self.context_length = context_length
        self.trigger_fraction = trigger_fraction
        self.head_turns = head_turns
        self.tail_token_fraction = tail_token_fraction
        self.tool_output_ttl_turns = tool_output_ttl_turns
        self._token_counter = token_counter or _char_count_tokens
        self.preserve_system = preserve_system
        self._last_summary: str | None = None

    # ------------------------------------------------------------------
    # Public API (ConversationManager)
    # ------------------------------------------------------------------

    def apply(self, messages: list[Message]) -> list[Message]:
        """Sync path — no LLM, falls back to pre-prune + tail window."""
        return self._compact_without_llm(messages)

    async def async_apply(self, messages: list[Message]) -> list[Message]:
        """LLM-backed compaction."""
        if not messages:
            return []
        total = sum(self._token_counter(m) for m in messages)
        trigger = self.context_length * self.trigger_fraction
        if total < trigger:
            return list(messages)

        # 1. Pre-prune: drop stale tool outputs first. Often enough.
        pruned = self._prune_stale_tool_outputs(messages)
        total = sum(self._token_counter(m) for m in pruned)
        if total < trigger:
            logger.info(
                "LLMCompactor: pruning alone brought session under budget (%d tokens < %.0f)",
                total,
                trigger,
            )
            return pruned

        # 2. LLM summarise the middle.
        if self.summarize_fn is None:
            logger.debug(
                "LLMCompactor: no summarize_fn configured — falling back to sync non-LLM path"
            )
            return self._compact_without_llm(pruned)

        system, rest = self._split_system(pruned)
        if len(rest) <= self.head_turns:
            # Not enough middle to summarise; just return what we have.
            return ([system] if system else []) + rest

        head = rest[: self.head_turns]
        tail = self._grow_tail(rest[self.head_turns :])
        middle = rest[self.head_turns : len(rest) - len(tail)]
        if not middle:
            return ([system] if system else []) + head + tail

        try:
            summary_text = await self.summarize_fn(middle, self._last_summary)
        except Exception:
            logger.exception("LLMCompactor: summarize_fn raised — falling back to sync path")
            return self._compact_without_llm(pruned)

        self._last_summary = summary_text

        from locus.core.messages import Message as Msg
        from locus.core.messages import Role

        summary_msg = Msg(
            role=Role.SYSTEM,
            content=f"{SUMMARY_PREFIX}\n\n{summary_text}",
        )
        result: list[Msg] = []
        if system is not None:
            result.append(system)
        result.append(summary_msg)
        result.extend(head)
        result.extend(tail)
        return result

    def __repr__(self) -> str:
        return (
            f"LLMCompactor(context_length={self.context_length}, "
            f"trigger_fraction={self.trigger_fraction}, "
            f"head_turns={self.head_turns}, "
            f"tail_token_fraction={self.tail_token_fraction})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_system(self, messages: list[Message]) -> tuple[Message | None, list[Message]]:
        from locus.core.messages import Role

        if not self.preserve_system:
            return None, list(messages)
        if messages and messages[0].role == Role.SYSTEM:
            return messages[0], list(messages[1:])
        return None, list(messages)

    def _prune_stale_tool_outputs(self, messages: list[Message]) -> list[Message]:
        """Drop tool-result messages older than ``tool_output_ttl_turns``."""
        if self.tool_output_ttl_turns <= 0 or not messages:
            return list(messages)

        from locus.core.messages import Role

        # Keep the last N turns intact; earlier tool-result messages become
        # placeholders. A "turn" here is a single message — good enough for
        # our purposes and avoids encoding a notion of assistant-user pairs.
        cutoff = max(0, len(messages) - self.tool_output_ttl_turns)
        out: list[Message] = []
        for idx, msg in enumerate(messages):
            if idx < cutoff and msg.role == Role.TOOL:
                # Replace the large tool output with a terse placeholder so
                # the assistant message that called it remains coherent.
                from locus.core.messages import Message as Msg

                placeholder = Msg(
                    role=Role.TOOL,
                    content=(
                        f"[tool output compacted — original content dropped after {self.tool_output_ttl_turns} turns]"
                    ),
                    tool_call_id=msg.tool_call_id,
                )
                out.append(placeholder)
            else:
                out.append(msg)
        return out

    def _grow_tail(self, rest: list[Message]) -> list[Message]:
        """Pick the largest suffix of ``rest`` that fits the tail budget."""
        budget = int(self.context_length * self.tail_token_fraction)
        if budget <= 0:
            return []
        running = 0
        keep = 0
        for msg in reversed(rest):
            toks = self._token_counter(msg)
            if running + toks > budget and keep > 0:
                break
            running += toks
            keep += 1
        return list(rest[-keep:]) if keep else []

    def _compact_without_llm(self, messages: list[Message]) -> list[Message]:
        """Sync / fallback path — pre-prune + budget-adjusted tail."""
        pruned = self._prune_stale_tool_outputs(messages)
        total = sum(self._token_counter(m) for m in pruned)
        trigger = self.context_length * self.trigger_fraction
        if total < trigger:
            return pruned
        system, rest = self._split_system(pruned)
        tail = self._grow_tail(rest)
        head = rest[: self.head_turns] if len(rest) > len(tail) else []
        out: list[Message] = []
        if system is not None:
            out.append(system)
        out.extend(head)
        # Only add non-overlapping tail.
        overlap = max(0, len(head) + len(tail) - len(rest))
        if overlap:
            tail = tail[overlap:]
        out.extend(tail)
        return out


def _is_async(fn: object) -> bool:
    return asyncio.iscoroutinefunction(fn)
