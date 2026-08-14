# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Conversation management for Tulip - manage message history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tulip.core.messages import Message


class ConversationManager(ABC):
    """
    Base class for conversation management strategies.

    Conversation managers handle how message history is maintained,
    including trimming, summarization, and other memory management.
    """

    @abstractmethod
    def apply(self, messages: list[Message]) -> list[Message]:
        """
        Apply conversation management to messages.

        Args:
            messages: Full message history

        Returns:
            Managed message list (potentially trimmed/summarized)
        """
        ...

    async def async_apply(self, messages: list[Message]) -> list[Message]:
        """
        Async version of apply. Supports async summarization functions.

        Default implementation delegates to the synchronous apply().
        Override in subclasses that need async operations (e.g., LLM summarization).
        """
        return self.apply(messages)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NullManager(ConversationManager):
    """
    Null conversation manager - keeps all messages unchanged.

    Use this when you want to preserve the entire conversation history
    without any modifications. Suitable for short conversations or
    when full history is required.
    """

    def apply(self, messages: list[Message]) -> list[Message]:
        """Return messages unchanged."""
        return messages.copy()


class SlidingWindowManager(ConversationManager):
    """
    Sliding window conversation manager - keeps last N messages.

    Preserves the system message (if present) and the last N messages from the
    conversation. This is a simple and effective strategy for managing
    conversation length.

    When the window would retain **no user turn at all**, the opening request is
    re-attached. That case is an agent loop — assistant/tool all the way down —
    where the first user message is the task, and losing it leaves the model
    working from a role description and a wall of tool output with no statement
    of what it was asked to do. It reads as drift rather than as an error, and
    on Qwen-family chat templates it fails outright with ``No user query found
    in messages.``. Twenty tool round-trips reach the default window, so this is
    the ordinary case for a tool-using agent, not an edge one.

    A chat is left alone: while any user turn survives in the window, the newest
    ones are the live context and the first is not privileged.

    Args:
        window_size: Maximum number of messages to keep (excluding system message)
        preserve_system: Whether to preserve the system message at the start
        preserve_first_user: Whether to re-attach the opening user turn when the
            window would otherwise contain none
    """

    def __init__(
        self,
        window_size: int = 20,
        preserve_system: bool = True,
        preserve_first_user: bool = True,
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.preserve_system = preserve_system
        self.preserve_first_user = preserve_first_user

    def apply(self, messages: list[Message]) -> list[Message]:
        """
        Apply sliding window to messages.

        Keeps the system message (if preserve_system is True) and the last
        window_size messages, re-attaching the opening user turn when the window
        would otherwise retain none (see the class docstring).
        """
        if not messages:
            return []

        from tulip.core.messages import Role

        result: list[Message] = []
        non_system_messages: list[Message] = []

        for msg in messages:
            if msg.role == Role.SYSTEM and self.preserve_system:
                result.append(msg)
            else:
                non_system_messages.append(msg)

        # Keep only the last window_size messages
        before_count = len(non_system_messages)
        if before_count > self.window_size:
            anchor: Message | None = None
            if self.preserve_first_user:
                anchor = next(
                    (m for m in non_system_messages if m.role == Role.USER),
                    None,
                )

            non_system_messages = non_system_messages[-self.window_size :]

            # The cut lands at a fixed offset, not a turn boundary, so it can
            # fall between an assistant's tool_calls and the tool messages
            # answering them. Those results are orphans the provider will
            # reject, so drop them here rather than leaving the agent loop to
            # strip them later.
            def _drop_leading_orphans(window: list[Message]) -> list[Message]:
                kept = {tc.id for m in window if m.role == Role.ASSISTANT for tc in m.tool_calls}
                start = 0
                while (
                    start < len(window)
                    and window[start].role == Role.TOOL
                    and window[start].tool_call_id not in kept
                ):
                    start += 1
                return window[start:]

            non_system_messages = _drop_leading_orphans(non_system_messages)

            # Re-attach the opening request only when the window kept no user
            # turn at all. In a chat the newest user turns are the live context
            # and the first one is not privileged; it is an agent loop —
            # assistant/tool all the way down — that ends up with no user turn
            # and therefore no statement of the task.
            #
            # Trimming orphans usually frees the slot; if it didn't, drop one
            # more turn from the front and re-normalise, since that drop can
            # itself orphan the results that followed it.
            if anchor is not None and not any(m.role == Role.USER for m in non_system_messages):
                if len(non_system_messages) >= self.window_size:
                    non_system_messages = _drop_leading_orphans(non_system_messages[1:])
                non_system_messages.insert(0, anchor)
            from tulip.observability.emit import (  # noqa: PLC0415
                EV_MEMORY_CONVERSATION_PRUNED,
                emit_sync,
            )

            emit_sync(
                EV_MEMORY_CONVERSATION_PRUNED,
                strategy="sliding_window",
                window_size=self.window_size,
                # Count what actually went, not what the raw slice implies:
                # orphan trimming and the re-attached anchor both move this.
                removed_count=before_count - len(non_system_messages),
            )

        result.extend(non_system_messages)
        return result

    def __repr__(self) -> str:
        return f"SlidingWindowManager(window_size={self.window_size}, preserve_system={self.preserve_system})"


class SummarizingManager(ConversationManager):
    """
    Summarizing conversation manager - summarizes older messages.

    When the conversation exceeds a threshold, older messages are
    summarized into a single summary message, preserving recent context.

    This manager requires a summarization function to be provided,
    which can use an LLM to generate summaries.

    Args:
        threshold: Number of messages before summarization kicks in
        keep_recent: Number of recent messages to always preserve
        summarize_fn: Async function that summarizes a list of messages
        preserve_system: Whether to preserve the system message
    """

    def __init__(
        self,
        threshold: int = 30,
        keep_recent: int = 10,
        summarize_fn: Any | None = None,
        preserve_system: bool = True,
    ):
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        if keep_recent >= threshold:
            raise ValueError("keep_recent must be less than threshold")

        self.threshold = threshold
        self.keep_recent = keep_recent
        self.summarize_fn = summarize_fn
        self.preserve_system = preserve_system
        self._summary_cache: dict[int, str] = {}

    def apply(self, messages: list[Message]) -> list[Message]:
        """
        Apply summarization to messages.

        If total messages exceed threshold, older messages are summarized.
        Note: If no summarize_fn is provided, falls back to a simple summary.
        """
        if not messages:
            return []

        from tulip.core.messages import Message as Msg
        from tulip.core.messages import Role

        result: list[Msg] = []
        non_system_messages: list[Msg] = []
        system_message: Msg | None = None

        for msg in messages:
            if msg.role == Role.SYSTEM and self.preserve_system:
                system_message = msg
            else:
                non_system_messages.append(msg)

        # If under threshold, no summarization needed
        if len(non_system_messages) <= self.threshold:
            if system_message:
                result.append(system_message)
            result.extend(non_system_messages)
            return result

        # Split into messages to summarize and recent messages
        to_summarize = non_system_messages[: -self.keep_recent]
        recent = non_system_messages[-self.keep_recent :]

        # Generate summary
        summary_text = self._generate_summary(to_summarize)

        # Build result
        if system_message:
            result.append(system_message)

        # Add summary as a system message
        summary_message = Msg(
            role=Role.SYSTEM,
            content=f"[Summary of previous conversation ({len(to_summarize)} messages)]:\n{summary_text}",
        )
        result.append(summary_message)
        result.extend(recent)

        return result

    def _generate_summary(self, messages: list[Message]) -> str:
        """
        Generate a summary of messages.

        If summarize_fn is provided, it's used. Otherwise, a simple
        extractive summary is generated.
        """
        # Create cache key from message contents
        cache_key = hash(
            tuple((m.role.value, m.content or "", len(m.tool_calls)) for m in messages)
        )

        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        if self.summarize_fn is not None:
            # If async function provided, we can't call it synchronously
            # This is a limitation - for async summarization, use async API
            summary = f"Conversation with {len(messages)} messages summarized."
        else:
            # Simple extractive summary
            summary_parts = []
            for msg in messages[-5:]:  # Last 5 messages before cutoff
                content = msg.content or ""
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    summary_parts.append(f"- {msg.role.value}: {preview}")

            summary = "\n".join(summary_parts) if summary_parts else "No significant content."

        self._summary_cache[cache_key] = summary
        return summary

    async def async_apply(self, messages: list[Message]) -> list[Message]:
        """
        Async apply with LLM summarization support.

        If summarize_fn is an async function, it will be properly awaited.
        Falls back to synchronous _generate_summary() otherwise.
        """
        if not messages:
            return []

        from tulip.core.messages import Message as Msg
        from tulip.core.messages import Role

        result: list[Msg] = []
        non_system_messages: list[Msg] = []
        system_message: Msg | None = None

        for msg in messages:
            if msg.role == Role.SYSTEM and self.preserve_system:
                system_message = msg
            else:
                non_system_messages.append(msg)

        if len(non_system_messages) <= self.threshold:
            if system_message:
                result.append(system_message)
            result.extend(non_system_messages)
            return result

        to_summarize = non_system_messages[: -self.keep_recent]
        recent = non_system_messages[-self.keep_recent :]

        # Use async summarize_fn if available
        import asyncio

        from tulip.observability.emit import (  # noqa: PLC0415
            EV_MEMORY_COMPACTOR_COMPLETED,
            EV_MEMORY_COMPACTOR_TRIGGERED,
            emit,
        )

        await emit(
            EV_MEMORY_COMPACTOR_TRIGGERED,
            strategy="summarizing",
            messages_before=len(non_system_messages),
            threshold=self.threshold,
        )
        import time as _time  # noqa: PLC0415

        _started = _time.perf_counter()

        if self.summarize_fn is not None and asyncio.iscoroutinefunction(self.summarize_fn):
            summary_text = await self.summarize_fn(to_summarize)
        else:
            summary_text = self._generate_summary(to_summarize)

        if system_message:
            result.append(system_message)

        summary_message = Msg(
            role=Role.SYSTEM,
            content=f"[Summary of previous conversation ({len(to_summarize)} messages)]:\n{summary_text}",
        )
        result.append(summary_message)
        result.extend(recent)
        await emit(
            EV_MEMORY_COMPACTOR_COMPLETED,
            strategy="summarizing",
            messages_before=len(non_system_messages),
            messages_after=len(result),
            summarized_count=len(to_summarize),
            duration_ms=(_time.perf_counter() - _started) * 1000,
        )
        return result

    def __repr__(self) -> str:
        return f"SummarizingManager(threshold={self.threshold}, keep_recent={self.keep_recent})"
