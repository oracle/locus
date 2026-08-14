# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for conversation management module."""

import pytest

from tulip.core.messages import Message, ToolCall, ToolResult
from tulip.memory.conversation import (
    NullManager,
    SlidingWindowManager,
)


class TestNullManager:
    """Tests for NullManager."""

    def test_returns_copy(self):
        """Test that apply returns a copy of messages."""
        manager = NullManager()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        result = manager.apply(messages)

        assert result == messages
        assert result is not messages  # Should be a copy

    def test_empty_messages(self):
        """Test with empty message list."""
        manager = NullManager()
        result = manager.apply([])
        assert result == []

    def test_repr(self):
        """Test string representation."""
        manager = NullManager()
        assert "NullManager" in repr(manager)


class TestSlidingWindowManager:
    """Tests for SlidingWindowManager."""

    def test_default_window_size(self):
        """Test default window size is 20."""
        manager = SlidingWindowManager()
        assert manager.window_size == 20
        assert manager.preserve_system is True

    def test_custom_window_size(self):
        """Test custom window size."""
        manager = SlidingWindowManager(window_size=10, preserve_system=False)
        assert manager.window_size == 10
        assert manager.preserve_system is False

    def test_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            SlidingWindowManager(window_size=0)

    def test_fewer_messages_than_window(self):
        """Test when there are fewer messages than window size."""
        manager = SlidingWindowManager(window_size=10)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        result = manager.apply(messages)

        assert len(result) == 2

    def test_more_messages_than_window(self):
        """Test when there are more messages than window size."""
        manager = SlidingWindowManager(window_size=3, preserve_system=False)
        messages = [Message(role="user", content=f"Message {i}") for i in range(10)]

        result = manager.apply(messages)

        assert len(result) == 3
        # Should keep the last 3 messages
        assert result[0].content == "Message 7"
        assert result[1].content == "Message 8"
        assert result[2].content == "Message 9"

    def test_preserves_system_message(self):
        """Test that system message is preserved."""
        manager = SlidingWindowManager(window_size=2, preserve_system=True)
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User 1"),
            Message(role="assistant", content="Assistant 1"),
            Message(role="user", content="User 2"),
            Message(role="assistant", content="Assistant 2"),
        ]

        result = manager.apply(messages)

        # System + last 2 messages
        assert len(result) == 3
        assert result[0].role == "system"
        assert result[0].content == "System prompt"

    def test_no_preserve_system(self):
        """Test without preserving system message."""
        manager = SlidingWindowManager(window_size=2, preserve_system=False)
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User 1"),
            Message(role="assistant", content="Assistant 1"),
            Message(role="user", content="User 2"),
        ]

        result = manager.apply(messages)

        # Last 2 messages only
        assert len(result) == 2
        assert result[0].content == "Assistant 1"
        assert result[1].content == "User 2"

    def test_empty_messages(self):
        """Test with empty message list."""
        manager = SlidingWindowManager()
        result = manager.apply([])
        assert result == []


class TestSlidingWindowAnchorsTheTask:
    """The opening user turn is the task; dropping it makes the agent drift.

    Twenty tool round-trips reach the default window, so this is the ordinary
    case for a tool-using agent rather than an edge one.
    """

    @staticmethod
    def _tool_heavy(pairs: int, parallel: int = 1) -> list[Message]:
        msgs = [
            Message(role="system", content="SYSTEM PROMPT"),
            Message(role="user", content="THE TASK: audit this repo"),
        ]
        for i in range(pairs):
            calls = [
                ToolCall(id=f"c{i}_{j}", name="read_file", arguments={"p": j})
                for j in range(parallel)
            ]
            msgs.append(Message.assistant("", tool_calls=calls))
            for j in range(parallel):
                msgs.append(
                    Message.tool(
                        ToolResult(tool_call_id=f"c{i}_{j}", name="read_file", content="body")
                    )
                )
        return msgs

    def test_task_survives_a_long_tool_run(self):
        manager = SlidingWindowManager(window_size=40)
        result = manager.apply(self._tool_heavy(20))

        assert any(m.role.value == "user" for m in result), "the task was dropped"
        assert any("THE TASK" in (m.content or "") for m in result)

    def test_window_budget_is_still_respected(self):
        manager = SlidingWindowManager(window_size=40)
        result = manager.apply(self._tool_heavy(20))

        non_system = [m for m in result if m.role.value != "system"]
        assert len(non_system) <= 40

    @pytest.mark.parametrize("parallel", [1, 3, 7])
    def test_no_orphaned_tool_results(self, parallel):
        """The cut can land mid-turn; orphaned results are rejected by providers."""
        manager = SlidingWindowManager(window_size=40)
        result = manager.apply(self._tool_heavy(20, parallel=parallel))

        kept_ids = {tc.id for m in result if m.role.value == "assistant" for tc in m.tool_calls}
        orphans = [m for m in result if m.role.value == "tool" and m.tool_call_id not in kept_ids]
        assert not orphans, f"{len(orphans)} orphaned tool result(s) survived the window"

    def test_short_conversations_are_untouched(self):
        manager = SlidingWindowManager(window_size=40)
        messages = [
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="yo"),
        ]
        assert manager.apply(messages) == messages

    def test_anchor_is_not_added_when_a_user_turn_survived(self):
        manager = SlidingWindowManager(window_size=40)
        result = manager.apply(self._tool_heavy(2))
        users = [m for m in result if m.role.value == "user"]
        assert len(users) == 1

    def test_preserve_first_user_can_be_disabled(self):
        manager = SlidingWindowManager(window_size=40, preserve_first_user=False)
        result = manager.apply(self._tool_heavy(20))
        assert not any(m.role.value == "user" for m in result)

    def test_a_chat_keeps_its_newest_user_turns(self):
        """In a chat the first user turn is not privileged — only an agent loop
        (assistant/tool all the way down) ends up with no user turn at all."""
        manager = SlidingWindowManager(window_size=3, preserve_system=False)
        messages = [Message(role="user", content=f"Message {i}") for i in range(10)]

        result = manager.apply(messages)

        assert [m.content for m in result] == ["Message 7", "Message 8", "Message 9"]
