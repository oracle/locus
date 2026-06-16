# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for conversation management module."""

import pytest

from locus.core.messages import Message
from locus.memory.conversation import (
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
