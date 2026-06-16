# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for tools context module."""

from unittest.mock import MagicMock

from locus.tools.context import ToolContext


class TestToolContext:
    """Tests for ToolContext."""

    def test_create_context(self):
        """Test creating a tool context with required fields."""
        ctx = ToolContext(
            tool_call_id="call123",
            tool_name="test_tool",
            run_id="run456",
            iteration=5,
        )
        assert ctx.tool_call_id == "call123"
        assert ctx.tool_name == "test_tool"
        assert ctx.run_id == "run456"
        assert ctx.iteration == 5
        assert ctx.agent_id is None
        assert ctx.state is None

    def test_create_context_with_all_fields(self):
        """Test creating context with all optional fields."""
        mock_state = MagicMock()
        ctx = ToolContext(
            tool_call_id="call123",
            tool_name="test_tool",
            run_id="run456",
            iteration=3,
            agent_id="agent789",
            state=mock_state,
            invocation_metadata={"user": "test"},
            tool_config={"timeout": 30},
        )
        assert ctx.agent_id == "agent789"
        assert ctx.state is mock_state
        assert ctx.invocation_metadata == {"user": "test"}
        assert ctx.tool_config == {"timeout": 30}

    def test_get_metadata(self):
        """Test getting metadata values."""
        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
            invocation_metadata={"key1": "value1", "key2": 42},
        )
        assert ctx.get_metadata("key1") == "value1"
        assert ctx.get_metadata("key2") == 42
        assert ctx.get_metadata("missing") is None
        assert ctx.get_metadata("missing", "default") == "default"

    def test_get_config(self):
        """Test getting config values."""
        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
            tool_config={"timeout": 30, "retries": 3},
        )
        assert ctx.get_config("timeout") == 30
        assert ctx.get_config("retries") == 3
        assert ctx.get_config("missing") is None
        assert ctx.get_config("missing", 10) == 10

    def test_messages_property_no_state(self):
        """Test messages property returns empty list when no state."""
        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
        )
        assert ctx.messages == []

    def test_messages_property_with_state(self):
        """Test messages property returns messages from state."""
        mock_state = MagicMock()
        mock_state.messages = ["msg1", "msg2"]

        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
            state=mock_state,
        )
        assert ctx.messages == ["msg1", "msg2"]

    def test_confidence_property_no_state(self):
        """Test confidence property returns 0.0 when no state."""
        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
        )
        assert ctx.confidence == 0.0

    def test_confidence_property_with_state(self):
        """Test confidence property returns confidence from state."""
        mock_state = MagicMock()
        mock_state.confidence = 0.85

        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
            state=mock_state,
        )
        assert ctx.confidence == 0.85

    def test_default_values(self):
        """Test default values for optional fields."""
        ctx = ToolContext(
            tool_call_id="call1",
            tool_name="test",
            run_id="run1",
            iteration=0,
        )
        assert ctx.invocation_metadata == {}
        assert ctx.tool_config == {}
        assert ctx.state is None
        assert ctx.agent_id is None
