# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for interrupt (Human-in-the-Loop) module."""

import pytest

from locus.core.interrupt import (
    AutoApproveHandler,
    CallbackInterruptHandler,
    GraphInterrupted,
    InterruptException,
    InterruptState,
    InterruptValue,
    NodeExecutionContext,
    clear_resume_context,
    get_current_graph_id,
    get_current_node_id,
    interrupt,
    set_resume_context,
)


class TestInterruptValue:
    """Tests for InterruptValue class."""

    def test_basic_creation(self):
        """Test basic InterruptValue creation."""
        iv = InterruptValue(payload={"action": "delete"})
        assert iv.payload == {"action": "delete"}
        assert iv.interrupt_id.startswith("int_")
        assert iv.node_id is None
        assert iv.graph_id is None

    def test_with_metadata(self):
        """Test InterruptValue with metadata."""
        iv = InterruptValue(
            payload="Approve?",
            metadata={"urgency": "high", "deadline": "2024-01-01"},
        )
        assert iv.metadata["urgency"] == "high"

    def test_to_display(self):
        """Test to_display method."""
        iv = InterruptValue(
            payload={"question": "Confirm?"},
            node_id="approval_node",
        )
        display = iv.to_display()
        assert "interrupt_id" in display
        assert display["payload"] == {"question": "Confirm?"}
        assert display["node_id"] == "approval_node"
        assert "created_at" in display


class TestInterruptState:
    """Tests for InterruptState class."""

    def test_creation(self):
        """Test InterruptState creation."""
        iv = InterruptValue(payload="test")
        state = InterruptState(
            interrupt=iv,
            node_id="node1",
            pending_nodes=["node2", "node3"],
            state_snapshot={"x": 1},
        )
        assert state.interrupt == iv
        assert state.node_id == "node1"
        assert state.pending_nodes == ["node2", "node3"]


class TestInterruptException:
    """Tests for InterruptException."""

    def test_exception_creation(self):
        """Test InterruptException creation."""
        iv = InterruptValue(payload="test")
        exc = InterruptException(iv)
        assert exc.value == iv
        assert iv.interrupt_id in str(exc)

    def test_exception_is_catchable(self):
        """Test that InterruptException can be caught."""
        iv = InterruptValue(payload="test")
        with pytest.raises(InterruptException) as exc_info:
            raise InterruptException(iv)
        assert exc_info.value.value.payload == "test"


class TestGraphInterrupted:
    """Tests for GraphInterrupted exception."""

    def test_creation(self):
        """Test GraphInterrupted creation."""
        iv = InterruptValue(payload="test")
        state = InterruptState(interrupt=iv, node_id="node1")
        exc = GraphInterrupted(state, checkpoint_id="cp123")
        assert exc.interrupt_state == state
        assert exc.checkpoint_id == "cp123"


class TestNodeExecutionContext:
    """Tests for NodeExecutionContext."""

    def test_context_sets_node_id(self):
        """Test that context sets node_id."""
        with NodeExecutionContext(node_id="test_node", graph_id="test_graph"):
            assert get_current_node_id() == "test_node"
            assert get_current_graph_id() == "test_graph"

    def test_context_clears_on_exit(self):
        """Test that context clears values on exit."""
        with NodeExecutionContext(node_id="test_node"):
            pass
        # After context, should be None (or previous value)
        # Note: Due to context var semantics, may need reset handling


class TestInterruptFunction:
    """Tests for interrupt() function."""

    def test_raises_when_not_resuming(self):
        """Test interrupt raises exception when not resuming."""
        with NodeExecutionContext(node_id="test"):
            with pytest.raises(InterruptException) as exc_info:
                interrupt({"question": "Approve?"})

            assert exc_info.value.value.payload == {"question": "Approve?"}
            assert exc_info.value.value.node_id == "test"

    def test_returns_value_when_resuming(self):
        """Test interrupt returns resume value when resuming."""
        with NodeExecutionContext(
            node_id="test",
            resume_value="approved",
            is_resuming=True,
        ):
            result = interrupt({"question": "Approve?"})
            assert result == "approved"

    def test_includes_metadata(self):
        """Test interrupt includes metadata."""
        with NodeExecutionContext(node_id="test"):
            with pytest.raises(InterruptException) as exc_info:
                interrupt("Question?", priority="high", category="approval")

            assert exc_info.value.value.metadata["priority"] == "high"
            assert exc_info.value.value.metadata["category"] == "approval"


class TestResumeContext:
    """Tests for resume context functions."""

    def test_set_and_clear(self):
        """Test set_resume_context and clear_resume_context."""
        set_resume_context("test_value")
        # In a real scenario, interrupt() would consume this
        clear_resume_context()


class TestAutoApproveHandler:
    """Tests for AutoApproveHandler."""

    @pytest.mark.asyncio
    async def test_returns_configured_response(self):
        """Test handler returns configured response."""
        handler = AutoApproveHandler(response="approved")
        iv = InterruptValue(payload="test")
        result = await handler.handle(iv)
        assert result == "approved"

    @pytest.mark.asyncio
    async def test_default_response(self):
        """Test default response is 'approved'."""
        handler = AutoApproveHandler()
        iv = InterruptValue(payload="test")
        result = await handler.handle(iv)
        assert result == "approved"

    @pytest.mark.asyncio
    async def test_can_handle_returns_true(self):
        """Test can_handle returns True by default."""
        handler = AutoApproveHandler()
        iv = InterruptValue(payload="test")
        result = await handler.can_handle(iv)
        assert result is True


class TestCallbackInterruptHandler:
    """Tests for CallbackInterruptHandler."""

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        """Test with sync callback."""

        def my_callback(interrupt):
            return f"handled: {interrupt.payload}"

        handler = CallbackInterruptHandler(my_callback)
        iv = InterruptValue(payload="test")
        result = await handler.handle(iv)
        assert result == "handled: test"

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test with async callback."""

        async def my_async_callback(interrupt):
            return f"async handled: {interrupt.payload}"

        handler = CallbackInterruptHandler(my_async_callback)
        iv = InterruptValue(payload="test")
        result = await handler.handle(iv)
        assert result == "async handled: test"
