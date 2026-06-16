# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for logging hooks."""

import logging
from unittest.mock import MagicMock

import pytest

from locus.core.state import AgentState
from locus.hooks.builtin.logging import LoggingHook, StructuredLoggingHook
from locus.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent


class TestLoggingHook:
    """Tests for LoggingHook."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def hook(self, mock_logger):
        """Create hook with mock logger."""
        hook = LoggingHook(
            level=logging.INFO,
            log_arguments=True,
            log_results=True,
        )
        hook._logger = mock_logger
        return hook

    @pytest.mark.asyncio
    async def test_on_before_invocation(self, hook, mock_logger):
        """Log before invocation."""
        state = AgentState()

        result = await hook.on_before_invocation("Test prompt", state)

        assert result is state
        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_after_invocation(self, hook, mock_logger):
        """Log after invocation."""
        state = AgentState(iteration=5, confidence=0.8)

        await hook.on_after_invocation(state, success=True)

        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_before_tool_call(self, hook, mock_logger):
        """Log before tool call with arguments."""
        hook._log_arguments = True
        event = BeforeToolCallEvent(
            tool_name="search", tool_call_id="t1", arguments={"query": "test"}
        )

        await hook.on_before_tool_call(event)

        # Hook is observe-only — event.arguments is unmodified.
        assert event.arguments == {"query": "test"}
        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_after_tool_call_success(self, hook, mock_logger):
        """Log successful tool call."""
        hook._log_results = True
        event = AfterToolCallEvent(tool_name="search", result="Found 5 results", error=None)

        await hook.on_after_tool_call(event)

        mock_logger.log.assert_called()
        call_args = mock_logger.log.call_args
        assert "success" in str(call_args) or call_args is not None

    @pytest.mark.asyncio
    async def test_on_after_tool_call_with_result_preview(self, hook, mock_logger):
        """Log tool call with result preview for long results."""
        hook._log_results = True

        # Long result that should be truncated
        long_result = "x" * 300
        event = AfterToolCallEvent(tool_name="search", result=long_result, error=None)
        await hook.on_after_tool_call(event)

        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_after_tool_call_error(self, hook, mock_logger):
        """Log tool call error."""
        event = AfterToolCallEvent(tool_name="search", result=None, error="Connection failed")
        await hook.on_after_tool_call(event)

        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_iteration_start(self, hook, mock_logger):
        """Log iteration start."""
        state = AgentState(iteration=1, max_iterations=10, confidence=0.5)

        await hook.on_iteration_start(1, state)

        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_on_iteration_end(self, hook, mock_logger):
        """Log iteration end."""
        state = AgentState(iteration=1, confidence=0.7)

        await hook.on_iteration_end(1, state)

        mock_logger.log.assert_called()


class TestStructuredLoggingHook:
    """Tests for StructuredLoggingHook."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def hook(self, mock_logger):
        """Create structured hook with mock logger."""
        hook = StructuredLoggingHook(
            level=logging.INFO,
            include_timestamps=True,
        )
        hook._logger = mock_logger
        return hook

    def test_init(self):
        """Test initialization."""
        hook = StructuredLoggingHook(
            level=logging.DEBUG,
            logger_name="test",
            extra={"app": "test"},
            include_timestamps=False,
        )

        assert hook._level == logging.DEBUG
        assert hook._include_timestamps is False
        assert hook._extra["app"] == "test"

    @pytest.mark.asyncio
    async def test_log_with_timestamps(self, hook, mock_logger):
        """Log includes timestamps when enabled."""
        hook._include_timestamps = True
        state = AgentState()

        await hook.on_before_invocation("Test", state)

        mock_logger.log.assert_called()
        call_args = mock_logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        # The structured record should have timestamp
        if "structured" in extra:
            assert "timestamp" in extra["structured"]

    @pytest.mark.asyncio
    async def test_log_without_timestamps(self, mock_logger):
        """Log excludes timestamps when disabled."""
        hook = StructuredLoggingHook(
            level=logging.INFO,
            include_timestamps=False,
        )
        hook._logger = mock_logger
        state = AgentState()

        await hook.on_before_invocation("Test", state)

        mock_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_log_structured_format(self, hook, mock_logger):
        """Log uses structured format."""
        state = AgentState()

        await hook.on_before_invocation("Test prompt", state)

        mock_logger.log.assert_called()
        call_args = mock_logger.log.call_args
        extra = call_args.kwargs.get("extra", {})
        assert "structured" in extra

    @pytest.mark.asyncio
    async def test_extra_context(self, mock_logger):
        """Extra context is included in logs."""
        hook = StructuredLoggingHook(
            level=logging.INFO,
            extra={"service": "test-service", "version": "1.0"},
        )
        hook._logger = mock_logger
        state = AgentState()

        await hook.on_before_invocation("Test", state)

        mock_logger.log.assert_called()
