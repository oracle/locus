# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for hooks/builtin module."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from locus.hooks.builtin.logging import LoggingHook
from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
)


class TestLoggingHook:
    """Tests for LoggingHook."""

    def test_create_default(self):
        """Test creating hook with defaults."""
        hook = LoggingHook()
        assert hook.priority == HookPriority.OBSERVABILITY_DEFAULT
        assert hook.name == "LoggingHook"
        assert hook._level == logging.INFO
        assert hook._log_arguments is False
        assert hook._log_results is False

    def test_create_custom(self):
        """Test creating hook with custom settings."""
        hook = LoggingHook(
            level=logging.DEBUG,
            logger_name="custom.logger",
            extra={"env": "test"},
            log_arguments=True,
            log_results=True,
            priority=100,
        )
        assert hook._level == logging.DEBUG
        assert hook._log_arguments is True
        assert hook._log_results is True
        assert hook.priority == 100

    @pytest.mark.asyncio
    async def test_on_before_invocation(self):
        """Test before_invocation logging."""
        hook = LoggingHook(level=logging.DEBUG)
        mock_state = MagicMock()

        with patch.object(hook._logger, "log") as mock_log:
            result = await hook.on_before_invocation("test prompt", mock_state)

        mock_log.assert_called_once()
        assert result is mock_state

    @pytest.mark.asyncio
    async def test_on_after_invocation(self):
        """Test after_invocation logging."""
        hook = LoggingHook()
        mock_state = MagicMock()

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_after_invocation(mock_state, True)

        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_before_tool_call(self):
        """Test before_tool_call logging."""
        hook = LoggingHook()
        args = {"key": "value"}
        event = BeforeToolCallEvent(tool_name="test_tool", tool_call_id="t1", arguments=args)

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_before_tool_call(event)

        mock_log.assert_called_once()
        # Hook is observe-only — event.arguments is unmodified.
        assert event.arguments == args

    @pytest.mark.asyncio
    async def test_on_before_tool_call_with_arguments(self):
        """Test logging with arguments enabled."""
        hook = LoggingHook(log_arguments=True)
        args = {"key": "value"}
        event = BeforeToolCallEvent(tool_name="test_tool", tool_call_id="t1", arguments=args)

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_before_tool_call(event)

        # Should include arguments in log
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "arguments" in str(call_args) or args in str(call_args)

    @pytest.mark.asyncio
    async def test_on_after_tool_call(self):
        """Test after_tool_call logging."""
        hook = LoggingHook()
        event = AfterToolCallEvent(tool_name="test_tool", result="result", error=None)

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_after_tool_call(event)

        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_after_tool_call_with_error(self):
        """Test after_tool_call logging with error."""
        hook = LoggingHook()
        event = AfterToolCallEvent(tool_name="test_tool", result=None, error="Error message")

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_after_tool_call(event)

        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_iteration_start(self):
        """Test iteration_start logging."""
        hook = LoggingHook()
        mock_state = MagicMock()

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_iteration_start(5, mock_state)

        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_iteration_end(self):
        """Test iteration_end logging."""
        hook = LoggingHook()
        mock_state = MagicMock()

        with patch.object(hook._logger, "log") as mock_log:
            await hook.on_iteration_end(5, mock_state)

        mock_log.assert_called_once()

    def test_log_method(self):
        """Test the _log helper method."""
        hook = LoggingHook(extra={"base": "context"})

        with patch.object(hook._logger, "log") as mock_log:
            hook._log("Test message", extra_key="extra_value")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        # Check extra context is merged
        extra = call_args.kwargs.get("extra", {})
        assert "base" in extra
        assert "extra_key" in extra

    def test_register_hooks(self):
        """Test register_hooks method."""
        hook = LoggingHook()
        hooks = hook.register_hooks()

        assert hooks["on_before_invocation"] is True
        assert hooks["on_after_invocation"] is True
        assert hooks["on_before_tool_call"] is True
        assert hooks["on_after_tool_call"] is True
        assert hooks["on_iteration_start"] is True
        assert hooks["on_iteration_end"] is True


class TestBuiltinHooksThroughRegistry:
    """End-to-end: register a built-in hook through HookRegistry and
    dispatch a tool call. This is the contract the README documents
    (`registry.add_provider(LoggingHook())`); regression-pinning it
    here prevents the kind of signature drift tracked in #45 from
    re-emerging silently. The unit tests above only call the hook
    methods directly — they don't exercise the dispatch path."""

    @pytest.mark.asyncio
    async def test_logging_hook_dispatches_through_registry(self):
        """LoggingHook registered via HookRegistry routes correctly."""
        from locus.hooks.registry import HookRegistry

        registry = HookRegistry()
        hook = LoggingHook()
        registry.add_provider(hook)

        with patch.object(hook._logger, "log") as mock_log:
            modified = await registry.emit_before_tool_call("search", {"q": "test"})
            await registry.emit_after_tool_call("search", "ok", None)

        # Two log calls: one before, one after.
        assert mock_log.call_count == 2
        # Modified args round-trip cleanly (LoggingHook is observe-only).
        assert modified == {"q": "test"}
