# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Logging hook provider for Locus."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState


class LoggingHook(HookProvider):
    """Hook provider that logs all lifecycle events.

    Provides structured logging for agent execution with configurable
    log levels and optional extra context.

    Example:
        # Basic usage
        registry.add_provider(LoggingHook())

        # With custom log level
        registry.add_provider(LoggingHook(level=logging.DEBUG))

        # With structured context
        registry.add_provider(LoggingHook(
            extra={"environment": "production", "service": "my-agent"}
        ))
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: str = "locus.agent",
        extra: dict[str, Any] | None = None,
        log_arguments: bool = False,
        log_results: bool = False,
        priority: int = HookPriority.OBSERVABILITY_DEFAULT,
    ) -> None:
        """Initialize logging hook.

        Args:
            level: Logging level (default: INFO)
            logger_name: Name for the logger
            extra: Extra context to include in all log records
            log_arguments: Whether to log tool arguments (may contain sensitive data)
            log_results: Whether to log tool results (may be verbose)
            priority: Hook priority
        """
        self._level = level
        self._logger = logging.getLogger(logger_name)
        self._extra = extra or {}
        self._log_arguments = log_arguments
        self._log_results = log_results
        self._priority = priority

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return self._priority

    @property
    def name(self) -> str:
        """Return hook name."""
        return "LoggingHook"

    def _log(self, message: str, **kwargs: Any) -> None:
        """Log a message with extra context.

        Args:
            message: Log message
            **kwargs: Additional context to include
        """
        extra = {**self._extra, **kwargs}
        self._logger.log(self._level, message, extra=extra)

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Log invocation start.

        Args:
            prompt: User prompt
            state: Agent state

        Returns:
            Unchanged state
        """
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self._log(
            "Agent invocation starting",
            run_id=state.run_id,
            agent_id=state.agent_id,
            prompt_preview=prompt_preview,
            prompt_length=len(prompt),
        )
        return state

    async def on_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """Log invocation completion.

        Args:
            state: Final agent state
            success: Whether execution succeeded
        """
        duration_ms = (state.updated_at - state.started_at).total_seconds() * 1000
        self._log(
            "Agent invocation completed",
            run_id=state.run_id,
            agent_id=state.agent_id,
            success=success,
            iterations=state.iteration,
            confidence=state.confidence,
            tool_calls=len(state.tool_executions),
            errors=len(state.errors),
            duration_ms=duration_ms,
        )

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Log tool call start.

        Args:
            event: Write-protected event carrying ``tool_name`` and
                ``arguments``. The hook only inspects them.
        """
        log_data: dict[str, Any] = {"tool_name": event.tool_name}
        if self._log_arguments:
            log_data["arguments"] = event.arguments
        else:
            log_data["argument_keys"] = list(event.arguments.keys())

        self._log("Tool call starting", **log_data)

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Log tool call completion.

        Args:
            event: Write-protected event carrying ``tool_name``,
                ``result``, and ``error``. The hook only inspects them.
        """
        log_data: dict[str, Any] = {
            "tool_name": event.tool_name,
            "success": event.error is None,
        }

        if event.error:
            log_data["error"] = event.error
        elif self._log_results and event.result is not None:
            result_str = str(event.result)
            log_data["result_preview"] = (
                result_str[:200] + "..." if len(result_str) > 200 else result_str
            )
            log_data["result_length"] = len(result_str)

        self._log("Tool call completed", **log_data)

    async def on_iteration_start(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Log iteration start.

        Args:
            iteration: Iteration number
            state: Current state
        """
        self._log(
            "Iteration starting",
            run_id=state.run_id,
            iteration=iteration,
            max_iterations=state.max_iterations,
            confidence=state.confidence,
        )

    async def on_iteration_end(
        self,
        iteration: int,
        state: AgentState,
    ) -> None:
        """Log iteration end.

        Args:
            iteration: Iteration number
            state: Current state
        """
        self._log(
            "Iteration completed",
            run_id=state.run_id,
            iteration=iteration,
            confidence=state.confidence,
            messages=len(state.messages),
        )


class StructuredLoggingHook(LoggingHook):
    """Logging hook with JSON-structured output.

    Extends LoggingHook to emit structured JSON logs suitable
    for log aggregation systems like ELK, Datadog, or CloudWatch.

    Example:
        import json
        import logging

        # Configure JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.getLogger("locus.agent").addHandler(handler)

        registry.add_provider(StructuredLoggingHook())
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: str = "locus.agent.structured",
        extra: dict[str, Any] | None = None,
        include_timestamps: bool = True,
        priority: int = HookPriority.OBSERVABILITY_DEFAULT,
    ) -> None:
        """Initialize structured logging hook.

        Args:
            level: Logging level
            logger_name: Logger name
            extra: Extra context for all logs
            include_timestamps: Whether to include ISO timestamps
            priority: Hook priority
        """
        super().__init__(
            level=level,
            logger_name=logger_name,
            extra=extra,
            log_arguments=False,
            log_results=False,
            priority=priority,
        )
        self._include_timestamps = include_timestamps

    def _log(self, message: str, **kwargs: Any) -> None:
        """Log structured message.

        Args:
            message: Log message
            **kwargs: Structured context
        """
        from datetime import UTC, datetime

        log_record = {
            "message": message,
            "event": message.lower().replace(" ", "_"),
            **self._extra,
            **kwargs,
        }

        if self._include_timestamps:
            log_record["timestamp"] = datetime.now(UTC).isoformat()

        # Log as structured data (handler should format as JSON)
        self._logger.log(self._level, message, extra={"structured": log_record})
