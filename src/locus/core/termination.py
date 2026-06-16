# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Composable termination conditions.

Pluggable stop conditions that can be combined with | (OR) and & (AND).
Replaces hardcoded termination logic with a flexible, declarative system.

Example:
    from locus.core.termination import (
        MaxIterations, TokenLimit, TextMention, TimeLimit, ToolCalled,
    )

    # Stop after 10 iterations OR when agent says "DONE"
    condition = MaxIterations(10) | TextMention("DONE")

    # Stop after 5 iterations AND token limit reached
    condition = MaxIterations(5) & TokenLimit(5000)

    agent = Agent(config=AgentConfig(
        model=model,
        termination=condition,
    ))
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.core.state import AgentState


class TerminationCondition(ABC):
    """Base class for composable termination conditions.

    Subclasses implement `should_terminate(state)` which returns
    (should_stop: bool, reason: str | None).

    Conditions are composable:
    - `a | b` — stop if either condition is met (OR)
    - `a & b` — stop if both conditions are met (AND)
    """

    @abstractmethod
    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        """Check if termination condition is met.

        Args:
            state: Current agent state.
            **context: Additional context (last_message, tool_names, etc.)

        Returns:
            Tuple of (should_stop, reason).
        """
        ...

    def reset(self) -> None:
        """Reset any internal state. Called between runs."""

    def __or__(self, other: TerminationCondition) -> TerminationCondition:
        """Combine with OR: stop if either condition met."""
        return OrCondition(self, other)

    def __and__(self, other: TerminationCondition) -> TerminationCondition:
        """Combine with AND: stop if both conditions met."""
        return AndCondition(self, other)


class OrCondition(TerminationCondition):
    """Stop if ANY child condition is met."""

    def __init__(self, *conditions: TerminationCondition) -> None:
        self._conditions = list(conditions)

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        for cond in self._conditions:
            stop, reason = cond.check(state, **context)
            if stop:
                return True, reason
        return False, None

    def reset(self) -> None:
        for cond in self._conditions:
            cond.reset()


class AndCondition(TerminationCondition):
    """Stop only if ALL child conditions are met."""

    def __init__(self, *conditions: TerminationCondition) -> None:
        self._conditions = list(conditions)

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        reasons: list[str] = []
        for cond in self._conditions:
            stop, reason = cond.check(state, **context)
            if not stop:
                return False, None
            if reason:
                reasons.append(reason)
        return True, " AND ".join(reasons) if reasons else "all_conditions_met"

    def reset(self) -> None:
        for cond in self._conditions:
            cond.reset()


# =============================================================================
# Built-in Conditions
# =============================================================================


class MaxIterations(TerminationCondition):
    """Stop after N iterations."""

    def __init__(self, max_iterations: int) -> None:
        self._max = max_iterations

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        if state.iteration >= self._max:
            return True, "max_iterations"
        return False, None


class TokenLimit(TerminationCondition):
    """Stop when token usage exceeds a limit."""

    def __init__(self, max_tokens: int) -> None:
        self._max = max_tokens

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        if state.total_tokens_used >= self._max:
            return True, "token_budget"
        return False, None


class TimeLimit(TerminationCondition):
    """Stop after a time duration."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._start: float | None = None

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        if self._start is None:
            self._start = time.time()
        if time.time() - self._start >= self._seconds:
            return True, "time_budget"
        return False, None

    def reset(self) -> None:
        self._start = None


class TextMention(TerminationCondition):
    """Stop when specific text appears in the last message."""

    def __init__(self, text: str, case_sensitive: bool = False) -> None:
        self._text = text
        self._case_sensitive = case_sensitive

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        last_content = context.get("last_message", "")
        if not last_content and state.messages:
            for msg in reversed(state.messages):
                if msg.role.value == "assistant" and msg.content:
                    last_content = msg.content
                    break

        if not last_content:
            return False, None

        if self._case_sensitive:
            found = self._text in last_content
        else:
            found = self._text.lower() in last_content.lower()

        if found:
            return True, f"text_mention:{self._text}"
        return False, None


class ToolCalled(TerminationCondition):
    """Stop when a specific tool is called."""

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        for te in state.tool_executions:
            if te.tool_name == self._tool_name:
                return True, f"tool_called:{self._tool_name}"
        return False, None


class ConfidenceMet(TerminationCondition):
    """Stop when confidence threshold is reached."""

    def __init__(self, threshold: float = 0.9) -> None:
        self._threshold = threshold

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        if state.confidence >= self._threshold:
            return True, "confidence_met"
        return False, None


class NoToolCalls(TerminationCondition):
    """Stop when the model produces no tool calls."""

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        no_tools = context.get("no_tool_calls", False)
        if no_tools:
            return True, "no_tools"
        return False, None


class CustomCondition(TerminationCondition):
    """Stop based on a custom function.

    Example:
        condition = CustomCondition(
            lambda state, **ctx: (state.iteration > 3 and "error" in str(state.messages[-1].content), "custom")
        )
    """

    def __init__(self, fn: Callable[..., tuple[bool, str | None]]) -> None:
        self._fn = fn

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        return self._fn(state, **context)
