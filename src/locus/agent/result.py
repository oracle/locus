# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent execution result - 100% Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, computed_field

from locus.core.messages import Message
from locus.core.state import AgentState, ReasoningStep, ToolExecution


T = TypeVar("T", bound=BaseModel)


class ExecutionMetrics(BaseModel):
    """Metrics from agent execution."""

    iterations: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    reflexion_evaluations: int = 0
    grounding_evaluations: int = 0
    # Anthropic prompt-caching token counts. Populated only when the
    # AnthropicModel is configured with `prompt_cache=True` and the
    # provider returns cache_creation_input_tokens / cache_read_input_tokens
    # on the response usage. Zero on other providers.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    model_config = {"frozen": True}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tools_success_rate(self) -> float:
        """Percentage of successful tool calls."""
        if self.tool_calls == 0:
            return 1.0
        return (self.tool_calls - self.tool_errors) / self.tool_calls

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_per_iteration(self) -> float:
        """Average tokens per iteration."""
        if self.iterations == 0:
            return 0.0
        return self.total_tokens / self.iterations


StopReason = Literal[
    "complete",  # Agent finished normally (no more tool calls)
    "terminal_tool",  # A terminal tool was called
    "confidence_met",  # Confidence threshold reached
    "max_iterations",  # Hit iteration limit
    "tool_loop",  # Detected tool loop
    "no_tools",  # No tool calls in response
    "grounding_failed",  # Grounding check failed
    "token_budget",  # Token budget exhausted
    "time_budget",  # Time budget exhausted
    "interrupted",  # Agent paused for user input
    "error",  # Execution error
    "cancelled",  # User cancelled
]


class AgentResult(BaseModel):
    """
    Result from an agent execution.

    Contains the final message, state, and execution metrics.
    """

    model_config = {"frozen": True}

    # Final output
    message: str = Field(
        ...,
        description="Final response message from the agent",
    )

    # Execution state
    state: AgentState = Field(
        ...,
        description="Final agent state",
    )

    # How execution ended
    stop_reason: StopReason = Field(
        ...,
        description="Why the agent stopped",
    )

    # Metrics
    metrics: ExecutionMetrics = Field(
        default_factory=ExecutionMetrics,
        description="Execution metrics",
    )

    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When execution started",
    )

    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When execution completed",
    )

    # Error info (if stop_reason == "error")
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )

    # Grounding info (if grounding was run)
    grounding_score: float | None = Field(
        default=None,
        description="Final grounding score",
    )

    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="Claims that couldn't be grounded",
    )

    # GSAR info (if AgentConfig.gsar was set). The framework lives in
    # locus.reasoning.gsar — see arXiv:2604.23366. The fields are typed
    # as ``Any`` here to keep ``locus.agent`` import-light; the actual
    # values are ``JudgeOutput``, ``float``, and ``Decision`` from
    # ``locus.reasoning.gsar*``.
    gsar_judgment: Any = Field(
        default=None,
        description=(
            "The :class:`~locus.reasoning.gsar_judge.JudgeOutput` "
            "produced by the configured GSAR judge over the agent's "
            "final message + tool-execution history. ``None`` when "
            "``AgentConfig.gsar`` is unset."
        ),
    )

    gsar_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Scalar score ``S`` from Eq. 2, recomputed from the "
            "judgment partition under the configured weight map and "
            "contradiction penalty. ``None`` when GSAR is unset."
        ),
    )

    gsar_decision: str | None = Field(
        default=None,
        description=(
            "The :class:`~locus.reasoning.gsar.Decision` (``proceed``, "
            "``regenerate``, ``replan``, ``abstain``) for ``gsar_score`` "
            "under the configured thresholds. ``None`` when GSAR is unset."
        ),
    )

    # Structured output (if Agent was configured with output_schema)
    parsed: BaseModel | None = Field(
        default=None,
        description=(
            "Final assistant message parsed into the configured ``output_schema``. "
            "``None`` when no schema is set or all parse retries failed."
        ),
    )

    parse_error: str | None = Field(
        default=None,
        description=(
            "Pydantic validation error from the last structured-output attempt, "
            "or ``None`` on success. Mutually exclusive with ``parsed``."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        """Whether execution completed successfully."""
        return self.stop_reason in ("complete", "terminal_tool", "confidence_met")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Final confidence score."""
        return self.state.confidence

    @computed_field  # type: ignore[prop-decorator]
    @property
    def iterations(self) -> int:
        """Number of iterations used."""
        return self.state.iteration

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """Alias for ``message``.

        Many AI SDKs surface the final assistant text as ``.text``;
        Locus's primary field is ``.message``. Both names now work.
        """
        return self.message

    @computed_field  # type: ignore[prop-decorator]
    @property
    def messages(self) -> tuple[Message, ...]:
        """All messages from the conversation."""
        return self.state.messages

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_executions(self) -> tuple[ToolExecution, ...]:
        """All tool executions."""
        return self.state.tool_executions

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reasoning_steps(self) -> tuple[ReasoningStep, ...]:
        """All reasoning steps."""
        return self.state.reasoning_steps

    @property
    def last_assistant_message(self) -> str | None:
        """Get the last assistant message content."""
        for msg in reversed(self.state.messages):
            if msg.role.value == "assistant" and msg.content:
                return msg.content
        return None

    def parsed_as(self, schema: type[T]) -> T:
        """Return ``parsed`` cast to ``schema``, with a runtime check.

        Use this when you want a typed handle on the structured output without
        casting yourself::

            picks = result.parsed_as(VendorList)
            for v in picks.vendors:
                ...

        Raises ``ValueError`` if ``parsed`` is None (parse failed or no schema
        configured) and ``TypeError`` if ``parsed`` is the wrong concrete type.
        """
        if self.parsed is None:
            if self.parse_error:
                raise ValueError(f"AgentResult has no parsed output: {self.parse_error}")
            raise ValueError("AgentResult has no parsed output (no output_schema was configured)")
        if not isinstance(self.parsed, schema):
            raise TypeError(f"Expected {schema.__name__}, got {type(self.parsed).__name__}")
        return self.parsed

    def to_dict(self) -> dict[str, Any]:
        """Export result to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        stop_reason: StopReason,
        metrics: ExecutionMetrics | None = None,
        started_at: datetime | None = None,
        error: str | None = None,
        grounding_score: float | None = None,
        ungrounded_claims: list[str] | None = None,
        parsed: BaseModel | None = None,
        parse_error: str | None = None,
        message: str | None = None,
        gsar_judgment: Any = None,
        gsar_score: float | None = None,
        gsar_decision: str | None = None,
    ) -> AgentResult:
        """
        Create a result from final state.

        Extracts the final message from the last assistant response unless an
        explicit ``message`` is supplied (used after a structuring re-prompt).
        """
        # Find the last assistant message if not provided
        final_message = message
        if final_message is None:
            final_message = ""
            for msg in reversed(state.messages):
                if msg.role.value == "assistant":
                    final_message = msg.content or ""
                    break

        return cls(
            message=final_message,
            state=state,
            stop_reason=stop_reason,
            metrics=metrics or ExecutionMetrics(),
            started_at=started_at or state.started_at,
            completed_at=datetime.now(UTC),
            error=error,
            grounding_score=grounding_score,
            ungrounded_claims=ungrounded_claims or [],
            parsed=parsed,
            parse_error=parse_error,
            gsar_judgment=gsar_judgment,
            gsar_score=gsar_score,
            gsar_decision=gsar_decision,
        )


class StreamingResult(BaseModel):
    """
    Partial result during streaming.

    Used to provide intermediate state during agent execution.
    """

    model_config = {"frozen": True}

    # Current state
    state: AgentState

    # Partial content (accumulated)
    partial_content: str = ""

    # Current iteration
    iteration: int = 0

    # Is complete?
    is_complete: bool = False

    # Final result (if complete)
    final: AgentResult | None = None
