# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent state management - 100% Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from tulip.core.loops import DEFAULT_READ_ONLY_TOOLS, ToolLoop, call_signature, detect_tool_loop
from tulip.core.media import estimate_tokens
from tulip.core.messages import Message, ToolCall


#: Kept under its old name for code that imported it from here.
_tool_call_signature = call_signature


class ToolExecution(BaseModel):
    """Record of a single tool execution."""

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # True when this execution short-circuited via idempotent dedup
    # (a prior call with identical arguments produced the result that was
    # reused). The tool body did NOT run again.
    idempotent_cache_hit: bool = False

    @property
    def success(self) -> bool:
        """Whether the execution succeeded."""
        return self.error is None


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning trace."""

    iteration: int
    thought: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolExecution] = Field(default_factory=list)
    reflection: str | None = None
    confidence_delta: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentState(BaseModel):
    """
    Immutable state for an agent execution.

    All updates return a new state instance (functional updates).
    """

    # Identity
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str | None = None

    # Conversation
    messages: tuple[Message, ...] = Field(default_factory=tuple)

    # Execution tracking
    iteration: int = 0
    max_iterations: int = 20
    tool_executions: tuple[ToolExecution, ...] = Field(default_factory=tuple)
    reasoning_steps: tuple[ReasoningStep, ...] = Field(default_factory=tuple)

    # Confidence (Reflexion)
    confidence: float = 0.0
    confidence_threshold: float = 0.85
    confidence_history: tuple[float, ...] = Field(default_factory=tuple)

    # Tool loop detection (see tulip.core.loops)
    tool_history: tuple[str, ...] = Field(default_factory=tuple)
    tool_loop_threshold: int = 3
    # Repeats a step made only of read-only tools needs; None = threshold + 1.
    tool_loop_read_only_threshold: int | None = None
    tool_loop_read_only_tools: frozenset[str] = DEFAULT_READ_ONLY_TOOLS
    # (loop signature, reasoning steps at the time) for each loop the model
    # was warned about, so one sighting gets one note. A loop stops the run
    # only once it repeats past the point where it was warned about.
    tool_loop_warnings: tuple[tuple[str, int], ...] = Field(default_factory=tuple)

    # Terminal tools
    terminal_tools: frozenset[str] = Field(
        default_factory=lambda: frozenset({"submit", "done", "finish", "complete"})
    )

    # Token tracking
    total_tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    # Anthropic prompt-cache token counts. Populated only when an
    # AnthropicModel is configured with prompt_cache=True. Zero on
    # other providers.
    cache_creation_tokens_used: int = 0
    cache_read_tokens_used: int = 0
    token_budget: int | None = None
    # Spend tracking. Prices come from model metadata (USD per million tokens);
    # ``None`` means unknown, and an unknown price leaves ``cost_usd_used`` at 0.
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    cost_usd_used: float = 0.0
    cost_budget_usd: float | None = None

    # Completion mode
    completion_mode: str = "auto"  # "auto" or "explicit"

    # Errors
    errors: tuple[str, ...] = Field(default_factory=tuple)

    # Opaque per-provider continuation state. Default None for the
    # vast majority of providers (chat/completions-style transports
    # are stateless). Server-stateful transports such as
    # ``OCIResponsesModel`` populate this with their continuation
    # token (e.g. ``{"previous_response_id": "resp_abc"}``) so that
    # the next turn references the server-held thread instead of
    # resending the full message history. Checkpointer persists
    # this; resume picks it up transparently.
    provider_state: dict[str, Any] | None = None

    # Custom state (user-defined)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"frozen": True}

    # =========================================================================
    # Functional updates (return new state)
    # =========================================================================

    def with_message(self, message: Message) -> AgentState:
        """Add a message to the conversation."""
        return self.model_copy(
            update={
                "messages": (*self.messages, message),
                "updated_at": datetime.now(UTC),
            }
        )

    def with_messages(self, messages: list[Message]) -> AgentState:
        """Add multiple messages to the conversation."""
        return self.model_copy(
            update={
                "messages": (*self.messages, *messages),
                "updated_at": datetime.now(UTC),
            }
        )

    def with_iteration(self, iteration: int) -> AgentState:
        """Update the current iteration."""
        return self.model_copy(
            update={
                "iteration": iteration,
                "updated_at": datetime.now(UTC),
            }
        )

    def next_iteration(self) -> AgentState:
        """Increment iteration counter."""
        return self.with_iteration(self.iteration + 1)

    def with_provider_state(self, provider_state: dict[str, Any] | None) -> AgentState:
        """Replace the provider continuation state.

        Server-stateful transports (e.g. ``OCIResponsesModel``) return
        a continuation token in ``ModelResponse.provider_state``; the
        agent calls this to thread the token into the next turn.
        """
        return self.model_copy(
            update={
                "provider_state": provider_state,
                "updated_at": datetime.now(UTC),
            }
        )

    def with_tool_execution(self, execution: ToolExecution) -> AgentState:
        """Record a tool execution."""
        return self.model_copy(
            update={
                "tool_executions": (*self.tool_executions, execution),
                "tool_history": (*self.tool_history, execution.tool_name),
                "updated_at": datetime.now(UTC),
            }
        )

    def with_reasoning_step(self, step: ReasoningStep) -> AgentState:
        """Add a reasoning step to the trace."""
        return self.model_copy(
            update={
                "reasoning_steps": (*self.reasoning_steps, step),
                "updated_at": datetime.now(UTC),
            }
        )

    def with_confidence(self, confidence: float) -> AgentState:
        """Update confidence score."""
        clamped = max(0.0, min(1.0, confidence))
        return self.model_copy(
            update={
                "confidence": clamped,
                "confidence_history": (*self.confidence_history, clamped),
                "updated_at": datetime.now(UTC),
            }
        )

    def adjust_confidence(self, delta: float, diminishing: bool = True) -> AgentState:
        """
        Adjust confidence with optional diminishing returns.

        Args:
            delta: Raw confidence adjustment (-1.0 to 1.0)
            diminishing: If True, positive deltas are scaled by (1 - current_confidence)
        """
        if diminishing and delta > 0:
            # Diminishing returns: harder to increase confidence as it gets higher
            effective_delta = delta * (1.0 - self.confidence)
        else:
            effective_delta = delta

        return self.with_confidence(self.confidence + effective_delta)

    def with_error(self, error: str) -> AgentState:
        """Record an error."""
        return self.model_copy(
            update={
                "errors": (*self.errors, error),
                "updated_at": datetime.now(UTC),
            }
        )

    def with_metadata(self, key: str, value: Any) -> AgentState:
        """Set a metadata value."""
        return self.model_copy(
            update={
                "metadata": {**self.metadata, key: value},
                "updated_at": datetime.now(UTC),
            }
        )

    def with_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> AgentState:
        """Record token usage from a model response.

        ``cache_creation_tokens`` and ``cache_read_tokens`` are populated
        only when Anthropic returns prompt-cache stats on the response
        usage (i.e., the AnthropicModel was configured with
        ``prompt_cache=True``). Default 0 for other providers.
        """
        return self.model_copy(
            update={
                "total_tokens_used": self.total_tokens_used + prompt_tokens + completion_tokens,
                "prompt_tokens_used": self.prompt_tokens_used + prompt_tokens,
                "completion_tokens_used": self.completion_tokens_used + completion_tokens,
                "cache_creation_tokens_used": (
                    self.cache_creation_tokens_used + cache_creation_tokens
                ),
                "cache_read_tokens_used": self.cache_read_tokens_used + cache_read_tokens,
                "cost_usd_used": self.cost_usd_used
                + (self.cost_of(prompt_tokens, completion_tokens) or 0.0),
                "updated_at": datetime.now(UTC),
            }
        )

    # =========================================================================
    # Queries
    # =========================================================================

    @property
    def priced(self) -> bool:
        """Whether both prices are known, so spend can be measured."""
        return self.input_price_per_mtok is not None and self.output_price_per_mtok is not None

    def cost_of(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """USD for a call of this size, or ``None`` when the prices are unknown.

        Cache-read and cache-write tokens are priced as ordinary input, which
        overstates a cached call rather than understating it.
        """
        if self.input_price_per_mtok is None or self.output_price_per_mtok is None:
            return None
        return (
            prompt_tokens * self.input_price_per_mtok
            + completion_tokens * self.output_price_per_mtok
        ) / 1_000_000

    def would_exceed_cost_budget(self, max_output_tokens: int) -> bool:
        """Whether the next call could cross the spend budget, at its worst case.

        The worst case is the current conversation as input (a char/4 estimate)
        plus ``max_output_tokens`` of output. Checked *before* the call, which is
        what stops one large turn: a check after the call only stops the next one.
        """
        if self.cost_budget_usd is None:
            return False
        prompt_estimate = sum(
            estimate_tokens(m.content)
            + sum(len(str(tc.arguments or "")) for tc in m.tool_calls) // 4
            for m in self.messages
        )
        worst = self.cost_of(prompt_estimate, max_output_tokens)
        return worst is not None and self.cost_usd_used + worst > self.cost_budget_usd

    @property
    def tool_loop(self) -> ToolLoop | None:
        """The loop the run's last steps are in, or ``None`` (see :mod:`tulip.core.loops`).

        A loop is the same step — calls by name and arguments, and their
        results — repeated back to back, or the same short cycle of steps
        repeated whole. Re-reading a file between other work, the same tool
        with other arguments, and a repeat whose result changed are progress.
        """
        return detect_tool_loop(
            self.reasoning_steps,
            threshold=self.tool_loop_threshold,
            read_only_threshold=self.tool_loop_read_only_threshold,
            read_only_tools=self.tool_loop_read_only_tools,
        )

    @property
    def has_tool_loop(self) -> bool:
        """Whether the run's last steps are a loop (warned about or not)."""
        return self.tool_loop is not None

    def tool_loop_warned(self, loop: ToolLoop) -> bool:
        """Whether the model has already been warned about ``loop``."""
        return any(sig == loop.signature for sig, _ in self.tool_loop_warnings)

    def with_tool_loop_warning(self, loop: ToolLoop) -> AgentState:
        """Record that the model was warned about ``loop`` at this step."""
        warnings = (*self.tool_loop_warnings, (loop.signature, len(self.reasoning_steps)))
        return self.model_copy(update={"tool_loop_warnings": warnings})

    @property
    def tool_loop_persists(self) -> bool:
        """Whether the run's loop went on past the point where it is warned about.

        This is what stops a run. The runtime warns the model when a loop
        reaches its threshold (:meth:`with_tool_loop_warning`); the warning is
        the model's chance to change approach, and one more repeat is the
        model ignoring it.
        """
        loop = self.tool_loop
        return loop is not None and loop.past_warning

    @property
    def last_tool_calls(self) -> list[ToolCall]:
        """Get tool calls from the last assistant message of the current turn.

        The scan stops at the most recent user message: calls made before it
        belong to an earlier turn, and treating them as "last" made a new turn
        on a thread whose previous turn ended with a terminal tool stop
        immediately with ``terminal_tool`` before the model saw the message.
        """
        for msg in reversed(self.messages):
            if msg.role.value == "user":
                return []
            if msg.role.value == "assistant" and msg.tool_calls:
                return list(msg.tool_calls)
        return []

    @property
    def called_terminal_tool(self) -> bool:
        """Check if a terminal tool was called."""
        last_calls = self.last_tool_calls
        return any(tc.name in self.terminal_tools for tc in last_calls)

    @property
    def should_terminate(self) -> tuple[bool, str | None]:
        """
        Check if the agent should terminate.

        In "auto" mode: stops on confidence, no_tools, tool_loop, or terminal_tool.
        In "explicit" mode: only stops on terminal_tool, max_iterations, or budgets.
        Use "explicit" for multi-step tasks that require verification before completion.

        Returns:
            Tuple of (should_stop, reason)
        """
        # Hard limits always apply
        if self.iteration >= self.max_iterations:
            return True, "max_iterations"

        if self.token_budget and self.total_tokens_used >= self.token_budget:
            return True, "token_budget"

        if self.cost_budget_usd is not None and self.cost_usd_used >= self.cost_budget_usd:
            return True, "cost_budget"

        # Terminal tool always stops (both modes)
        if self.called_terminal_tool:
            return True, "terminal_tool"

        # In explicit mode, only hard limits and terminal_tool can stop
        if self.completion_mode == "explicit":
            return False, None

        # Auto mode: additional soft termination signals
        if self.confidence >= self.confidence_threshold:
            return True, "confidence_met"

        if self.tool_loop_persists:
            return True, "tool_loop"

        if self.iteration > 0 and self._has_assistant_message() and not self.last_tool_calls:
            # Don't fire "no_tools" when the checkpointer has just appended a
            # new user message — the agent hasn't had a chance to think about
            # it yet, so terminating here would skip the model call entirely
            # and return a stale response.
            last = self.messages[-1] if self.messages else None
            if last is None or last.role.value != "user":
                return True, "no_tools"

        return False, None

    def _has_assistant_message(self) -> bool:
        """Check if there's at least one assistant message."""
        return any(m.role.value == "assistant" for m in self.messages)

    @property
    def total_tokens(self) -> int:
        """Total tokens used. Returns real count if tracked, else char/4 estimate."""
        if self.total_tokens_used > 0:
            return self.total_tokens_used
        # Fallback: rough estimate at 4 chars per token, images counted as images
        return sum(
            estimate_tokens(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls) // 4
            for m in self.messages
        )

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize state for checkpointing."""
        return self.model_dump(mode="json")

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> AgentState:
        """Restore state from checkpoint."""
        return cls.model_validate(data)
