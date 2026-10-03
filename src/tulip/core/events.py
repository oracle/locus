# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Event types for streaming and hooks - 100% Pydantic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializeAsAny

from tulip.core.messages import ToolCall


class TulipEvent(BaseModel):
    """Base class for all Tulip events."""

    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    #: Which agent produced this event, from ``AgentConfig.name`` (falling back
    #: to ``agent_id``). Stamped once as the event leaves the loop, so a caller
    #: consuming a merged stream can tell the researcher's tool call from the
    #: writer's without threading identity through every call site.
    #:
    #: ``None`` when the agent was never named — attribution is worth having
    #: and not worth inventing, and a positional label like ``"agent-3"`` would
    #: be stable only until someone reorders the list.
    #:
    #: The innermost agent wins: a nested run's events arrive already stamped
    #: and are never relabelled by the orchestrator around them, which is the
    #: whole point of attributing output to the specialist that produced it.
    agent_name: str | None = None

    model_config = {"frozen": True}


# =============================================================================
# Loop Events
# =============================================================================


class ThinkEvent(TulipEvent):
    """Agent produced reasoning and/or tool calls.

    ``reasoning`` carries the assistant's ordinary prose for the turn — the
    text you show a user — not hidden chain-of-thought. It is the streaming
    path's answer to "where is the assistant's text?" (#165): interim text
    arrives here, the final answer on ``TerminateEvent.final_message``, and
    token-by-token deltas on ``ModelChunkEvent`` (opt-in via
    ``stream_tokens=True``).
    """

    event_type: Literal["think"] = "think"
    iteration: int
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def content(self) -> str | None:
        """The assistant's text for this turn — alias for ``reasoning``.

        ``content`` is the first guess from outside; ``reasoning`` reads like
        chain-of-thought but is actually the visible prose. Both names work.
        """
        return self.reasoning


class ToolStartEvent(TulipEvent):
    """Tool execution started."""

    event_type: Literal["tool_start"] = "tool_start"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]


class ToolCompleteEvent(TulipEvent):
    """Tool execution completed."""

    event_type: Literal["tool_complete"] = "tool_complete"
    tool_name: str
    tool_call_id: str
    result: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    #: Machine-readable output the tool returned alongside its text — an MCP
    #: server's ``structuredContent``, or ``ToolOutput.structured_content``
    #: from a local tool. The model reads ``result``; a UI that renders a
    #: widget from the data reads this. Present on failures too, so an error
    #: payload is not lost. ``None`` when the tool returned plain text.
    structured_content: dict[str, Any] | None = None
    #: Non-text content the tool returned (images, audio, embedded
    #: resources, resource links), each a JSON-mode dict in the tool
    #: protocol's own shape (for MCP: ``{"type": "image", "data": ...,
    #: "mimeType": ...}``). ``None`` when there was none.
    content_blocks: list[dict[str, Any]] | None = None

    @property
    def success(self) -> bool:
        """Whether the tool execution succeeded."""
        return self.error is None


class ToolProgressEvent(TulipEvent):
    """A running tool reported progress.

    Emitted between the tool's :class:`ToolStartEvent` and its
    :class:`ToolCompleteEvent` — for an MCP tool, once per
    ``notifications/progress`` the server sends. Only tools that declare
    ``emits_progress=True`` (every MCP tool does) are streamed live; a tool
    calls :func:`tulip.tools.context.report_progress` to send one.
    """

    event_type: Literal["tool_progress"] = "tool_progress"
    tool_name: str
    tool_call_id: str
    progress: float
    total: float | None = None
    message: str | None = None


class SubagentEvent(TulipEvent):
    """An event from a subagent, delivered live on its parent's stream.

    A delegating tool (:func:`tulip.agent.tasks.task_tool`, or any tool that
    calls :func:`tulip.agent.subagent.run_subagent`) runs a whole child loop
    inside one tool call. Without this the parent's stream goes quiet for
    the length of that call, and a front end shows a frozen agent over a busy
    one.

    The child's event is wrapped, not passed through: a child's bare
    :class:`TerminateEvent` would read to every consumer of the parent's
    stream as the *parent* finishing, and its tool events would be counted as
    the parent's. A consumer that does not know this type skips it.

    Emitted between the delegating call's :class:`ToolStartEvent` and its
    :class:`ToolCompleteEvent`, for a tool declared with
    ``emits_progress=True``. A grandchild's events arrive as a
    ``SubagentEvent`` wrapping a ``SubagentEvent``.
    """

    event_type: Literal["subagent"] = "subagent"
    #: The parent's tool call that is running the child.
    tool_call_id: str
    tool_name: str
    #: The child's resumable id, when it has one.
    task_id: str | None = None
    #: The child's own event, attributed to the child by ``agent_name``.
    event: SerializeAsAny[TulipEvent]


class ReflectEvent(TulipEvent):
    """Reflexion evaluation completed."""

    event_type: Literal["reflect"] = "reflect"
    iteration: int
    assessment: str  # "on_track", "stuck", "new_findings", "loop_detected"
    confidence_delta: float
    new_confidence: float
    guidance: str | None = None


class GroundingEvent(TulipEvent):
    """Grounding evaluation completed."""

    event_type: Literal["grounding"] = "grounding"
    score: float
    claims_evaluated: int
    ungrounded_claims: list[str] = Field(default_factory=list)
    requires_replan: bool = False


class FinalAnswerVerificationEvent(TulipEvent):
    """``AgentConfig.final_answer_verifier`` judged a final-answer draft.

    ``passed`` is the verdict. On a rejection ``feedback`` is what the
    verifier returned and ``replanning`` says whether the model is called
    again with it (False once ``max_replans`` are spent — the draft is then
    returned as the answer anyway). ``error`` is set when the verifier raised;
    the draft is then accepted (the verifier fails open) and ``passed`` is
    False. ``attempt`` is 0 for the first draft. ``replaced`` is True when
    ``final_answer_fallback`` replaced a draft that failed its last attempt.

    ``continuation`` is True when the verifier sent the model back to
    unfinished work (a :class:`~tulip.agent.completion.Continuation`) rather
    than rejecting an answer: the reply stays in the conversation and
    ``reason`` names why the run did not stop (``announced_step``,
    ``no_changes``, ``unchecked_edits``).
    """

    event_type: Literal["final_answer_verification"] = "final_answer_verification"
    passed: bool
    attempt: int
    replanning: bool = False
    feedback: str | None = None
    error: str | None = None
    #: The draft failed with no replan left and ``final_answer_fallback``
    #: replaced it: the run answers with the fallback text instead.
    replaced: bool = False
    #: The verifier asked the model to keep working rather than to rewrite
    #: its answer (see :class:`~tulip.agent.completion.Continuation`).
    continuation: bool = False
    #: Why, when the verifier gave a reason (a continuation always does).
    reason: str | None = None


class ModelRetryEvent(TulipEvent):
    """A model call failed transiently and the loop is about to retry it.

    Emitted once per retry, before the backoff sleep, so a long-running
    agent's UI can show "rate limited, retrying in 12 s" instead of going
    quiet. With ``stream_tokens=True`` it arrives live, between chunks;
    without it, after the call finally returns (or just before the error
    ``TerminateEvent`` when every retry failed). See
    :class:`~tulip.agent.config.ModelRetryConfig`.
    """

    event_type: Literal["model_retry"] = "model_retry"
    #: 1 for the first retry, 2 for the second, and so on.
    attempt: int
    #: Seconds the loop waits before the retry.
    delay_seconds: float
    #: ``FailoverReason`` value: ``rate_limit``, ``overloaded``,
    #: ``server_error`` or ``timeout``.
    reason: str
    status_code: int | None = None
    #: The failure, as ``"ExceptionType: message"``.
    error: str
    #: Whether the delay came from the provider's ``retry-after``.
    from_retry_after: bool = False


class CompactionEvent(TulipEvent):
    """The run's context was compacted to keep it inside the model's window.

    ``stage`` says what it took: ``"prune"`` cleared old tool outputs,
    ``"summarize"`` replaced older history with a model-written summary, and
    ``"truncate"`` dropped it without one. Token counts are estimates of the
    whole request (messages plus tool definitions). ``exhausted`` means the
    context could not be brought under ``threshold`` (or compaction was
    thrashing) and the run ends with ``context_exhausted``; ``detail`` says why.
    It doubles as the compact boundary: everything before it in the stream is
    no longer verbatim in the model's context.
    """

    event_type: Literal["compaction"] = "compaction"
    iteration: int
    stage: Literal["prune", "summarize", "truncate"]
    tokens_before: int
    tokens_after: int
    threshold: int
    context_window: int
    messages_before: int
    messages_after: int
    summary: str | None = None
    exhausted: bool = False
    detail: str | None = None


class TerminateEvent(TulipEvent):
    """Agent execution terminated.

    Not just a lifecycle signal: ``final_message`` is the payload — the
    agent's final answer on the streaming path, the counterpart of
    ``AgentResult.message`` (#165). Also readable as ``.content``.
    """

    event_type: Literal["terminate"] = "terminate"
    reason: (
        str  # "complete", "max_iterations", "confidence_met", "terminal_tool", "tool_loop", "error"
    )
    iterations_used: int
    final_confidence: float
    total_tool_calls: int
    final_message: str | None = None  # Final assistant message content
    # Cumulative token usage for the run segment that ended here, read off the
    # AgentState counters (prompt/completion/total, plus
    # cache_read_input_tokens / cache_creation_input_tokens when the provider
    # reported cache activity Anthropic's way, outside ``prompt_tokens``, and
    # cached_tokens / cache_write_tokens when it reported it OpenAI's way,
    # inside them). None when the model reported no usage — consumers must
    # treat absence as "unmetered", not 0.
    usage: dict[str, int] | None = None
    # What the segment cost in USD, from the model's metadata prices. None
    # when the model is unpriced: a stream consumer cannot tell "free" from
    # "unknown" otherwise, and a supervisor enforcing its own spend limit
    # needs the number the loop already computed rather than a second table.
    cost_usd: float | None = None
    # What the provider itself reported the segment's calls cost (OpenRouter's
    # ``usage.cost``), delegated subagents included. None when no call
    # reported one. Where both are set this is the bill and ``cost_usd`` the
    # list-price estimate, which ignores prompt caching.
    reported_cost_usd: float | None = None
    # Why the run failed, when ``reason == "error"``. The loop yields this
    # event and then re-raises, so a consumer that stops at the event (a
    # stream-json writer, a socket front end) otherwise has no message to show.
    error: str | None = None

    @property
    def content(self) -> str | None:
        """The agent's final answer — alias for ``final_message``."""
        return self.final_message


class InterruptEvent(TulipEvent):
    """Agent paused for user input.

    When a tool calls interrupt() (e.g., ask_user), the agent yields this
    event and pauses. The caller should present the question to the user
    and call agent.resume(response) to continue.
    """

    event_type: Literal["interrupt"] = "interrupt"
    question: str
    options: list[str] | None = None
    #: Structured input request: a list of {name, label, type, placeholder,
    #: required} dicts. A question that needs SEVERAL answers ("payment id,
    #: amount, reason") declares them here so a console can render a form
    #: instead of a free-text box. None keeps the plain-question shape.
    fields: list[dict[str, Any]] | None = None
    interrupt_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustomEvent(TulipEvent):
    """An application-defined event emitted from a hook for the UI only.

    A hook that wants to tell the client something the model must not see
    (render a hotel carousel, show a progress chip) calls
    ``event.emit(CustomEvent(name="hotel_list", data={...}))`` on the hook
    event it was handed. The runtime yields it from ``Agent.run()`` right
    after the hook returns, in order with the surrounding tool events. It is
    never added to ``state.messages``, never shown to the model, and never
    checkpointed — it exists only on the event stream.
    """

    event_type: Literal["custom"] = "custom"
    #: Application-chosen discriminator (``"hotel_list"``, ``"progress"``).
    name: str
    #: JSON-serialisable payload. Kept a plain dict so every transport
    #: (SSE, websockets, A2A) can ship it with ``model_dump(mode="json")``.
    data: dict[str, Any] = Field(default_factory=dict)
    #: Filled by the runtime from the hook event's run context, so a consumer
    #: multiplexing several runs can route the event without bookkeeping.
    run_id: str | None = None
    thread_id: str | None = None
    #: The tool call the emitting hook was handling, when there was one.
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Read-only identity of the run a hook is observing.

    Attached to every hook event as ``event.run`` (``None`` only when a hook
    event is constructed outside a run, e.g. in a unit test). A single Agent
    instance serves many concurrent runs; this is how a hook tells them apart
    without reaching for globals or context variables.

    Attributes:
        run_id: ``AgentState.run_id`` of the run (a new id per user turn;
            a resumed run keeps the id of the turn it continues).
        thread_id: The conversation the run belongs to (``None`` when the
            caller passed none).
        metadata: The run's invocation metadata, exactly what tools see as
            ``ctx.invocation_metadata``. A read-only mapping.
        agent_name: ``AgentConfig.name`` (or ``agent_id``), when set.
    """

    run_id: str
    thread_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    agent_name: str | None = None

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        thread_id: str | None,
        metadata: Mapping[str, Any] | None,
        agent_name: str | None = None,
    ) -> RunInfo:
        """Construct with ``metadata`` frozen into a read-only snapshot."""
        return cls(
            run_id=run_id,
            thread_id=thread_id,
            metadata=MappingProxyType(dict(metadata or {})),
            agent_name=agent_name,
        )


# =============================================================================
# Model Events
# =============================================================================


class ModelChunkEvent(TulipEvent):
    """Streaming chunk from model.

    **Fires only with** ``agent.run(prompt, stream_tokens=True)`` — without
    the flag no chunk events arrive and nothing says why (#164). Assistant
    text still arrives without it, batched per turn: interim prose on
    ``ThinkEvent.reasoning``/``.content``, the final answer on
    ``TerminateEvent.final_message``/``.content``.
    """

    event_type: Literal["model_chunk"] = "model_chunk"
    content: str | None = None
    # The model that is actually answering, as the provider names it in the
    # stream (OpenAI-compat: ``chunk.model``). Behind a router this is not a
    # constant — a fallback model can serve the turn while the primary
    # restarts — and a UI announcing "who am I talking to" needs the served
    # name, not the requested one. Best-effort: None when the transport does
    # not say.
    model: str | None = None
    # Chain-of-thought delta from reasoning models (Qwen/DeepSeek via
    # vLLM surface it as ``delta.reasoning_content``). Separate from
    # ``content`` so streaming consumers can render CoT distinctly or
    # accumulate it independently.
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    done: bool = False
    # Set on the terminal chunk. ``usage`` arrives only if the caller asked
    # for it (OpenAI: ``stream_options={"include_usage": True}``). Without
    # these a streaming consumer cannot meter a turn, and cannot tell a
    # natural stop from a ``length`` truncation — which on reasoning models
    # surfaces as an empty reply rather than an error.
    usage: dict[str, int] | None = None
    stop_reason: str | None = None
    # The provider's own figure for the call, in USD, on the terminal chunk
    # when it reports one (see ``ModelResponse.cost_usd``).
    cost_usd: float | None = None


class ModelCompleteEvent(TulipEvent):
    """Model completion finished."""

    event_type: Literal["model_complete"] = "model_complete"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    stop_reason: str | None = None


# =============================================================================
# Multi-Agent Events
# =============================================================================


class SpecialistStartEvent(TulipEvent):
    """Specialist agent started."""

    event_type: Literal["specialist_start"] = "specialist_start"
    specialist_id: str
    specialist_type: str
    task: str


class SpecialistCompleteEvent(TulipEvent):
    """Specialist agent completed."""

    event_type: Literal["specialist_complete"] = "specialist_complete"
    specialist_id: str
    specialist_type: str
    result: str | None = None
    confidence: float
    duration_ms: float


class OrchestratorDecisionEvent(TulipEvent):
    """Orchestrator made a routing decision."""

    event_type: Literal["orchestrator_decision"] = "orchestrator_decision"
    decision: str  # "invoke_specialist", "correlate", "summarize", "finalize"
    specialists_selected: list[str] = Field(default_factory=list)
    reasoning: str | None = None


# =============================================================================
# Causal Events
# =============================================================================


class CausalNodeEvent(TulipEvent):
    """Causal inference node identified."""

    event_type: Literal["causal_node"] = "causal_node"
    node_id: str
    label: str
    node_type: str  # "root_cause", "symptom", "intermediate"
    evidence: list[str] = Field(default_factory=list)


class CausalEdgeEvent(TulipEvent):
    """Causal relationship identified."""

    event_type: Literal["causal_edge"] = "causal_edge"
    source_id: str
    target_id: str
    relationship: str  # "causes", "correlates_with", "precedes"
    confidence: float


# =============================================================================
# Hook Events
# =============================================================================


class HookEvent(TulipEvent):
    """Base class for hook lifecycle events."""


class BeforeInvocationEvent(HookEvent):
    """Fired before agent invocation starts."""

    event_type: Literal["before_invocation"] = "before_invocation"
    prompt: str
    agent_id: str | None = None


class AfterInvocationEvent(HookEvent):
    """Fired after agent invocation completes."""

    event_type: Literal["after_invocation"] = "after_invocation"
    success: bool
    iterations: int
    confidence: float
    duration_ms: float


class BeforeToolCallEvent(HookEvent):
    """Fired before a tool is called."""

    event_type: Literal["before_tool_call"] = "before_tool_call"
    tool_name: str
    arguments: dict[str, Any]
    # Writable: hooks can modify arguments
    modified_arguments: dict[str, Any] | None = None


class AfterToolCallEvent(HookEvent):
    """Fired after a tool call completes."""

    event_type: Literal["after_tool_call"] = "after_tool_call"
    tool_name: str
    tool_call_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: float


# =============================================================================
# Type aliases
# =============================================================================

LoopEvent = (
    ThinkEvent
    | ToolStartEvent
    | ToolProgressEvent
    | SubagentEvent
    | ToolCompleteEvent
    | ReflectEvent
    | GroundingEvent
    | FinalAnswerVerificationEvent
    | ModelRetryEvent
    | CompactionEvent
    | TerminateEvent
)
AgentEvent = LoopEvent | SpecialistStartEvent | SpecialistCompleteEvent | OrchestratorDecisionEvent
AllEvents = (
    AgentEvent
    | ModelChunkEvent
    | ModelCompleteEvent
    | CausalNodeEvent
    | CausalEdgeEvent
    | HookEvent
)
