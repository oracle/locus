# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Agent runtime loop methods, extracted from ``Agent`` as a mixin.

Holds the ReAct loop body (``run``, ``_run_from_state``, and the
collection of model / tool / reflexion / grounding helpers it dispatches
to). Mixed into ``Agent`` via ``class Agent(AgentRuntimeMixin, BaseModel):``
so the public-facade class stays focused on construction, public
entry points, and properties.

Method bodies are byte-identical to their pre-extraction form on
``Agent`` — they reach into the same ``self.config`` and ``self._*``
private attributes. Pydantic v2 supports this kind of mixin as long as
the mixin doesn't declare its own fields.

Type-only forward references on the mixin tell mypy what attributes
``self`` is expected to carry at runtime, so the strict pass keeps
working without scattering ``# type: ignore`` annotations across the
moved methods.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tulip.agent.config import AgentConfig
from tulip.agent.result import StopReason
from tulip.core.events import (
    GroundingEvent,
    InterruptEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    TulipEvent,
)
from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.core.state import AgentState, ReasoningStep, ToolExecution
from tulip.models.base import ModelResponse
from tulip.tools.executor import ToolContextFactory, ToolExecutor
from tulip.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from tulip.agent.hook_orchestrator import HookOrchestrator
    from tulip.memory.conversation import ConversationManager
    from tulip.memory.manager import BaseMemoryManager
    from tulip.reasoning.grounding import GroundingEvaluator
    from tulip.reasoning.reflexion import Reflector


def _bus_bridge(
    fn: Callable[..., AsyncIterator[TulipEvent]],
) -> Callable[..., AsyncIterator[TulipEvent]]:
    """Decorate an ``async def run(...)`` style generator so each yielded
    :class:`TulipEvent` is also published on the SSE bus.

    No-op when no ``run_context`` is active — the underlying ``emit()``
    short-circuits on the contextvar. Failures inside the bridge are
    swallowed by ``emit()``; the original event is always passed
    through to the caller, so the bridge cannot break agent execution.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[TulipEvent]:
        # Local import — no cost when telemetry is unused.
        from tulip.observability.agent_bridge import (  # noqa: PLC0415
            bridge_tulip_event,
        )

        async for event in fn(*args, **kwargs):
            try:
                await bridge_tulip_event(event)
            except Exception:  # noqa: BLE001 — telemetry never breaks the loop
                pass
            yield event

    return wrapper


# Suppress an unused-import warning from the type alias.
_ = Awaitable  # noqa: SLF001 — placeholder so mypy knows we imported it intentionally


def _normalize_stop_reason(raw: str | None) -> StopReason:
    """Map a free-form ``TerminateEvent.reason`` to the ``StopReason`` Literal.

    Lifted alongside the runtime methods so the mixin stays
    self-contained. The original copy on ``tulip.agent.agent`` is
    re-exported from this module for back-compat with any external
    importer.
    """
    valid: frozenset[str] = frozenset(
        {
            "complete",
            "terminal_tool",
            "confidence_met",
            "max_iterations",
            "tool_loop",
            "no_tools",
            "grounding_failed",
            "token_budget",
            "time_budget",
            "interrupted",
            "error",
            "cancelled",
        }
    )
    if not raw:
        return "complete"
    if raw in valid:
        return raw  # type: ignore[return-value]
    if "tool_called:" in raw:
        return "terminal_tool"
    if "text_mention:" in raw:
        return "complete"
    for known in valid:
        if known in raw:
            return known  # type: ignore[return-value]
    return "complete"


class AgentRuntimeMixin:
    """Mixin holding the ReAct loop body and reasoning helpers.

    Mixed into ``Agent`` so the public-facade class stays small. The
    type annotations below describe the agent attributes that the
    mixin's methods read at runtime — they are *not* declared by the
    mixin itself. ``Agent``'s ``PrivateAttr`` declarations remain the
    source of truth for runtime construction.
    """

    if TYPE_CHECKING:
        # These attributes live on the concrete ``Agent`` instance the
        # mixin is mixed into. Declaring them here gives mypy enough
        # information to type-check the moved method bodies without
        # creating runtime fields.
        config: AgentConfig
        _model: Any
        _tool_registry: ToolRegistry
        _executor: ToolExecutor
        _hooks: list[Any]
        _hook_orchestrator: HookOrchestrator | None
        _conversation_manager: ConversationManager | None
        _memory_manager: BaseMemoryManager | None
        _reflector: Reflector | None
        _grounding_evaluator: GroundingEvaluator | None
        _grounding_model: Any
        _auxiliary_model: Any
        _last_run_state: AgentState | None
        _interrupt_state: AgentState | None
        _interrupt_prompt: str | None
        _has_unverified_writes: bool
        _interrupt_thread_id: str | None
        _interrupt_metadata: dict[str, Any] | None
        _cancel_signal: threading.Event | None
        _initialized: bool

        @property
        def is_cancelled(self) -> bool: ...

        def _initialize(self) -> None: ...

    @_bus_bridge
    async def run(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TulipEvent]:
        """
        Run the agent with streaming events.

        Args:
            prompt: User prompt to process
            thread_id: Optional thread ID for checkpointing
            metadata: Additional metadata for tools

        Yields:
            TulipEvent instances for each step
        """
        self._initialize()

        # Create initial state
        state = await self._create_initial_state(prompt, thread_id, metadata)

        # Track metrics
        started_at = datetime.now(UTC)
        _total_tokens = 0
        _tool_calls_count = 0
        _tool_errors_count = 0
        _reflexion_evals = 0
        _grounding_evals = 0
        _last_assistant_content: str | None = None
        _last_no_tool_calls = False

        # Reset any user-supplied composable termination condition so
        # time-windowed checks (TimeLimit) start their clock at run start.
        if self.config.termination is not None:
            self.config.termination.reset()

        # Run hooks: before_invocation
        state = await self._run_before_invocation_hooks(prompt, state)

        # Inject long-term memories into the system prompt.
        if self._memory_manager is not None:
            state = await self._memory_manager.on_session_start(state)

        try:
            # Main ReAct loop
            while True:
                # Check time budget
                if self.config.time_budget_seconds is not None:
                    elapsed = (datetime.now(UTC) - started_at).total_seconds()
                    if elapsed >= self.config.time_budget_seconds:
                        yield TerminateEvent(
                            reason="time_budget",
                            iterations_used=state.iteration,
                            final_confidence=state.confidence,
                            total_tool_calls=len(state.tool_executions),
                            final_message=_last_assistant_content,
                        )
                        break

                # Check external cancellation
                if self.is_cancelled:
                    yield TerminateEvent(
                        reason="cancelled",
                        iterations_used=state.iteration,
                        final_confidence=state.confidence,
                        total_tool_calls=_tool_calls_count,
                        final_message="Agent cancelled by external signal.",
                    )
                    break

                # User-supplied composable termination condition runs first
                # so MaxIterations(...) | TextMention("DONE") and friends
                # actually fire before the hard-coded fallbacks.
                if self.config.termination is not None:
                    user_stop, user_reason = self.config.termination.check(
                        state,
                        last_message=_last_assistant_content or "",
                        no_tool_calls=_last_no_tool_calls,
                    )
                    if user_stop:
                        yield TerminateEvent(
                            reason=user_reason or "complete",
                            iterations_used=state.iteration,
                            final_confidence=state.confidence,
                            total_tool_calls=len(state.tool_executions),
                            final_message=_last_assistant_content,
                        )
                        break

                # Check termination conditions
                should_stop, stop_reason = state.should_terminate
                if should_stop and stop_reason:
                    if stop_reason == "max_iterations" and state.iteration > 0:
                        # Inject summary request and do one final call WITHOUT tools
                        state = state.with_message(
                            Message.system(
                                "[Iteration Limit Reached]\n"
                                "You have used all available iterations. "
                                "Provide a final summary of your findings and conclusions "
                                "based on the work done so far. Do NOT call any more tools."
                            )
                        )
                        # Call model without tool schemas to force text response.
                        # Use the auxiliary (cheap) model when configured —
                        # this is just a final summary, no need to spend
                        # primary-model budget.
                        messages = list(state.messages)
                        if self._conversation_manager:
                            if hasattr(self._conversation_manager, "async_apply"):
                                messages = await self._conversation_manager.async_apply(messages)
                            else:
                                messages = self._conversation_manager.apply(messages)
                        messages = self._validate_messages(messages)

                        summary_model = self._auxiliary_model or self._model
                        response = await summary_model.complete(
                            messages=messages,
                            tools=None,  # No tools — force text summary
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )
                        prompt_toks = response.usage.get("prompt_tokens", 0)
                        completion_toks = response.usage.get("completion_tokens", 0)
                        cache_creation_toks = response.usage.get("cache_creation_input_tokens", 0)
                        cache_read_toks = response.usage.get("cache_read_input_tokens", 0)
                        _total_tokens += prompt_toks + completion_toks
                        state = state.with_token_usage(
                            prompt_toks,
                            completion_toks,
                            cache_creation_tokens=cache_creation_toks,
                            cache_read_tokens=cache_read_toks,
                        )

                        summary = (
                            response.message.content
                            or _last_assistant_content
                            or self._build_fallback_summary(state)
                        )
                        yield TerminateEvent(
                            reason="max_iterations",
                            iterations_used=state.iteration,
                            final_confidence=state.confidence,
                            total_tool_calls=len(state.tool_executions),
                            final_message=summary,
                        )
                        break

                    # All other stop reasons: hard stop
                    yield TerminateEvent(
                        reason=stop_reason,
                        iterations_used=state.iteration,
                        final_confidence=state.confidence,
                        total_tool_calls=len(state.tool_executions),
                        final_message=_last_assistant_content,
                    )
                    break

                # Increment iteration
                state = state.next_iteration()

                # Planning: inject plan prompt on first iteration
                if self.config.planning and state.iteration == 1:
                    state = state.with_message(
                        Message.system(
                            "[Planning Phase]\n"
                            "Before taking any action, create a step-by-step plan.\n"
                            "Format your plan as a numbered list:\n"
                            "1. First step\n"
                            "2. Second step\n"
                            "...\n\n"
                            "After stating your plan, begin executing step 1.\n"
                            "Do NOT call tools without a plan."
                        )
                    )

                # Budget warning in explicit mode — nudge model to complete
                if self.config.completion_mode == "explicit":
                    remaining = self.config.max_iterations - state.iteration
                    if remaining == 2:
                        state = state.with_message(
                            Message.system(
                                f"[Budget Warning] You have {remaining} iterations left. "
                                "Start wrapping up. Call task_complete(summary='your findings') "
                                "to finish, or you'll hit the iteration limit."
                            )
                        )
                    elif remaining == 0:
                        state = state.with_message(
                            Message.system(
                                "[Final Iteration] This is your LAST iteration. "
                                "You MUST call task_complete now with a summary of everything "
                                "you've found. Do NOT call any other tools."
                            )
                        )

                # Get model response
                response, state = await self._get_model_response(state)
                prompt_toks = response.usage.get("prompt_tokens", 0)
                completion_toks = response.usage.get("completion_tokens", 0)
                cache_creation_toks = response.usage.get("cache_creation_input_tokens", 0)
                cache_read_toks = response.usage.get("cache_read_input_tokens", 0)
                _total_tokens += prompt_toks + completion_toks
                state = state.with_token_usage(
                    prompt_toks,
                    completion_toks,
                    cache_creation_tokens=cache_creation_toks,
                    cache_read_tokens=cache_read_toks,
                )
                _last_assistant_content = response.message.content
                # Track for the user-supplied termination condition. Updated again
                # below if a Cohere-style text tool call is parsed out of the body.
                _last_no_tool_calls = not response.message.tool_calls

                # Store plan from first iteration if planning enabled
                if self.config.planning and state.iteration == 1 and response.message.content:
                    state = state.with_metadata("plan", response.message.content)

                # Emit think event
                yield ThinkEvent(
                    iteration=state.iteration,
                    reasoning=response.message.content,
                    tool_calls=list(response.message.tool_calls),
                )

                # If no structured tool calls, try parsing from text (Cohere fallback)
                if not response.message.tool_calls and response.message.content:
                    parsed_calls = self._parse_text_tool_calls(response.message.content)
                    if parsed_calls:
                        response = ModelResponse(
                            message=Message(
                                role=response.message.role,
                                content=response.message.content,
                                tool_calls=parsed_calls,
                                tool_call_id=response.message.tool_call_id,
                                name=response.message.name,
                            ),
                            usage=response.usage,
                            stop_reason=response.stop_reason,
                        )
                        # Update the assistant message in state with parsed tool calls
                        messages = list(state.messages)
                        messages[-1] = response.message
                        state = state.model_copy(update={"messages": tuple(messages)})
                        _last_no_tool_calls = False

                # If still no tool calls — in auto mode we're done, in explicit mode we continue
                if not response.message.tool_calls and self.config.completion_mode != "explicit":
                    # Apply grounding before final response if enabled
                    if (
                        self.config.grounding
                        and self.config.grounding.enabled
                        and self.config.grounding.check_before_final
                        and self._grounding_evaluator
                        and response.message.content
                        and len(state.tool_executions) > 0
                    ):
                        grounding_event, state = await self._apply_grounding(
                            state, response.message.content
                        )
                        _grounding_evals += 1
                        yield grounding_event

                        # If grounding fails, inject guidance and continue loop
                        if grounding_event.requires_replan and _grounding_evals <= (
                            self.config.grounding.max_replans
                        ):
                            from tulip.reasoning.grounding import GroundingResult

                            replan_guidance = self._grounding_evaluator.get_replan_guidance(
                                GroundingResult(
                                    score=grounding_event.score,
                                    ungrounded_claims=grounding_event.ungrounded_claims,
                                    requires_replan=True,
                                )
                            )
                            state = state.with_message(
                                Message.system(f"[Grounding Check Failed]\n{replan_guidance}")
                            )
                            continue  # Re-enter loop for replanning

                    # Empty-content safety net (fixes #280).
                    #
                    # The model can return zero tool_calls AND zero
                    # ``content`` for several reasons that all look the
                    # same to the runtime:
                    #
                    #   - Reasoning-only iteration (gpt-5.x / o-series /
                    #     Gemini 2.5 thinking mode generate
                    #     ``completion_tokens`` that land in a separate
                    #     reasoning channel — ``message.content`` is
                    #     None even though tokens were consumed).
                    #   - The model decided "I'm done" without writing
                    #     anything after a long context (observed on
                    #     Gemini with > 50K-token system prompts).
                    #   - The model returned an empty assistant message
                    #     between tool calls and the runtime collapses
                    #     to this branch on the trailing iteration.
                    #
                    # Without the safety net, ``TerminateEvent.final_message``
                    # becomes None → ``AgentResult.message`` becomes ""
                    # → callers see empty output despite real work done.
                    # Mirror the MaxIterations summary-injection path:
                    # force one no-tools completion to extract the
                    # model's actual answer based on the conversation
                    # so far. Costs ~one extra call only when the bug
                    # shape would otherwise produce empty output.
                    final_content = response.message.content
                    if not final_content:
                        state = state.with_message(
                            Message.system(
                                "[Final answer requested]\n"
                                "Your previous turn produced no visible response. "
                                "Provide your final answer to the user's question "
                                "based on the conversation so far. Do NOT call any "
                                "more tools — write the answer as plain text."
                            )
                        )
                        messages = list(state.messages)
                        if self._conversation_manager:
                            if hasattr(self._conversation_manager, "async_apply"):
                                messages = await self._conversation_manager.async_apply(messages)
                            else:
                                messages = self._conversation_manager.apply(messages)
                        messages = self._validate_messages(messages)

                        summary_model = self._auxiliary_model or self._model
                        try:
                            summary_resp = await summary_model.complete(
                                messages=messages,
                                tools=None,  # No tools — force text
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                            )
                            s_prompt_toks = summary_resp.usage.get("prompt_tokens", 0)
                            s_completion_toks = summary_resp.usage.get("completion_tokens", 0)
                            s_cc_toks = summary_resp.usage.get("cache_creation_input_tokens", 0)
                            s_cr_toks = summary_resp.usage.get("cache_read_input_tokens", 0)
                            _total_tokens += s_prompt_toks + s_completion_toks
                            state = state.with_token_usage(
                                s_prompt_toks,
                                s_completion_toks,
                                cache_creation_tokens=s_cc_toks,
                                cache_read_tokens=s_cr_toks,
                            )
                            final_content = (
                                summary_resp.message.content
                                or _last_assistant_content
                                or self._build_fallback_summary(state)
                            )
                        except Exception:  # noqa: BLE001
                            # Summary call failed — fall back to the
                            # last assistant content we saw or a
                            # deterministic stub. Better than empty.
                            final_content = _last_assistant_content or self._build_fallback_summary(
                                state
                            )

                    yield TerminateEvent(
                        reason="complete",
                        iterations_used=state.iteration,
                        final_confidence=state.confidence,
                        total_tool_calls=len(state.tool_executions),
                        final_message=final_content,
                    )
                    break

                # Execute tool calls.
                #
                # Three phases so ``tool_execution="concurrent"`` is real
                # (#210): per-call serial hook/cache work is hoisted out
                # of the executor call so the survivors can run in a
                # single ``asyncio.gather``. Without this split, the
                # earlier per-call loop fed the executor singletons and
                # ``ConcurrentExecutor`` collapsed to ``SequentialExecutor``.
                #
                #   Phase 1 — per call, serial: emit ToolStartEvent, run
                #     before-hooks, resolve cancel/idempotent-cache short-
                #     circuits (recorded on state immediately so a later
                #     same-args call in this batch still cache-hits).
                #   Phase 2 — one batched ``executor.execute(...)`` over
                #     the surviving calls.
                #   Phase 3 — per result, in tool_call order: interrupt
                #     detection, result truncation/offload, state update,
                #     ToolCompleteEvent, after-hook (with retry), write/
                #     verification tracking.
                tool_results: list[ToolResult] = []
                reasoning_step_tools: list[ToolExecution] = []

                # Phase 1 — pre-execute.
                slots: list[dict[str, Any]] = []
                to_execute_indices: list[int] = []
                to_execute_calls: list[ToolCall] = []
                # Within-batch idempotent dedup: when an ``@tool(idempotent=True)``
                # appears twice in the same batch with the same arguments, the
                # second slot becomes a ``batch_cache_ref`` pointing at the first
                # slot. Phase 3 copies the first slot's result back. Without
                # this, the README contract ("idempotent body fires once per
                # run") is silently violated whenever a model fans out
                # duplicates in one response — see #210 follow-up.
                batch_idempotent_seen: dict[tuple[str, str], int] = {}

                for tool_call in response.message.tool_calls:
                    _tool_calls_count += 1

                    yield ToolStartEvent(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        arguments=tool_call.arguments,
                    )

                    tool_event = await self._run_before_tool_hooks(
                        tool_call.name, tool_call.id, tool_call.arguments
                    )

                    if tool_event.cancel:
                        cancel_msg = (
                            tool_event.cancel
                            if isinstance(tool_event.cancel, str)
                            else "Cancelled by hook"
                        )
                        cancel_result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=cancel_msg,
                            error=None,
                            duration_ms=0.0,
                        )
                        cancel_execution = ToolExecution(
                            tool_name=cancel_result.name,
                            tool_call_id=cancel_result.tool_call_id,
                            arguments=tool_call.arguments,
                            result=cancel_result.content,
                        )
                        # NOTE: state is updated in Phase 3 (tool_call order),
                        # not here, so concurrent batches keep deterministic
                        # ordering on ``state.tool_executions``.
                        slots.append(
                            {
                                "tool_call": tool_call,
                                "arguments": tool_call.arguments,
                                "kind": "cancel",
                                "result": cancel_result,
                                "execution": cancel_execution,
                            }
                        )
                        continue

                    modified_args = tool_event.arguments

                    # Idempotent dedup: if the tool declared idempotent=True
                    # and a prior call in this run used the same arguments,
                    # reuse the prior result instead of invoking the body.
                    # Without this, ``@tool(idempotent=True)`` is silently a
                    # no-op for the main Agent.run() path (despite being
                    # advertised on the README hero example).
                    cached = self._maybe_cached_idempotent_result(
                        state, tool_call.name, modified_args, tool_call.id
                    )
                    if cached is not None:
                        cache_execution = ToolExecution(
                            tool_name=cached.name,
                            tool_call_id=cached.tool_call_id,
                            arguments=modified_args,
                            result=cached.content if cached.success else None,
                            error=cached.error,
                            duration_ms=cached.duration_ms,
                            idempotent_cache_hit=True,
                        )
                        slots.append(
                            {
                                "tool_call": tool_call,
                                "arguments": modified_args,
                                "kind": "cache",
                                "result": cached,
                                "execution": cache_execution,
                            }
                        )
                        continue

                    # Within-batch dedup for idempotent tools whose duplicate
                    # appears alongside the original in this same response.
                    # Without this, a model emitting ``[submit(X), submit(X)]``
                    # in one assistant message would fire the body twice — a
                    # regression vs the pre-batching per-call loop, which
                    # sneaked the dedup in via serial state updates.
                    idempotent_key = self._idempotent_batch_key(tool_call.name, modified_args)
                    if idempotent_key is not None:
                        prior_slot = batch_idempotent_seen.get(idempotent_key)
                        if prior_slot is not None:
                            slots.append(
                                {
                                    "tool_call": tool_call,
                                    "arguments": modified_args,
                                    "kind": "batch_cache_ref",
                                    "ref_slot": prior_slot,
                                    "result": None,
                                    "execution": None,
                                }
                            )
                            continue
                        batch_idempotent_seen[idempotent_key] = len(slots)

                    slots.append(
                        {
                            "tool_call": tool_call,
                            "arguments": modified_args,
                            "kind": "execute",
                            "result": None,
                            "execution": None,
                        }
                    )
                    to_execute_indices.append(len(slots) - 1)
                    to_execute_calls.append(
                        tool_call.model_copy(update={"arguments": modified_args})
                    )

                # Phase 2 — stream the survivors through the executor.
                # Constructed after Phase 1 so ``state`` already reflects
                # cancel / cache short-circuits, matching the pre-extraction
                # per-call-loop semantics.
                #
                # Using ``execute_streaming`` (yields ``(input_idx, result)``
                # in completion order for ConcurrentExecutor, input order for
                # SequentialExecutor) gives us two things in one pass:
                #   * Opt-in completion-order ``ToolCompleteEvent`` for UI
                #     streaming (``AgentConfig.tool_event_order='completion'``).
                #   * Active sibling cancellation on interrupt: a ``break``
                #     out of the ``async for`` triggers the executor's
                #     ``finally`` clause, which cancels in-flight sibling
                #     tasks. The pre-streaming ``gather`` impl had no such
                #     hook — every sibling completed and only the fold was
                #     halted.
                if to_execute_calls:
                    ctx_factory = ToolContextFactory(
                        run_id=state.run_id,
                        agent_id=state.agent_id,
                        iteration=state.iteration,
                        state=state,
                        invocation_metadata=metadata or {},
                    )
                    batch_start = time.perf_counter()
                    interrupted_slot_idx: int | None = None
                    try:
                        async for input_idx, batched_result in self._executor.execute_streaming(
                            to_execute_calls,
                            self._tool_registry,
                            ctx_factory,
                        ):
                            slot_idx = to_execute_indices[input_idx]
                            slots[slot_idx]["result"] = batched_result

                            if self.config.tool_event_order == "completion":
                                # Stream the complete event the moment the
                                # tool finishes. Phase 3 will skip its own
                                # ToolCompleteEvent emission for execute-kind
                                # slots when in completion mode.
                                yield ToolCompleteEvent(
                                    tool_name=batched_result.name,
                                    tool_call_id=batched_result.tool_call_id,
                                    result=(
                                        batched_result.content if batched_result.success else None
                                    ),
                                    error=batched_result.error,
                                    duration_ms=batched_result.duration_ms,
                                )

                            # Interrupt detection — break to trigger the
                            # executor's cancellation of in-flight siblings.
                            # Phase 3 will run the InterruptEvent emission
                            # path on the interrupting slot exactly like the
                            # pre-streaming code did.
                            if (
                                batched_result.content
                                and '"__interrupt__": true' in batched_result.content
                            ):
                                interrupted_slot_idx = slot_idx
                                break
                    except Exception as e:  # noqa: BLE001 — ``_execute_one`` already traps user-tool exceptions; this catches a failure of the executor itself (registry blow-up, ctx_factory error) so a single bad call doesn't abort the whole turn.
                        batch_duration = (time.perf_counter() - batch_start) * 1000
                        for slot_idx, tc in zip(to_execute_indices, to_execute_calls, strict=True):
                            if slots[slot_idx]["result"] is None:
                                slots[slot_idx]["result"] = ToolResult(
                                    tool_call_id=tc.id,
                                    name=tc.name,
                                    content="",
                                    error=str(e),
                                    duration_ms=batch_duration,
                                )

                    # Synthesize "cancelled by sibling interrupt" results
                    # for slots whose tasks were cancelled (the interrupt
                    # break happened before they completed). Folded as
                    # ``kind="cancel"`` so Phase 3 treats them identically
                    # to hook-cancelled calls — Start/Complete pair emits,
                    # state records the cancellation, no after-hook fires.
                    if interrupted_slot_idx is not None:
                        for slot_idx in to_execute_indices:
                            slot = slots[slot_idx]
                            if slot["result"] is not None:
                                continue
                            tc = slot["tool_call"]
                            cancelled_result = ToolResult(
                                tool_call_id=tc.id,
                                name=tc.name,
                                content="Cancelled: sibling tool requested interrupt",
                                error=None,
                                duration_ms=0.0,
                            )
                            cancelled_execution = ToolExecution(
                                tool_name=tc.name,
                                tool_call_id=tc.id,
                                arguments=slot["arguments"],
                                result=cancelled_result.content,
                            )
                            slot["result"] = cancelled_result
                            slot["execution"] = cancelled_execution
                            slot["kind"] = "cancel"

                # Phase 3 — per-result fold, in tool_call order.
                for slot in slots:
                    tool_call = slot["tool_call"]
                    modified_args = slot["arguments"]
                    kind = slot["kind"]
                    result = slot["result"]

                    if kind == "cancel":
                        tool_results.append(result)
                        state = state.with_tool_execution(slot["execution"])
                        reasoning_step_tools.append(slot["execution"])
                        yield ToolCompleteEvent(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            result=result.content,
                            duration_ms=0.0,
                        )
                        continue
                    if kind == "cache":
                        tool_results.append(result)
                        state = state.with_tool_execution(slot["execution"])
                        reasoning_step_tools.append(slot["execution"])
                        yield ToolCompleteEvent(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            result=result.content,
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )
                        continue
                    if kind == "batch_cache_ref":
                        # Same-args duplicate of an earlier slot in this same
                        # batch. The referenced slot's result is now resolved
                        # (Phase 2 ran ``asyncio.gather`` to completion before
                        # this loop). Reuse it and rebrand the result/execution
                        # for this call's id.
                        ref_slot = slots[slot["ref_slot"]]
                        ref_result: ToolResult = ref_slot["result"]
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=ref_result.content,
                            error=ref_result.error,
                            duration_ms=0.0,
                        )
                        batch_cache_execution = ToolExecution(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            arguments=modified_args,
                            result=result.content if result.success else None,
                            error=result.error,
                            duration_ms=0.0,
                            idempotent_cache_hit=True,
                        )
                        tool_results.append(result)
                        state = state.with_tool_execution(batch_cache_execution)
                        reasoning_step_tools.append(batch_cache_execution)
                        yield ToolCompleteEvent(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            result=result.content,
                            error=result.error,
                            duration_ms=0.0,
                        )
                        continue

                    # Interrupt marker from ``ask_user``. Sibling calls in
                    # this batch have already executed in parallel — we
                    # still pause here, leaving them un-folded into state.
                    # An interrupt is rare and ``ask_user`` is the only
                    # tool that emits it; we trust users not to fan it
                    # out alongside high-cost siblings.
                    if result.content and '"__interrupt__": true' in result.content:
                        import json as _json

                        try:
                            interrupt_data = _json.loads(result.content)
                            if interrupt_data.get("__interrupt__"):
                                self._last_run_state = state
                                self._interrupt_state = state
                                self._interrupt_prompt = prompt
                                self._interrupt_thread_id = thread_id
                                self._interrupt_metadata = metadata
                                yield InterruptEvent(
                                    question=interrupt_data.get("question", ""),
                                    options=interrupt_data.get("options"),
                                    interrupt_id=result.tool_call_id,
                                )
                                return  # Pause the generator
                        except (ValueError, KeyError):
                            pass

                    # Cap oversized tool results so they don't blow the
                    # model's context window. When ``tool_result_store`` is
                    # configured we offload the full payload through it and
                    # inline a recoverable reference key; otherwise we fall
                    # back to lossy head-truncation.
                    if (
                        self.config.max_tool_result_length > 0
                        and result.content
                        and len(result.content) > self.config.max_tool_result_length
                    ):
                        if self.config.tool_result_store is not None:
                            result = self.config.tool_result_store.maybe_offload(
                                result,
                                run_id=state.run_id,
                                iteration=state.iteration,
                            )
                        else:
                            original_len = len(result.content)
                            result = ToolResult(
                                tool_call_id=result.tool_call_id,
                                name=result.name,
                                content=(
                                    result.content[: self.config.max_tool_result_length]
                                    + f"\n[OUTPUT TRUNCATED — original: {original_len} chars]"
                                ),
                                error=result.error,
                                duration_ms=result.duration_ms,
                            )

                    tool_results.append(result)

                    execution = ToolExecution(
                        tool_name=result.name,
                        tool_call_id=result.tool_call_id,
                        arguments=modified_args,
                        result=result.content if result.success else None,
                        error=result.error,
                        duration_ms=result.duration_ms,
                    )
                    state = state.with_tool_execution(execution)
                    reasoning_step_tools.append(execution)

                    if result.error:
                        _tool_errors_count += 1

                    # Skip the per-slot ToolCompleteEvent yield in
                    # completion mode — Phase 2 already streamed it the
                    # moment this tool finished. Sequential mode (default)
                    # still emits here so consumers see events in
                    # tool_call order.
                    if self.config.tool_event_order == "sequential":
                        yield ToolCompleteEvent(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            result=result.content if result.success else None,
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )

                    after_tool_event = await self._run_after_tool_hooks(
                        result.name,
                        result.content if result.success else None,
                        result.error,
                        tool_call_id=result.tool_call_id,
                        arguments=modified_args,
                    )

                    if after_tool_event.retry:
                        try:
                            retry_ctx_factory = ToolContextFactory(
                                run_id=state.run_id,
                                agent_id=state.agent_id,
                                iteration=state.iteration,
                                state=state,
                                invocation_metadata=metadata or {},
                            )
                            [result] = await self._executor.execute(
                                [tool_call.model_copy(update={"arguments": modified_args})],
                                self._tool_registry,
                                retry_ctx_factory,
                            )
                        except Exception as e:  # noqa: BLE001 — user tool bodies can raise anything; surface as ToolResult.error
                            result = ToolResult(
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                                content="",
                                error=str(e),
                                duration_ms=0.0,
                            )

                    if result.name in self.config.verify_tools:
                        self._has_unverified_writes = True
                    if result.name in self.config.verification_tools:
                        self._has_unverified_writes = False

                # Add tool results to messages
                for result in tool_results:
                    state = state.with_message(Message.tool(result))

                # Inject verification reminder if write-like tools were used
                if self.config.verify_tools:
                    tools_used = {e.tool_name for e in reasoning_step_tools}
                    wrote = tools_used & self.config.verify_tools
                    if wrote:
                        state = state.with_message(
                            Message.system(
                                "[Verification Reminder] You modified files/data. "
                                "Before completing, verify your changes:\n"
                                "- Run tests or checks if available\n"
                                "- Read back modified files to confirm correctness\n"
                                "- Fix any issues found\n"
                                "Do NOT call task_complete until verified."
                            )
                        )

                # Apply Reflexion if enabled
                if (
                    self.config.reflexion
                    and self.config.reflexion.enabled
                    and self._reflector
                    and state.iteration % self.config.reflexion.evaluate_every_n_iterations == 0
                ):
                    reflect_event, state = await self._apply_reflexion(state, reasoning_step_tools)
                    _reflexion_evals += 1
                    yield reflect_event

                    # Inject guidance when agent is stuck or looping
                    if self.config.reflexion.include_guidance and reflect_event.guidance:
                        guidance = f"[Agent Self-Reflection]\n{reflect_event.guidance}"
                        # Add replan suggestion if planning is enabled and agent is stuck
                        if self.config.planning and reflect_event.assessment in (
                            "stuck",
                            "loop_detected",
                        ):
                            guidance += (
                                "\n\n[Replan] Your current approach isn't working. "
                                "Create a NEW plan with a different strategy, then execute it."
                            )
                        state = state.with_message(Message.system(guidance))

                # Record reasoning step
                reasoning_step = ReasoningStep(
                    iteration=state.iteration,
                    thought=response.message.content,
                    tool_calls=list(response.message.tool_calls),
                    tool_results=reasoning_step_tools,
                    reflection=None,  # Will be updated if reflexion was applied
                    confidence_delta=0.0,
                )
                state = state.with_reasoning_step(reasoning_step)

                # Checkpoint if enabled
                if (
                    self.config.checkpointer
                    and self.config.checkpoint_every_n_iterations > 0
                    and state.iteration % self.config.checkpoint_every_n_iterations == 0
                ):
                    _cp_thread = thread_id or state.run_id
                    await self.config.checkpointer.save(
                        state,
                        _cp_thread,
                    )
                    from tulip.observability.emit import (  # noqa: PLC0415
                        EV_CHECKPOINT_SAVED,
                        emit,
                    )

                    await emit(
                        EV_CHECKPOINT_SAVED,
                        thread_id=_cp_thread,
                        iteration=state.iteration,
                        backend=type(self.config.checkpointer).__name__,
                        trigger="every_n_iterations",
                    )

        except Exception as e:
            # Emit error termination
            state = state.with_error(str(e))
            yield TerminateEvent(
                reason="error",
                iterations_used=state.iteration,
                final_confidence=state.confidence,
                total_tool_calls=len(state.tool_executions),
            )
            raise

        finally:
            # Clear cancel signal
            if self._cancel_signal is not None:
                self._cancel_signal.clear()

            # Save output to state if output_key configured
            if self.config.output_key:
                final_msg = ""
                for msg in reversed(state.messages):
                    if msg.role.value == "assistant" and msg.content:
                        final_msg = msg.content
                        break
                if final_msg:
                    state = state.with_metadata(self.config.output_key, final_msg)

            # Store final state for run_sync access
            self._last_run_state = state

            # Run hooks: after_invocation
            _duration_ms = (datetime.now(UTC) - started_at).total_seconds() * 1000  # noqa: F841
            await self._run_after_invocation_hooks(state, len(state.errors) == 0)

            # Extract and persist long-term memories from this session.
            if self._memory_manager is not None:
                await self._memory_manager.on_session_end(state)

            # Final checkpoint
            if self.config.checkpointer and thread_id:
                await self.config.checkpointer.save(state, thread_id)
                from tulip.observability.emit import (  # noqa: PLC0415
                    EV_CHECKPOINT_SAVED,
                    emit,
                )

                await emit(
                    EV_CHECKPOINT_SAVED,
                    thread_id=thread_id,
                    iteration=state.iteration,
                    backend=type(self.config.checkpointer).__name__,
                    trigger="final",
                )

    @_bus_bridge
    async def _run_from_state(
        self,
        state: AgentState,
        prompt: str,
        thread_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[TulipEvent]:
        """Continue execution from a given state (used for resume)."""
        self._initialize()

        started_at = datetime.now(UTC)
        _total_tokens = 0
        _tool_calls_count = 0
        _tool_errors_count = 0
        _reflexion_evals = 0
        _grounding_evals = 0
        _last_assistant_content: str | None = None
        _last_no_tool_calls = False

        # Extract last assistant content from state
        for msg in reversed(state.messages):
            if msg.role == Role.ASSISTANT and msg.content:
                _last_assistant_content = msg.content
                break

        # Reset user-supplied composable termination state; resume = fresh clock.
        if self.config.termination is not None:
            self.config.termination.reset()

        try:
            while True:
                # Same loop as run() — check termination, get response, execute tools
                if self.config.time_budget_seconds is not None:
                    elapsed = (datetime.now(UTC) - started_at).total_seconds()
                    if elapsed >= self.config.time_budget_seconds:
                        yield TerminateEvent(
                            reason="time_budget",
                            iterations_used=state.iteration,
                            final_confidence=state.confidence,
                            total_tool_calls=len(state.tool_executions),
                            final_message=_last_assistant_content,
                        )
                        break

                if self.config.termination is not None:
                    user_stop, user_reason = self.config.termination.check(
                        state,
                        last_message=_last_assistant_content or "",
                        no_tool_calls=_last_no_tool_calls,
                    )
                    if user_stop:
                        yield TerminateEvent(
                            reason=user_reason or "complete",
                            iterations_used=state.iteration,
                            final_confidence=state.confidence,
                            total_tool_calls=len(state.tool_executions),
                            final_message=_last_assistant_content,
                        )
                        break

                should_stop, stop_reason = state.should_terminate
                if should_stop and stop_reason:
                    yield TerminateEvent(
                        reason=stop_reason,
                        iterations_used=state.iteration,
                        final_confidence=state.confidence,
                        total_tool_calls=len(state.tool_executions),
                        final_message=_last_assistant_content,
                    )
                    break

                state = state.next_iteration()
                response, state = await self._get_model_response(state)
                prompt_toks = response.usage.get("prompt_tokens", 0)
                completion_toks = response.usage.get("completion_tokens", 0)
                cache_creation_toks = response.usage.get("cache_creation_input_tokens", 0)
                cache_read_toks = response.usage.get("cache_read_input_tokens", 0)
                _total_tokens += prompt_toks + completion_toks
                state = state.with_token_usage(
                    prompt_toks,
                    completion_toks,
                    cache_creation_tokens=cache_creation_toks,
                    cache_read_tokens=cache_read_toks,
                )
                _last_assistant_content = response.message.content
                _last_no_tool_calls = not response.message.tool_calls

                yield ThinkEvent(
                    iteration=state.iteration,
                    reasoning=response.message.content,
                    tool_calls=list(response.message.tool_calls),
                )

                if not response.message.tool_calls and self.config.completion_mode != "explicit":
                    yield TerminateEvent(
                        reason="complete",
                        iterations_used=state.iteration,
                        final_confidence=state.confidence,
                        total_tool_calls=len(state.tool_executions),
                        final_message=response.message.content,
                    )
                    break

                if not response.message.tool_calls:
                    continue

                # Execute tools (simplified — reuse main logic)
                for tc in response.message.tool_calls:
                    yield ToolStartEvent(
                        tool_name=tc.name, tool_call_id=tc.id, arguments=tc.arguments
                    )
                    start_time = time.perf_counter()
                    try:
                        ctx_factory = ToolContextFactory(
                            run_id=state.run_id,
                            agent_id=state.agent_id,
                            iteration=state.iteration,
                            state=state,
                            invocation_metadata=metadata or {},
                        )
                        [result] = await self._executor.execute(
                            [tc],
                            self._tool_registry,
                            ctx_factory,
                        )
                    except Exception as e:  # noqa: BLE001 — catches tool errors and InterruptException; branched below
                        from tulip.core.interrupt import InterruptException

                        if isinstance(e, InterruptException):
                            self._last_run_state = state
                            self._interrupt_state = state
                            self._interrupt_prompt = prompt
                            self._interrupt_thread_id = thread_id
                            self._interrupt_metadata = metadata
                            payload = e.value.payload if hasattr(e, "value") else {}
                            question = (
                                payload.get("question", str(payload))
                                if isinstance(payload, dict)
                                else str(payload)
                            )
                            options = payload.get("options") if isinstance(payload, dict) else None
                            yield InterruptEvent(
                                question=question,
                                options=options,
                                interrupt_id=e.value.interrupt_id
                                if hasattr(e, "value")
                                else "unknown",
                            )
                            return
                        result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            content="",
                            error=str(e),
                            duration_ms=(time.perf_counter() - start_time) * 1000,
                        )

                    state = state.with_tool_execution(
                        ToolExecution(
                            tool_name=result.name,
                            tool_call_id=result.tool_call_id,
                            arguments=tc.arguments,
                            result=result.content if result.success else None,
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )
                    )
                    state = state.with_message(Message.tool(result))

                    yield ToolCompleteEvent(
                        tool_name=result.name,
                        tool_call_id=result.tool_call_id,
                        result=result.content if result.success else None,
                        error=result.error,
                        duration_ms=result.duration_ms,
                    )

        finally:
            self._last_run_state = state

    async def _create_initial_state(
        self,
        prompt: str,
        thread_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> AgentState:
        """Create initial agent state."""
        # Try to load from checkpoint
        if self.config.checkpointer and thread_id:
            existing = await self.config.checkpointer.load(thread_id)
            if existing:
                from tulip.observability.emit import (  # noqa: PLC0415
                    EV_CHECKPOINT_LOADED,
                    emit,
                )

                await emit(
                    EV_CHECKPOINT_LOADED,
                    thread_id=thread_id,
                    iteration=existing.iteration,
                    backend=type(self.config.checkpointer).__name__,
                )
                # Add new user message and continue
                resumed: AgentState = existing.with_message(Message.user(prompt))
                return resumed

        # Create fresh state
        state = AgentState(
            agent_id=self.config.agent_id,
            max_iterations=self.config.max_iterations,
            confidence_threshold=(
                self.config.reflexion.confidence_threshold if self.config.reflexion else 0.85
            ),
            tool_loop_threshold=self.config.tool_loop_threshold,
            terminal_tools=frozenset(self.config.terminal_tools),
            token_budget=self.config.token_budget,
            completion_mode=self.config.completion_mode,
            metadata=metadata or {},
        )

        # Resolve system prompt (string or callable)
        prompt_value = self.config.system_prompt
        if callable(prompt_value):
            prompt_value = prompt_value({"prompt": prompt, "metadata": metadata or {}})
        prompt_str = str(prompt_value)

        # When output_schema is set, append a schema instruction so even
        # providers without ``response_format`` support produce valid JSON.
        if self.config.output_schema is not None:
            from tulip.core.structured import create_schema_prompt

            prompt_str = (
                f"{prompt_str}\n\n"
                f"=== Final-answer schema ===\n"
                f"{create_schema_prompt(self.config.output_schema)}"
            )

        state = state.with_message(Message.system(prompt_str))
        state = state.with_message(Message.user(prompt))

        return state

    async def _get_final_state(
        self,
        prompt: str,
        thread_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> AgentState:
        """Get the final state after run (for run_sync)."""
        # This is a fallback - in normal operation, state is tracked in run()
        # For run_sync, we need to reconstruct the final state
        state = await self._create_initial_state(prompt, thread_id, metadata)
        return state

    @staticmethod
    def _validate_messages(messages: list[Message]) -> list[Message]:
        """Validate message sequence and remove orphaned tool calls/results.

        Many LLM providers (Anthropic and others) reject requests where
        assistant messages with tool_calls don't have matching tool result
        messages. This method ensures message pairs are consistent.
        """
        # Collect all tool_call IDs that have matching tool results
        tool_result_ids: set[str] = set()
        for msg in messages:
            if msg.role == Role.TOOL and msg.tool_call_id:
                tool_result_ids.add(msg.tool_call_id)

        # Collect all tool_call IDs from assistant messages
        tool_call_ids: set[str] = set()
        for msg in messages:
            if msg.role == Role.ASSISTANT:
                for tc in msg.tool_calls:
                    tool_call_ids.add(tc.id)

        validated: list[Message] = []
        for msg in messages:
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Keep only tool calls that have matching results
                valid_calls = [tc for tc in msg.tool_calls if tc.id in tool_result_ids]
                if valid_calls:
                    validated.append(
                        Message(
                            role=msg.role,
                            content=msg.content,
                            tool_calls=valid_calls,
                            tool_call_id=msg.tool_call_id,
                            name=msg.name,
                        )
                    )
                elif msg.content:
                    # Has content but orphaned tool calls — keep as text-only message
                    validated.append(
                        Message(
                            role=msg.role,
                            content=msg.content,
                            tool_calls=[],
                            tool_call_id=msg.tool_call_id,
                            name=msg.name,
                        )
                    )
                # else: no content and no valid tool calls — drop entirely
            elif msg.role == Role.TOOL and msg.tool_call_id:
                # Keep only tool results whose tool_call exists
                if msg.tool_call_id in tool_call_ids:
                    validated.append(msg)
                # else: orphaned tool result — drop
            else:
                validated.append(msg)

        return validated

    def _parse_text_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from model text output (text fallback).

        Some models output tool calls as text like ``search(query="test")``
        instead of structured function calls. This parses them by matching
        against the registered tool registry.

        Returns parsed ToolCall list, or empty list if no matches found.
        """
        import re

        if not text or not self._tool_registry:
            return []

        # Build case-insensitive lookup: normalized_name -> real_name
        tool_lookup: dict[str, str] = {}
        for name in self._tool_registry.tools:
            normalized = name.lower().replace("_", "").replace("-", "")
            tool_lookup[normalized] = name

        # Match patterns like: tool_name(arg1="val1", arg2=val2)
        # Handles: search(query="test"), search(query='test'), search(query=test)
        pattern = re.compile(
            r"\b([a-zA-Z_][a-zA-Z0-9_-]*)\s*\(\s*(.*?)\s*\)",
            re.DOTALL,
        )

        parsed: list[ToolCall] = []
        for match in pattern.finditer(text):
            func_name = match.group(1)
            args_str = match.group(2)

            # Match against registry (case-insensitive, ignore underscores/hyphens)
            normalized = func_name.lower().replace("_", "").replace("-", "")
            real_name = tool_lookup.get(normalized)
            if not real_name:
                continue

            # Parse arguments: key="value" or key='value' or key=value
            args: dict[str, Any] = {}
            arg_pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?))\s*[,)]')
            # Add trailing ) to help match last arg
            args_text = args_str + ")"
            for arg_match in arg_pattern.finditer(args_text):
                key = arg_match.group(1)
                value = arg_match.group(2) or arg_match.group(3) or arg_match.group(4)
                if value is not None:
                    args[key] = value

            # Validate arguments against tool's schema before accepting
            tool_obj = self._tool_registry.get(real_name)
            if tool_obj:
                schema = tool_obj.to_openai_schema().get("function", {})
                params = schema.get("parameters", {})
                valid_params = set(params.get("properties", {}).keys())
                # Drop any argument not declared in the tool's schema
                args = {k: v for k, v in args.items() if k in valid_params}

            parsed.append(ToolCall(name=real_name, arguments=args))

        return parsed

    async def _get_model_response(
        self,
        state: AgentState,
    ) -> tuple[ModelResponse, AgentState]:
        """Get a response from the model."""
        # Server-stateful transports (e.g. OCIResponsesModel) own the
        # conversation history server-side — we send only the input
        # added since the last call (the initial system+user on turn
        # one, tool results / next user message on subsequent turns)
        # and pass the continuation token via ``provider_state``.
        # ConversationManager strategies (window/summarize on the
        # full history) have nothing to operate on and are skipped.
        #
        # Check the instance attribute with ``is True`` — that's both
        # MagicMock-safe (Mock() is True → False) and honours per-
        # instance overrides such as OCIResponsesModel's ``store=False``
        # ZDR mode, where the same class advertises False on certain
        # instances. Class-level check would miss those.
        server_stateful = getattr(self._model, "server_stateful", False) is True

        if server_stateful:
            messages = self._messages_since_last_assistant(state)
        else:
            # Apply conversation manager if present
            messages = list(state.messages)
            if self._conversation_manager:
                if hasattr(self._conversation_manager, "async_apply"):
                    messages = await self._conversation_manager.async_apply(messages)
                else:
                    messages = self._conversation_manager.apply(messages)

            # Validate message pairs (remove orphaned tool calls/results)
            messages = self._validate_messages(messages)

        # Get tool schemas
        tool_schemas = self._tool_registry.to_openai_schemas()

        # Pre-model hooks: allow hooks to modify messages before model call
        messages = await self._run_before_model_hooks(messages, tool_schemas or None)

        # When ``output_schema`` is set AND the provider ships native
        # structured output (OpenAI's ``response_format`` shape), pass
        # the JSON schema through directly. The provider parses + returns
        # a typed response without the prompted-JSON fallback. Otherwise
        # the schema only lives in the system prompt (see
        # ``_create_initial_state``) and is parsed post-hoc.
        native_response_format: dict[str, Any] | None = None
        if self.config.output_schema is not None and getattr(
            self._model, "supports_structured_output", False
        ):
            from tulip.core.structured import build_response_format

            native_response_format = build_response_format(
                self.config.output_schema,
                strict=self.config.output_schema_strict,
            )

        # Call model with hook-driven retry support
        # Hooks can request retries via event.retry = True
        max_model_retries = 5
        for _model_attempt in range(max_model_retries):
            complete_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": tool_schemas or None,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if native_response_format is not None:
                complete_kwargs["response_format"] = native_response_format

            if server_stateful:
                complete_kwargs["provider_state"] = state.provider_state

            response = await self._model.complete(**complete_kwargs)

            # Post-model hooks: event.retry = True to re-call
            after_event = await self._run_after_model_hooks(response, messages)

            if after_event.retry:
                continue  # Retry model call
            response = after_event.response
            break

        # Add assistant message to state
        state = state.with_message(response.message)

        # Server-stateful transports return a continuation token in
        # ``response.provider_state``; thread it into AgentState so
        # the next turn references the server-held thread.
        if server_stateful and response.provider_state is not None:
            state = state.with_provider_state(response.provider_state)

        return response, state

    def _messages_since_last_assistant(self, state: AgentState) -> list[Message]:
        """Return the slice of state.messages that the model hasn't seen yet.

        Used on the server-stateful path (OCIResponsesModel et al.).
        Walks state.messages backwards: everything after the most
        recent assistant message is the new input to send. If no
        assistant message exists yet (turn one), returns the full
        list — the model class is responsible for splitting system
        vs user appropriately for its wire protocol.
        """
        msgs = list(state.messages)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "assistant":
                return msgs[i + 1 :]
        return msgs

    def _idempotent_batch_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Return a hashable (name, args) key for within-batch idempotent
        dedup, or ``None`` if the tool is unknown / not idempotent.

        Used by the Phase 1 pre-execute pass in :meth:`run` to detect when
        the same idempotent call appears twice in one assistant response;
        the duplicate is collapsed to a ``batch_cache_ref`` slot so the
        body still fires only once per (name, args) per run.
        """
        tool = self._tool_registry.get(tool_name) if self._tool_registry else None
        if tool is None or not getattr(tool, "idempotent", False):
            return None
        # ``default=str`` keeps the key well-defined for argument types
        # ``json.dumps`` cannot serialise natively (e.g. datetimes); we
        # only need stable equality, not round-trip fidelity.
        import json as _json  # local import — only used here

        return (tool_name, _json.dumps(arguments, sort_keys=True, default=str))

    def _maybe_cached_idempotent_result(
        self,
        state: AgentState,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> ToolResult | None:
        """Return a cached ``ToolResult`` for an idempotent re-call, or None.

        A tool decorated with ``@tool(idempotent=True)`` should fire its body
        at most once per (name, arguments) pair within a run. If we find a
        prior execution on ``state.tool_executions`` with the same name and
        structurally-equal arguments, reuse its output and skip the executor.

        Returns None when:
          * the tool is unknown to the registry,
          * the tool didn't declare ``idempotent=True``, or
          * no prior execution matches.
        """
        tool = self._tool_registry.get(tool_name) if self._tool_registry else None
        if tool is None or not getattr(tool, "idempotent", False):
            return None

        # Late import to avoid circularity (loop.nodes -> agent.agent).
        from tulip.loop.nodes import _find_matching_execution

        prior = _find_matching_execution(state, tool_name, dict(arguments))
        if prior is None:
            return None
        return ToolResult(
            tool_call_id=tool_call_id,
            name=tool_name,
            content=prior.result if prior.result is not None else "",
            error=prior.error,
            duration_ms=0.0,
        )

    async def _structure_output(
        self,
        state: AgentState,
        final_message: str,
    ) -> tuple[BaseModel | None, str | None, AgentState]:
        """Coerce the agent's final answer into ``config.output_schema``.

        Tries to parse ``final_message`` directly; on validation failure,
        re-prompts the model up to ``output_schema_retries`` times with the
        Pydantic error details inlined so it can repair the JSON. Supporting
        providers receive a strict ``response_format`` for constrained
        decoding.

        Returns a triple ``(parsed, parse_error, state)`` — exactly one of
        ``parsed`` / ``parse_error`` is non-None.
        """
        from tulip.core.structured import (
            build_response_format,
            format_validation_errors,
            parse_structured,
        )

        schema = self.config.output_schema
        if schema is None:
            return None, None, state

        # First attempt: parse what the agent already produced.
        attempt = parse_structured(final_message, schema, strict=False)
        if attempt.success:
            return attempt.parsed, None, state

        last_error = attempt.error or "structured-output parse failed"
        last_validation_errors = attempt.validation_errors

        response_format = build_response_format(schema, strict=self.config.output_schema_strict)

        for _retry in range(self.config.output_schema_retries):
            error_detail = format_validation_errors(last_validation_errors)
            repair_prompt = (
                "[Schema Repair] Your previous response did not match the "
                "required JSON schema.\n"
                f"Validation errors:\n{error_detail}\n\n"
                "Return ONLY a valid JSON object that matches the schema. "
                "Do not wrap it in markdown fences. Do not add commentary."
            )
            state = state.with_message(Message.system(repair_prompt))
            messages = self._validate_messages(list(state.messages))

            try:
                response = await self._model.complete(
                    messages=messages,
                    tools=None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format=response_format,
                )
            except TypeError:
                # Provider doesn't accept response_format kwarg — retry without.
                response = await self._model.complete(
                    messages=messages,
                    tools=None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            new_message = response.message.content or ""
            state = state.with_message(response.message)

            attempt = parse_structured(new_message, schema, strict=False)
            if attempt.success:
                return attempt.parsed, None, state

            last_error = attempt.error or last_error
            last_validation_errors = attempt.validation_errors

        return None, last_error, state

    async def _apply_reflexion(
        self,
        state: AgentState,
        iteration_executions: list[ToolExecution] | None = None,
    ) -> tuple[ReflectEvent, AgentState]:
        """Apply Reflexion using the real Reflector.

        Delegates to reasoning.reflexion.Reflector for loop detection,
        execution analysis, confidence calculation, and guidance generation.
        """
        from tulip.reasoning.reflexion import ReflectionResult

        if self._reflector is None:
            # Fallback: no-op reflection
            return (
                ReflectEvent(
                    iteration=state.iteration,
                    assessment="on_track",
                    confidence_delta=0.0,
                    new_confidence=state.confidence,
                    guidance=None,
                ),
                state,
            )

        # Delegate to the real Reflector
        reflection: ReflectionResult = self._reflector.reflect(
            state, iteration_executions=iteration_executions
        )

        # Update state confidence
        state = self._reflector.adjust_state_confidence(state, reflection)

        # Create guidance message text
        guidance_text = self._reflector.create_guidance_message(reflection)

        return (
            ReflectEvent(
                iteration=state.iteration,
                assessment=reflection.assessment.value,
                confidence_delta=reflection.confidence_delta,
                new_confidence=state.confidence,
                guidance=guidance_text,
            ),
            state,
        )

    async def _apply_grounding(
        self,
        state: AgentState,
        final_response: str,
    ) -> tuple[GroundingEvent, AgentState]:
        """Apply grounding evaluation using LLM-as-judge.

        Extracts claims from the final response, gathers evidence from
        tool results, and uses the GroundingEvaluator to validate.
        """
        if self._grounding_evaluator is None or self._grounding_model is None:
            return (
                GroundingEvent(
                    score=1.0,
                    claims_evaluated=0,
                    ungrounded_claims=[],
                    requires_replan=False,
                ),
                state,
            )

        # Extract claims and evidence
        claims = self._extract_claims(final_response)
        evidence = self._gather_evidence(state)

        if not claims or not evidence:
            return (
                GroundingEvent(
                    score=1.0,
                    claims_evaluated=0,
                    ungrounded_claims=[],
                    requires_replan=False,
                ),
                state,
            )

        # Use LLM-as-judge
        from tulip.reasoning.grounding import GroundingResult

        grounding_result: GroundingResult = await self._grounding_evaluator.evaluate_with_llm(
            claims=claims,
            evidence=evidence,
            model=self._grounding_model,
        )

        return (
            GroundingEvent(
                score=grounding_result.score,
                claims_evaluated=len(grounding_result.claims),
                ungrounded_claims=grounding_result.ungrounded_claims,
                requires_replan=grounding_result.requires_replan,
            ),
            state,
        )

    @staticmethod
    def _extract_claims(response: str) -> list[str]:
        """Extract evaluable claims from the agent's response."""
        import re

        sentences = re.split(r"(?<=[.!])\s+", response.strip())
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()  # noqa: PLW2901
            if (
                len(sentence) > 20
                and not sentence.endswith("?")
                and not sentence.lower().startswith(("i ", "i'm ", "i'll ", "let me"))
            ):
                claims.append(sentence)
        return claims

    @staticmethod
    def _gather_evidence(state: AgentState) -> list[str]:
        """Gather evidence from tool execution results."""
        evidence = []
        for execution in state.tool_executions:
            if execution.success and execution.result:
                result_text = execution.result
                if len(result_text) > 500:
                    result_text = result_text[:500] + "..."
                evidence.append(f"[{execution.tool_name}]: {result_text}")
        return evidence

    @staticmethod
    def _build_fallback_summary(state: AgentState) -> str:
        """Build a summary from state when model returns no content on grace iteration."""
        parts = [
            f"Completed {state.iteration} iterations with {len(state.tool_executions)} tool calls."
        ]
        # Include last few tool results
        for execution in state.tool_executions[-3:]:
            if execution.success and execution.result:
                preview = (
                    execution.result[:150] + "..."
                    if len(execution.result) > 150
                    else execution.result
                )
                parts.append(f"- {execution.tool_name}: {preview}")
        return "\n".join(parts)

    async def _run_gsar_judgment(
        self,
        state: AgentState,
        final_message: str,
    ) -> tuple[Any, float | None, str | None]:
        """Run the GSAR judge over the agent's final answer + tool history.

        Returns ``(judgment, score, decision_value)`` where:

        - ``judgment`` is a ``JudgeOutput`` (or ``None`` if the
          judge raised and the safe-default fallback was used).
        - ``score`` is the recomputed scalar ``S`` from the judgment's
          partition under the configured weight map and contradiction
          penalty.
        - ``decision_value`` is the string form of
          :class:`~tulip.reasoning.gsar.Decision` (``"proceed"``, etc.),
          or ``"abstain"`` when the judge abstained.

        Returns ``(None, None, None)`` when ``self.config.gsar`` is unset.
        """
        if self.config.gsar is None:
            return None, None, None

        from tulip.reasoning.gsar import (
            EvidenceType,
            GSARThresholds,
            decide,
            gsar_score,
        )
        from tulip.reasoning.gsar_judge import StructuredOutputGSARJudge

        cfg = self.config.gsar

        # Default judge: a StructuredOutputGSARJudge over the agent's
        # primary model. Documented as "almost never what you want for
        # production" — the paper recommends a different judge model
        # from the generator.
        judge = cfg.judge
        if judge is None:
            judge = StructuredOutputGSARJudge(model=self._model)

        # Build the evidence corpus from tool executions on the final
        # state. Format mirrors the shape the default judge prompt
        # expects: one ``[tool=NAME args=…] result``-flavoured line per
        # execution, skipping idempotent cache hits and errored calls.
        evidence_lines: list[str] = []
        for ex in state.tool_executions:
            if ex.error:
                continue
            line = f"[tool={ex.tool_name} args={ex.arguments}] {ex.result or ''}"
            evidence_lines.append(line)
        evidence_corpus = "\n".join(evidence_lines) or "(no tool executions)"

        # Translate optional weight_map (str-keyed) into the typed map.
        weight_map: dict[EvidenceType, float] | None = None
        if cfg.weight_map is not None:
            weight_map = {EvidenceType(k): v for k, v in cfg.weight_map.items()}

        try:
            judgment = await judge.judge(
                report_synthesis=final_message,
                evidence_corpus=evidence_corpus,
            )
        except Exception:  # noqa: BLE001 — paper §6 "Robustness": never
            # let a judge failure crash the agent. Surface ``None`` so
            # the caller can decide whether to ship or replan.
            return None, None, None

        partition = judgment.to_partition()
        score = gsar_score(
            partition,
            weight_map=weight_map,
            contradiction_penalty=cfg.contradiction_penalty,
        )

        if judgment.abstained:
            decision_value = "abstain"
        else:
            thresholds = GSARThresholds(proceed=cfg.tau_proceed, regenerate=cfg.tau_regenerate)
            decision_value = decide(score, thresholds=thresholds).value

        return judgment, score, decision_value

    # Hook lifecycle dispatch is delegated to HookOrchestrator; these
    # thin wrappers preserve the original method names so internal
    # callers don't need to change. They all run after ``_initialize`` so
    # ``_hook_orchestrator`` is non-None — assert once via a helper.

    def _orch(self) -> HookOrchestrator:
        """Return the hook orchestrator, asserting it has been initialised."""
        assert self._hook_orchestrator is not None, (
            "Agent._hook_orchestrator accessed before initialize_agent() ran"
        )
        return self._hook_orchestrator

    async def _run_before_invocation_hooks(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        return await self._orch().run_before_invocation(prompt, state)

    async def _run_after_invocation_hooks(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        await self._orch().run_after_invocation(state, success)

    async def _run_before_model_hooks(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None,
    ) -> list[Any]:
        return await self._orch().run_before_model(messages, tools)

    async def _run_after_model_hooks(
        self,
        response: Any,
        messages: list[Any],
    ) -> Any:
        return await self._orch().run_after_model(response, messages)

    async def _run_before_tool_hooks(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> Any:
        return await self._orch().run_before_tool(
            tool_name,
            tool_call_id,
            arguments,
        )

    async def _run_after_tool_hooks(
        self,
        tool_name: str,
        result: Any,
        error: str | None,
        *,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        return await self._orch().run_after_tool(
            tool_name,
            result,
            error,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )

    # Properties for easy access
