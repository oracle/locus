# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Bridge yielded :class:`LocusEvent` instances onto the SSE bus.

The agent's ``run()`` async generator yields typed :class:`LocusEvent`
objects (``ThinkEvent``, ``ToolStartEvent``, ``ReflectEvent``, etc.).
Without this bridge, only the immediate ``async for`` consumer sees
them — the SSE workbench sees nothing of the agent's inner work.

We tap the iterator with a single ``bridge_locus_event()`` call that
maps each event type to the canonical ``EV_AGENT_*`` constants and
publishes via ``emit()``. Because ``emit()`` itself short-circuits
when no ``run_context`` is active, this entire bridge is a no-op for
SDK users who don't subscribe.

Why a separate module: ``runtime_loop.py`` already imports a long
list of SDK internals; we don't want to grow it further with
event-type dispatch. Keeping the dispatch table here keeps the bridge
greppable and testable.
"""

from __future__ import annotations

from locus.core.events import (
    GroundingEvent,
    InterruptEvent,
    LocusEvent,
    ModelChunkEvent,
    ModelCompleteEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.observability.emit import (
    EV_AGENT_GROUNDING,
    EV_AGENT_INTERRUPT,
    EV_AGENT_MODEL_CHUNK,
    EV_AGENT_MODEL_COMPLETED,
    EV_AGENT_REFLECT,
    EV_AGENT_TERMINATE,
    EV_AGENT_THINK,
    EV_AGENT_TOKENS_USED,
    EV_AGENT_TOOL_COMPLETED,
    EV_AGENT_TOOL_STARTED,
    emit,
)


_PREVIEW_LEN = 240


def _preview(text: str | None) -> str:
    if not text:
        return ""
    return text[:_PREVIEW_LEN]


async def bridge_locus_event(event: LocusEvent) -> None:
    """Publish ``event`` on the SSE bus under the canonical ``agent.*``
    event_type. No-op when no ``run_context`` is active (the underlying
    ``emit()`` checks the contextvar first).

    Each branch carries the same fields as the source ``LocusEvent``,
    plus tags useful for the workbench renderer to pick a swimlane:

    * ``ThinkEvent`` → ``agent.think`` with reasoning preview + tool
      call count.
    * ``ToolStartEvent`` / ``ToolCompleteEvent`` → started/completed
      pair sharing the source ``tool_call_id`` as ``span_id``.
    * ``ReflectEvent`` → ``agent.reflect`` with the assessment
      category + new confidence.
    * ``GroundingEvent`` → ``agent.grounding`` with score + replan
      flag.
    * ``ModelChunkEvent`` → ``agent.model.chunk`` (streaming tokens;
      caller is responsible for any throttling).
    * ``ModelCompleteEvent`` → ``agent.model.completed`` AND a
      separate ``agent.tokens.used`` event so token-cost dashboards
      can subscribe without parsing model-completion payloads.
    * ``InterruptEvent`` → ``agent.interrupt`` so HITL UIs can pop
      modals on the bus.
    * ``TerminateEvent`` → ``agent.terminate`` with the canonical
      stop reason + iteration count.

    Failures inside this bridge are swallowed by the underlying
    ``emit()`` — telemetry must never break the agent loop.
    """
    if isinstance(event, ThinkEvent):
        await emit(
            EV_AGENT_THINK,
            iteration=event.iteration,
            reasoning_preview=_preview(event.reasoning),
            has_tool_calls=bool(event.tool_calls),
            tool_call_count=len(event.tool_calls),
        )
    elif isinstance(event, ToolStartEvent):
        await emit(
            EV_AGENT_TOOL_STARTED,
            tool_name=event.tool_name,
            span_id=event.tool_call_id,
            arg_keys=sorted(event.arguments.keys()),
        )
    elif isinstance(event, ToolCompleteEvent):
        await emit(
            EV_AGENT_TOOL_COMPLETED,
            tool_name=event.tool_name,
            span_id=event.tool_call_id,
            success=event.success,
            duration_ms=event.duration_ms,
            output_preview=_preview(event.result),
            error=event.error,
        )
    elif isinstance(event, ReflectEvent):
        await emit(
            EV_AGENT_REFLECT,
            iteration=event.iteration,
            assessment=event.assessment,
            confidence_delta=event.confidence_delta,
            new_confidence=event.new_confidence,
            guidance_preview=_preview(event.guidance),
        )
    elif isinstance(event, GroundingEvent):
        await emit(
            EV_AGENT_GROUNDING,
            score=event.score,
            claims_evaluated=event.claims_evaluated,
            ungrounded_count=len(event.ungrounded_claims),
            requires_replan=event.requires_replan,
        )
    elif isinstance(event, ModelChunkEvent):
        await emit(
            EV_AGENT_MODEL_CHUNK,
            content_preview=_preview(event.content),
            done=event.done,
            has_tool_calls=bool(event.tool_calls),
        )
    elif isinstance(event, ModelCompleteEvent):
        await emit(
            EV_AGENT_MODEL_COMPLETED,
            content_preview=_preview(event.content),
            tool_call_count=len(event.tool_calls),
            stop_reason=event.stop_reason,
        )
        # Separate token-usage event so cost dashboards can subscribe
        # without parsing the completion payload.
        if event.usage:
            await emit(
                EV_AGENT_TOKENS_USED,
                prompt_tokens=event.usage.get("prompt_tokens", 0),
                completion_tokens=event.usage.get("completion_tokens", 0),
                total_tokens=event.usage.get("total_tokens", 0),
            )
    elif isinstance(event, InterruptEvent):
        await emit(
            EV_AGENT_INTERRUPT,
            interrupt_id=event.interrupt_id,
            question_preview=_preview(event.question),
            options=event.options,
        )
    elif isinstance(event, TerminateEvent):
        await emit(
            EV_AGENT_TERMINATE,
            reason=event.reason,
            iterations_used=event.iterations_used,
            final_confidence=event.final_confidence,
            total_tool_calls=event.total_tool_calls,
            final_message_preview=_preview(event.final_message),
        )
