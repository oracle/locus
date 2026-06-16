# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Canonical router event types + emission helpers.

Each function publishes a :class:`StreamEvent` on the singleton
:class:`EventBus` with a stable ``event_type`` string. Callers (the
:class:`Router`, :class:`CognitiveCompiler`) get a typed surface that
keeps the event names in one place — string drift across modules is
the most common bug in pub/sub plumbing.

Event types follow ``router.<phase>.<status>`` naming:

* ``router.frame.extracted`` — the LLM produced a valid GoalFrame
* ``router.frame.failed`` — extractor couldn't produce a parseable frame
* ``router.protocol.selected`` — the registry picked a protocol
* ``router.protocol.no_match`` — registry couldn't match the frame
* ``router.policy.verdict`` — policy gate produced allow/deny/approval
* ``router.runnable.compiled`` — builder emitted a Runnable
* ``router.runnable.executing`` — runnable.execute started
* ``router.runnable.executed`` — runnable.execute finished
* ``router.runnable.failed`` — execution raised
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from locus.observability.event_bus import StreamEvent, get_event_bus


if TYPE_CHECKING:
    from locus.router.goal_frame import GoalFrame
    from locus.router.policy import PolicyVerdict
    from locus.router.protocol import Protocol


def _frame_payload(frame: GoalFrame) -> dict[str, Any]:
    """JSON-safe snapshot of a GoalFrame for the wire."""
    return {
        "primary_goal": frame.primary_goal.value,
        "secondary_goals": [g.value for g in frame.secondary_goals],
        "domain": frame.domain,
        "complexity": frame.complexity.value,
        "risk": frame.risk.value,
        "approval_required": frame.approval_required,
        "required_capabilities": list(frame.required_capabilities),
        "success_criteria": list(frame.success_criteria),
    }


async def emit_frame_extracted(run_id: str, frame: GoalFrame) -> None:
    """Frame extraction succeeded — publishes the parsed schema."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.frame.extracted",
            data={"frame": _frame_payload(frame)},
        ),
    )


async def emit_frame_failed(run_id: str, error: str) -> None:
    """Extractor returned something the schema rejected."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.frame.failed",
            data={"error": error},
        ),
    )


async def emit_protocol_selected(
    run_id: str,
    frame: GoalFrame,
    protocol: Protocol,
    *,
    method: str = "rule_based",
    rationale: str | None = None,
) -> None:
    """Registry picked a protocol for the frame.

    ``method`` records how the pick was made:

    - ``"rule_based"`` — the default ``_rank_key`` tuple comparison.
    - ``"single_candidate"`` — only one protocol survived filtering;
      no ranker call needed.
    - ``"llm_picked"`` — the opt-in :class:`LLMProtocolPicker` made
      the call; ``rationale`` carries its one-sentence justification.
    - ``"rule_based_fallback"`` — the picker raised or returned an
      unknown id, and ``_rank_key`` resolved it instead. The matching
      ``router.protocol.picker_fallback`` event carries the underlying
      error.
    """
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.protocol.selected",
            data={
                "protocol_id": protocol.id,
                "protocol_description": protocol.description,
                "primary_goal": frame.primary_goal.value,
                "is_canonical": frame.primary_goal in protocol.primary_for,
                "cost": protocol.cost,
                "latency": protocol.latency,
                "risk_max": protocol.risk_max.value,
                "method": method,
                "rationale": rationale,
            },
        ),
    )


async def emit_protocol_no_match(run_id: str, frame: GoalFrame, error: str) -> None:
    """Registry couldn't find a protocol matching the frame."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.protocol.no_match",
            data={
                "primary_goal": frame.primary_goal.value,
                "risk": frame.risk.value,
                "error": error,
            },
        ),
    )


async def emit_picker_fallback(run_id: str, frame: GoalFrame, error: str) -> None:
    """The LLM protocol picker raised or returned an unknown id, so
    the compiler fell back to the rule-based ranker for this dispatch.

    Operators see this when the emergent path degrades — it does not
    indicate a failed dispatch (the fallback still produced a valid
    protocol via ``_rank_key``).
    """
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.protocol.picker_fallback",
            data={
                "primary_goal": frame.primary_goal.value,
                "risk": frame.risk.value,
                "error": error,
            },
        ),
    )


async def emit_policy_verdict(
    run_id: str,
    frame: GoalFrame,
    protocol: Protocol,
    verdict: PolicyVerdict,
) -> None:
    """Policy gate decided allow / deny / require_approval."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.policy.verdict",
            data={
                "protocol_id": protocol.id,
                "allow": verdict.allow,
                "require_approval": verdict.require_approval,
                "reason": verdict.reason,
                "frame_risk": frame.risk.value,
            },
        ),
    )


async def emit_runnable_compiled(
    run_id: str,
    protocol_id: str,
    runnable_type: str,
    capability_count: int,
) -> None:
    """Builder emitted a Runnable. ``runnable_type`` is the adapter
    class name (e.g. ``AgentRunnable``, ``PipelineRunnable``)."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.runnable.compiled",
            data={
                "protocol_id": protocol_id,
                "runnable_type": runnable_type,
                "capability_count": capability_count,
            },
        ),
    )


async def emit_runnable_executing(run_id: str, protocol_id: str) -> None:
    """About to call ``runnable.execute(task)``."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.runnable.executing",
            data={"protocol_id": protocol_id},
        ),
    )


async def emit_runnable_executed(run_id: str, protocol_id: str, text_length: int) -> None:
    """Execution returned a RunnableResult successfully."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.runnable.executed",
            data={
                "protocol_id": protocol_id,
                "text_length": text_length,
            },
        ),
    )


async def emit_runnable_failed(run_id: str, protocol_id: str, error: str) -> None:
    """Execution raised before producing a result."""
    await get_event_bus().publish(
        StreamEvent(
            run_id=run_id,
            event_type="router.runnable.failed",
            data={
                "protocol_id": protocol_id,
                "error": error,
            },
        ),
    )
