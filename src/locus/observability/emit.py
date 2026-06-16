# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Lightweight emit helpers — single line per instrumentation site.

The contract for every emission site in the SDK:

* **Sync call sites** use :func:`emit_sync`. It schedules a publish on
  the running event loop *if there is one*; otherwise drops. Never
  blocks, never raises.
* **Async call sites** use :func:`emit`. Awaits the bus publish so
  events appear in deterministic order on the consumer side.

Both check :func:`current_run_id` first. When the contextvar is unset
they return immediately — no bus singleton instantiation, no event
construction, no allocation. SDK users who don't use telemetry pay
exactly one ``ContextVar.get()`` per emission site.

Module-level event-name constants pin the canonical wire types so
changes are greppable and consumers (the workbench, third-party
monitors) can rely on them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from locus.observability.context import current_owner_loop, current_run_id


logger = logging.getLogger(__name__)

# ``loop.create_task`` returns a Task that the GC may collect before it
# runs unless someone holds a strong reference. We pin every fire-and-
# forget telemetry task on this set and remove it on completion, which
# is the recipe Python's docs recommend for long-lived emit-only tasks.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


# Canonical event-type names — change here, propagates to every
# instrumentation site. Listed verbatim in the workbench's renderer
# so a typo here breaks one place, not many.

# --- Multi-agent ---
EV_ORCHESTRATOR_ROUTING = "multiagent.orchestrator.routing"
EV_ORCHESTRATOR_DECISION = "multiagent.orchestrator.decision"
EV_ORCHESTRATOR_SPECIALISTS_INVOKED = "multiagent.orchestrator.specialists_invoked"
EV_ORCHESTRATOR_SUMMARY = "multiagent.orchestrator.summary"
EV_SPECIALIST_STARTED = "multiagent.specialist.started"
EV_SPECIALIST_COMPLETED = "multiagent.specialist.completed"
EV_HANDOFF_INITIATED = "multiagent.handoff.initiated"
EV_HANDOFF_COMPLETED = "multiagent.handoff.completed"

# --- Composition pipelines ---
EV_PIPELINE_STAGE_STARTED = "composition.stage.started"
EV_PIPELINE_STAGE_COMPLETED = "composition.stage.completed"
EV_PIPELINE_FANOUT_STARTED = "composition.fanout.started"
EV_PIPELINE_FANOUT_COMPLETED = "composition.fanout.completed"
EV_LOOP_ITERATION_STARTED = "composition.loop.iteration.started"
EV_LOOP_ITERATION_COMPLETED = "composition.loop.iteration.completed"
EV_LOOP_TERMINATED = "composition.loop.terminated"

# --- Skills ---
EV_SKILL_ACTIVATED = "skills.activated"

# --- Memory / checkpointing ---
EV_CHECKPOINT_SAVED = "memory.checkpoint.saved"
EV_CHECKPOINT_LOADED = "memory.checkpoint.loaded"

# --- Agent ReAct loop (bridge from yielded LocusEvent → bus) ---
EV_AGENT_THINK = "agent.think"
EV_AGENT_TOOL_STARTED = "agent.tool.started"
EV_AGENT_TOOL_COMPLETED = "agent.tool.completed"
EV_AGENT_REFLECT = "agent.reflect"
EV_AGENT_GROUNDING = "agent.grounding"
EV_AGENT_MODEL_CHUNK = "agent.model.chunk"
EV_AGENT_MODEL_COMPLETED = "agent.model.completed"
EV_AGENT_TOKENS_USED = "agent.tokens.used"
EV_AGENT_INTERRUPT = "agent.interrupt"
EV_AGENT_TERMINATE = "agent.terminate"

# --- StateGraph node lifecycle ---
EV_GRAPH_NODE_STARTED = "multiagent.graph.node.started"
EV_GRAPH_NODE_COMPLETED = "multiagent.graph.node.completed"
EV_GRAPH_NODE_ROUTED = "multiagent.graph.node.routed"

# --- A2A protocol ---
EV_A2A_TASK_RECEIVED = "a2a.task.received"
EV_A2A_TASK_PROCESSING = "a2a.task.processing"
EV_A2A_TASK_COMPLETED = "a2a.task.completed"
EV_A2A_CLIENT_SEND = "a2a.client.send"
EV_A2A_CLIENT_RECEIVED = "a2a.client.received"

# --- RAG ---
EV_RAG_QUERY_STARTED = "rag.query.started"
EV_RAG_QUERY_COMPLETED = "rag.query.completed"
EV_RAG_EMBEDDING_GENERATED = "rag.embedding.generated"
EV_RAG_STORE_UPSERT = "rag.store.upsert"
EV_RAG_STORE_SEARCH = "rag.store.search"

# --- Memory operations ---
EV_MEMORY_COMPACTOR_TRIGGERED = "memory.compactor.triggered"
EV_MEMORY_COMPACTOR_COMPLETED = "memory.compactor.completed"
EV_MEMORY_CONVERSATION_ADDED = "memory.conversation.added"
EV_MEMORY_CONVERSATION_PRUNED = "memory.conversation.pruned"
EV_MEMORY_MANAGER_INJECTED = "memory.manager.injected"
EV_MEMORY_MANAGER_EXTRACTED = "memory.manager.extracted"

# --- DeepAgent ---
EV_DEEPAGENT_SUBAGENT_SPAWNED = "deepagent.subagent.spawned"
EV_DEEPAGENT_SUBAGENT_COMPLETED = "deepagent.subagent.completed"
EV_DEEPAGENT_FS_READ = "deepagent.fs.read"
EV_DEEPAGENT_FS_WRITE = "deepagent.fs.write"
EV_DEEPAGENT_TODO_ADDED = "deepagent.todo.added"
EV_DEEPAGENT_TODO_COMPLETED = "deepagent.todo.completed"

# --- Research workflow ---
EV_RESEARCH_EXECUTE_STARTED = "research.execute.started"
EV_RESEARCH_EXECUTE_COMPLETED = "research.execute.completed"
EV_RESEARCH_CAUSAL_BUILT = "research.causal.built"
EV_RESEARCH_SUMMARIZE_COMPLETED = "research.summarize.completed"
EV_RESEARCH_GROUNDING_EVALUATED = "research.grounding.evaluated"
EV_RESEARCH_REGENERATE_STARTED = "research.regenerate.started"
EV_RESEARCH_REGENERATE_COMPLETED = "research.regenerate.completed"
EV_RESEARCH_REPLAN = "research.replan"
EV_RESEARCH_COMPLETED = "research.completed"

# --- Hooks (built-in bridges) ---
EV_HOOK_MODEL_RETRY = "agent.model.retry"
EV_HOOK_STEERING_APPLIED = "agent.steering.applied"
EV_HOOK_GUARDRAIL_TRIGGERED = "agent.guardrail.triggered"


async def emit(event_type: str, /, **data: Any) -> None:
    """Publish a :class:`StreamEvent` if a run_id is in the current
    context. No-op otherwise.

    Use from any ``async def`` instrumentation site. Awaits the bus
    publish so events appear in the order they were emitted on the
    consumer side.

    The bus singleton is imported lazily here so simply importing
    this module doesn't construct it.
    """
    rid = current_run_id()
    if rid is None:
        return
    # Lazy import — keeps the bus singleton from being constructed
    # until the first real emission.
    from locus.observability.event_bus import StreamEvent, get_event_bus  # noqa: PLC0415

    bus = get_event_bus()
    # Cache the running loop on the bus so worker-thread ``emit_sync``
    # callers (sync ``@tool`` functions hopping through the executor)
    # can hop publishes back into the right event loop.
    if getattr(bus, "_owner_loop", None) is None:
        try:
            bus._owner_loop = asyncio.get_running_loop()  # noqa: SLF001 — bus owns this attr
        except RuntimeError:
            pass
    try:
        await bus.publish(
            StreamEvent(run_id=rid, event_type=event_type, data=data),
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the SDK
        logger.debug("emit failed for %s", event_type, exc_info=True)


def emit_sync(event_type: str, /, **data: Any) -> None:
    """Sync-call equivalent of :func:`emit`.

    Schedules a publish on the running event loop. Two cases handled:

    * Called from inside the loop's thread — uses ``loop.create_task``
      and pins the task on ``_BACKGROUND_TASKS``.
    * Called from a worker thread (``asyncio.to_thread`` / the @tool
      decorator's executor) — uses ``run_coroutine_threadsafe`` to
      hop back to the bus's event loop. The contextvar copy that the
      decorator passes into the executor preserves ``run_id`` across
      the thread boundary.

    If neither path resolves a loop, the event is dropped — telemetry
    must never spin up a fresh loop and fight asyncio singletons.
    """
    rid = current_run_id()
    if rid is None:
        return
    from locus.observability.event_bus import StreamEvent, get_event_bus  # noqa: PLC0415

    bus = get_event_bus()
    event = StreamEvent(run_id=rid, event_type=event_type, data=data)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in *this* thread. We may still be inside a
        # worker thread spawned by ``asyncio.to_thread`` /
        # ``loop.run_in_executor`` — in which case the run_context's
        # owner loop is reachable via the contextvar populated on
        # ``run_context`` entry (and copied into the worker thread by
        # the ``@tool`` decorator's ``ctx.run()``).
        owner_loop = current_owner_loop()
        if owner_loop is None or not owner_loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(bus.publish(event), owner_loop)
        except Exception:  # noqa: BLE001 — telemetry must never break the SDK
            logger.debug("emit_sync threadsafe schedule failed", exc_info=True)
        return

    coro = bus.publish(event)
    task = loop.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
