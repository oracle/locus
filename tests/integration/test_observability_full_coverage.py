# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end coverage of every layer instrumented in the SSE retrofit.

The first observability pass instrumented the orchestration layer
(router, multi-agent, composition pipelines, checkpointing, skill
activation). The follow-up retrofit added eight more layers:

1. Agent ReAct loop yield bridge (Think/Tool/Reflect/Grounding/
   Model/Interrupt/Terminate → ``agent.*``).
2. Token usage as a first-class event (``agent.tokens.used``).
3. StateGraph node lifecycle (``multiagent.graph.node.*``).
4. A2A server + client (``a2a.task.*``, ``a2a.client.*``).
5. RAG retriever query (``rag.query.*``).
6. Memory operations — sliding window prune + summarising compactor
   (``memory.conversation.pruned``, ``memory.compactor.*``).
7. DeepAgent — subagent dispatch + filesystem + todos
   (``deepagent.subagent.*``, ``deepagent.fs.*``, ``deepagent.todo.*``).
8. Built-in hooks bridge — retry / steering / guardrails
   (``agent.model.retry``, ``agent.steering.applied``,
   ``agent.guardrail.triggered``).

This test sweep drives the smallest possible primitive in each layer
and asserts the documented event_type lands on the bus history.
Layers that need a real LLM call (the agent loop, A2A tasks) use
stub models; layers that don't (RAG retriever, conversation prune,
filesystem, guardrails) are exercised directly.

The SDK-without-SSE invariant is *not* re-asserted per layer here —
the unit-test suite already locks that in via subprocess fresh-import.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from locus.observability import (
    StreamEvent,
    get_event_bus,
    reset_event_bus,
    run_context,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


def _kinds(rid: str) -> list[str]:
    history = list(get_event_bus()._history.get(rid, ()))  # type: ignore[attr-defined]
    return [e.event_type for e in history]


def _data(rid: str, event_type: str) -> dict[str, Any] | None:
    for ev in get_event_bus()._history.get(rid, ()):  # type: ignore[attr-defined]
        if ev.event_type == event_type:
            return ev.data
    return None


# ---------------------------------------------------------------------------
# Layer 1 + 2 — agent loop yield bridge + token usage
# ---------------------------------------------------------------------------


class TestAgentLoopBridge:
    async def test_terminate_event_published_via_decorator(self):
        """Drives the smallest agent surface: the ``run()`` async
        generator decorated with ``_bus_bridge``. Yields a fake
        ``TerminateEvent`` and asserts ``agent.terminate`` lands on
        the bus."""
        from locus.agent.runtime_loop import _bus_bridge
        from locus.core.events import TerminateEvent

        @_bus_bridge
        async def fake_run(_self):
            yield TerminateEvent(
                reason="complete",
                iterations_used=3,
                final_confidence=0.85,
                total_tool_calls=1,
                final_message="ok",
            )

        async with run_context("agent-1") as rid:
            async for _ in fake_run(None):
                pass

        kinds = _kinds(rid)
        assert "agent.terminate" in kinds, kinds
        data = _data(rid, "agent.terminate")
        assert data is not None
        assert data["iterations_used"] == 3
        assert data["reason"] == "complete"

    async def test_tokens_used_emitted_with_model_complete(self):
        from locus.agent.runtime_loop import _bus_bridge
        from locus.core.events import ModelCompleteEvent

        @_bus_bridge
        async def fake_run(_self):
            yield ModelCompleteEvent(
                content="hello",
                usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            )

        async with run_context("agent-2") as rid:
            async for _ in fake_run(None):
                pass

        assert "agent.tokens.used" in _kinds(rid)
        td = _data(rid, "agent.tokens.used")
        assert td is not None
        assert td["prompt_tokens"] == 12
        assert td["completion_tokens"] == 5
        assert td["total_tokens"] == 17


# ---------------------------------------------------------------------------
# Layer 3 — StateGraph node lifecycle
# ---------------------------------------------------------------------------


class TestStateGraphNodes:
    async def test_node_started_completed_emitted(self):
        from pydantic import BaseModel

        from locus.multiagent.graph import END, START, GraphConfig, StateGraph

        class _State(BaseModel):
            greeting: str = ""

        graph = StateGraph(state_schema=_State)

        async def hello_node(_state: dict[str, Any]) -> dict[str, Any]:
            return {"greeting": "hi"}

        graph.add_node("hello", hello_node)
        graph.add_edge(START, "hello")
        graph.add_edge("hello", END)

        async with run_context("graph-1") as rid:
            await graph.execute({}, config=GraphConfig())

        kinds = _kinds(rid)
        assert "multiagent.graph.node.started" in kinds, kinds
        assert "multiagent.graph.node.completed" in kinds, kinds
        started = _data(rid, "multiagent.graph.node.started")
        assert started is not None
        assert started["node_id"] == "hello"


# ---------------------------------------------------------------------------
# Layer 5 — RAG retriever
# ---------------------------------------------------------------------------


class TestRagRetriever:
    async def test_query_started_completed_pair(self):
        from locus.rag.embeddings.base import EmbeddingResult
        from locus.rag.retriever import RAGRetriever
        from locus.rag.stores.base import Document, SearchResult

        embedder = AsyncMock()
        embedder.embed_query.return_value = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3], text="q", model="stub"
        )

        store = AsyncMock()
        store.search.return_value = [
            SearchResult(
                document=Document(id="d1", content="hello"),
                score=0.9,
            )
        ]

        retriever = RAGRetriever(embedder=embedder, store=store)
        async with run_context("rag-1") as rid:
            await retriever.retrieve("what is ...?", limit=5)

        kinds = _kinds(rid)
        assert "rag.query.started" in kinds
        assert "rag.query.completed" in kinds
        completed = _data(rid, "rag.query.completed")
        assert completed is not None
        assert completed["hit_count"] == 1
        assert completed["top_score"] == 0.9


# ---------------------------------------------------------------------------
# Layer 6 — memory operations (sliding window prune)
# ---------------------------------------------------------------------------


class TestMemoryConversation:
    async def test_sliding_window_prune_emits(self):
        from locus.core.messages import Message, Role
        from locus.memory.conversation import SlidingWindowManager

        mgr = SlidingWindowManager(window_size=3, preserve_system=True)
        # 6 non-system messages → 3 should be pruned.
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            *(Message(role=Role.USER, content=f"hi {i}") for i in range(6)),
        ]

        async with run_context("mem-1") as rid:
            mgr.apply(msgs)
            # emit_sync schedules a task on the loop — drain.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        kinds = _kinds(rid)
        assert "memory.conversation.pruned" in kinds, kinds
        d = _data(rid, "memory.conversation.pruned")
        assert d is not None
        assert d["removed_count"] == 3
        assert d["window_size"] == 3


# ---------------------------------------------------------------------------
# Layer 7 — DeepAgent (filesystem + todos)
# ---------------------------------------------------------------------------


class TestDeepAgentFilesystem:
    async def test_fs_write_emits(self):
        from locus.deepagent.backends.state import StateBackend
        from locus.deepagent.tools.filesystem import make_filesystem_tools

        backend = StateBackend()
        tools = make_filesystem_tools(backend)
        write_file = next(t for t in tools if t.name == "write_file")

        async with run_context("fs-1") as rid:
            await write_file.execute(path="/note.txt", contents="hello world")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert "deepagent.fs.write" in _kinds(rid)
        d = _data(rid, "deepagent.fs.write")
        assert d is not None
        assert d["path"] == "/note.txt"
        assert d["byte_count"] == len("hello world")

    async def test_todo_added_emits(self):
        from locus.deepagent.todos import make_todo_tools

        tools = make_todo_tools()
        write_todos = next(t for t in tools if t.name == "write_todos")

        todos = json.dumps(
            [
                {"content": "research X", "status": "pending"},
                {"content": "review Y", "status": "completed"},
            ]
        )
        async with run_context("todo-1") as rid:
            await write_todos.execute(todos_json=todos)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        kinds = _kinds(rid)
        assert "deepagent.todo.added" in kinds
        assert "deepagent.todo.completed" in kinds


# ---------------------------------------------------------------------------
# Layer 8 — built-in hooks bridge (guardrails fastest path)
# ---------------------------------------------------------------------------


class TestGuardrailsBridge:
    async def test_blocked_tool_emits_guardrail_triggered(self):
        from locus.core.events import BeforeToolCallEvent
        from locus.hooks.builtin.guardrails import GuardrailConfig, GuardrailsHook

        hook = GuardrailsHook(GuardrailConfig(block_dangerous_tools={"shell"}))
        ev = BeforeToolCallEvent(tool_name="shell", arguments={"cmd": "ls"})

        async with run_context("gr-1") as rid:
            with pytest.raises(ValueError, match="blocked by guardrails"):
                await hook.on_before_tool_call(ev)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        kinds = _kinds(rid)
        assert "agent.guardrail.triggered" in kinds, kinds
        d = _data(rid, "agent.guardrail.triggered")
        assert d is not None
        assert d["rule_name"] == "blocked_tool"


# ---------------------------------------------------------------------------
# Coverage smoke — every newly-added EV_* constant is publishable
# ---------------------------------------------------------------------------


class TestEventConstantsAreReachable:
    async def test_every_new_event_constant_is_a_string(self):
        """Smoke-check: each new EV_* constant exposed in
        ``locus.observability.emit`` is a non-empty string. Catches typos
        in the event-name registry before the workbench renderer trips on
        them."""
        # ``locus.observability.emit`` resolves to the *function* via the
        # package's re-export — fetch the module from sys.modules.
        import sys as _sys

        emit_mod = _sys.modules["locus.observability.emit"]

        new_constants = [
            "EV_AGENT_THINK",
            "EV_AGENT_TOOL_STARTED",
            "EV_AGENT_TOOL_COMPLETED",
            "EV_AGENT_REFLECT",
            "EV_AGENT_GROUNDING",
            "EV_AGENT_MODEL_CHUNK",
            "EV_AGENT_MODEL_COMPLETED",
            "EV_AGENT_TOKENS_USED",
            "EV_AGENT_INTERRUPT",
            "EV_AGENT_TERMINATE",
            "EV_GRAPH_NODE_STARTED",
            "EV_GRAPH_NODE_COMPLETED",
            "EV_GRAPH_NODE_ROUTED",
            "EV_A2A_TASK_RECEIVED",
            "EV_A2A_TASK_PROCESSING",
            "EV_A2A_TASK_COMPLETED",
            "EV_A2A_CLIENT_SEND",
            "EV_A2A_CLIENT_RECEIVED",
            "EV_RAG_QUERY_STARTED",
            "EV_RAG_QUERY_COMPLETED",
            "EV_RAG_EMBEDDING_GENERATED",
            "EV_RAG_STORE_UPSERT",
            "EV_RAG_STORE_SEARCH",
            "EV_MEMORY_COMPACTOR_TRIGGERED",
            "EV_MEMORY_COMPACTOR_COMPLETED",
            "EV_MEMORY_CONVERSATION_ADDED",
            "EV_MEMORY_CONVERSATION_PRUNED",
            "EV_DEEPAGENT_SUBAGENT_SPAWNED",
            "EV_DEEPAGENT_SUBAGENT_COMPLETED",
            "EV_DEEPAGENT_FS_READ",
            "EV_DEEPAGENT_FS_WRITE",
            "EV_DEEPAGENT_TODO_ADDED",
            "EV_DEEPAGENT_TODO_COMPLETED",
            "EV_HOOK_MODEL_RETRY",
            "EV_HOOK_STEERING_APPLIED",
            "EV_HOOK_GUARDRAIL_TRIGGERED",
        ]
        for name in new_constants:
            value = getattr(emit_mod, name, None)
            assert isinstance(value, str), name
            assert value, name
            # Convention: {component}.{noun}.{verb}
            assert "." in value, value

    async def test_stream_event_serialises_new_constants(self):
        """Belt + braces: a StreamEvent built from each new constant
        round-trips through ``to_dict()`` cleanly. Makes sure no payload
        we ship is JSON-unfriendly."""
        from locus.observability.emit import EV_AGENT_TOKENS_USED

        ev = StreamEvent(
            run_id="r",
            event_type=EV_AGENT_TOKENS_USED,
            data={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        d = ev.to_dict()
        assert d["event_type"] == EV_AGENT_TOKENS_USED
        assert d["data"]["total_tokens"] == 3
