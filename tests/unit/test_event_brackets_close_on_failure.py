# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Every ``started`` event gets its ``completed`` when the work in between raises.

Each site publishes ``completed`` with ``success`` false and ``error`` set to
the exception's type name before the exception reaches the caller, so a
subscriber never sees a bracket left open; the normal path carries
``success`` true.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from locus.agent.composition import LoopAgent, SequentialPipeline
from locus.core.messages import Message, Role
from locus.deepagent.workflow import make_execute_node, make_regenerate_summary_node
from locus.memory.conversation import SummarizingManager
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus
from locus.rag.embeddings.base import EmbeddingResult
from locus.rag.retriever import RAGRetriever
from locus.rag.stores.memory import InMemoryVectorStore


class _Events:
    """Collects the events a subscriber of the run saw, in order."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.seen: list[tuple[str, dict[str, Any]]] = []
        self._rid: str | None = None
        self._run: Any = None
        self._collector: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _Events:
        bus = get_event_bus()
        self._run = run_context()
        self._rid = await self._run.__aenter__()

        async def collect() -> None:
            assert self._rid is not None
            async for event in bus.subscribe(self._rid):
                if event.event_type.startswith(self.prefix):
                    self.seen.append((event.event_type, dict(event.data)))

        self._collector = asyncio.create_task(collect())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._rid is not None
        assert self._collector is not None
        await get_event_bus().close_stream(self._rid)
        await self._collector
        await self._run.__aexit__(None, None, None)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.seen]

    def only(self, name: str) -> dict[str, Any]:
        matches = [data for event, data in self.seen if event == name]
        assert len(matches) == 1, self.seen
        return matches[0]

    def nth(self, name: str, index: int) -> dict[str, Any]:
        return [data for event, data in self.seen if event == name][index]


class _Reply:
    def __init__(self, message: str) -> None:
        self.message = message


class _OkAgent:
    async def arun(self, prompt: str) -> _Reply:
        return _Reply(f"ok: {prompt[:10]}")


class _BoomAgent:
    async def arun(self, prompt: str) -> _Reply:
        raise RuntimeError("boom")


class TestCompositionBrackets:
    async def test_a_failing_stage_closes_its_bracket_before_the_pipeline_reports(self) -> None:
        pipeline = SequentialPipeline(agents=[_OkAgent(), _BoomAgent()])
        async with _Events("composition.") as events:
            result = await pipeline.run("task")

        assert result.success is False
        assert "boom" in (result.error or "")
        assert events.names == [
            "composition.stage.started",
            "composition.stage.completed",
            "composition.stage.started",
            "composition.stage.completed",
        ]
        first, second = (events.nth("composition.stage.completed", i) for i in (0, 1))
        assert first["stage"] == 0
        assert first["success"] is True
        assert "error" not in first
        assert second["stage"] == 1
        assert second["success"] is False
        assert second["error"] == "RuntimeError"
        assert second["output_length"] == 0
        assert second["duration_ms"] >= 0

    async def test_a_failing_loop_iteration_closes_its_bracket(self) -> None:
        agent = LoopAgent(agent=_BoomAgent(), max_loops=3)
        async with _Events("composition.loop.") as events:
            result = await agent.run("task")

        assert result.success is False
        assert events.names == [
            "composition.loop.iteration.started",
            "composition.loop.iteration.completed",
        ]
        completed = events.only("composition.loop.iteration.completed")
        assert completed["iteration"] == 0
        assert completed["success"] is False
        assert completed["error"] == "RuntimeError"
        assert completed["condition_met"] is False
        assert completed["duration_ms"] >= 0

    async def test_a_loop_iteration_that_returns_reports_success(self) -> None:
        agent = LoopAgent(agent=_OkAgent(), max_loops=2, condition=lambda out: True)
        async with _Events("composition.loop.") as events:
            result = await agent.run("task")

        assert result.success is True
        completed = events.only("composition.loop.iteration.completed")
        assert completed["success"] is True
        assert completed["condition_met"] is True
        assert "error" not in completed
        assert completed["duration_ms"] >= 0


class _FakeAgent:
    """Stands in for ``Agent`` inside the research nodes."""

    fail_run = False
    fail_run_sync = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def run(self, prompt: str) -> Any:
        if self.fail_run:
            raise ValueError("model down")
        return
        yield  # pragma: no cover

    def run_sync(self, prompt: str) -> _Reply:
        if self.fail_run_sync:
            raise ValueError("model down")
        return _Reply("rewritten")


class TestResearchBrackets:
    # The node factories bind ``Agent`` when they are called, so the patch
    # has to be active while the node is built as well as while it runs.

    async def test_a_failing_execute_node_closes_its_bracket(self) -> None:
        boom = type("Boom", (_FakeAgent,), {"fail_run": True})
        with patch("locus.agent.agent.Agent", boom):
            node = make_execute_node(model=object(), tools=[])
            async with _Events("research.execute.") as events:
                with pytest.raises(ValueError, match="model down"):
                    await node({"prompt": "topic"})

        assert events.names == ["research.execute.started", "research.execute.completed"]
        completed = events.only("research.execute.completed")
        assert completed == {"fact_count": 0, "success": False, "error": "ValueError"}

    async def test_a_failing_regenerate_node_closes_its_bracket(self) -> None:
        boom = type("Boom", (_FakeAgent,), {"fail_run_sync": True})
        with patch("locus.agent.agent.Agent", boom):
            node = make_regenerate_summary_node(model=object())
            async with _Events("research.regenerate.") as events:
                with pytest.raises(ValueError, match="model down"):
                    await node({"ungrounded_claims": ["c"], "evidence": ["e"], "summary": "s"})

        assert events.names == ["research.regenerate.started", "research.regenerate.completed"]
        completed = events.only("research.regenerate.completed")
        assert completed == {"regeneration": 1, "success": False, "error": "ValueError"}

    async def test_regenerate_reports_success_on_the_normal_path(self) -> None:
        with patch("locus.agent.agent.Agent", _FakeAgent):
            node = make_regenerate_summary_node(model=object())
            async with _Events("research.regenerate.") as events:
                update = await node({"ungrounded_claims": ["c"], "evidence": ["e"], "summary": "s"})

        assert update["summary"] == "rewritten"
        assert events.only("research.regenerate.completed") == {"regeneration": 1, "success": True}


class TestCompactorBracket:
    async def test_a_failing_summariser_closes_the_compactor_bracket(self) -> None:
        async def summarise(messages: list[Message]) -> str:
            raise TimeoutError("summary model timed out")

        manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
        messages = [Message(role=Role.USER, content=f"m{i}") for i in range(5)]
        async with _Events("memory.compactor.") as events:
            with pytest.raises(TimeoutError):
                await manager.async_apply(messages)

        assert events.names == ["memory.compactor.triggered", "memory.compactor.completed"]
        completed = events.only("memory.compactor.completed")
        assert completed["success"] is False
        assert completed["error"] == "TimeoutError"
        assert completed["messages_before"] == 5
        assert completed["messages_after"] == 5
        assert completed["summarized_count"] == 0
        assert completed["duration_ms"] >= 0

    async def test_a_summary_that_returns_reports_success(self) -> None:
        async def summarise(messages: list[Message]) -> str:
            return "summary"

        manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
        messages = [Message(role=Role.USER, content=f"m{i}") for i in range(5)]
        async with _Events("memory.compactor.") as events:
            result = await manager.async_apply(messages)

        assert len(result) == 2
        completed = events.only("memory.compactor.completed")
        assert completed["success"] is True
        assert completed["summarized_count"] == 4
        assert "error" not in completed


class _Embedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def embed_query(self, query: str) -> EmbeddingResult:
        if self.fail:
            raise ConnectionError("embedding endpoint unreachable")
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3], text=query, model="fake")


class TestRagBracket:
    async def test_a_failing_embedder_closes_the_query_bracket(self) -> None:
        retriever = RAGRetriever(
            embedder=_Embedder(fail=True), store=InMemoryVectorStore(dimension=3)
        )
        async with _Events("rag.query.") as events:
            with pytest.raises(ConnectionError):
                await retriever.retrieve("what is locus")

        assert events.names == ["rag.query.started", "rag.query.completed"]
        completed = events.only("rag.query.completed")
        assert completed["success"] is False
        assert completed["error"] == "ConnectionError"
        assert completed["hit_count"] == 0
        assert completed["top_score"] is None
        assert completed["store_type"] == "InMemoryVectorStore"
        assert completed["reranker_type"] is None
        assert completed["duration_ms"] >= 0

    async def test_a_failing_store_closes_the_query_bracket(self) -> None:
        class _BrokenStore(InMemoryVectorStore):
            async def search(self, *args: Any, **kwargs: Any) -> Any:
                raise OSError("store offline")

        retriever = RAGRetriever(embedder=_Embedder(), store=_BrokenStore(dimension=3))
        async with _Events("rag.query.") as events:
            with pytest.raises(OSError):
                await retriever.retrieve("what is locus")

        completed = events.only("rag.query.completed")
        assert completed["success"] is False
        assert completed["error"] == "OSError"
        assert completed["store_type"] == "_BrokenStore"

    async def test_a_query_that_returns_reports_success(self) -> None:
        retriever = RAGRetriever(embedder=_Embedder(), store=InMemoryVectorStore(dimension=3))
        async with _Events("rag.query.") as events:
            result = await retriever.retrieve("what is locus")

        assert result.total_results == 0
        completed = events.only("rag.query.completed")
        assert completed["success"] is True
        assert completed["hit_count"] == 0
        assert "error" not in completed
