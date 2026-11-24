# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Five events that were declared and never emitted now reach a subscriber.

``multiagent.graph.node.routed`` reports each dynamic routing decision of a
``StateGraph`` (a conditional edge or a ``Command`` with ``goto``),
``a2a.task.processing`` each time an A2A task turns ``working``, and
``rag.embedding.generated`` / ``rag.store.search`` / ``rag.store.upsert`` the
embedder and store calls of every ``RAGRetriever`` query and ingest path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from locus.a2a import A2AClient, A2AClientError, A2AServer, Message, TaskState, TextPart
from locus.agent import Agent
from locus.core.command import Command
from locus.core.events import ThinkEvent
from locus.core.messages import Role
from locus.multiagent.graph import END, START, StateGraph
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus
from locus.rag.embeddings.base import EmbeddingResult
from locus.rag.multimodal import ContentType, ProcessedContent
from locus.rag.retriever import RAGRetriever
from locus.rag.stores.memory import InMemoryVectorStore
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


class _Events:
    """Collects the events a subscriber of the run saw, in order."""

    def __init__(self, *prefixes: str) -> None:
        self.prefixes = prefixes
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
                if event.event_type.startswith(self.prefixes):
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

    def all(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.seen if event == name]

    def only(self, name: str) -> dict[str, Any]:
        matches = self.all(name)
        assert len(matches) == 1, self.seen
        return matches[0]


# ---------------------------------------------------------------------------
# multiagent.graph.node.routed
# ---------------------------------------------------------------------------

ROUTED = "multiagent.graph.node.routed"
STARTED = "multiagent.graph.node.started"


def _two_branch_graph(router: Any, targets: dict[str, str] | None = None) -> StateGraph:
    graph = StateGraph()
    graph.add_node("classify", lambda inputs: {"kind": inputs.get("kind", "refund")})
    graph.add_node("refund", lambda inputs: {"handled_by": "refund"})
    graph.add_node("escalate", lambda inputs: {"handled_by": "escalate"})
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", router, targets)
    graph.add_edge("refund", END)
    graph.add_edge("escalate", END)
    return graph


class TestGraphNodeRouted:
    @pytest.mark.parametrize("kind", ["refund", "escalate"])
    async def test_a_conditional_edge_reports_the_branch_it_picked(self, kind: str) -> None:
        graph = _two_branch_graph(lambda state: state["kind"])
        async with _Events("multiagent.graph.node.") as events:
            result = await graph.execute({"kind": kind})

        assert result.success
        routed = events.only(ROUTED)
        assert routed == {
            "graph_id": graph.id,
            "node_id": "classify",
            "targets": [kind],
            "iteration": 1,
            "source": "conditional_edge",
        }
        # The decision is published before the node it names starts.
        started_after = [
            data["node_id"]
            for name, data in events.seen[events.names.index(ROUTED) + 1 :]
            if name == STARTED
        ]
        assert started_after == [kind]

    async def test_targets_are_the_resolved_node_ids_not_the_router_return(self) -> None:
        graph = _two_branch_graph(
            lambda state: {"decision": state["kind"]}["decision"],
            {"refund": "refund", "other": "escalate"},
        )
        async with _Events(ROUTED) as events:
            await graph.execute({"kind": "other"})

        assert events.only(ROUTED)["targets"] == ["escalate"]

    async def test_a_router_that_ends_the_run_reports_the_end_marker(self) -> None:
        graph = StateGraph()
        graph.add_node("check", lambda inputs: {"done": True})
        graph.add_node("more", lambda inputs: {"done": True})
        graph.add_edge(START, "check")
        graph.add_conditional_edges("check", lambda state: END if state["done"] else "more")
        graph.add_edge("more", END)
        async with _Events(ROUTED) as events:
            await graph.execute({})

        routed = events.only(ROUTED)
        assert routed["targets"] == [END]
        assert routed["source"] == "conditional_edge"

    async def test_a_command_goto_reports_source_command(self) -> None:
        graph = StateGraph()
        graph.add_node("triage", lambda inputs: Command(update={"seen": True}, goto="audit"))
        graph.add_node("audit", lambda inputs: {"audited": True})
        graph.add_node("skipped", lambda inputs: {"skipped": True})
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "skipped")
        graph.add_edge("audit", END)
        graph.add_edge("skipped", END)
        async with _Events("multiagent.graph.node.") as events:
            result = await graph.execute({})

        assert result.execution_order == ["triage", "audit"]
        routed = events.only(ROUTED)
        assert routed["node_id"] == "triage"
        assert routed["targets"] == ["audit"]
        assert routed["source"] == "command"
        assert set(routed) == {"graph_id", "node_id", "targets", "iteration", "source"}

    async def test_a_command_wins_over_a_conditional_edge_on_the_same_node(self) -> None:
        graph = StateGraph()
        graph.add_node("triage", lambda inputs: Command(goto="audit"))
        graph.add_node("audit", lambda inputs: {"audited": True})
        graph.add_node("other", lambda inputs: {"other": True})
        graph.add_edge(START, "triage")
        graph.add_conditional_edges("triage", lambda state: "other")
        graph.add_edge("audit", END)
        graph.add_edge("other", END)
        async with _Events(ROUTED) as events:
            await graph.execute({})

        routed = events.only(ROUTED)
        assert routed["source"] == "command"
        assert routed["targets"] == ["audit"]

    async def test_regular_edges_emit_nothing(self) -> None:
        graph = StateGraph()
        graph.add_node("first", lambda inputs: {"a": 1})
        graph.add_node("second", lambda inputs: {"b": 2})
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        async with _Events("multiagent.graph.node.") as events:
            result = await graph.execute({})

        assert result.execution_order == ["first", "second"]
        assert events.all(ROUTED) == []
        assert events.names.count(STARTED) == 2

    async def test_every_pass_of_a_loop_reports_its_own_decision(self) -> None:
        graph = StateGraph()
        graph.add_node("step", lambda inputs: {"count": inputs.get("count", 0) + 1})
        graph.add_edge(START, "step")
        graph.add_conditional_edges("step", lambda state: "step" if state["count"] < 3 else END)
        async with _Events(ROUTED) as events:
            await graph.execute({})

        decisions = events.all(ROUTED)
        assert [d["targets"] for d in decisions] == [["step"], ["step"], [END]]
        assert [d["iteration"] for d in decisions] == [1, 2, 3]


# ---------------------------------------------------------------------------
# a2a.task.processing
# ---------------------------------------------------------------------------

PROCESSING = "a2a.task.processing"


@tool
def ask_region(question: str) -> str:
    """Pause the run for a person: the interrupt marker any tool may return."""
    return json.dumps({"__interrupt__": True, "question": question, "options": None})


def _answering_agent() -> Agent:
    return Agent(
        model=FunctionModel(lambda messages, tools: text("done")),
        reflexion=False,
        grounding=False,
    )


def _question_agent() -> Agent:
    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        answers = [m.content for m in messages if m.role == Role.TOOL]
        if answers:
            return text(f"Deployed to {answers[-1]}.")
        return tool_call("ask_region", question="Which region?")

    return Agent(model=FunctionModel(handler), tools=[ask_region], reflexion=False, grounding=False)


class _BoomAgent:
    async def run(self, prompt: str) -> Any:
        raise RuntimeError("boom")
        yield  # pragma: no cover


class _CancelledMidRun:
    """Asks the server to cancel its own task, then yields so the run notices."""

    def __init__(self) -> None:
        self.server: A2AServer | None = None

    async def run(self, prompt: str, thread_id: str | None = None) -> Any:
        assert self.server is not None
        assert thread_id is not None
        assert self.server._store.cancel(thread_id)
        yield ThinkEvent(iteration=0, reasoning="still going", tool_calls=[])


def _http(server: A2AServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False),
        base_url="http://testserver",
    )


def _message(content: str, *, task_id: str | None = None) -> Message:
    return Message(
        role="user",
        parts=[TextPart(text=content)],
        messageId=f"m-{content[:8]}",
        taskId=task_id,
    )


class TestA2ATaskProcessing:
    async def test_a_task_reports_when_it_starts_working(self) -> None:
        server = A2AServer(agent=_answering_agent(), allow_unauthenticated=True)
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("hello"))

        assert task.status.state == TaskState.completed
        assert events.names == ["a2a.task.received", PROCESSING, "a2a.task.completed"]
        assert events.only(PROCESSING) == {
            "task_id": task.id,
            "context_id": task.contextId,
            "resumed": False,
        }

    async def test_the_http_json_binding_reports_it_too(self) -> None:
        server = A2AServer(agent=_answering_agent(), allow_unauthenticated=True)
        async with _Events(PROCESSING) as events, _http(server) as http:
            client = A2AClient("http://testserver", binding="http+json", http_client=http)
            task = await client.send_message(_message("hello"))

        assert events.only(PROCESSING) == {
            "task_id": task.id,
            "context_id": task.contextId,
            "resumed": False,
        }

    async def test_reasoning_updates_do_not_emit_again(self) -> None:
        class _Thinker:
            async def run(self, prompt: str) -> Any:
                for i in range(3):
                    yield ThinkEvent(iteration=i, reasoning=f"thought {i}", tool_calls=[])

        server = A2AServer(agent=_Thinker(), allow_unauthenticated=True)
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            await client.send_message(_message("think"))

        assert events.names.count(PROCESSING) == 1

    async def test_a_resumed_task_reports_a_second_transition(self) -> None:
        server = A2AServer(agent=_question_agent(), allow_unauthenticated=True)
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            paused = await client.send_message(_message("deploy the service"))
            assert paused.status.state == TaskState.input_required
            resumed = await client.send_message(_message("eu-frankfurt-1", task_id=paused.id))

        assert resumed.status.state == TaskState.completed
        first, second = events.all(PROCESSING)
        assert first == {"task_id": paused.id, "context_id": paused.contextId, "resumed": False}
        assert second == {"task_id": paused.id, "context_id": paused.contextId, "resumed": True}
        assert events.names == [
            "a2a.task.received",
            PROCESSING,
            "a2a.task.completed",
            "a2a.task.received",
            PROCESSING,
            "a2a.task.completed",
        ]

    async def test_a_failed_task_reports_processing_before_it_fails(self) -> None:
        server = A2AServer(agent=_BoomAgent(), allow_unauthenticated=True)
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            with pytest.raises(A2AClientError):
                await client.send_message(_message("hello"))

        assert events.names == ["a2a.task.received", PROCESSING, "a2a.task.completed"]
        assert events.only(PROCESSING)["resumed"] is False
        assert events.only("a2a.task.completed")["success"] is False
        failed = server._store.get(events.only(PROCESSING)["task_id"])
        assert failed is not None
        assert failed.status.state == TaskState.failed

    async def test_a_cancelled_task_reports_processing_once(self) -> None:
        agent = _CancelledMidRun()
        server = A2AServer(agent=agent, allow_unauthenticated=True)
        agent.server = server
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("hello"))

        assert task.status.state == TaskState.canceled
        assert events.names == ["a2a.task.received", PROCESSING, "a2a.task.completed"]
        assert events.only(PROCESSING)["task_id"] == task.id

    async def test_a_streamed_task_reports_processing(self) -> None:
        server = A2AServer(agent=_answering_agent(), allow_unauthenticated=True)
        async with _Events("a2a.task.") as events, _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            frames = [f async for f in client.send_message_streaming(_message("hello"))]

        processing = events.only(PROCESSING)
        assert processing["task_id"] == frames[0]["id"]
        assert processing["resumed"] is False
        assert events.names.index("a2a.task.received") < events.names.index(PROCESSING)


# ---------------------------------------------------------------------------
# rag.embedding.generated, rag.store.search, rag.store.upsert
# ---------------------------------------------------------------------------

EMBEDDED = "rag.embedding.generated"
SEARCH = "rag.store.search"
UPSERT = "rag.store.upsert"


class _Embedder:
    """Three-dimensional vectors, one per text; counts what it was asked for."""

    def _result(self, value: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3], text=value, model="fake")

    async def embed(self, value: str) -> EmbeddingResult:
        return self._result(value)

    async def embed_query(self, query: str) -> EmbeddingResult:
        return self._result(query)

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self._result(t) for t in texts]


class _HybridStore(InMemoryVectorStore):
    """An in-memory store that also answers the blended search."""

    async def hybrid_search(
        self, query_text: str, query_embedding: list[float], **kwargs: Any
    ) -> list[Any]:
        return await self.search(
            query_embedding=query_embedding,
            limit=kwargs["limit"],
            threshold=kwargs.get("threshold"),
            metadata_filter=kwargs.get("metadata_filter"),
        )


def _retriever(**kwargs: Any) -> RAGRetriever:
    kwargs.setdefault("store", InMemoryVectorStore(dimension=3))
    return RAGRetriever(embedder=_Embedder(), **kwargs)


async def _processed(*args: Any, **kwargs: Any) -> ProcessedContent:
    return ProcessedContent(text="a diagram of the pipeline", content_type=ContentType.IMAGE)


class TestRagQueryEvents:
    async def test_a_query_reports_the_embedding_and_the_search_inside_its_bracket(self) -> None:
        retriever = _retriever()
        await retriever.add_document("Locus is an agent SDK.", doc_id="d1")
        async with _Events("rag.") as events:
            result = await retriever.retrieve("what is locus", limit=4)

        assert events.names == ["rag.query.started", EMBEDDED, SEARCH, "rag.query.completed"]
        assert events.only(EMBEDDED) == {
            "embedder_type": "_Embedder",
            "count": 1,
            "dimension": 3,
            "purpose": "query",
        }
        assert events.only(SEARCH) == {
            "store_type": "InMemoryVectorStore",
            "limit": 4,
            "hit_count": len(result.documents),
            "hybrid": False,
        }
        assert result.documents

    async def test_hybrid_is_true_only_when_the_blended_search_ran(self) -> None:
        blended = _retriever(store=_HybridStore(dimension=3), hybrid=True)
        fallback = _retriever(hybrid=True)
        for retriever in (blended, fallback):
            await retriever.add_document("Locus is an agent SDK.")
        async with _Events(SEARCH) as events:
            await blended.retrieve("locus")
            await blended.retrieve("locus", hybrid=False)
            await fallback.retrieve("locus")

        assert [data["hybrid"] for data in events.all(SEARCH)] == [True, False, False]
        assert [data["store_type"] for data in events.all(SEARCH)] == [
            "_HybridStore",
            "_HybridStore",
            "InMemoryVectorStore",
        ]

    async def test_a_failing_search_reports_the_embedding_and_no_search(self) -> None:
        class _BrokenStore(InMemoryVectorStore):
            async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
                raise ConnectionError("store unreachable")

        retriever = _retriever(store=_BrokenStore(dimension=3))
        async with _Events("rag.") as events:
            with pytest.raises(ConnectionError):
                await retriever.retrieve("locus")

        assert events.names == ["rag.query.started", EMBEDDED, "rag.query.completed"]


class TestRagIngestEvents:
    async def test_a_chunked_document_reports_one_embedding_call_and_one_upsert(self) -> None:
        retriever = _retriever(chunk_size=40, chunk_overlap=0)
        content = "\n\n".join(f"Paragraph {i} about the harness." for i in range(6))
        async with _Events("rag.") as events:
            ids = await retriever.add_document(content, doc_id="guide")

        assert len(ids) > 1
        assert events.names == [EMBEDDED, UPSERT]
        assert events.only(EMBEDDED) == {
            "embedder_type": "_Embedder",
            "count": len(ids),
            "dimension": 3,
            "purpose": "documents",
        }
        assert events.only(UPSERT) == {
            "store_type": "InMemoryVectorStore",
            "doc_id": "guide",
            "chunk_count": len(ids),
        }

    async def test_a_short_document_is_one_chunk(self) -> None:
        retriever = _retriever()
        async with _Events("rag.") as events:
            ids = await retriever.add_document("short")

        assert events.only(EMBEDDED)["count"] == 1
        assert events.only(UPSERT) == {
            "store_type": "InMemoryVectorStore",
            "doc_id": ids[0],
            "chunk_count": 1,
        }

    async def test_add_documents_reports_each_document(self) -> None:
        retriever = _retriever()
        async with _Events("rag.") as events:
            await retriever.add_documents(["one", "two", "three"])

        assert events.names == [EMBEDDED, UPSERT] * 3

    async def test_add_file_reports_the_embedding_and_the_upsert(self, tmp_path: Path) -> None:
        note = tmp_path / "note.txt"
        note.write_text("Locus keeps an audit trail of every admitted action.", encoding="utf-8")
        retriever = _retriever()
        async with _Events("rag.") as events:
            ids = await retriever.add_file(note, doc_id="note")

        assert ids == ["note"]
        assert events.names == [EMBEDDED, UPSERT]
        assert events.only(EMBEDDED)["purpose"] == "documents"
        assert events.only(UPSERT) == {
            "store_type": "InMemoryVectorStore",
            "doc_id": "note",
            "chunk_count": 1,
        }

    async def test_add_image_reports_the_embedding_and_the_upsert(self) -> None:
        retriever = _retriever()
        with patch("locus.rag.multimodal.ImageProcessor.process", _processed):
            async with _Events("rag.") as events:
                doc_id = await retriever.add_image(b"\x89PNG", doc_id="diagram")

        assert doc_id == "diagram"
        assert events.names == [EMBEDDED, UPSERT]
        assert events.only(EMBEDDED) == {
            "embedder_type": "_Embedder",
            "count": 1,
            "dimension": 3,
            "purpose": "documents",
        }
        assert events.only(UPSERT) == {
            "store_type": "InMemoryVectorStore",
            "doc_id": "diagram",
            "chunk_count": 1,
        }

    async def test_add_audio_reports_the_embedding_and_the_upsert(self) -> None:
        retriever = _retriever()
        with patch("locus.rag.multimodal.AudioProcessor.process", _processed):
            async with _Events("rag.") as events:
                doc_id = await retriever.add_audio(b"RIFF", doc_id="call")

        assert events.names == [EMBEDDED, UPSERT]
        assert events.only(UPSERT)["doc_id"] == doc_id

    async def test_add_pdf_reports_through_add_document(self) -> None:
        retriever = _retriever()
        with patch("locus.rag.multimodal.PDFProcessor.process", _processed):
            async with _Events("rag.") as events:
                await retriever.add_pdf(b"%PDF-1.7", doc_id="manual")

        assert events.names == [EMBEDDED, UPSERT]
        assert events.only(UPSERT)["doc_id"] == "manual"

    async def test_no_text_content_is_on_any_of_the_three(self) -> None:
        retriever = _retriever()
        async with _Events("rag.embedding.", "rag.store.") as events:
            await retriever.add_document("a private sentence", doc_id="p")
            await retriever.retrieve("a private question")

        for _, data in events.seen:
            assert all(not isinstance(v, str) or "private" not in v for v in data.values())
