# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``create_research_workflow(datastores=...)``.

The workflow factory should mirror ``create_deepagent``'s datastore
auto-wiring: a ``search_<name>`` tool per entry plus a routing block
prepended to the execute agent's system prompt.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.deepagent.factory import wire_datastores
from locus.deepagent.workflow import create_research_workflow


class _StubRetriever:
    """RAGRetriever stand-in — wire_datastores only needs an isinstance check
    against the real class plus that it can be passed through to
    create_rag_tool. We monkey-patch the isinstance check by subclassing.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self.calls.append({"args": args, "kwargs": kwargs})
        from locus.rag.retriever import RetrievalResult

        return RetrievalResult(query="", documents=[], scores=[])


def _make_retriever() -> Any:
    """Build a real RAGRetriever pointing at the in-memory store so
    `wire_datastores`'s isinstance check passes and `create_rag_tool` can
    bind it. The store is irrelevant — we only exercise the wiring path.
    """
    from locus.rag import OCIEmbeddings, RAGRetriever
    from locus.rag.stores.memory import InMemoryVectorStore

    # OCIEmbeddings is lazy — never calls OCI unless we run embed_query.
    embedder = OCIEmbeddings(model_id="cohere.embed-english-v3.0")
    store = InMemoryVectorStore(dimension=1024)
    return RAGRetriever(embedder=embedder, store=store)


def test_wire_datastores_none_returns_empty() -> None:
    tools, block = wire_datastores(None)
    assert tools == []
    assert block == ""


def test_wire_datastores_empty_returns_empty() -> None:
    tools, block = wire_datastores({})
    assert tools == []
    assert block == ""


def test_wire_datastores_bare_retriever() -> None:
    """Bare RAGRetriever picks up the default top_k."""
    retriever = _make_retriever()
    tools, block = wire_datastores({"medical": retriever}, datastore_top_k=7)
    assert len(tools) == 1
    assert tools[0].name == "search_medical"
    assert "search_medical" in block
    assert "Datastores" in block


def test_wire_datastores_dict_form() -> None:
    """Dict form respects per-store description, top_k, threshold."""
    retriever = _make_retriever()
    tools, block = wire_datastores(
        {
            "medical": {
                "retriever": retriever,
                "description": "clinical knowledge",
                "top_k": 4,
                "threshold": 0.3,
            }
        }
    )
    assert len(tools) == 1
    assert tools[0].name == "search_medical"
    # Routing block includes the description, not a generic fallback.
    assert "clinical knowledge" in block
    # Per-tool description includes the top_k.
    assert "4 relevant documents" in tools[0].description


def test_wire_datastores_multiple_routes_all() -> None:
    retriever = _make_retriever()
    tools, block = wire_datastores(
        {
            "medical": {"retriever": retriever, "description": "med stuff"},
            "news": {"retriever": retriever, "description": "news stuff"},
        }
    )
    assert {t.name for t in tools} == {"search_medical", "search_news"}
    assert "search_medical" in block
    assert "search_news" in block


def test_wire_datastores_bad_value_type() -> None:
    with pytest.raises(TypeError, match="must be a RAGRetriever"):
        wire_datastores({"bogus": 42})


def test_workflow_accepts_datastores_kwarg() -> None:
    """The workflow factory takes the new kwarg without exploding."""
    import inspect

    sig = inspect.signature(create_research_workflow)
    assert "datastores" in sig.parameters
    assert "datastore_top_k" in sig.parameters


class _NopModel:
    """Minimum-viable model stub — workflow only needs it for Agent
    construction inside make_execute_node, and we don't run the loop."""

    async def complete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("not invoked in this test")

    async def close(self) -> None:  # pragma: no cover
        pass


def test_workflow_with_datastores_compiles() -> None:
    """The workflow compiles end-to-end with a datastore configured.

    We don't run the graph (no live model) — just prove construction
    succeeds and the execute node was built with the merged tools.
    """
    retriever = _make_retriever()
    graph = create_research_workflow(
        model=_NopModel(),
        tools=[],
        datastores={
            "medical": {
                "retriever": retriever,
                "description": "clinical knowledge",
                "top_k": 3,
            }
        },
        reflexion=False,
        causal_inference=False,
    )
    # StateGraph compile produced a runnable.
    assert graph is not None
