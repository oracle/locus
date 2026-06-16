# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``create_deepagent(datastores=...)``.

Verifies the tool auto-wiring and system-prompt injection without spinning
up a model — uses ``model="mock:test"`` and inspects the constructed
``Agent`` object's tools and prompt.
"""

from __future__ import annotations

import pytest

from locus.deepagent.factory import create_deepagent


class _StubRetriever:
    """Minimal RAGRetriever surrogate so ``isinstance`` and ``create_rag_tool``
    accept it. We don't exercise actual retrieval here."""

    def __init__(self) -> None:
        self.embedder = None
        self.store = None

    async def retrieve(self, *args, **kwargs):  # pragma: no cover - never called
        from locus.rag.retriever import RetrievalResult

        return RetrievalResult(query="", documents=[], total_results=0)


def _make_stub_retriever() -> object:
    """Build a real RAGRetriever with stub embedder + store.

    Going through the actual constructor keeps the ``isinstance`` check in
    ``create_deepagent`` honest — we shouldn't have to hack around it.
    """
    from locus.rag.embeddings.base import BaseEmbedding, EmbeddingConfig, EmbeddingResult
    from locus.rag.retriever import RAGRetriever
    from locus.rag.stores.memory import InMemoryVectorStore

    class StubEmbedder(BaseEmbedding):
        @property
        def config(self) -> EmbeddingConfig:
            return EmbeddingConfig(dimension=8, max_tokens=1, batch_size=1)

        @property
        def capabilities(self):  # type: ignore[override]
            from locus.rag.embeddings.base import EmbeddingCapabilities

            return EmbeddingCapabilities()

        async def embed(self, text: str) -> EmbeddingResult:
            return EmbeddingResult(embedding=[0.0] * 8, text=text, model="stub")

        async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
            return [await self.embed(t) for t in texts]

        async def embed_query(self, query: str) -> EmbeddingResult:
            return await self.embed(query)

        async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
            return await self.embed_batch(documents)

    return RAGRetriever(embedder=StubEmbedder(), store=InMemoryVectorStore(dimension=8))


def _build_agent(monkeypatch: pytest.MonkeyPatch, **overrides):
    """Build a deepagent against a stub OCI model registration.

    Mirrors the helper used in ``test_deepagent.py``: a fake ``OCI_PROFILE``
    env var lets the registry resolve ``oci:`` model strings without
    actually opening an OCI client (the client is lazy and never used in
    these structural tests).
    """
    monkeypatch.setenv("OCI_PROFILE", "DEFAULT")
    base = {
        "model": "oci:openai.gpt-4o-mini",
        "system_prompt": "Answer questions.",
        "tools": [],
        "reflexion": False,
        "grounding": False,
        "max_iterations": 1,
    }
    base.update(overrides)
    return create_deepagent(**base)


def test_no_datastores_leaves_prompt_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``datastores`` is not set, the prompt is the original."""
    agent = _build_agent(monkeypatch)
    assert "Datastores" not in agent.config.system_prompt


def test_single_datastore_auto_wires_tool_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single retriever produces a ``search_<name>`` tool + prompt block."""
    retriever = _make_stub_retriever()
    agent = _build_agent(
        monkeypatch,
        datastores={
            "medical": {
                "retriever": retriever,
                "description": "iron metabolism, anemia",
                "top_k": 3,
            },
        },
    )

    tool_names = {t.name for t in agent.config.tools}
    assert "search_medical" in tool_names
    # Prompt routing block lists the auto-wired tool with its description
    sp = agent.config.system_prompt
    assert "# Datastores" in sp
    assert "search_medical" in sp
    assert "iron metabolism, anemia" in sp


def test_bare_retriever_value_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing the retriever directly (no dict wrapper) also works."""
    retriever = _make_stub_retriever()
    agent = _build_agent(monkeypatch, datastores={"docs": retriever})
    names = {t.name for t in agent.config.tools}
    assert "search_docs" in names


def test_multiple_datastores_get_distinct_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each entry produces its own namespaced tool."""
    agent = _build_agent(
        monkeypatch,
        datastores={
            "medical": {"retriever": _make_stub_retriever(), "description": "med"},
            "legal": {"retriever": _make_stub_retriever(), "description": "law"},
        },
    )
    names = {t.name for t in agent.config.tools}
    assert {"search_medical", "search_legal"} <= names


def test_invalid_datastore_value_raises_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-RAGRetriever, non-dict value is rejected."""
    with pytest.raises(TypeError, match="must be a RAGRetriever"):
        _build_agent(monkeypatch, datastores={"bad": object()})


def test_max_output_tokens_lands_on_agent_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_output_tokens`` is forwarded to AgentConfig.max_tokens for the
    per-completion cap (different from ``max_tokens`` which is the
    total-run termination budget)."""
    agent = _build_agent(monkeypatch, max_output_tokens=8192)
    assert agent.config.max_tokens == 8192


def test_max_output_tokens_default_leaves_agent_default_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``max_output_tokens`` is omitted, AgentConfig keeps its default."""
    agent = _build_agent(monkeypatch)
    # Default is the AgentConfig default (4096 at time of writing). We just
    # assert we *didn't* clobber it with our 80k termination budget.
    assert agent.config.max_tokens != 80_000
