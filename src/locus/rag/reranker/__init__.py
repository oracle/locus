# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Rerankers — reorder retriever candidates by relevance to a query.

The retrieve-then-rerank pattern materially improves answer grounding in
RAG: pull a wide candidate set (top-K) from the vector store cheaply,
then have a *cross-encoder* reranker score each candidate against the
query before the top-N hits the LLM. The reranker sees both query and
candidate together, so it catches relevance signals an embedding-only
score misses.

This subpackage adds:

  * :class:`Reranker` — abstract base every reranker implements.
  * :class:`CohereReranker` — OCI Generative AI Cohere rerank-v3.5 (and,
    once GA on the on-demand wire, rerank-v4) implementation.

Closes #216. Wire one into a retriever with ``RAGRetriever(reranker=...)``
to swap the default semantic-only ordering for a reranked one.
"""

from locus.rag.reranker.base import Reranker
from locus.rag.reranker.cohere_oci import CohereReranker


__all__ = ["CohereReranker", "Reranker"]
