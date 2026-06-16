# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI Generative AI Cohere reranker — closes #216.

OCI ships Cohere's rerank-v3.5 and (on-demand) rerank-v4 on the
``generative_ai_inference`` endpoint. This reranker calls
``rerank_text`` with the candidate documents, parses the resulting
``document_ranks`` list, and returns the candidates reordered by
``relevance_score``.

Usage::

    from locus.rag.reranker import CohereReranker

    reranker = CohereReranker(
        model="cohere.rerank-v3.5",  # or "cohere.rerank-v4" when GA
        compartment_id="ocid1.compartment.oc1..xxx",
        profile_name="DEFAULT",
        region="us-chicago-1",
        top_n=5,
    )
    top = await reranker.rerank("hepcidin in iron metabolism", candidates)

Plugged into a retriever::

    retriever = RAGRetriever(
        embedder=embedder,
        store=store,
        reranker=reranker,  # opt-in; ``None`` keeps semantic-only order
    )
    # ``retriever.retrieve`` over-fetches ``rerank_candidate_pool`` hits
    # (default 50), then the reranker trims to ``limit``.
    results = await retriever.retrieve("query", limit=5)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from locus.rag.reranker.base import Reranker
from locus.rag.stores.base import SearchResult


# Default model — Cohere Reranker V4 (fast variant) is GA on OCI
# Generative AI's on-demand pricing tier in us-chicago-1 as of May 2026,
# which closes #216. ``cohere.rerank-v4.0-pro`` is the larger / more
# accurate variant on the same wire; ``cohere.rerank-v3.5`` is the
# previous-generation fallback.
DEFAULT_COHERE_RERANK_MODEL = "cohere.rerank-v4.0-fast"


class CohereReranker(Reranker):
    """Reranker backed by OCI Generative AI Cohere ``rerank_text``.

    Args:
        model: OCI Cohere rerank model id. Defaults to
            ``cohere.rerank-v3.5`` (GA today). Set
            ``cohere.rerank-v4`` once the V4 on-demand wire is live in
            your region.
        compartment_id: OCI compartment OCID for the inference call.
            Required unless the OCI config profile carries one (most
            profiles do — auto-derived from the profile's tenancy when
            unset).
        profile_name: OCI config profile name. Defaults to ``DEFAULT``.
        config_file: Path to the OCI config file. Defaults to
            ``~/.oci/config``.
        service_endpoint: Override the service endpoint URL. Default is
            derived from ``region``.
        region: OCI region (default ``us-chicago-1``).
        top_n: Trim the reranked output to the top N candidates. ``None``
            returns every candidate, reordered.
        max_chunks_per_document: Pass-through to the OCI request; limits
            how many overlapping windows each document is split into for
            scoring. ``None`` lets the service pick the default.
        max_tokens_per_document: Pass-through to the OCI request;
            truncates each candidate before scoring.

    Notes:
        The OCI SDK call is sync; the reranker dispatches it to a
        threadpool via :func:`asyncio.to_thread` so callers can ``await``
        it from the same async retriever context as the embedding +
        vector-store calls.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_COHERE_RERANK_MODEL,
        compartment_id: str | None = None,
        profile_name: str = "DEFAULT",
        auth_type: str | None = None,
        config_file: str = "~/.oci/config",
        service_endpoint: str | None = None,
        region: str = "us-chicago-1",
        top_n: int | None = None,
        max_chunks_per_document: int | None = None,
        max_tokens_per_document: int | None = None,
        _client: Any = None,
    ) -> None:
        self.model = model
        self.compartment_id = compartment_id
        self.profile_name = profile_name
        # Resolve auth_type: explicit arg first, then OCI_AUTH_TYPE env
        # (matches the rest of the locus OCI surface).
        self.auth_type = auth_type or os.environ.get("OCI_AUTH_TYPE", "api_key") or "api_key"
        self.config_file = os.path.expanduser(config_file)
        self.service_endpoint = (
            service_endpoint or f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        )
        self.region = region
        self.top_n = top_n
        self.max_chunks_per_document = max_chunks_per_document
        self.max_tokens_per_document = max_tokens_per_document
        # Injection seam for unit tests — _client overrides the OCI
        # client so we don't need real OCI auth in tests.
        self._client_override = _client
        self._cached_client: Any = None

    def _build_client(self) -> Any:
        """Construct (and cache) the OCI GenAI inference client.

        Reuses :class:`locus.models.providers.oci.client.OCIClient` so
        we honour every auth mode the rest of locus already supports —
        ``api_key``, ``security_token``, ``session_token``,
        ``instance_principal``, ``resource_principal`` — without
        duplicating the signer-building plumbing here.
        """
        if self._client_override is not None:
            return self._client_override
        if self._cached_client is not None:
            return self._cached_client

        from locus.models.providers.oci.client import (  # noqa: PLC0415
            OCIAuthType,
            OCIClient,
            OCIClientConfig,
        )

        oci_client = OCIClient(
            OCIClientConfig(
                profile_name=self.profile_name,
                config_file=self.config_file,
                auth_type=OCIAuthType(self.auth_type),
                compartment_id=self.compartment_id,
                service_endpoint=self.service_endpoint,
            )
        )
        # Compartment auto-derived from the profile's tenancy if not
        # explicitly set — same convention as OCIModel / OCIChatCompletionsModel.
        if not self.compartment_id:
            self.compartment_id = oci_client.compartment_id
        self._cached_client = oci_client.client
        return self._cached_client

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """See :meth:`Reranker.rerank`."""
        if not candidates:
            return []

        # OCI SDK calls are blocking; run them off the event loop.
        ranks = await asyncio.to_thread(self._call_rerank_sync, query, candidates)
        return self._apply_ranks(candidates, ranks)

    def _call_rerank_sync(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[tuple[int, float]]:
        """Make the synchronous OCI call. Returns a list of
        ``(original_index, relevance_score)`` tuples in descending score
        order."""
        from oci.generative_ai_inference.models import (  # noqa: PLC0415
            OnDemandServingMode,
            RerankTextDetails,
        )

        client = self._build_client()
        documents = [c.document.content or "" for c in candidates]
        request = RerankTextDetails(
            input=query,
            documents=documents,
            serving_mode=OnDemandServingMode(model_id=self.model),
            compartment_id=self.compartment_id or "",
            top_n=self.top_n if self.top_n is not None else len(candidates),
            max_chunks_per_document=self.max_chunks_per_document,
            max_tokens_per_document=self.max_tokens_per_document,
            is_echo=False,
        )
        response = client.rerank_text(rerank_text_details=request)
        # response.data → RerankTextResult{document_ranks: [RerankDocumentRank{index, relevance_score}, ...]}
        return [(rank.index, float(rank.relevance_score)) for rank in response.data.document_ranks]

    @staticmethod
    def _apply_ranks(
        candidates: list[SearchResult],
        ranks: list[tuple[int, float]],
    ) -> list[SearchResult]:
        """Rebuild the candidate list in reranked order.

        Preserves the original embedding score on ``.distance`` so
        callers comparing semantic vs reranker scores can still see both.
        """
        reordered: list[SearchResult] = []
        for original_idx, score in ranks:
            if original_idx < 0 or original_idx >= len(candidates):
                # OCI returned an out-of-range index — skip rather than
                # blow up. Surfaces in observability if it ever fires.
                continue
            original = candidates[original_idx]
            reordered.append(
                SearchResult(
                    document=original.document,
                    score=score,
                    distance=original.score if original.distance is None else original.distance,
                )
            )
        return reordered
