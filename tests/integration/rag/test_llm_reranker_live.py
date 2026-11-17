# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live test for ``LLMReranker`` on an OCI Generative AI chat model.

Gated on ``OCI_PROFILE``. ``LOCUS_RERANK_MODEL`` picks the model
(``oci:openai.gpt-4.1`` by default); ``OCI_REGION`` / ``OCI_COMPARTMENT_ID``
are read by the OCI provider as usual.
"""

from __future__ import annotations

import os

import pytest

from locus.rag.reranker.llm import LLMReranker
from locus.rag.stores.base import Document, SearchResult


pytestmark = [pytest.mark.integration, pytest.mark.requires_oci]

CANDIDATES = [
    (
        "ach_return",
        "ACH return R01 (insufficient funds): the bank account had no balance to "
        "cover the debit. Reattempt once after 2 business days.",
    ),
    (
        "refund_sla",
        "Refund SLA: card refunds settle in 5-10 business days; the funds leave the "
        "merchant account immediately but appear on the statement after the issuer "
        "posts them.",
    ),
    (
        "chargeback_4853",
        "Reason code 4853 (cardholder dispute): merchandise not received. Request "
        "proof of delivery and tracking; represent within 30 days.",
    ),
    (
        "decline_51",
        "Decline code 51 (insufficient funds): the issuer rejected the auth for a low "
        "balance. Safe to retry after payday; do not force-post.",
    ),
    ("onboarding", "New merchant onboarding checklist: KYC documents, bank letter, MCC."),
]


def _candidates() -> list[SearchResult]:
    # Deliberately give the refund passage the *worst* vector score so the
    # reranker, not the retriever order, has to surface it.
    scores = [0.9, 0.2, 0.8, 0.7, 0.6]
    return [
        SearchResult(document=Document(id=doc_id, content=text), score=score)
        for (doc_id, text), score in zip(CANDIDATES, scores, strict=True)
    ]


async def test_llm_reranker_surfaces_the_relevant_passage() -> None:
    if not os.environ.get("OCI_PROFILE"):
        pytest.skip("OCI_PROFILE not set")
    reranker = LLMReranker(os.environ.get("LOCUS_RERANK_MODEL", "oci:openai.gpt-4.1"), top_n=3)

    ranked = await reranker.rerank(
        "how long until a customer sees their card refund on the statement?",
        _candidates(),
    )

    ids = [r.document.id for r in ranked]
    assert len(ranked) == 3
    assert ids[0] == "refund_sla", ids
    assert ranked[0].score > ranked[-1].score
    assert ranked[0].distance == 0.2  # the retriever's score is preserved
    assert "onboarding" not in ids
