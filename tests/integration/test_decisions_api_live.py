# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``DecisionsAPIDecider`` against the real Decisions API.

Two targets, each skipped unless its switch is set:

- OCI Generative AI: ``LOCUS_OCI_DECISIONS=1`` plus the usual OCI environment
  (``OCI_PROFILE`` or ``LOCUS_OCI_PROFILE``, ``OCI_REGION``, ``OCI_COMPARTMENT``);
  the model is ``openai.gpt-6-luna`` on ``/openai/v1/decisions``.
- OpenAI: ``LOCUS_OPENAI_DECISIONS=1`` and ``OPENAI_API_KEY``; the model is
  ``gpt-6-luna``.

Each call is one request for three questions about one message; the assertions
are about the shape of the answers and the one fact the text leaves no doubt
about (the customer asks for a refund), not about exact probabilities.
"""

from __future__ import annotations

import os

import pytest

from locus.decision import Choice, DecisionsAPIDecider, Score, YesNo, decide_sync


pytestmark = pytest.mark.integration

TEXT = (
    "Customer writes: the package arrived with a broken screen and I want my money "
    "back for order 8812."
)
FIELDS = [
    YesNo("damaged", "Does the customer report a damaged item?"),
    Choice(
        "intent", "What does the customer want?", ("refund", "replacement", "information", "other")
    ),
    Score("urgency", "How urgent is this message?", ("low", "medium", "high")),
]


def _check(decision) -> None:  # noqa: ANN001 - a Decision; typed by the caller
    assert set(decision) == {"damaged", "intent", "urgency"}
    assert decision["damaged"].p_yes > 0.5
    assert decision["intent"].label == "refund"
    assert decision["intent"].coverage == pytest.approx(1.0, abs=0.02)
    assert 0.0 <= decision["urgency"].expected <= 1.0
    assert sum(decision["urgency"].distribution.values()) == pytest.approx(1.0)
    assert set(decision.meta["confidence"]) == {"intent", "urgency"}
    assert decision.meta["usage"]["input_tokens"] > 0
    assert decision.provider == "decisions"


@pytest.mark.skipif(
    os.environ.get("LOCUS_OCI_DECISIONS") != "1", reason="LOCUS_OCI_DECISIONS is not 1"
)
async def test_decisions_api_on_oci() -> None:
    pytest.importorskip("oci")
    decider = DecisionsAPIDecider.on_oci()
    try:
        decision = await decider.decide(TEXT, FIELDS)
        _check(decision)
        assert decision.model == "openai.gpt-6-luna"
        # The synchronous path, one field, through the module helper.
        alone = decide_sync(decider, TEXT, [FIELDS[0]])
        assert alone["damaged"].p_yes > 0.5
    finally:
        await decider.aclose()
        decider.close()


@pytest.mark.skipif(
    os.environ.get("LOCUS_OPENAI_DECISIONS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="LOCUS_OPENAI_DECISIONS is not 1 or OPENAI_API_KEY is not set",
)
async def test_decisions_api_at_openai() -> None:
    decider = DecisionsAPIDecider(api_key=os.environ["OPENAI_API_KEY"])
    try:
        decision = await decider.decide(TEXT, FIELDS)
        _check(decision)
        assert decision.model == "gpt-6-luna"
    finally:
        await decider.aclose()
