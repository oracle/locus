# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIEmbeddings dimension auto-detection.

Validates that any model_id (including unknown ones) works without needing
an entry in ``MODEL_DIMENSION_HINTS`` — the real dimension is read off the
first successful embed response and cached for subsequent calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.rag.embeddings.oci import (
    DEFAULT_DIMENSION,
    MODEL_DIMENSION_HINTS,
    OCIEmbeddings,
)


def _make_embed_response(embeddings: list[list[float]]) -> Any:
    """Mimic the ``oci.generative_ai_inference`` response envelope."""
    return SimpleNamespace(data=SimpleNamespace(embeddings=embeddings))


@pytest.fixture
def embedder_with_mock_client(monkeypatch: pytest.MonkeyPatch) -> OCIEmbeddings:
    """An OCIEmbeddings instance whose underlying client is mocked."""
    e = OCIEmbeddings(
        # Synthetic id not in MODEL_DIMENSION_HINTS — verifies the
        # auto-detection fallback. Real models with dimension hints
        # (including ``cohere.embed-v4.0``) take the fast path instead.
        model_id="vendor.experimental-embed-xyz",
        compartment_id="ocid1.compartment.oc1..test",
        profile_name="DEFAULT",
        auth_type="api_key",
    )
    fake_client = MagicMock()
    e._client = fake_client
    e._oci_config_dict = {"tenancy": "ocid1.tenancy.oc1..test"}

    async def get_client() -> Any:
        return fake_client

    monkeypatch.setattr(e, "_get_client", get_client)
    return e


def test_unknown_model_defaults_before_first_call(
    embedder_with_mock_client: OCIEmbeddings,
) -> None:
    """Before any embed call, an unknown model reports the default dim."""
    e = embedder_with_mock_client
    assert e.oci_config.model_id == "vendor.experimental-embed-xyz"
    assert e.oci_config.model_id not in MODEL_DIMENSION_HINTS
    assert e._detected_dimension is None
    assert e.config.dimension == DEFAULT_DIMENSION  # 1024


def test_known_model_uses_hint_before_first_call() -> None:
    """v3 models hit the fast-path hint without needing a call."""
    e = OCIEmbeddings(
        model_id="cohere.embed-english-light-v3.0",
        compartment_id="x",
        profile_name="DEFAULT",
    )
    assert e._detected_dimension is None
    assert e.config.dimension == 384


@pytest.mark.asyncio
async def test_embed_query_caches_dimension(
    embedder_with_mock_client: OCIEmbeddings,
) -> None:
    """First ``embed_query`` updates ``_detected_dimension`` and config."""
    e = embedder_with_mock_client
    e._client.embed_text = MagicMock(return_value=_make_embed_response([[0.0] * 1536]))
    result = await e.embed_query("hello")

    assert len(result.embedding) == 1536
    assert e._detected_dimension == 1536
    assert e.config.dimension == 1536


@pytest.mark.asyncio
async def test_embed_batch_caches_dimension(
    embedder_with_mock_client: OCIEmbeddings,
) -> None:
    """First ``embed_batch`` updates the cached dimension."""
    e = embedder_with_mock_client
    e._client.embed_text = MagicMock(return_value=_make_embed_response([[0.0] * 768, [1.0] * 768]))
    results = await e.embed_batch(["a", "b"])

    assert len(results) == 2
    assert e._detected_dimension == 768
    assert e.config.dimension == 768


@pytest.mark.asyncio
async def test_dimension_cache_is_idempotent(
    embedder_with_mock_client: OCIEmbeddings,
) -> None:
    """Subsequent calls don't overwrite the first-call dimension."""
    e = embedder_with_mock_client
    # First response: dim=1536
    e._client.embed_text = MagicMock(return_value=_make_embed_response([[0.0] * 1536]))
    await e.embed_query("first")
    assert e._detected_dimension == 1536

    # Second response: spuriously dim=42 (shouldn't change the cache)
    e._client.embed_text = MagicMock(return_value=_make_embed_response([[0.0] * 42]))
    await e.embed_query("second")
    assert e._detected_dimension == 1536


@pytest.mark.asyncio
async def test_empty_response_doesnt_corrupt_cache(
    embedder_with_mock_client: OCIEmbeddings,
) -> None:
    """An empty embeddings list leaves the cache untouched."""
    e = embedder_with_mock_client
    e._client.embed_text = MagicMock(return_value=_make_embed_response([]))
    # ``embed_batch([])`` shouldn't blow up
    out = await e.embed_batch([])
    assert out == []
    assert e._detected_dimension is None
