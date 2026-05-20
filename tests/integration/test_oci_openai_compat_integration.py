# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for OCIChatCompletionsModel (V1 transport).

Parallels ``test_oci_integration.py`` but targets the
``/openai/v1/chat/completions`` endpoint. Verifies real wire behavior:

- ``complete()`` returns model output.
- ``stream()`` yields multiple content chunks before ``done`` (proves real
  SSE — the regular OCIModel only fakes streaming).
- Multi-turn conversations preserve context.

Skipped automatically if ``OCI_PROFILE`` and ``~/.oci/config`` aren't set.

Environment:
    OCI_PROFILE          OCI config profile (required to run these tests)
    OCI_REGION           Region for the inference endpoint (default: us-chicago-1)
    OCI_MODEL_ID         Model id used for tests
                         (default: meta.llama-3.3-70b-instruct)
"""

from __future__ import annotations

import os

import pytest

from locus.core.messages import Message
from locus.models.providers.oci import OCIChatCompletionsModel


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_oci,
]


DEFAULT_TEST_MODEL = "meta.llama-3.3-70b-instruct"
DEFAULT_TEST_REGION = "us-chicago-1"


def _test_model_id() -> str:
    return os.environ.get("OCI_MODEL_ID", DEFAULT_TEST_MODEL)


def _test_region() -> str:
    return os.environ.get("OCI_REGION", DEFAULT_TEST_REGION)


def _make_model(*, max_tokens: int = 64) -> OCIChatCompletionsModel:
    """Build an ``OCIChatCompletionsModel`` for integration tests via OCI profile."""
    profile = os.environ.get("OCI_PROFILE")
    if not profile:
        pytest.skip("OCI_PROFILE not set")
    model_id = _test_model_id()
    # Cohere R-series isn't supported by OCI on /openai/v1 (returns 400
    # "Unsupported OpenAI operation"). Documented limitation — skip
    # rather than fail when the suite's default test model is Cohere R.
    if model_id.lower().startswith("cohere.command-r"):
        pytest.skip(f"{model_id} is not supported on /openai/v1 — use OCIModel for Cohere R-series")
    return OCIChatCompletionsModel(
        model=model_id,
        profile=profile,
        region=_test_region(),
        max_tokens=max_tokens,
        temperature=0.0,
    )


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self):
        model = _make_model()
        response = await model.complete(
            [Message.user("Reply with exactly the word 'pong' and nothing else.")]
        )
        assert response.message is not None
        assert response.content
        assert "pong" in response.content.lower()

    @pytest.mark.asyncio
    async def test_complete_reports_usage(self):
        model = _make_model()
        response = await model.complete([Message.user("Hi")])
        # OCI may not always surface usage, but if it does, validate shape.
        if response.usage:
            assert response.usage.get("prompt_tokens", 0) > 0


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_completes_with_done(self):
        model = _make_model()
        chunks = []
        async for chunk in model.stream([Message.user("Reply with 'ok'.")]):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert chunks[-1].done is True

    @pytest.mark.asyncio
    async def test_stream_emits_multiple_content_chunks(self):
        """SSE proof: a long answer should arrive in multiple chunks.

        The regular ``OCIModel.stream()`` fakes this by chunking a finished
        response client-side. ``OCIChatCompletionsModel`` uses real openai-SDK SSE.
        Most providers stream token-by-token; some (Gemini) coalesce short
        outputs into a single chunk. The prompt below is long enough that
        all providers we support today emit at least two content chunks.
        """
        model = _make_model(max_tokens=512)
        content_chunks = 0
        async for chunk in model.stream(
            [Message.user("Count slowly from 1 to 25, putting each number on its own line.")]
        ):
            if chunk.content:
                content_chunks += 1
        assert content_chunks >= 2


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_history_preserved_via_message_list(self):
        model = _make_model(max_tokens=64)

        first = await model.complete(
            [
                Message.system("Reply briefly."),
                Message.user("Remember: my favorite color is blue."),
            ]
        )
        assert first.content

        second = await model.complete(
            [
                Message.system("Reply briefly."),
                Message.user("Remember: my favorite color is blue."),
                Message.assistant(first.content),
                Message.user("What did I say my favorite color is? One word."),
            ]
        )
        assert second.content
        assert "blue" in second.content.lower()
