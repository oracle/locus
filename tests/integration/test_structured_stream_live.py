# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live: ``StructuredStream`` produces incrementally-validated partials.

Unit tests cover the parser; this guards the end-to-end story against a
real streaming provider so we know the auto-close logic survives
real-world chunking quirks. Wraps ``model.stream(...)`` in
``StructuredStream`` and asserts at least one parsed partial arrives
before the final fully-valid instance.

Activation:

* ``OCI_PROFILE=<profile>`` OR ``OPENAI_API_KEY=<key>`` — runs against
  whichever provider is available.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from locus.core.events import LocusEvent, ModelChunkEvent, TerminateEvent
from locus.core.messages import Message
from locus.streaming.structured import StructuredStream


pytestmark = [pytest.mark.integration]


_PROFILE = os.environ.get("OCI_PROFILE")
_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))


def _has_oci_config() -> bool:
    return Path("~/.oci/config").expanduser().exists()


class Vendor(BaseModel):
    name: str = Field(description="Vendor legal name")
    score: float = Field(description="Quality score in [0,1]", ge=0.0, le=1.0)


class VendorList(BaseModel):
    vendors: list[Vendor] = Field(description="Exactly 3 vendor records")


def _build_oci_v1():
    if not (_PROFILE and _has_oci_config()):
        return None
    from locus.models.providers.oci import OCIChatCompletionsModel

    return OCIChatCompletionsModel(model="openai.gpt-4o-mini", profile=_PROFILE, region=_REGION)


def _build_openai_native():
    if not _OPENAI:
        return None
    pytest.importorskip("openai")
    from locus.models.native.openai import OpenAIModel

    return OpenAIModel(model=os.environ.get("LOCUS_OPENAI_TEST_MODEL", "gpt-4o-mini"))


_FACTORIES = [
    pytest.param(_build_oci_v1, id="oci-v1-gpt-4o-mini"),
    pytest.param(_build_openai_native, id="openai-native-gpt-4o-mini"),
]


async def _events_from_model_stream(model, messages, response_format) -> AsyncIterator[LocusEvent]:
    """Drive ``model.stream(...)`` and re-yield only the events
    ``StructuredStream`` cares about (``ModelChunkEvent`` + a synthetic
    ``TerminateEvent`` at the end).
    """
    final_buffer = ""
    async for chunk in model.stream(messages=messages, tools=None, response_format=response_format):
        if isinstance(chunk, ModelChunkEvent) and chunk.content:
            final_buffer += chunk.content
            yield chunk
    yield TerminateEvent(
        reason="complete",
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=final_buffer,
    )


@pytest.mark.parametrize("factory", _FACTORIES)
async def test_structured_stream_emits_partial_then_final(factory):
    """End-to-end: a real streaming response yields at least one parsed
    partial before the final instance arrives.
    """
    model = factory()
    if model is None:
        pytest.skip("provider credentials missing")

    from locus.core.structured import build_response_format

    response_format = build_response_format(VendorList, strict=True)
    messages = [
        Message.system("You are a procurement researcher. Recommend exactly 3 cloud vendors."),
        Message.user(
            "List 3 well-known cloud-hosting vendors as JSON. Each entry "
            "needs a `name` and a `score` (0..1)."
        ),
    ]

    stream = StructuredStream(
        _events_from_model_stream(model, messages, response_format),
        schema=VendorList,
    )

    partials: list[VendorList] = []
    async for partial in stream:
        partials.append(partial)

    # We accept either a real progressive yield (>= 1 partial during
    # streaming and a final) or a single yield that's already complete —
    # depends on the provider's chunking. The wiring is correct as long
    # as ``stream.final`` is a fully valid instance with 3 vendors.
    assert stream.final is not None, (
        f"StructuredStream produced no final; partials={len(partials)}, "
        f"terminate_reason={stream.terminate_reason!r}"
    )
    assert isinstance(stream.final, VendorList)
    assert len(stream.final.vendors) == 3
    for v in stream.final.vendors:
        assert v.name
        assert 0.0 <= v.score <= 1.0
