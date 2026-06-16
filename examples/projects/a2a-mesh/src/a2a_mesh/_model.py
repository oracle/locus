# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Shared model factory for the mesh services.

Reads ``LOCUS_MODEL_PROVIDER`` (``mock`` | ``oci`` | ``openai``) and
returns a model instance. Defaults to a small inline MockModel so the
demo runs end-to-end without credentials or network access to a model
provider.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.base import ModelResponse


class MockModel(BaseModel):
    """A deterministic stand-in for a real LLM. Used when no creds are set."""

    max_tokens: int = 256

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> ModelResponse:
        last = (messages[-1].content or "") if messages else ""
        if tools:
            return ModelResponse(
                message=Message.assistant(
                    content=f"[mock] would consult tools={[t.get('name') for t in tools]} on {last!r}"
                ),
                usage={"prompt_tokens": 8, "completion_tokens": 16},
                stop_reason="end_turn",
            )
        return ModelResponse(
            message=Message.assistant(content=f"[mock reply] {last[:120]}"),
            usage={"prompt_tokens": 8, "completion_tokens": 16},
            stop_reason="end_turn",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        resp = await self.complete(messages, tools, **kwargs)
        text = resp.content or ""
        for i in range(0, len(text), 16):
            yield ModelChunkEvent(content=text[i : i + 16])
        yield ModelChunkEvent(done=True)


def _oci_model() -> Any:
    from locus.models import OCIChatCompletionsModel

    return OCIChatCompletionsModel(
        model=os.environ.get("LOCUS_MODEL_ID", "openai.gpt-5.5-2026-04-23"),
        profile=os.environ.get("LOCUS_OCI_PROFILE", "DEFAULT"),
        region=os.environ.get("LOCUS_OCI_REGION", "us-chicago-1"),
        compartment_id=os.environ.get("LOCUS_OCI_COMPARTMENT"),
    )


def _openai_model() -> Any:
    from locus.models import OpenAIModel

    return OpenAIModel(model=os.environ.get("LOCUS_MODEL_ID", "gpt-5.5-2026-04-23"))


def get_model() -> Any:
    provider = os.environ.get("LOCUS_MODEL_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockModel()
    if provider == "oci":
        return _oci_model()
    if provider == "openai":
        return _openai_model()
    msg = f"Unknown LOCUS_MODEL_PROVIDER: {provider!r}"
    raise ValueError(msg)
