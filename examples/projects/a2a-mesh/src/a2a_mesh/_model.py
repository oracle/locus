# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Shared model factory for the mesh services.

Reads ``TULIP_MODEL_PROVIDER`` (``mock`` | ``openai`` | ``anthropic``) and
returns a model instance. Defaults to a small inline MockModel so the
demo runs end-to-end without credentials or network access to a model
provider.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message
from tulip.models.base import ModelResponse


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


def _openai_model() -> Any:
    from tulip.models import OpenAIModel

    return OpenAIModel(model=os.environ.get("TULIP_MODEL_ID", "gpt-4o"))


def _anthropic_model() -> Any:
    from tulip.models.native.anthropic import AnthropicModel

    return AnthropicModel(model=os.environ.get("TULIP_MODEL_ID", "claude-sonnet-4-6"))


def get_model() -> Any:
    provider = os.environ.get("TULIP_MODEL_PROVIDER", "mock").lower()
    if provider == "mock":
        return MockModel()
    if provider == "openai":
        return _openai_model()
    if provider == "anthropic":
        return _anthropic_model()
    msg = f"Unknown TULIP_MODEL_PROVIDER: {provider!r}"
    raise ValueError(msg)
