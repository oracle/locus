"""Unit tests for native ``response_format`` pass-through on the agent loop.

When ``Agent(output_schema=Pydantic)`` is configured AND the provider
exposes ``supports_structured_output`` as True, the loop should pass
``response_format=`` to ``model.complete()`` directly — skipping the
prompted-JSON fallback.

When the provider returns False (Anthropic), the loop falls
back to the prompted-JSON path and ``response_format`` is NOT passed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from tulip.core.messages import Message


pytest.importorskip("openai")
pytest.importorskip("anthropic")


class SamplePayload(BaseModel):
    name: str
    score: float


class _StubModel:
    """Minimal model stub that records the kwargs it received."""

    def __init__(self, supports: bool) -> None:
        self.supports_structured_output = supports
        self.complete = AsyncMock()
        self.stream = AsyncMock()
        self._captured_kwargs: dict[str, Any] = {}

    async def complete(self, **kwargs: Any) -> Any:  # type: ignore[override,no-redef]
        self._captured_kwargs = kwargs
        from tulip.models.base import ModelResponse

        return ModelResponse(
            message=Message.assistant('{"name": "ok", "score": 0.9}'),
            usage={"input_tokens": 10, "output_tokens": 4},
        )


def test_supports_structured_output_capability_on_openai_model():
    """OpenAIModel reports True; structured output passes through natively."""
    from tulip.models.native.openai import OpenAIModel

    model = OpenAIModel(model="gpt-4o", api_key="sk-test")
    assert model.supports_structured_output is True


def test_supports_structured_output_capability_on_anthropic_model():
    """AnthropicModel reports False; falls back to prompted JSON."""
    from tulip.models.native.anthropic import AnthropicModel

    model = AnthropicModel(model="claude-sonnet-4-20250514", api_key="sk-test")
    assert model.supports_structured_output is False


def test_build_response_format_returns_openai_shape():
    """``build_response_format`` already returns the right shape — sanity check."""
    from tulip.core.structured import build_response_format

    rf = build_response_format(SamplePayload, strict=True)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "SamplePayload"
    assert rf["json_schema"]["strict"] is True
    assert "schema" in rf["json_schema"]
    # required fields propagated:
    schema = rf["json_schema"]["schema"]
    assert "name" in schema.get("required", [])
    assert "score" in schema.get("required", [])


# ---------------------------------------------------------------------------
# inline_schema_refs — unit tests
# ---------------------------------------------------------------------------


def test_inline_schema_refs_no_defs():
    """Schema without $defs passes through unchanged."""
    from tulip.core.structured import inline_schema_refs

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = inline_schema_refs(schema)
    assert result == schema


def test_inline_schema_refs_inlines_single_ref():
    """$ref inside a property is replaced with the definition inline."""
    from tulip.core.structured import inline_schema_refs

    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"type": "object", "properties": {"id": {"type": "integer"}}}},
    }
    result = inline_schema_refs(schema)
    assert "$defs" not in result
    assert "$ref" not in str(result)
    assert result["properties"]["item"]["type"] == "object"
    assert result["properties"]["item"]["properties"]["id"]["type"] == "integer"


def test_inline_schema_refs_inlines_nested():
    """Nested $ref (list items) are also inlined."""
    from tulip.core.structured import inline_schema_refs

    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Row"}}},
        "$defs": {"Row": {"type": "object", "properties": {"v": {"type": "number"}}}},
    }
    result = inline_schema_refs(schema)
    assert "$defs" not in result
    assert result["properties"]["items"]["items"]["type"] == "object"


def test_inline_schema_refs_nested_pydantic_integration():
    """Nested Pydantic model produces a fully inlined schema (no $defs/$ref)."""
    from pydantic import BaseModel, Field

    from tulip.core.structured import inline_schema_refs

    class Inner(BaseModel):
        score: float = Field(ge=0.0, le=1.0)

    class Outer(BaseModel):
        items: list[Inner]

    raw = Outer.model_json_schema()
    assert "$defs" in raw  # Pydantic emits $defs for nested models

    inlined = inline_schema_refs(raw)
    assert "$defs" not in inlined
    assert "$ref" not in str(inlined)
