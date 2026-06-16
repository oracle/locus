"""Unit tests for native ``response_format`` pass-through on the agent loop.

When ``Agent(output_schema=Pydantic)`` is configured AND the provider
exposes ``supports_structured_output`` as True, the loop should pass
``response_format=`` to ``model.complete()`` directly — skipping the
prompted-JSON fallback.

When the provider returns False (Anthropic, Ollama, OCI's native SDK),
the loop falls back to the prompted-JSON path and ``response_format``
is NOT passed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from locus.core.messages import Message


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
        from locus.models.base import ModelResponse

        return ModelResponse(
            message=Message.assistant('{"name": "ok", "score": 0.9}'),
            usage={"input_tokens": 10, "output_tokens": 4},
        )


def test_supports_structured_output_capability_on_openai_model():
    """OpenAIModel reports True; structured output passes through natively."""
    from locus.models.native.openai import OpenAIModel

    model = OpenAIModel(model="gpt-4o", api_key="sk-test")
    assert model.supports_structured_output is True


def test_supports_structured_output_capability_on_anthropic_model():
    """AnthropicModel reports False; falls back to prompted JSON."""
    from locus.models.native.anthropic import AnthropicModel

    model = AnthropicModel(model="claude-sonnet-4-20250514", api_key="sk-test")
    assert model.supports_structured_output is False


def test_supports_structured_output_capability_on_ollama_model():
    """OllamaModel reports False."""
    from locus.models.native.ollama import OllamaModel

    model = OllamaModel(model="llama3.3")
    assert model.supports_structured_output is False


def test_supports_structured_output_capability_on_oci_native_model():
    """OCIModel (native SDK transport) reports False; use OCIChatCompletionsModel for native."""
    pytest.importorskip("oci")
    from locus.models.providers.oci import OCIModel

    # Model id chosen to route to the native SDK transport.
    try:
        model = OCIModel(model_id="cohere.command-r-08-2024", profile_name="DEFAULT")
    except Exception:
        pytest.skip("OCI client construction requires real config")
    assert model.supports_structured_output is False


def test_oci_openai_compat_inherits_capability():
    """OCIChatCompletionsModel inherits from OpenAIModel; reports True."""
    pytest.importorskip("oci")
    from unittest.mock import MagicMock, patch

    from locus.models.providers.oci.openai_compat import OCIChatCompletionsModel

    fake_cfg = {
        "tenancy": "ocid1.tenancy.oc1..aaa",
        "user": "ocid1.user.oc1..aaa",
        "fingerprint": "aa:bb:cc",
        "key_file": "/dev/null",
        "region": "us-chicago-1",
    }
    fake_signer = MagicMock()
    fake_signer.region = "us-chicago-1"

    with (
        patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value=fake_cfg,
        ),
        patch(
            "locus.models.providers.oci.openai_compat._build_signer_from_profile",
            return_value=fake_signer,
        ),
    ):
        model = OCIChatCompletionsModel(model="openai.gpt-5", profile="DEFAULT")
    assert model.supports_structured_output is True


def test_build_response_format_returns_openai_shape():
    """``build_response_format`` already returns the right shape — sanity check."""
    from locus.core.structured import build_response_format

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
    from locus.core.structured import inline_schema_refs

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = inline_schema_refs(schema)
    assert result == schema


def test_inline_schema_refs_inlines_single_ref():
    """$ref inside a property is replaced with the definition inline."""
    from locus.core.structured import inline_schema_refs

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
    from locus.core.structured import inline_schema_refs

    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Row"}}},
        "$defs": {"Row": {"type": "object", "properties": {"v": {"type": "number"}}}},
    }
    result = inline_schema_refs(schema)
    assert "$defs" not in result
    assert result["properties"]["items"]["items"]["type"] == "object"


def test_inline_schema_refs_gemini_integration():
    """Nested Pydantic model produces schema that Gemini would accept after inlining."""
    from pydantic import BaseModel, Field

    from locus.core.structured import inline_schema_refs

    class Inner(BaseModel):
        score: float = Field(ge=0.0, le=1.0)

    class Outer(BaseModel):
        items: list[Inner]

    raw = Outer.model_json_schema()
    assert "$defs" in raw  # Pydantic emits $defs for nested models

    inlined = inline_schema_refs(raw)
    assert "$defs" not in inlined
    assert "$ref" not in str(inlined)


# ---------------------------------------------------------------------------
# OCIChatCompletionsModel Gemini $ref inlining — unit tests
# ---------------------------------------------------------------------------


def test_oci_openai_model_requires_inlined_refs_for_gemini():
    """google.* models trigger ref inlining; others do not."""
    pytest.importorskip("oci")
    from unittest.mock import MagicMock, patch

    from locus.models.providers.oci.openai_compat import OCIChatCompletionsModel

    fake_cfg = {
        "tenancy": "t",
        "user": "u",
        "fingerprint": "f",
        "key_file": "/dev/null",
        "region": "us-chicago-1",
    }
    fake_signer = MagicMock()

    with (
        patch(
            "locus.models.providers.oci.openai_compat._load_profile_config", return_value=fake_cfg
        ),
        patch(
            "locus.models.providers.oci.openai_compat._build_signer_from_profile",
            return_value=fake_signer,
        ),
    ):
        gemini = OCIChatCompletionsModel(model="google.gemini-2.5-flash", profile="DEFAULT")
        gpt = OCIChatCompletionsModel(model="openai.gpt-5", profile="DEFAULT")

    assert gemini._requires_inlined_schema_refs is True
    assert gpt._requires_inlined_schema_refs is False
