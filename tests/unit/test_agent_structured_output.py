# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for Agent's ``output_schema`` structured-output coercion."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from locus.agent import Agent
from locus.core.messages import Message
from locus.core.structured import build_response_format, format_validation_errors
from locus.models.base import ModelResponse


class Vendor(BaseModel):
    """Single vendor record."""

    name: str = Field(description="Legal name")
    score: float = Field(description="0..1 quality score")
    notes: str | None = None


class VendorList(BaseModel):
    """A list of vendor picks."""

    vendors: list[Vendor]


class _ScriptedModel:
    """Test double that returns a scripted sequence of model responses.

    Each call pops the next ``ModelResponse``. Intended for verifying the
    structuring pipeline without a real provider.
    """

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        if not self._responses:
            raise AssertionError("ScriptedModel exhausted")
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("ScriptedModel does not stream")


def _assistant(content: str) -> ModelResponse:
    return ModelResponse(message=Message.assistant(content=content), usage={})


# =============================================================================
# build_response_format
# =============================================================================


class TestBuildResponseFormat:
    def test_strict_dict_shape(self):
        rf = build_response_format(Vendor, strict=True)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Vendor"
        assert rf["json_schema"]["strict"] is True
        # Title is stripped (OpenAI strict mode rejects it at the root).
        assert "title" not in rf["json_schema"]["schema"]
        # Field names survive.
        props = rf["json_schema"]["schema"]["properties"]
        assert "name" in props
        assert "score" in props

    def test_strict_false(self):
        rf = build_response_format(Vendor, strict=False)
        assert rf["json_schema"]["strict"] is False


# =============================================================================
# format_validation_errors
# =============================================================================


class TestFormatValidationErrors:
    def test_empty(self):
        assert format_validation_errors([]) == "(no error details)"

    def test_renders_loc_msg_type(self):
        rendered = format_validation_errors(
            [
                {
                    "loc": ("vendors", 0, "score"),
                    "msg": "Input should be a valid number",
                    "type": "float_parsing",
                },
                {"loc": (), "msg": "Field required", "type": "missing"},
            ]
        )
        assert "vendors.0.score: Input should be a valid number [float_parsing]" in rendered
        assert "<root>: Field required [missing]" in rendered


# =============================================================================
# Agent + output_schema
# =============================================================================


class TestAgentOutputSchemaSuccess:
    def test_first_attempt_parses(self):
        """Agent's first answer is already valid JSON — no retry needed."""
        good_json = (
            '{"vendors": [{"name": "Acme", "score": 0.9, "notes": null}, '
            '{"name": "Globex", "score": 0.7, "notes": "EMEA only"}]}'
        )
        model = _ScriptedModel([_assistant(good_json)])
        agent = Agent(model=model, tools=[], output_schema=VendorList)

        result = agent.run_sync("Find me 2 vendors.")

        assert result.parsed is not None
        assert isinstance(result.parsed, VendorList)
        assert [v.name for v in result.parsed.vendors] == ["Acme", "Globex"]
        assert result.parse_error is None
        # Only the main run call — no repair re-prompt.
        assert len(model.calls) == 1
        # Message replaced with canonical JSON dump for downstream consumers.
        assert result.message.startswith("{")
        assert "vendors" in result.message

    def test_extracts_from_markdown_fence(self):
        fenced = 'Here you go:\n```json\n{"vendors": [{"name": "Acme", "score": 0.5}]}\n```'
        model = _ScriptedModel([_assistant(fenced)])
        agent = Agent(model=model, tools=[], output_schema=VendorList)

        result = agent.run_sync("Find one.")
        assert result.parsed is not None
        assert result.parsed.vendors[0].name == "Acme"


class TestAgentOutputSchemaRetry:
    def test_invalid_then_valid_repairs(self):
        """First answer fails validation; retry with errors yields valid JSON."""
        # First response is missing required fields — must trigger a repair pass.
        bad = '{"vendors": [{"name": "Acme"}]}'  # missing score
        good = '{"vendors": [{"name": "Acme", "score": 0.5}]}'
        model = _ScriptedModel([_assistant(bad), _assistant(good)])
        agent = Agent(model=model, tools=[], output_schema=VendorList)

        result = agent.run_sync("Find vendors.")

        assert result.parsed is not None
        assert result.parse_error is None
        assert result.parsed.vendors[0].score == 0.5
        # 1 main run + 1 repair re-prompt.
        assert len(model.calls) == 2
        # The repair call must include a Schema Repair system message.
        repair_messages = [
            m.content for m in model.calls[1]["messages"] if m.role.value == "system"
        ]
        assert any("Schema Repair" in (c or "") for c in repair_messages)
        # The repair call ships ``response_format`` to the provider.
        assert "response_format" in model.calls[1]["kwargs"]
        rf = model.calls[1]["kwargs"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "VendorList"

    def test_all_retries_fail_returns_error(self):
        """When every attempt fails validation, ``parse_error`` is populated."""
        always_bad = "totally not json"
        model = _ScriptedModel([_assistant(always_bad)] * 5)
        agent = Agent(
            model=model,
            tools=[],
            output_schema=VendorList,
            output_schema_retries=2,
        )

        result = agent.run_sync("Try anyway.")

        assert result.parsed is None
        assert result.parse_error is not None
        # 1 main + 2 retries.
        assert len(model.calls) == 3

    def test_zero_retries_disables_repair(self):
        bad = '{"oops": "wrong shape"}'
        model = _ScriptedModel([_assistant(bad)])
        agent = Agent(
            model=model,
            tools=[],
            output_schema=VendorList,
            output_schema_retries=0,
        )

        result = agent.run_sync("Try once.")

        assert result.parsed is None
        assert result.parse_error is not None
        # No repair pass.
        assert len(model.calls) == 1


class TestAgentOutputSchemaPrompt:
    def test_schema_appended_to_system_prompt(self):
        good = '{"vendors": []}'
        model = _ScriptedModel([_assistant(good)])
        agent = Agent(
            model=model,
            tools=[],
            system_prompt="You are a procurement officer.",
            output_schema=VendorList,
        )

        agent.run_sync("Find vendors.")

        first_call_messages = model.calls[0]["messages"]
        system_msg = next(m for m in first_call_messages if m.role.value == "system")
        assert "procurement officer" in (system_msg.content or "")
        assert "Final-answer schema" in (system_msg.content or "")
        # Pydantic schema should include the field name.
        assert "vendors" in (system_msg.content or "")


class TestOutputSchemaConfigValidation:
    def test_rejects_non_basemodel(self):
        with pytest.raises(TypeError, match=r"must be a pydantic\.BaseModel subclass"):
            Agent(model="openai:gpt-4o", output_schema=dict)  # type: ignore[arg-type]
