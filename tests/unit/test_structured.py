# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for structured output module."""

import pytest
from pydantic import BaseModel, Field

from locus.core.structured import (
    StructuredOutput,
    StructuredOutputError,
    create_output_instructions,
    create_schema_prompt,
    extract_json,
    parse_structured,
)


class SampleSchema(BaseModel):
    """Sample schema for testing."""

    name: str = Field(description="The name")
    age: int = Field(description="The age")
    active: bool = Field(default=True, description="Is active")


class NestedSchema(BaseModel):
    """Schema with nested structure."""

    user: SampleSchema
    tags: list[str] = Field(default_factory=list)


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_code_block_json(self):
        """Extract JSON from ```json code block."""
        content = """Some text
```json
{"name": "test", "age": 25}
```
More text"""
        result = extract_json(content)
        assert result == '{"name": "test", "age": 25}'

    def test_extract_from_plain_code_block(self):
        """Extract JSON from plain ``` code block."""
        content = """Here's the result:
```
{"name": "test", "age": 25}
```"""
        result = extract_json(content)
        assert result == '{"name": "test", "age": 25}'

    def test_extract_from_code_block_with_language(self):
        """Extract JSON from code block with language identifier."""
        content = """```javascript
{"name": "test", "age": 25}
```"""
        result = extract_json(content)
        assert result == '{"name": "test", "age": 25}'

    def test_extract_raw_json(self):
        """Extract raw JSON object from text."""
        content = 'The result is {"name": "test", "age": 25} as expected.'
        result = extract_json(content)
        assert result == '{"name": "test", "age": 25}'

    def test_extract_nested_json(self):
        """Extract nested JSON with balanced braces."""
        content = '{"outer": {"inner": {"value": 1}}}'
        result = extract_json(content)
        assert result == '{"outer": {"inner": {"value": 1}}}'

    def test_extract_plain_text(self):
        """Return plain text when no JSON found."""
        content = "Just plain text"
        result = extract_json(content)
        assert result == "Just plain text"

    def test_extract_strips_whitespace(self):
        """Strip whitespace from content."""
        content = '   \n  {"name": "test"}  \n   '
        result = extract_json(content)
        assert result == '{"name": "test"}'


class TestParseStructured:
    """Tests for parse_structured function."""

    def test_parse_valid_json(self):
        """Parse valid JSON into schema."""
        content = '{"name": "Alice", "age": 30}'
        result = parse_structured(content, SampleSchema)

        assert result.success
        assert result.parsed is not None
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 30
        assert result.parsed.active is True  # default
        assert result.error is None

    def test_parse_from_code_block(self):
        """Parse JSON from code block."""
        content = """```json
{"name": "Bob", "age": 25, "active": false}
```"""
        result = parse_structured(content, SampleSchema)

        assert result.success
        assert result.parsed.name == "Bob"
        assert result.parsed.age == 25
        assert result.parsed.active is False

    def test_parse_invalid_json_strict(self):
        """Raise error for invalid JSON in strict mode."""
        content = "not valid json"

        with pytest.raises(StructuredOutputError) as exc_info:
            parse_structured(content, SampleSchema, strict=True)

        assert "JSON parse error" in str(exc_info.value)
        assert exc_info.value.raw_content == content

    def test_parse_invalid_json_non_strict(self):
        """Return error result for invalid JSON in non-strict mode."""
        content = "not valid json"
        result = parse_structured(content, SampleSchema, strict=False)

        assert not result.success
        assert result.parsed is None
        assert "JSON parse error" in result.error

    def test_parse_validation_error_strict(self):
        """Raise error for validation failure in strict mode."""
        content = '{"name": "Alice"}'  # missing required 'age'

        with pytest.raises(StructuredOutputError) as exc_info:
            parse_structured(content, SampleSchema, strict=True)

        assert "Validation error" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0

    def test_parse_validation_error_non_strict(self):
        """Return error result for validation failure in non-strict mode."""
        content = '{"name": "Alice"}'  # missing required 'age'
        result = parse_structured(content, SampleSchema, strict=False)

        assert not result.success
        assert result.parsed is None
        assert "Validation error" in result.error

    def test_parse_nested_schema(self):
        """Parse nested schema."""
        content = '{"user": {"name": "Test", "age": 20}, "tags": ["a", "b"]}'
        result = parse_structured(content, NestedSchema)

        assert result.success
        assert result.parsed.user.name == "Test"
        assert result.parsed.tags == ["a", "b"]


class TestStructuredOutput:
    """Tests for StructuredOutput model."""

    def test_success_property_true(self):
        """Success is True when parsed is set."""
        output = StructuredOutput(
            raw='{"name": "test"}',
            parsed=SampleSchema(name="test", age=25),
        )
        assert output.success is True

    def test_success_property_false_no_parsed(self):
        """Success is False when parsed is None."""
        output = StructuredOutput(raw="invalid", error="Parse error")
        assert output.success is False

    def test_success_property_false_with_error(self):
        """Success is False when error is set."""
        output = StructuredOutput(
            raw="test",
            parsed=SampleSchema(name="test", age=25),
            error="Some error",
        )
        assert output.success is False

    def test_unwrap_returns_parsed(self):
        """Unwrap returns parsed value."""
        parsed = SampleSchema(name="test", age=25)
        output = StructuredOutput(raw='{"name": "test"}', parsed=parsed)

        assert output.unwrap() == parsed

    def test_unwrap_raises_on_no_parsed(self):
        """Unwrap raises error when no parsed value."""
        output = StructuredOutput(raw="invalid", error="Parse error")

        with pytest.raises(StructuredOutputError) as exc_info:
            output.unwrap()

        assert "Parse error" in str(exc_info.value)

    def test_unwrap_raises_default_message(self):
        """Unwrap raises with default message."""
        output = StructuredOutput(raw="invalid")

        with pytest.raises(StructuredOutputError) as exc_info:
            output.unwrap()

        assert "No parsed output" in str(exc_info.value)


class TestStructuredOutputError:
    """Tests for StructuredOutputError."""

    def test_error_with_message(self):
        """Create error with message."""
        error = StructuredOutputError("Test error", "raw content")
        assert str(error) == "Test error"
        assert error.raw_content == "raw content"
        assert error.errors == []

    def test_error_with_errors_list(self):
        """Create error with errors list."""
        errors = [{"loc": ["name"], "msg": "required"}]
        error = StructuredOutputError("Validation failed", "raw", errors)
        assert error.errors == errors


class TestCreateSchemaPrompt:
    """Tests for create_schema_prompt function."""

    def test_creates_prompt_with_schema(self):
        """Create prompt with JSON schema."""
        prompt = create_schema_prompt(SampleSchema)

        assert "JSON object" in prompt
        assert "schema" in prompt.lower()
        assert "name" in prompt
        assert "age" in prompt
        assert "```json" in prompt

    def test_removes_title_from_schema(self):
        """Title is removed from schema in prompt."""
        prompt = create_schema_prompt(SampleSchema)
        # Title "SampleSchema" should not appear in the JSON schema
        assert '"title": "SampleSchema"' not in prompt


class TestCreateOutputInstructions:
    """Tests for create_output_instructions function."""

    def test_creates_field_instructions(self):
        """Create instructions with field details."""
        instructions = create_output_instructions(SampleSchema)

        assert "name" in instructions
        assert "age" in instructions
        assert "active" in instructions
        assert "(required)" in instructions
        assert "(optional)" in instructions
        assert "The name" in instructions
        assert "The age" in instructions

    def test_includes_type_information(self):
        """Include type information for fields."""
        instructions = create_output_instructions(SampleSchema)

        assert "string" in instructions
        assert "integer" in instructions
        assert "boolean" in instructions

    def test_ends_with_json_only_instruction(self):
        """Ends with instruction to return only JSON."""
        instructions = create_output_instructions(SampleSchema)
        assert "only" in instructions.lower()
        assert "json" in instructions.lower()
