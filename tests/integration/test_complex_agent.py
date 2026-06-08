# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Integration tests for complex agent scenarios."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from tulip.core.structured import parse_structured


pytestmark = pytest.mark.integration


# =============================================================================
# Structured Output Schemas
# =============================================================================


class SimpleAnswer(BaseModel):
    """Simple answer schema."""

    answer: str = Field(description="The answer")
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")


# =============================================================================
# Tests
# =============================================================================


class TestStructuredOutputs:
    """Test structured output parsing."""

    def test_parse_simple_json(self):
        """Parse simple JSON response."""
        content = '{"answer": "Paris", "confidence": 0.95}'

        result = parse_structured(content, SimpleAnswer, strict=False)
        assert result.success
        assert result.parsed.answer == "Paris"
        assert result.parsed.confidence == 0.95

    def test_parse_json_in_markdown(self):
        """Parse JSON wrapped in markdown code block."""
        content = """Here is my answer:

```json
{
    "answer": "42",
    "confidence": 1.0
}
```

Hope this helps!"""

        result = parse_structured(content, SimpleAnswer, strict=False)
        assert result.success
        assert result.parsed.answer == "42"

    def test_parse_invalid_json(self):
        """Handle invalid JSON gracefully."""
        content = "This is not JSON at all."

        result = parse_structured(content, SimpleAnswer, strict=False)
        assert not result.success
        assert "error" in result.error.lower()

    def test_parse_missing_fields(self):
        """Handle missing required fields."""
        content = '{"answer": "test"}'  # Missing confidence

        result = parse_structured(content, SimpleAnswer, strict=False)
        assert not result.success


class TestCheckpointBackends:
    """Test checkpoint backend implementations."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        """Test in-memory checkpoint backend."""
        from tulip.core.state import AgentState
        from tulip.memory.backends import MemoryCheckpointer

        backend = MemoryCheckpointer()

        # Create a state
        state = AgentState()

        # Save and load
        checkpoint_id = await backend.save(state, "test_thread")
        assert checkpoint_id is not None

        loaded = await backend.load("test_thread")
        assert loaded is not None

        # List
        threads = backend.get_thread_ids()
        assert "test_thread" in threads
