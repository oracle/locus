# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for complex agent scenarios."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field

from locus import Agent
from locus.core.structured import parse_structured
from locus.tools import tool
from tests._safe_math import safe_math_eval


pytestmark = pytest.mark.integration


def load_local_config() -> dict:
    """Load local config if available."""
    config_path = Path(__file__).parent.parent.parent / "config.local.yaml"
    if config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


# =============================================================================
# Structured Output Schemas
# =============================================================================


class SimpleAnswer(BaseModel):
    """Simple answer schema."""

    answer: str = Field(description="The answer")
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")


class AnalysisResult(BaseModel):
    """Analysis result schema."""

    summary: str
    findings: list[str]
    recommendation: str


# =============================================================================
# Complex Tools
# =============================================================================


@tool
async def query_database(table: str, filters: dict | None = None) -> str:
    """Query a mock database."""
    data = {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
        ],
        "products": [
            {"id": 1, "name": "Widget", "price": 9.99},
            {"id": 2, "name": "Gadget", "price": 19.99},
        ],
    }
    result = data.get(table, [])
    return json.dumps(result)


@tool
async def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error: {e}"


@tool
async def search_web(query: str) -> str:
    """Simulate web search."""
    return json.dumps(
        {
            "results": [
                {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
            ]
        }
    )


@tool
async def analyze_data(data: list, analysis_type: str = "summary") -> str:
    """Analyze data."""
    if analysis_type == "summary":
        return f"Analyzed {len(data)} records. Summary: Data looks good."
    return f"Analysis type '{analysis_type}' not supported."


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.requires_oci
class TestComplexAgent:
    """Complex agent integration tests."""

    @pytest.fixture
    def model(self):
        """Create OCI model."""
        from locus.models import OCIModel

        config = load_local_config().get("oci", {})
        return OCIModel(
            model_id=config.get("models", {}).get(
                "gpt", os.getenv("OCI_MODEL_ID", "openai.gpt-5.4")
            ),
            profile_name=config.get("profile_name", "DEFAULT"),
            auth_type=config.get("auth_type", "api_key"),
            region=config.get("region", "eu-frankfurt-1"),
            compartment_id=config.get("compartment_id"),
            service_endpoint=config.get("service_endpoint"),
            max_tokens=512,
        )

    @pytest.mark.asyncio
    async def test_simple_agent(self, model):
        """Test simple agent execution."""
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant. Be concise.",
        )

        result = agent.run_sync("What is 2+2? Just the number.")
        assert result.success
        assert "4" in result.message

    @pytest.mark.asyncio
    async def test_agent_with_tools(self, model):
        """Test agent with multiple tools."""
        agent = Agent(
            model=model,
            tools=[calculate, search_web],
            system_prompt="You are a helpful assistant with tools.",
        )

        # Test calculation
        result = agent.run_sync("Calculate 15 * 7")
        assert result.success
        assert "105" in result.message

    @pytest.mark.asyncio
    async def test_agent_streaming(self, model):
        """Test agent streaming execution."""
        agent = Agent(
            model=model,
            system_prompt="Be concise.",
        )

        events = []
        async for event in agent.run("What is the capital of Japan?"):
            events.append(event)

        # Should have at least think and terminate events
        event_types = [e.event_type for e in events]
        assert "think" in event_types
        assert "terminate" in event_types

        # Find terminate event
        terminate = next(e for e in events if e.event_type == "terminate")
        assert terminate.final_message is not None
        assert "Tokyo" in terminate.final_message

    @pytest.mark.asyncio
    async def test_multi_step_task(self, model):
        """Test multi-step task execution."""
        agent = Agent(
            model=model,
            tools=[query_database, analyze_data],
            system_prompt="""You are a data analyst. When asked to analyze data:
            1. First query the database
            2. Then analyze the results
            3. Provide a summary""",
            max_iterations=5,
        )

        events = []
        async for event in agent.run("Analyze the users in our database"):
            events.append(event)
            if event.event_type == "tool_complete":
                print(f"Tool: {event.tool_name} -> {event.result[:50]}...")

        terminate = next(e for e in events if e.event_type == "terminate")
        assert terminate.iterations_used >= 1


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

    @pytest.mark.requires_oci
    @pytest.mark.asyncio
    async def test_structured_output_live(self):
        """Test structured output with live model."""
        from locus.core.structured import create_schema_prompt
        from locus.models import OCIModel

        config = load_local_config().get("oci", {})
        model = OCIModel(
            model_id=config.get("models", {}).get(
                "gpt", os.getenv("OCI_MODEL_ID", "openai.gpt-5.4")
            ),
            profile_name=config.get("profile_name", "DEFAULT"),
            auth_type=config.get("auth_type", "api_key"),
            region=config.get("region", "eu-frankfurt-1"),
            compartment_id=config.get("compartment_id"),
            service_endpoint=config.get("service_endpoint"),
        )

        schema_prompt = create_schema_prompt(SimpleAnswer)
        agent = Agent(
            model=model,
            system_prompt=f"Answer questions. {schema_prompt}",
        )

        result = agent.run_sync("What is the capital of France?")

        # Try to parse as structured
        parsed = parse_structured(result.message, SimpleAnswer, strict=False)
        # May or may not succeed depending on model output format
        print(f"Raw: {result.message}")
        print(f"Parsed: {parsed.success}, {parsed.error if not parsed.success else 'OK'}")


class TestCheckpointBackends:
    """Test checkpoint backend implementations."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        """Test in-memory checkpoint backend."""
        from locus.core.state import AgentState
        from locus.memory.backends import MemoryCheckpointer

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
