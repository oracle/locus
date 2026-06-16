# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the tool system."""

import pytest

from locus.tools import tool
from locus.tools.context import ToolContext
from locus.tools.decorator import Tool
from locus.tools.registry import ToolRegistry, create_registry
from locus.tools.schema import generate_schema, python_type_to_json_type


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_simple_tool(self):
        """Create a simple tool."""

        @tool
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "Greet someone by name."
        assert "name" in greet.parameters["properties"]

    def test_tool_with_defaults(self):
        """Tool with default parameters."""

        @tool
        def search(query: str, limit: int = 10) -> list[str]:
            """Search for items."""
            return [f"Result {i}" for i in range(limit)]

        assert "query" in search.parameters["required"]
        assert "limit" not in search.parameters["required"]
        assert search.parameters["properties"]["limit"]["default"] == 10

    def test_tool_with_custom_name(self):
        """Tool with custom name."""

        @tool(name="custom_search")
        def search(query: str) -> str:
            """Search."""
            return query

        assert search.name == "custom_search"

    def test_tool_with_custom_description(self):
        """Tool with custom description."""

        @tool(description="Custom description here")
        def my_tool(x: int) -> int:
            """Original docstring."""
            return x

        assert my_tool.description == "Custom description here"

    def test_tool_direct_call(self):
        """Tool can be called directly."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5

    @pytest.mark.asyncio
    async def test_async_tool(self):
        """Async tool execution."""

        @tool
        async def async_fetch(url: str) -> str:
            """Fetch a URL."""
            return f"Fetched: {url}"

        result = await async_fetch.execute(url="https://example.com")
        assert result == "Fetched: https://example.com"

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self):
        """Sync tool executed via execute()."""

        @tool
        def sync_add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        result = await sync_add.execute(a=2, b=3)
        assert result == "5"  # Result is stringified

    @pytest.mark.asyncio
    async def test_tool_with_context(self):
        """Tool that receives context."""

        @tool
        def contextual(query: str, ctx: ToolContext) -> str:
            """Tool with context."""
            return f"Query: {query}, Iteration: {ctx.iteration}"

        ctx = ToolContext(
            tool_call_id="call_1",
            tool_name="contextual",
            run_id="run_1",
            iteration=5,
        )

        result = await contextual.execute(ctx=ctx, query="test")
        assert "Iteration: 5" in result

    @pytest.mark.asyncio
    async def test_tool_returning_none(self):
        """Tool that returns None gets success message."""

        @tool
        def void_tool(x: int) -> None:
            """A void tool."""

        result = await void_tool.execute(x=42)
        assert result == "Success (no output)"

    @pytest.mark.asyncio
    async def test_tool_returning_pydantic_model(self):
        """Tool that returns Pydantic model gets JSON serialized."""
        from pydantic import BaseModel

        class MyResult(BaseModel):
            name: str
            value: int

        @tool
        def model_tool(x: int) -> MyResult:
            """A tool returning a model."""
            return MyResult(name="test", value=x)

        result = await model_tool.execute(x=42)
        assert '"name": "test"' in result or '"name":"test"' in result
        assert '"value": 42' in result or '"value":42' in result


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Register a tool."""

        @tool
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry = ToolRegistry()
        registry.register(my_tool)

        assert "my_tool" in registry
        assert len(registry) == 1

    def test_register_duplicate_fails(self):
        """Registering duplicate name fails."""

        @tool
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry = ToolRegistry()
        registry.register(my_tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(my_tool)

    def test_get_tool(self):
        """Get a registered tool."""

        @tool
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry = ToolRegistry()
        registry.register(my_tool)

        retrieved = registry.get("my_tool")
        assert retrieved is my_tool

    def test_get_nonexistent_returns_none(self):
        """Get nonexistent tool returns None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_raise(self):
        """Get or raise for nonexistent tool."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_or_raise("nonexistent")

    def test_unregister(self):
        """Unregister a tool."""

        @tool
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry = ToolRegistry()
        registry.register(my_tool)
        registry.unregister("my_tool")

        assert "my_tool" not in registry

    def test_to_openai_schemas(self):
        """Generate OpenAI schemas for all tools."""

        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        registry = create_registry(tool_a, tool_b)
        schemas = registry.to_openai_schemas()

        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"tool_a", "tool_b"}

    def test_register_many(self):
        """Register multiple tools at once."""

        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        registry = ToolRegistry()
        registry.register_many([tool_a, tool_b])

        assert len(registry) == 2
        assert "tool_a" in registry
        assert "tool_b" in registry

    def test_get_or_raise_found(self):
        """Get or raise returns tool when found."""

        @tool
        def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry = ToolRegistry()
        registry.register(my_tool)

        result = registry.get_or_raise("my_tool")
        assert result is my_tool

    def test_list_tools(self):
        """List all registered tool names."""

        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        registry = create_registry(tool_a, tool_b)
        names = registry.list_tools()

        assert len(names) == 2
        assert "tool_a" in names
        assert "tool_b" in names

    def test_iter(self):
        """Iterate over tools in registry."""

        @tool
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        registry = create_registry(tool_a, tool_b)
        tools = list(registry)

        assert len(tools) == 2
        assert tool_a in tools
        assert tool_b in tools


class TestSchema:
    """Tests for schema generation."""

    def test_string_type(self):
        """String type conversion."""
        result = python_type_to_json_type(str)
        assert result == {"type": "string"}

    def test_int_type(self):
        """Int type conversion."""
        result = python_type_to_json_type(int)
        assert result == {"type": "integer"}

    def test_float_type(self):
        """Float type conversion."""
        result = python_type_to_json_type(float)
        assert result == {"type": "number"}

    def test_bool_type(self):
        """Bool type conversion."""
        result = python_type_to_json_type(bool)
        assert result == {"type": "boolean"}

    def test_list_type(self):
        """List type conversion."""
        result = python_type_to_json_type(list[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_dict_type(self):
        """Dict type conversion."""
        result = python_type_to_json_type(dict[str, int])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "integer"}

    def test_generate_schema_simple(self):
        """Generate schema for simple function."""

        def my_func(name: str, count: int) -> str:
            """Do something with name and count."""
            return f"{name}: {count}"

        schema = generate_schema(my_func)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_func"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "count" in schema["function"]["parameters"]["properties"]

    def test_generate_schema_with_defaults(self):
        """Schema generation with default values."""

        def my_func(required: str, optional: int = 10) -> str:
            """A function."""
            return f"{required}: {optional}"

        schema = generate_schema(my_func)
        params = schema["function"]["parameters"]

        assert "required" in params["required"]
        assert "optional" not in params["required"]
        assert params["properties"]["optional"]["default"] == 10
