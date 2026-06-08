# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Notebook 42: MCP integration — publish and consume tools across processes.

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process. Tulip speaks
both sides of it.

- Publish a Tulip agent as an MCP server — tools and the agent's own
  ``run_agent`` become MCP methods.
- Connect a Tulip agent to an external MCP server and use its tools as
  ordinary ``@tool``-decorated callables.
- Convert tool schemas in both directions
  (``tulip_tool_to_mcp`` / ``mcp_tool_to_tulip``).
- Handle ``tools/list`` and ``tools/call`` requests programmatically.

The configured provider drives the agent. The MCP layer is transport-only
— the same agent works against any provider.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_47_mcp_integration.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_47_mcp_integration.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Optional: ``pip install fastmcp`` to exercise live request handling.

See https://modelcontextprotocol.io for the MCP specification.
"""

import ast
import asyncio
import json
import operator as _op

# Import shared config for model
from config import get_model, print_config

from tulip.agent import Agent
from tulip.integrations.fastmcp import (
    TulipMCPServer,
    create_mcp_server,
    tulip_tool_to_mcp,
)
from tulip.tools import tool


_SAFE_MATH_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_SAFE_MATH_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}


def _safe_math_eval(expression: str) -> float:
    # AST-only arithmetic — no names, calls, or attribute access so the
    # calculator tool can't be turned into a sandbox escape.
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BIN_OPS:
            return _SAFE_MATH_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_UNARY_OPS:
            return _SAFE_MATH_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    return _eval(tree)


# =============================================================================
# Part 1: Three ordinary Tulip tools. Nothing MCP-specific about them yet.
# =============================================================================


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 55, "condition": "cloudy"},
        "tokyo": {"temp": 68, "condition": "partly cloudy"},
    }
    data = weather_data.get(city.lower(), {"temp": 70, "condition": "unknown"})
    return f"Weather in {city}: {data['temp']}°F, {data['condition']}"


@tool
def search_database(query: str, limit: int = 5) -> list[dict]:
    """Search the database for matching records."""
    return [
        {"id": 1, "title": f"Result for '{query}' - Item 1"},
        {"id": 2, "title": f"Result for '{query}' - Item 2"},
    ][:limit]


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(_safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "Error: Invalid expression"


def example_tulip_tools():
    print("=== Part 1: Tulip Tools ===\n")

    print("Tool: get_weather")
    print(f"  Name: {get_weather.name}")
    print(f"  Description: {get_weather.description}")
    print(f"  Parameters: {json.dumps(get_weather.parameters, indent=4)}")

    print("\nDirect execution:")
    result = get_weather("Tokyo")
    print(f"  get_weather('Tokyo') = {result}")
    print()


# =============================================================================
# Part 2: Schema conversion — Tulip tool -> MCP shape and back.
# =============================================================================


def example_tool_conversion():
    print("=== Part 2: Tool Conversion ===\n")

    mcp_schema = tulip_tool_to_mcp(get_weather)

    print("Tulip tool converted to MCP schema:")
    print(json.dumps(mcp_schema, indent=2))
    print()

    print("MCP tools can be converted to Tulip tools using mcp_tool_to_tulip()")
    print("This allows using external MCP server tools in Tulip agents.")
    print()


# =============================================================================
# Part 3: Publish an agent as an MCP server. Tools + run_agent become
#         callable methods over stdio or SSE.
# =============================================================================


def example_mcp_server():
    print("=== Part 3: MCP Server ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_weather, search_database, calculate],
        system_prompt="You are a helpful assistant with access to weather, search, and calculator tools.",
    )

    server = create_mcp_server(
        agent=agent,
        name="tulip-assistant",
        version="1.0.0",
    )

    print(f"MCP Server created: {server.name} v{server.version}")
    print("Agent tools will be exposed as MCP tools")
    print()

    print("To run the server:")
    print("  server.run()  # Starts stdio transport")
    print("  server.run(transport='sse')  # Starts SSE transport")
    print()

    print("The server exposes:")
    print("  - All agent tools (get_weather, search_database, calculate)")
    print("  - run_agent(prompt) - Run the full agent")
    print("  - run_agent_stream(prompt) - Run with streaming")
    print()

    return server


# =============================================================================
# Part 4: Handle MCP requests programmatically — no full transport needed.
# =============================================================================


async def example_mcp_requests():
    print("=== Part 4: MCP Requests ===\n")

    try:
        import fastmcp  # noqa: F401
    except ImportError:
        print("Note: fastmcp package not installed.")
        print("Install with: pip install fastmcp")
        print()
        print("Without fastmcp, the server structure is shown but requests can't be processed.")
        print("The server.handle_request() method requires fastmcp for full functionality.")
        print()
        return

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_weather, calculate],
        system_prompt="You are helpful.",
    )

    server = TulipMCPServer(agent=agent, name="test-server")

    list_request = {"method": "tools/list", "params": {}}
    list_response = await server.handle_request(list_request)

    print("Request: tools/list")
    print(f"Response: {json.dumps(list_response, indent=2)[:500]}...")
    print()

    call_request = {
        "method": "tools/call",
        "params": {
            "name": "get_weather",
            "arguments": {"city": "London"},
        },
    }
    call_response = await server.handle_request(call_request)

    print("Request: tools/call (get_weather)")
    print(f"Response: {json.dumps(call_response, indent=2)}")
    print()


# =============================================================================
# Part 5: Consume an external MCP server's tools as Tulip tools.
# =============================================================================


def example_mcp_client():
    print("=== Part 5: MCP Client ===\n")

    print("MCPClient allows Tulip agents to use tools from external MCP servers.")
    print()

    print("Example usage:")
    print("""
    # Connect to an MCP server
    client = MCPClient(server_command=["python", "weather_server.py"])
    await client.connect()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {tools}")

    # Call a tool
    result = await client.call_tool("get_weather", {"city": "Paris"})
    print(f"Result: {result}")

    # Convert MCP tools to Tulip tools
    tulip_tools = client.to_tulip_tools(tools)

    # Use in a Tulip agent
    agent = Agent(
        model=model,
        tools=tulip_tools,  # Tools from the MCP server!
        system_prompt="Use the available tools.",
    )

    # Close connection
    await client.close()
    """)
    print()


# =============================================================================
# Part 6: End-to-end — build agent, expose it, hit it with tools/list and
#         a run_agent call that goes through the whole loop.
# =============================================================================


async def example_complete_integration():
    print("=== Part 6: Complete Integration ===\n")

    try:
        import fastmcp  # noqa: F401

        has_fastmcp = True
    except ImportError:
        has_fastmcp = False

    model = get_model(max_tokens=300)

    agent = Agent(
        model=model,
        tools=[get_weather, search_database, calculate],
        system_prompt="""You are a helpful assistant.
Use the available tools to answer questions:
- get_weather: Check weather in cities
- search_database: Search for information
- calculate: Do math calculations""",
    )

    server = create_mcp_server(agent, name="multi-tool-assistant")

    print(f"Created MCP server: {server.name}")
    print(f"Agent tools: {[t.name for t in [get_weather, search_database, calculate]]}")
    print()

    if not has_fastmcp:
        print("Note: fastmcp not installed - showing structure only.")
        print("Install with: pip install fastmcp")
        print()
        print("With fastmcp installed, the server can:")
        print("  - Handle tools/list requests")
        print("  - Handle tools/call requests")
        print("  - Run as stdio or SSE transport")
        print()
        return

    print("Testing MCP server with simulated requests:\n")

    tools_response = await server.handle_request({"method": "tools/list"})
    tool_names = [t["name"] for t in tools_response.get("tools", [])]
    print(f"Available tools: {tool_names}")

    # run_agent exercises a full agent loop through MCP.
    run_response = await server.handle_request(
        {
            "method": "tools/call",
            "params": {
                "name": "run_agent",
                "arguments": {"prompt": "What's the weather in Tokyo?"},
            },
        }
    )
    print(f"\nAgent response: {run_response}")

    print()
    print("This server can now be used by any MCP-compatible client!")
    print()


# =============================================================================
# Part 7: Practical notes — tool design, errors, security, performance.
# =============================================================================


def example_best_practices():
    print("=== Part 7: Best Practices ===\n")

    print("1. Tool Design")
    print("-" * 40)
    print("   - Use clear, descriptive tool names")
    print("   - Write detailed docstrings (they become descriptions)")
    print("   - Use type hints for parameters")
    print("   - Return strings or JSON-serializable data")
    print()

    print("2. Error Handling")
    print("-" * 40)
    print("   - Return error messages as strings, don't raise exceptions")
    print("   - Validate inputs before processing")
    print("   - Include helpful error messages")
    print()

    print("3. Security")
    print("-" * 40)
    print("   - Validate all inputs")
    print("   - Limit what tools can access")
    print("   - Use hooks for additional validation")
    print("   - Don't expose sensitive operations")
    print()

    print("4. Performance")
    print("-" * 40)
    print("   - Keep tools focused and fast")
    print("   - Use async for I/O operations")
    print("   - Consider caching for repeated calls")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 42: MCP Integration")
    print("=" * 60)
    print()

    print_config()
    print()

    example_tulip_tools()
    example_tool_conversion()
    example_mcp_server()
    await example_mcp_requests()
    example_mcp_client()
    await example_complete_integration()
    example_best_practices()

    print("=" * 60)
    print("Done. Next: notebook 42 — playbooks.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
