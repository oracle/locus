# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 45: MCP integration — wire support tooling into an agent.

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process — exactly how a
support desk wires its order system, help center, and billing services
into an agent without bundling them. Tulip speaks both sides of it.

- Publish a Tulip support agent as an MCP server — tools and the
  agent's own ``run_agent`` become MCP methods.
- Connect a Tulip agent to an external MCP server (an order system, a
  knowledge base) and use its tools as ordinary ``@tool``-decorated
  callables.
- Convert tool schemas in both directions
  (``tulip_tool_to_mcp`` / ``mcp_tool_to_tulip``).
- Handle ``tools/list`` and ``tools/call`` requests programmatically.
- A **refund-eligibility probe** exposed as an MCP tool: it reads order
  signals (days since delivery, item condition, order value), a
  deterministic classifier returns an ``EligibilityVerdict``, and a
  grounding step either ships a grounded eligibility decision or abstains
  when signal coverage is too low.

The configured provider drives the agent. The MCP layer is transport-only
— the same agent works against any provider. All tool outputs here are
mock data (invented orders, fixed signals) — no live order system.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_45_mcp_integration.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_45_mcp_integration.py

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
from collections.abc import Mapping

# Import shared config for model
from config import get_model, print_config
from pydantic import BaseModel

from tulip.agent import Agent
from tulip.integrations.fastmcp import (
    TulipMCPServer,
    create_mcp_server,
    tulip_tool_to_mcp,
)
from tulip.reasoning.gsar import (
    Claim,
    Decision,
    EvidenceType,
    Partition,
    decide,
    gsar_score,
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
# Part 1: Three ordinary Tulip support tools. Nothing MCP-specific yet.
#         All data is mock — invented order ids, canned articles.
# =============================================================================


@tool
def order_status(order_id: str) -> str:
    """Look up the shipping status for an order."""
    order_data = {
        "ORD-1001": {"status": "delivered", "days_ago": 2},
        "ORD-1002": {"status": "in_transit", "days_ago": 0},
        "ORD-1003": {"status": "delivered", "days_ago": 45},
    }
    data = order_data.get(order_id, {"status": "not_found", "days_ago": 0})
    return f"Order {order_id}: {data['status']} ({data['days_ago']} days ago)"


@tool
def search_help_articles(query: str, limit: int = 5) -> list[dict]:
    """Search the help center for matching articles."""
    return [
        {"id": 1, "title": f"Help article for '{query}' - Article 1"},
        {"id": 2, "title": f"Help article for '{query}' - Article 2"},
    ][:limit]


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression (e.g. a prorated-refund formula)."""
    try:
        return str(_safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "Error: Invalid expression"


# The expected signal schema for a refund-eligibility check: days since
# delivery, an item-condition score, and the order value. Real desks read
# these from the order/returns system; here they are fixed mock numbers so
# the notebook stays offline and deterministic.
_ELIGIBILITY_SIGNALS = ("days_since_delivery", "item_condition_score", "order_value_usd")


class EligibilityVerdict(BaseModel):
    """A refund-eligibility decision plus its confidence and coverage."""

    decision: str
    reason: str
    confidence: float
    signal_coverage: float


def _classify_eligibility(signals: Mapping[str, float]) -> EligibilityVerdict:
    """Deterministic mock eligibility classifier (no model file, no network).

    Maps an order-signal vector to a refund decision over a fixed rule.
    ``signal_coverage`` is the fraction of the expected schema actually
    present — low coverage yields low confidence so the grounding step
    abstains rather than promising a refund.
    """
    coverage = sum(1 for s in _ELIGIBILITY_SIGNALS if s in signals) / len(_ELIGIBILITY_SIGNALS)
    # Inside the 30-day window and returned in good condition: a clean refund.
    within_window = signals.get("days_since_delivery", 9e9) <= 30
    good_condition = signals.get("item_condition_score", 0) >= 0.8
    eligible = within_window and good_condition
    return EligibilityVerdict(
        decision="refund" if eligible else "manual_review",
        reason="within window, good condition" if eligible else "needs an agent to review",
        confidence=round(0.9 * coverage, 4),
        signal_coverage=round(coverage, 4),
    )


@tool
def check_refund_eligibility(signals_json: str) -> str:
    """Check refund eligibility from order signals.

    Pass a JSON object of order signals (days_since_delivery,
    item_condition_score, order_value_usd). Returns the classifier verdict
    as JSON. Read-only assessment — it never issues the refund itself.
    """
    try:
        signals = {k: float(v) for k, v in json.loads(signals_json).items()}
    except (ValueError, TypeError):
        return '{"error": "signals_json must be a JSON object of numbers"}'
    return _classify_eligibility(signals).model_dump_json()


def example_tulip_tools():
    print("=== Part 1: Tulip Support Tools ===\n")

    print("Tool: order_status")
    print(f"  Name: {order_status.name}")
    print(f"  Description: {order_status.description}")
    print(f"  Parameters: {json.dumps(order_status.parameters, indent=4)}")

    print("\nDirect execution:")
    result = order_status("ORD-1001")
    print(f"  order_status('ORD-1001') = {result}")

    print("\nTool: check_refund_eligibility (refund-eligibility probe)")
    elig = check_refund_eligibility(
        '{"days_since_delivery": 2, "item_condition_score": 0.95, "order_value_usd": 60}'
    )
    print(f"  check_refund_eligibility(full signals) = {elig}")
    print()


# =============================================================================
# Part 2: Schema conversion — Tulip tool -> MCP shape and back.
# =============================================================================


def example_tool_conversion():
    print("=== Part 2: Tool Conversion ===\n")

    mcp_schema = tulip_tool_to_mcp(order_status)

    print("Tulip tool converted to MCP schema:")
    print(json.dumps(mcp_schema, indent=2))
    print()

    print("MCP tools can be converted to Tulip tools using mcp_tool_to_tulip()")
    print("This lets a Tulip agent use tools from external MCP servers —")
    print("an order system, a returns service, a knowledge base — without bundling them.")
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
        tools=[order_status, search_help_articles, calculate, check_refund_eligibility],
        system_prompt=(
            "You are a customer-support assistant with order status lookup, help-center "
            "search, a refund calculator, and a refund-eligibility probe."
        ),
    )

    server = create_mcp_server(
        agent=agent,
        name="tulip-support-desk",
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
    print(
        "  - All agent tools (order_status, search_help_articles, calculate, check_refund_eligibility)"
    )
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
        tools=[order_status, calculate],
        system_prompt="You are a customer-support assistant.",
    )

    server = TulipMCPServer(agent=agent, name="test-support-server")

    list_request = {"method": "tools/list", "params": {}}
    list_response = await server.handle_request(list_request)

    print("Request: tools/list")
    print(f"Response: {json.dumps(list_response, indent=2)[:500]}...")
    print()

    call_request = {
        "method": "tools/call",
        "params": {
            "name": "order_status",
            "arguments": {"order_id": "ORD-1002"},
        },
    }
    call_response = await server.handle_request(call_request)

    print("Request: tools/call (order_status)")
    print(f"Response: {json.dumps(call_response, indent=2)}")
    print()


# =============================================================================
# Part 5: Consume an external MCP server's tools as Tulip tools.
# =============================================================================


def example_mcp_client():
    print("=== Part 5: MCP Client ===\n")

    print("MCPClient lets Tulip agents use tools from external MCP servers —")
    print("e.g. your team's order system or returns service running out of process.")
    print()

    print("Example usage:")
    print("""
    # Connect to a support-tools MCP server
    client = MCPClient(server_command=["python", "orders_server.py"])
    await client.connect()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {tools}")

    # Call a tool
    result = await client.call_tool("order_status", {"order_id": "ORD-1001"})
    print(f"Result: {result}")

    # Convert MCP tools to Tulip tools
    tulip_tools = client.to_tulip_tools(tools)

    # Use in a Tulip agent
    agent = Agent(
        model=model,
        tools=tulip_tools,  # Tools from the MCP server!
        system_prompt="Resolve tickets with the available tools.",
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
        tools=[order_status, search_help_articles, calculate],
        system_prompt="""You are a customer-support assistant.
Use the available tools to answer questions:
- order_status: Check the shipping status of an order
- search_help_articles: Search help-center articles
- calculate: Do prorated-refund math""",
    )

    server = create_mcp_server(agent, name="support-desk-assistant")

    print(f"Created MCP server: {server.name}")
    print(f"Agent tools: {[t.name for t in [order_status, search_help_articles, calculate]]}")
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
                "arguments": {"prompt": "What's the status of ORD-1001?"},
            },
        }
    )
    print(f"\nAgent response: {run_response}")

    print()
    print("This server can now be used by any MCP-compatible client!")
    print()


# =============================================================================
# Part 7: The eligibility probe over MCP — read order signals, classify, then
#         ground the verdict (or abstain on low coverage). The grounding step
#         only "ships" a refund decision when it rests on tool evidence; thin
#         inference abstains. All numbers are mock.
# =============================================================================


def _ground_eligibility(partition: Partition) -> tuple[float, Decision]:
    """Score the evidence partition and turn it into a proceed/abstain call."""
    score = gsar_score(partition)
    return score, decide(score)


async def example_eligibility_probe():
    print("=== Part 7: Refund-Eligibility Probe ===\n")

    # Expose the probe as an MCP tool alongside the support tools.
    print("Tool:", check_refund_eligibility.name)
    print(f"  Description: {check_refund_eligibility.description}\n")

    order_id = "ORD-1001"

    # Full signal coverage: days since delivery, item condition, and order
    # value all read. The probe call is the tool-output evidence; the
    # verdict's typed fields are specific data lifted from that output.
    full = {"days_since_delivery": 2.0, "item_condition_score": 0.95, "order_value_usd": 60.0}
    verdict = _classify_eligibility(full)
    print(f"Order signals (full): {full}")
    print(
        f"  Verdict: {verdict.decision} — {verdict.reason} "
        f"(confidence={verdict.confidence}, coverage={verdict.signal_coverage})"
    )

    grounded = Partition(
        grounded=[
            Claim(
                text=f"Returns probe of {order_id} read days_since_delivery="
                f"{full['days_since_delivery']}, condition={full['item_condition_score']}.",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=[f"tool:check_refund_eligibility:{order_id}:full"],
            ),
            Claim(
                text=f"Classifier mapped the signal vector to '{verdict.decision}'.",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["classifier:refund_rules:row=within_window"],
            ),
        ],
    )
    score, call = _ground_eligibility(grounded)
    print("  ground_eligibility (full coverage):")
    if call == Decision.PROCEED:
        print(f"    SHIP  S={score:.4f}  decision={verdict.decision}")
    else:
        print(f"    ABSTAIN  S={score:.4f}  ({call.name})")

    # Low coverage: only days_since_delivery was observable, so the classifier
    # returns low confidence. The thin inference does not clear the bar — the
    # grounding step abstains rather than promising a refund.
    sparse = {"days_since_delivery": 2.0}
    weak_verdict = _classify_eligibility(sparse)
    print(f"\nOrder signals (sparse): {sparse}")
    print(
        f"  Verdict coverage={weak_verdict.signal_coverage}, confidence={weak_verdict.confidence}"
    )
    ungrounded = Partition(
        ungrounded=[
            Claim(
                text=f"Order {order_id} is probably '{weak_verdict.decision}'.",
                type=EvidenceType.INFERENCE,
            ),
        ],
    )
    score2, call2 = _ground_eligibility(ungrounded)
    print("  ground_eligibility (low coverage):")
    if call2 == Decision.PROCEED:
        print(f"    SHIP  S={score2:.4f}  decision={weak_verdict.decision}")
    else:
        print(f"    ABSTAIN  S={score2:.4f}  ({call2.name})")
    print()


# =============================================================================
# Part 8: Practical notes — tool design, errors, safety, performance.
# =============================================================================


def example_best_practices():
    print("=== Part 8: Best Practices ===\n")

    print("1. Tool Design")
    print("-" * 40)
    print("   - Use clear, descriptive tool names (order_status, not lookup)")
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

    print("3. Safety")
    print("-" * 40)
    print("   - Validate all inputs — tool output from an order system is untrusted")
    print("   - Limit what tools can access; read-only lookups by default")
    print("   - Use hooks for additional validation and audit logging")
    print("   - Don't expose refund or account-mutating operations over MCP")
    print()

    print("4. Performance")
    print("-" * 40)
    print("   - Keep tools focused and fast")
    print("   - Use async for I/O operations")
    print("   - Consider caching for repeated order lookups")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 45: MCP Integration")
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
    await example_eligibility_probe()
    example_best_practices()

    print("=" * 60)
    print("Done. Next: notebook 46 — playbooks.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
