# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 46: MCP integration — wire support tooling into an agent.

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process — exactly how a
support desk wires its order system, help center, and billing services
into an agent without bundling them. Locus speaks both sides of it.

- Publish a Locus support agent as an MCP server — tools and the
  agent's own ``run_agent`` become MCP methods.
- Convert tool schemas in both directions
  (``locus_tool_to_mcp`` / ``mcp_tool_to_locus``) and call the
  round-tripped tool.
- Handle ``tools/list`` and ``tools/call`` requests programmatically,
  including a ``run_agent`` call that drives the whole agent loop.
- A **refund-eligibility probe**: it reads order signals (days since
  delivery, item condition, order value), a deterministic classifier
  returns an ``EligibilityVerdict``, and a GSAR grounding step either
  ships a grounded eligibility decision or abstains when signal coverage
  is too low.

Consuming an external MCP server (an order system, a knowledge base)
from a Locus agent is the other half of the story — ``MCPClient`` and the
hooks pattern are covered in ``docs/concepts/mcp.md``, along with the
tool-design, error-handling, and safety notes for tools exposed over MCP.

The configured provider drives the agent. The MCP layer is transport-only
— the same agent works against any provider. All tool outputs here are
mock data (invented orders, fixed signals) — no live order system.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_46_mcp_integration.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_46_mcp_integration.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- Optional: ``pip install fastmcp`` to exercise live request handling.

See https://modelcontextprotocol.io for the MCP specification.
"""

import asyncio
import json
from collections.abc import Mapping

from config import get_model, print_config, safe_math_eval
from pydantic import BaseModel

from locus.agent import Agent
from locus.integrations.fastmcp import (
    LocusMCPServer,
    create_mcp_server,
    locus_tool_to_mcp,
    mcp_tool_to_locus,
)
from locus.reasoning.gsar import (
    Claim,
    Decision,
    EvidenceType,
    Partition,
    decide,
    gsar_score,
)
from locus.tools import tool


def _has_fastmcp() -> bool:
    # Request handling needs the fastmcp package; the schema and server
    # construction below do not, so the notebook degrades instead of failing.
    try:
        import fastmcp  # noqa: F401
    except ImportError:
        return False
    return True


# =============================================================================
# Part 1: Three ordinary Locus support tools. Nothing MCP-specific yet.
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
        return str(safe_math_eval(expression))
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


def example_locus_tools():
    print("=== Part 1: Locus Support Tools ===\n")

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
# Part 2: Schema conversion — Locus tool -> MCP shape -> Locus tool again.
# =============================================================================


async def example_tool_conversion():
    print("=== Part 2: Tool Conversion ===\n")

    mcp_schema = locus_tool_to_mcp(order_status)

    print("Locus tool converted to MCP schema:")
    print(json.dumps(mcp_schema, indent=2))
    print()

    # The reverse direction is what an external MCP server's tools go
    # through: name, description, and inputSchema come off the wire, and
    # the callable is whatever actually talks to that server. Here the
    # callable is the original tool, so the round trip can be checked
    # end-to-end — same name, same flat parameter schema, same answer.
    async def _call_order_system(**kwargs):
        return order_status(**kwargs)

    round_tripped = mcp_tool_to_locus(
        name=mcp_schema["name"],
        description=mcp_schema["description"],
        func=_call_order_system,
        parameters=mcp_schema["inputSchema"],
    )
    print("MCP schema converted back to a Locus tool:")
    print(f"  name={round_tripped.name!r}")
    print(f"  parameters preserved: {round_tripped.parameters == order_status.parameters}")
    answer = await round_tripped.execute(order_id="ORD-1003")
    print(f"  execute(order_id='ORD-1003') = {answer}")
    print()


# =============================================================================
# Part 3: Publish an agent as an MCP server. Tools + run_agent become
#         callable methods over stdio or SSE.
# =============================================================================


def example_mcp_server() -> LocusMCPServer:
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
        name="locus-support-desk",
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
# Part 4: Handle MCP requests programmatically — no transport needed.
#         tools/list, a direct tools/call, and a run_agent call that drives
#         the whole agent loop through the server.
# =============================================================================


async def example_mcp_requests(server: LocusMCPServer):
    print("=== Part 4: MCP Requests ===\n")

    if not _has_fastmcp():
        print("Note: fastmcp package not installed.")
        print("Install with: pip install fastmcp")
        print()
        print("Without fastmcp, the server structure is shown but requests can't be processed.")
        print("The server.handle_request() method requires fastmcp for full functionality.")
        print()
        return

    list_response = await server.handle_request({"method": "tools/list", "params": {}})
    tool_names = [t["name"] for t in list_response.get("tools", [])]
    print("Request: tools/list")
    print(f"Available tools: {tool_names}")
    print()

    call_response = await server.handle_request(
        {
            "method": "tools/call",
            "params": {"name": "order_status", "arguments": {"order_id": "ORD-1002"}},
        }
    )
    print("Request: tools/call (order_status)")
    print(f"Response: {json.dumps(call_response, indent=2)}")
    print()

    # run_agent exercises a full agent loop through MCP — the model picks
    # the tool, the server runs it, the answer comes back as one response.
    run_response = await server.handle_request(
        {
            "method": "tools/call",
            "params": {
                "name": "run_agent",
                "arguments": {"prompt": "What's the status of ORD-1001?"},
            },
        }
    )
    print("Request: tools/call (run_agent)")
    print(f"Agent response: {run_response}")
    print()


# =============================================================================
# Part 5: GSAR gate on the probe's verdict — read order signals, classify,
#         then ground the verdict (or abstain on low coverage). The grounding
#         step only "ships" a refund decision when it rests on tool evidence;
#         thin inference abstains. All numbers are mock.
# =============================================================================


def _ground_eligibility(partition: Partition) -> tuple[float, Decision]:
    """Score the evidence partition and turn it into a proceed/abstain call."""
    score = gsar_score(partition)
    return score, decide(score)


def example_eligibility_gate():
    print("=== Part 5: GSAR gate on the refund-eligibility verdict ===\n")

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


async def main():
    print("=" * 60)
    print("Notebook 46: MCP Integration")
    print("=" * 60)
    print()

    print_config()
    print()

    example_locus_tools()
    await example_tool_conversion()
    server = example_mcp_server()
    await example_mcp_requests(server)
    example_eligibility_gate()

    print("=" * 60)
    print("Done. Next: notebook 47 — plugins.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
