# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 45: MCP integration — wire security tooling into an agent.

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process — exactly how a
SOC wires its scanners, intel feeds, and enrichment services into an
agent without bundling them. Tulip speaks both sides of it.

- Publish a Tulip security agent as an MCP server — tools and the
  agent's own ``run_agent`` become MCP methods.
- Connect a Tulip agent to an external MCP server (an intel feed, a
  scanner) and use its tools as ordinary ``@tool``-decorated callables.
- Convert tool schemas in both directions
  (``tulip_tool_to_mcp`` / ``mcp_tool_to_tulip``).
- Handle ``tools/list`` and ``tools/call`` requests programmatically.
- A **timing side-channel inference-fingerprinting probe** exposed as an
  MCP tool: it measures streaming-timing features against an endpoint
  (the Gateway), a deterministic classifier returns a
  ``FingerprintVerdict``, and ``ground_fingerprint`` either ships a
  grounded ``(model, engine, hardware)`` finding or abstains when feature
  coverage is too low (MITRE ATLAS AML.T0024 / AML.T0040).

The configured provider drives the agent. The MCP layer is transport-only
— the same agent works against any provider. All tool outputs here are
mock data (RFC 5737 addresses, invented verdicts) — no live scanning.

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

from tulip.agent import Agent
from tulip.integrations.fastmcp import (
    TulipMCPServer,
    create_mcp_server,
    tulip_tool_to_mcp,
)
from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import (
    AtlasTechnique,
    FingerprintVerdict,
    Indicator,
    IndicatorType,
    Severity,
    ground_fingerprint,
    is_finding,
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
# Part 1: Three ordinary Tulip security tools. Nothing MCP-specific yet.
#         All data is mock — RFC 5737 documentation addresses, fake verdicts.
# =============================================================================


@tool
def ip_reputation(ip: str) -> str:
    """Look up the reputation verdict for an IP address."""
    reputation_data = {
        "192.0.2.44": {"score": 91, "verdict": "malicious"},
        "198.51.100.7": {"score": 55, "verdict": "suspicious"},
        "203.0.113.10": {"score": 3, "verdict": "clean"},
    }
    data = reputation_data.get(ip, {"score": 0, "verdict": "unknown"})
    return f"Reputation for {ip}: score {data['score']}/100, {data['verdict']}"


@tool
def search_threat_intel(query: str, limit: int = 5) -> list[dict]:
    """Search the threat-intel database for matching reports."""
    return [
        {"id": 1, "title": f"Intel report for '{query}' - Report 1"},
        {"id": 2, "title": f"Intel report for '{query}' - Report 2"},
    ][:limit]


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression (e.g. a risk-score formula)."""
    try:
        return str(_safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "Error: Invalid expression"


# The expected timing feature schema for a remote-API fingerprint: TTFT,
# tokens-per-second, and inter-token cadence variance. Real deployments
# measure these against the Gateway over many requests; here they are fixed
# mock numbers so the notebook stays offline and deterministic.
_FINGERPRINT_FEATURES = ("ttft_ms", "tokens_per_sec", "cadence_cv")


def _classify_fingerprint(features: Mapping[str, float]) -> FingerprintVerdict:
    """Deterministic mock fingerprint classifier (no model file, no network).

    Maps a timing feature vector to a ``(model, engine, hardware)`` verdict
    over a fixed lookup. ``feature_coverage`` is the fraction of the expected
    schema actually present — low coverage yields low confidence so the
    grounding step abstains rather than asserting a fingerprint.
    """
    coverage = sum(1 for f in _FINGERPRINT_FEATURES if f in features) / len(_FINGERPRINT_FEATURES)
    # A fast TTFT + high throughput pattern that the reference table maps to a
    # known open-weights model behind vLLM on a datacenter GPU.
    looks_vllm = features.get("ttft_ms", 9e9) < 120 and features.get("tokens_per_sec", 0) > 80
    return FingerprintVerdict(
        model="open-weights-8b" if looks_vllm else "unknown",
        engine="vLLM" if looks_vllm else "unknown",
        hardware="datacenter-gpu" if looks_vllm else "unknown",
        confidence=round(0.9 * coverage, 4),
        feature_coverage=round(coverage, 4),
    )


@tool
def fingerprint_endpoint(features_json: str) -> str:
    """Fingerprint a model endpoint from timing side-channel features.

    Pass a JSON object of timing features (ttft_ms, tokens_per_sec,
    cadence_cv). Returns the classifier verdict as JSON. Measurement only —
    no privileges, no exploit (MITRE ATLAS AML.T0040 inference-API access).
    """
    try:
        features = {k: float(v) for k, v in json.loads(features_json).items()}
    except (ValueError, TypeError):
        return '{"error": "features_json must be a JSON object of numbers"}'
    return _classify_fingerprint(features).model_dump_json()


def example_tulip_tools():
    print("=== Part 1: Tulip Security Tools ===\n")

    print("Tool: ip_reputation")
    print(f"  Name: {ip_reputation.name}")
    print(f"  Description: {ip_reputation.description}")
    print(f"  Parameters: {json.dumps(ip_reputation.parameters, indent=4)}")

    print("\nDirect execution:")
    result = ip_reputation("192.0.2.44")
    print(f"  ip_reputation('192.0.2.44') = {result}")

    print("\nTool: fingerprint_endpoint (timing side-channel probe)")
    fp = fingerprint_endpoint('{"ttft_ms": 95, "tokens_per_sec": 140, "cadence_cv": 0.07}')
    print(f"  fingerprint_endpoint(full features) = {fp}")
    print()


# =============================================================================
# Part 2: Schema conversion — Tulip tool -> MCP shape and back.
# =============================================================================


def example_tool_conversion():
    print("=== Part 2: Tool Conversion ===\n")

    mcp_schema = tulip_tool_to_mcp(ip_reputation)

    print("Tulip tool converted to MCP schema:")
    print(json.dumps(mcp_schema, indent=2))
    print()

    print("MCP tools can be converted to Tulip tools using mcp_tool_to_tulip()")
    print("This lets a Tulip agent use tools from external MCP servers —")
    print("a scanner, a sandbox, an intel feed — without bundling them.")
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
        tools=[ip_reputation, search_threat_intel, calculate, fingerprint_endpoint],
        system_prompt=(
            "You are a SOC enrichment assistant with IP reputation, threat-intel "
            "search, a risk-score calculator, and a timing-fingerprint probe."
        ),
    )

    server = create_mcp_server(
        agent=agent,
        name="tulip-sec-enrichment",
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
        "  - All agent tools (ip_reputation, search_threat_intel, calculate, fingerprint_endpoint)"
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
        tools=[ip_reputation, calculate],
        system_prompt="You are a SOC enrichment assistant.",
    )

    server = TulipMCPServer(agent=agent, name="test-enrichment-server")

    list_request = {"method": "tools/list", "params": {}}
    list_response = await server.handle_request(list_request)

    print("Request: tools/list")
    print(f"Response: {json.dumps(list_response, indent=2)[:500]}...")
    print()

    call_request = {
        "method": "tools/call",
        "params": {
            "name": "ip_reputation",
            "arguments": {"ip": "198.51.100.7"},
        },
    }
    call_response = await server.handle_request(call_request)

    print("Request: tools/call (ip_reputation)")
    print(f"Response: {json.dumps(call_response, indent=2)}")
    print()


# =============================================================================
# Part 5: Consume an external MCP server's tools as Tulip tools.
# =============================================================================


def example_mcp_client():
    print("=== Part 5: MCP Client ===\n")

    print("MCPClient lets Tulip agents use tools from external MCP servers —")
    print("e.g. your team's scanner or intel-feed server running out of process.")
    print()

    print("Example usage:")
    print("""
    # Connect to a security-tools MCP server
    client = MCPClient(server_command=["python", "intel_server.py"])
    await client.connect()

    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {tools}")

    # Call a tool
    result = await client.call_tool("ip_reputation", {"ip": "192.0.2.44"})
    print(f"Result: {result}")

    # Convert MCP tools to Tulip tools
    tulip_tools = client.to_tulip_tools(tools)

    # Use in a Tulip agent
    agent = Agent(
        model=model,
        tools=tulip_tools,  # Tools from the MCP server!
        system_prompt="Enrich indicators with the available tools.",
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
        tools=[ip_reputation, search_threat_intel, calculate],
        system_prompt="""You are a SOC enrichment assistant.
Use the available tools to answer questions:
- ip_reputation: Check the reputation of an IP address
- search_threat_intel: Search threat-intel reports
- calculate: Do risk-score math""",
    )

    server = create_mcp_server(agent, name="soc-enrichment-assistant")

    print(f"Created MCP server: {server.name}")
    print(f"Agent tools: {[t.name for t in [ip_reputation, search_threat_intel, calculate]]}")
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
                "arguments": {"prompt": "What's the reputation of 192.0.2.44?"},
            },
        }
    )
    print(f"\nAgent response: {run_response}")

    print()
    print("This server can now be used by any MCP-compatible client!")
    print()


# =============================================================================
# Part 7: The fingerprint probe over MCP — measure timing features against an
#         endpoint, classify, then ground the verdict (or abstain on low
#         coverage). MITRE ATLAS AML.T0024 (exfiltration via inference API) /
#         AML.T0040 (inference-API access). All numbers are mock.
# =============================================================================


async def example_fingerprint_probe():
    print("=== Part 7: Inference-Fingerprinting Probe ===\n")

    # Expose the probe as an MCP tool alongside the enrichment tools.
    print("Tool:", fingerprint_endpoint.name)
    print(f"  Description: {fingerprint_endpoint.description}\n")

    endpoint = "203.0.113.10:8000"  # the Gateway, RFC 5737 documentation IP

    # Full feature coverage: TTFT, throughput, and cadence variance all
    # measured. The probe call is the tool-output evidence; the verdict's
    # typed fields are specific data lifted from that output.
    full = {"ttft_ms": 95.0, "tokens_per_sec": 140.0, "cadence_cv": 0.07}
    verdict = _classify_fingerprint(full)
    print(f"Measured features (full): {full}")
    print(
        f"  VerificationResult: {verdict.model} / {verdict.engine} / {verdict.hardware} "
        f"(confidence={verdict.confidence}, coverage={verdict.feature_coverage})"
    )

    grounded = ground_fingerprint(
        verdict=verdict,
        asset=endpoint,
        partition=Partition(
            grounded=[
                Claim(
                    text=f"Streaming timing probe of {endpoint} returned "
                    f"TTFT={full['ttft_ms']}ms, {full['tokens_per_sec']} tok/s.",
                    type=EvidenceType.TOOL_MATCH,
                    evidence_refs=[f"tool:fingerprint_endpoint:{endpoint}:full"],
                ),
                Claim(
                    text=f"Classifier mapped the timing vector to {verdict.model} "
                    f"on {verdict.engine}.",
                    type=EvidenceType.SPECIFIC_DATA,
                    evidence_refs=["classifier:reference_table:row=open-weights-8b"],
                ),
            ],
        ),
        severity=Severity.MEDIUM,
        indicators=[Indicator(type=IndicatorType.ENDPOINT, value=endpoint)],
        taxonomy=[AtlasTechnique.INFERENCE_API_ACCESS],  # AML.T0040
    )
    print("  ground_fingerprint (full coverage):")
    if is_finding(grounded):
        print(f"    FINDING  S={grounded.gsar_score:.4f}  {grounded.title}")
    else:
        print(f"    ABSTAINED  ({grounded.reason})")

    # Low coverage: only TTFT was observable, so the classifier returns low
    # confidence. Under the Covenant the inference-grade evidence does not
    # clear the bar — ground_fingerprint abstains rather than naming a model.
    sparse = {"ttft_ms": 95.0}
    weak_verdict = _classify_fingerprint(sparse)
    print(f"\nMeasured features (sparse): {sparse}")
    print(
        f"  VerificationResult coverage={weak_verdict.feature_coverage}, confidence={weak_verdict.confidence}"
    )
    abstained = ground_fingerprint(
        verdict=weak_verdict,
        asset=endpoint,
        partition=Partition(
            ungrounded=[
                Claim(
                    text=f"Endpoint {endpoint} is probably {weak_verdict.model}.",
                    type=EvidenceType.INFERENCE,
                ),
            ],
        ),
        severity=Severity.MEDIUM,
        taxonomy=[AtlasTechnique.INFERENCE_API_ACCESS],
    )
    print("  ground_fingerprint (low coverage):")
    if is_finding(abstained):
        print(f"    FINDING  S={abstained.gsar_score:.4f}  {abstained.title}")
    else:
        print(f"    ABSTAINED  S={abstained.gsar_score:.4f}  ({abstained.reason})")
    print()


# =============================================================================
# Part 8: Practical notes — tool design, errors, security, performance.
# =============================================================================


def example_best_practices():
    print("=== Part 8: Best Practices ===\n")

    print("1. Tool Design")
    print("-" * 40)
    print("   - Use clear, descriptive tool names (ip_reputation, not lookup)")
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
    print("   - Validate all inputs — tool output from a scanner is untrusted")
    print("   - Limit what tools can access; read-only enrichment by default")
    print("   - Use hooks for additional validation and audit logging")
    print("   - Don't expose containment or destructive operations over MCP")
    print()

    print("4. Performance")
    print("-" * 40)
    print("   - Keep tools focused and fast")
    print("   - Use async for I/O operations")
    print("   - Consider caching for repeated indicator lookups")
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
    await example_fingerprint_probe()
    example_best_practices()

    print("=" * 60)
    print("Done. Next: notebook 46 — playbooks.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
