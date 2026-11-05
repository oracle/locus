# MCP integration

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process — the way a
support desk wires its order system, help center and billing services
into an agent without bundling them. The SDK speaks both sides of the
protocol. This notebook builds the publishing side: a support agent
whose tools, and the agent itself, can be served to any MCP client.
Consuming an external MCP server from an agent (`MCPClient`, and
the hooks pattern for side effects in the host process) is the other
half, covered in
[MCP](../concepts/mcp.md#getting-started-consume-an-mcp-server).

The configured provider drives the agent; the MCP layer is transport
only, so the same agent works against any provider. All tool outputs are
mock data (invented order ids, fixed signals) — no live order system is
involved.

What this notebook covers:

- Publish a support agent as a MCP server with
  `create_mcp_server`: the agent's tools, plus `run_agent` and
  `run_agent_stream`, are registered as MCP tools.
- Convert tool schemas in both directions (`locus_tool_to_mcp` /
  `mcp_tool_to_locus`) and call the round-tripped tool.
- Handle `tools/list` and `tools/call` requests programmatically through
  `LocusMCPServer.handle_request`, including a `run_agent` call that
  drives the whole agent loop.
- Gate a deterministic **refund-eligibility probe** with
  [GSAR (typed grounding)](../concepts/gsar.md): `gsar_score` scores the
  evidence partition and `decide` returns `PROCEED` only when that score
  reaches the proceed threshold; the example ships the refund decision
  on `PROCEED` and abstains on any other decision.

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `oci`, `openai` or `anthropic` with the
  matching credentials, or `mock` for no credentials.
- Optional: `pip install fastmcp` to exercise request handling in Part 4.

## Key lines

`locus_tool_to_mcp` gives a tool its MCP shape (`name`,
`description`, `inputSchema`); `mcp_tool_to_locus` is the reverse path an
external server's tools take, and because the schema it is handed is kept
as-is, the round trip keeps the same name, the same parameter schema and
the same answer:

{{ locus_source examples/notebook_46_mcp_integration.py::example_tool_conversion }}

The support agent is an ordinary `Agent` with four tools;
`create_mcp_server` wraps it in a `LocusMCPServer`. `server.run()` serves
it over stdio by default or over HTTP (`transport="http"`; `"sse"`
selects the older HTTP+SSE transport, SSE being Server-Sent Events), and
the server it builds registers every agent tool plus `run_agent` and
`run_agent_stream` as MCP tools:

{{ locus_source examples/notebook_46_mcp_integration.py::example_mcp_server }}

`LocusMCPServer.handle_request` answers `tools/list` and `tools/call`
in process, without a transport: `tools/list` returns the agent's tools
in their MCP shape (`run_agent` is not in that list; it is registered on
the server that `run()` serves), `tools/call` with a tool name runs that
tool and returns its text as a content block, and `tools/call` on
`run_agent` runs the whole agent loop through `Agent.run_sync`. Building
the handler imports `fastmcp`, so when the package is not installed the
step prints an install note and returns:

{{ locus_source examples/notebook_46_mcp_integration.py::example_mcp_requests }}

## Run it

```bash
python examples/notebook_46_mcp_integration.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. Request handling in
Part 4 needs the optional `fastmcp` package; without it the script prints an
install note and skips that part, and the rest runs unchanged.

Offline, with the bundled mock model:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_46_mcp_integration.py
```

## Output

Running it offline — no credentials, bundled mock model — prints the MCP schema derived from a tool and its round trip back with parameters preserved, the `tools/list` catalogue and a `tools/call` result (Part 4 ran because the optional `fastmcp` package was installed for this capture; without it the script prints an install note there instead), then the GSAR ship/abstain trail (SHIP at S=1.0000 on full refund signals, ABSTAIN/REPLAN at S=0.0000 on sparse ones):

```text
============================================================
Notebook 46: MCP Integration
============================================================

Model Provider: mock
Using mock model (no API calls)

=== Part 1: Locus Support Tools ===

Tool: order_status
  Name: order_status
  Description: Look up the shipping status for an order.
  Parameters: {
    "type": "object",
    "properties": {
        "order_id": {
            "type": "string"
        }
    },
    "required": [
        "order_id"
    ]
}

Direct execution:
  order_status('ORD-1001') = Order ORD-1001: delivered (2 days ago)

Tool: check_refund_eligibility (refund-eligibility probe)
  check_refund_eligibility(full signals) = {"decision":"refund","reason":"within window, good condition","confidence":0.9,"signal_coverage":1.0}

=== Part 2: Tool Conversion ===

Locus tool converted to MCP schema:
{
  "name": "order_status",
  "description": "Look up the shipping status for an order.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string"
      }
    },
    "required": [
      "order_id"
    ]
  }
}

MCP schema converted back to a Locus tool:
  name='order_status'
  parameters preserved: True
  execute(order_id='ORD-1003') = Order ORD-1003: delivered (45 days ago)

=== Part 3: MCP Server ===

MCP Server created: locus-support-desk v1.0.0
Agent tools will be exposed as MCP tools

To run the server:
  server.run()  # Starts stdio transport
  server.run(transport='sse')  # Starts SSE transport

The server exposes:
  - All agent tools (order_status, search_help_articles, calculate, check_refund_eligibility)
  - run_agent(prompt) - Run the full agent
  - run_agent_stream(prompt) - Run with streaming

=== Part 4: MCP Requests ===

Request: tools/list
Available tools: ['order_status', 'search_help_articles', 'calculate', 'check_refund_eligibility']

Request: tools/call (order_status)
Response: {
  "content": [
    {
      "type": "text",
      "text": "Order ORD-1002: in_transit (0 days ago)"
    }
  ]
}

Request: tools/call (run_agent)
Agent response: {'content': [{'type': 'text', 'text': 'This is a mock response for testing purposes.'}]}

=== Part 5: GSAR gate on the refund-eligibility verdict ===

Tool: check_refund_eligibility
  Description: Check refund eligibility from order signals.

Pass a JSON object of order signals (days_since_delivery,
item_condition_score, order_value_usd). Returns the classifier verdict
as JSON. Read-only assessment — it never issues the refund itself.

Order signals (full): {'days_since_delivery': 2.0, 'item_condition_score': 0.95, 'order_value_usd': 60.0}
  Verdict: refund — within window, good condition (confidence=0.9, coverage=1.0)
  ground_eligibility (full coverage):
    SHIP  S=1.0000  decision=refund

Order signals (sparse): {'days_since_delivery': 2.0}
  Verdict coverage=0.3333, confidence=0.3
  ground_eligibility (low coverage):
    ABSTAIN  S=0.0000  (REPLAN)

============================================================
Done. Next: notebook 47 — plugins.
============================================================
```
<!-- notebook-output:end -->

## See also

- [MCP](../concepts/mcp.md) — both sides of the protocol: consuming a
  server with `MCPClient`, the
  [hooks pattern](../concepts/mcp.md#side-effects-in-the-host-process-use-hooks-not-wrappers)
  for side effects in the host process, and the
  [transports](../concepts/mcp.md#transports) the server can run on.
- [GSAR (typed grounding)](../concepts/gsar.md) — the scoring and
  decision layer behind the refund-eligibility gate.
- [Tools](../concepts/tools.md) — the `@tool` decorator the MCP layer
  wraps.
- <https://modelcontextprotocol.io> — the MCP specification.

## Source

??? note "Full source — `examples/notebook_46_mcp_integration.py`"

    ````python
    --8<-- "examples/notebook_46_mcp_integration.py"
    ````
