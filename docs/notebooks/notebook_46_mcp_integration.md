# MCP Integration

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process — exactly how a
support desk wires its order system, help center, and billing services
into an agent without bundling them. Locus speaks both sides of it.

- Publish a Locus support agent as an MCP server — tools and the agent's
  own `run_agent` become MCP methods.
- Connect a Locus agent to an external MCP server (an order system, a
  knowledge base) and use its tools as ordinary `@tool`-decorated
  callables.
- Convert tool schemas in both directions
  (`locus_tool_to_mcp` / `mcp_tool_to_locus`).
- Handle `tools/list` and `tools/call` requests programmatically.
- Gate a deterministic **refund-eligibility probe** with GSAR grounding
  (`gsar_score` / `decide`): ship the refund decision only when the
  evidence partition clears the threshold, abstain otherwise.

The configured provider drives the agent. The MCP layer is transport-only
— the same agent works against any provider. All tool outputs are mock
data (invented order ids, fixed signals) — no live order system.

## Run it

OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_46_mcp_integration.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_46_mcp_integration.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.
- Optional: `pip install fastmcp` to exercise live request handling.

See <https://modelcontextprotocol.io> for the MCP specification.

## Source

````python
--8<-- "examples/notebook_46_mcp_integration.py"
````
