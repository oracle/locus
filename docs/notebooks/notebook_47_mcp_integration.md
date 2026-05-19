# MCP Integration

MCP (Model Context Protocol) is the open standard that lets AI
assistants call tools running in a different process. Locus speaks
both sides of it.

- Publish a Locus agent as an MCP server — tools and the agent's own
  `run_agent` become MCP methods.
- Connect a Locus agent to an external MCP server and use its tools as
  ordinary `@tool`-decorated callables.
- Convert tool schemas in both directions
  (`locus_tool_to_mcp` / `mcp_tool_to_locus`).
- Handle `tools/list` and `tools/call` requests programmatically.

OCI GenAI drives the agent by default. The MCP layer is transport-only
— the same agent works against any provider.

## Run it

OCI GenAI is the default (auto-detected from `~/.oci/config`):

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_47_mcp_integration.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_47_mcp_integration.py
```

## Prerequisites

- An OCI profile with GenAI access, or `LOCUS_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.
- Optional: `pip install fastmcp` to exercise live request handling.

See <https://modelcontextprotocol.io> for the MCP specification.

## Source

```python
--8<-- "examples/notebook_47_mcp_integration.py"
```
