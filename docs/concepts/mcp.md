# MCP — Model Context Protocol

The [Model Context Protocol](https://modelcontextprotocol.io) is an
Anthropic-spec interop standard for tools. Define a tool once,
expose it over MCP, and any MCP-compatible client (Claude Desktop,
Cline, Strands, another agent built with Tulip) can call it. Or consume tools from existing
MCP servers (filesystem, git, postgres, github, sequential-thinking)
without writing any glue.

**The SDK speaks MCP both ways**. That's a deliberate differentiator —
most agent frameworks consume MCP servers but don't expose their own
tools as MCP. Round-trip means an SDK-built agent can be either side
of the conversation.

## When to use MCP

| You want… | Use MCP |
|---|---|
| Your SDK agent to use Anthropic's published filesystem / git / postgres servers | ✓ — `MCPClient` |
| Your `@tool` library to be callable by Claude Desktop / Cline / other agents | ✓ — `TulipMCPServer` |
| Two SDK agents to share tools across processes / machines | ✓ — works, but [A2A](multi-agent/a2a.md) is the better protocol |
| In-process multi-agent — share tools by importing | use the [tools](tools.md) directly, not MCP |
| Reproducible tests | use a mock model + plain `@tool` — MCP adds I/O |

## Getting started — consume an MCP server

### 1. Install the MCP extras

```bash
pip install "tulip-agents[mcp]"
```

### 2. Spawn the server and wrap it with `MCPClient`

```python
from tulip.integrations.fastmcp import MCPClient

# Spawn Anthropic's filesystem server as a subprocess (stdio transport):
fs = MCPClient.stdio(
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"],
)
```

`MCPClient.stdio` runs the subprocess, opens an MCP session over its
stdin/stdout, and discovers what tools the server exposes.

### 3. Pass the tools straight into an Agent

```python
from tulip.agent import Agent
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[*fs.tools()],          # MCP tools become SDK tools
    system_prompt="You can read files in /data.",
)
result = agent.run_sync("Summarise the README in /data.")
```

`fs.tools()` returns a list of SDK `Tool` objects with full
schemas, descriptions, and call-through plumbing. The agent doesn't
know they're MCP — they look like any other `@tool`.

### Side effects in the host process — use hooks, not wrappers

A common shape for MCP integrators: the *real* effect of a tool call
lives in the host process (an HTTP response queue, a transaction batch,
a UI command stream), not inside the tool body that returns a string to
the model. The instinct is to wrap each MCP tool with a per-tool
`@tool` that calls `_action_queue().append(...)` before returning.

Don't. Use a single `HookProvider` instead:

```python
from tulip.hooks.provider import HookPriority, HookProvider

class MCPActionQueueHook(HookProvider):
    """Mirror every tool call into a host-side queue, keyed by call id."""

    priority = HookPriority.BUSINESS_DEFAULT

    def __init__(self, queue: list[dict]) -> None:
        self._queue = queue

    async def on_after_tool_call(self, event):
        if event.error is None:
            self._queue.append({
                "id": event.tool_call_id,
                "tool": event.tool_name,
                "args": event.arguments,
                "result": event.result,
            })

agent = Agent(
    model=...,
    tools=[*mcp_client.tools()],   # all 24 MCP tools, untouched
    hooks=[MCPActionQueueHook(queue)],
)
```

One hook covers every MCP-sourced tool. The `tool_call_id` correlates
with the model's `tool_calls[].id`, so parallel tool calls don't get
mixed up. See [hooks](hooks.md#on_after_tool_call-what-the-event-carries)
for the full event surface.

## Getting started — expose your tools as MCP

### 1. Wrap a tool list in `TulipMCPServer`

```python
from tulip.integrations.fastmcp import TulipMCPServer

server = TulipMCPServer(tools=[search_vendors, submit_po])
```

### 2. Pick a transport

```python
server.run_stdio()                    # for desktop clients
server.run_http(port=7400)            # for HTTP MCP clients
```

`run_stdio()` is what Claude Desktop, Cline, and most MCP clients
expect. `run_http()` runs an HTTP MCP server (transport + JSON-RPC)
that any HTTP MCP client can reach.

### 3. Point a client at it

For Claude Desktop, edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-tulip-tools": {
      "command": "python",
      "args": ["-m", "my_package.mcp_server"]
    }
  }
}
```

Restart Claude Desktop. Your `search_vendors` and `submit_po` tools
appear in the model's tool list.

## What you get out of the box

### Schema preservation

`@tool`'s docstring + type hints become the MCP tool's name,
description, and JSON schema — losslessly. The MCP client sees the
same parameter types, defaults, and descriptions an SDK agent
would.

### Both transports

| Transport | Use case |
|---|---|
| **stdio** — process pipes | Desktop clients (Claude Desktop, Cline). The MCP server is spawned as a subprocess. |
| **HTTP** — JSON-RPC over POST | Browser-side or networked clients. Good for shared tool servers. |

### Idempotency carries through

A tool tagged `@tool(idempotent=True)` keeps that semantic when
exposed via MCP. The dedup happens SDK-side; the MCP client
doesn't need to know.

## Round-trip example

A common shape: an SDK agent A consumes a filesystem MCP server,
*and* exposes its own tools as MCP for another agent B to consume:

```python
# Agent A — consumes filesystem, exposes its own analytics tools
fs = MCPClient.stdio(command=[...])      # consumer side
analytics = TulipMCPServer(              # producer side
    tools=[summarise_csv, plot_histogram],
)
analytics.run_http(port=7400, in_background=True)

agent_a = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[*fs.tools(), summarise_csv, plot_histogram],
)
```

Same `MCPClient` API on the consumer side, same `TulipMCPServer` on
the producer side, same tool definitions. The transport is an
implementation detail.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `MCP server failed to start` | The MCP server subprocess crashed before establishing the session. Run the command manually to see the error. |
| `Tool 'X' not found in MCP discovery` | The server exposes a different name than you expected. Print `[t.name for t in fs.tools()]` to see the actual list. |
| `Schema validation failed on call` | MCP tool returned an arg type that doesn't match its declared schema. Common with hand-written MCP servers; the standard ones are fine. |
| Claude Desktop doesn't show your SDK tools | `claude_desktop_config.json` not picked up — check the file lives at the right path and Claude has been restarted. |
| Hangs on `MCPClient.stdio` startup | The MCP subprocess is waiting for input on stdin (some servers expect a handshake). Pass `wait_for_init=True` and a timeout. |

## Source and notebook

- [`tulip.integrations.fastmcp`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/integrations/fastmcp.py) — built on FastMCP.
- [`notebook_45_mcp_integration.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_45_mcp_integration.py) — consumer + producer end-to-end.

## See also

- [Tools](tools.md) — the `@tool` decorator MCP wraps.
- [A2A](multi-agent/a2a.md) — purpose-built protocol for cross-process SDK-to-SDK agent meshes.
