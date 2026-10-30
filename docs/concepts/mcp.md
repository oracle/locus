# MCP — Model Context Protocol

Build a tool once — issue a refund, roll out a deploy, enrich an
IOC (indicator of compromise — an IP, domain, or file hash) — and let
any agent call it. [MCP](https://modelcontextprotocol.io)
is the protocol for that: wrap a function like `lookup_order` or `issue_refund`
in an MCP server, and any MCP client (Claude Desktop, Cline, a Locus
agent, your own tooling) invokes it without bespoke glue. The same SDK
also *consumes* existing MCP servers — a billing service, a
case-management server, a threat-intel feed — so your agent can reach
tools it didn't ship.

**The SDK speaks MCP both ways**. `MCPClient` turns a remote
server's tools into SDK tools, and `LocusMCPServer` publishes every
tool on an agent (plus `run_agent` / `run_agent_stream`) over stdio
or HTTP (streamable HTTP, plus the older SSE transport). Round-trip
means a Locus-built agent can be either side: pull data from a
billing server *and* serve its own refund tools back to the
operator's desktop.

## When to use MCP

| You want… | Use MCP |
|---|---|
| Your SDK agent to use an external billing / case-management / threat-intel MCP server | ✓ — `MCPClient` |
| Your `lookup_order` / `issue_refund` library callable by Claude Desktop / Cline / other agents | ✓ — `LocusMCPServer` |
| Two SDK agents to share side-effecting tools across processes / machines | ✓ — works, but [A2A](multi-agent/a2a.md) is the better protocol |
| In-process multi-agent — share tools by importing | use the [tools](tools.md) directly, not MCP |
| Reproducible tests | use a mock model + plain `@tool` — MCP adds I/O |

## Getting started — consume an MCP server

### 1. Install the MCP extras

```bash
pip install "locus-sdk[mcp]"   # MCPClient (consumer side)
pip install fastmcp            # also required by LocusMCPServer (producer side)
```

### 2. Spawn the server and wrap it with `MCPClient`

```python
import asyncio
from locus.integrations.fastmcp import MCPClient


async def main():
    # Point MCPClient at a threat-intel MCP server launched over stdio:
    ti = MCPClient(server_command=["python", "-m", "ti_feed.mcp_server"])
    await ti.connect()


asyncio.run(main())
```

`MCPClient(server_command=[...])` describes a stdio MCP server;
`await ti.connect()` spawns the subprocess, opens an MCP session over
its stdin/stdout, and prepares tool discovery. (For an HTTP server, pass
`base_url=` instead of `server_command=`.)

### 3. Pass the tools straight into an Agent

{{ locus_model_tabs }}

```python
from locus.agent import Agent

# Discover the server's tools and convert them to SDK tools:
mcp_tools = ti.to_locus_tools(await ti.list_tools())

agent = Agent(
    model="{{ locus_model }}",
    tools=mcp_tools,
    system_prompt="Triage the alert. Enrich every indicator before you act.",
)
result = agent.run_sync("Is 198.51.100.23 a known malicious endpoint?")
```

`await ti.list_tools()` returns the server's tool descriptors; passing
them through `ti.to_locus_tools(...)` produces SDK `Tool` objects with
full schemas, descriptions, and call-through plumbing. The agent doesn't
know they're MCP — they look like any other `@tool`.

### Side effects in the host process — use hooks, not wrappers

A common shape for MCP integrators: the *real* effect of a
side-effecting call lives in the host process (a case log, a ticketing
batch, a console command stream), not inside the tool body that
returns a string to the model. The instinct is to wrap each MCP tool
with a per-tool `@tool` that calls `_case_log().append(...)` before
returning.

Prefer a single `HookProvider` instead — the `ActionQueueHook`
from [Hooks](hooks.md#on_after_tool_call-what-the-event-carries) mirrors
every tool call into a host-side list, so a post-hoc review can replay
exactly what the agent did:

```python
call_log: list[dict] = []
agent = Agent(
    model="{{ locus_example_model }}",
    tools=ti.to_locus_tools(await ti.list_tools()),  # every MCP-sourced tool, untouched
    hooks=[ActionQueueHook(call_log)],
)
```

One hook covers every MCP-sourced tool. The `tool_call_id` correlates
with the model's `tool_calls[].id`, so parallel enrichments don't get
mixed up. See [hooks](hooks.md#on_after_tool_call-what-the-event-carries)
for the full event surface. Note this list is a faithful trace for
replay, not a tamper-evident record — for decisions an auditor must
trust, route the action through the hash-chained
[`AuditTrail`](agentic-ai-security.md) (each entry commits to the one
before it, so editing any record breaks verification).

## Getting started — expose your tools as MCP

### 1. Wrap an agent in `LocusMCPServer`

`LocusMCPServer` exposes a Locus **agent** (and the tools registered on
it) over MCP. Build the agent with the tools you want to publish:

```python
from locus.agent import Agent
from locus.integrations.fastmcp import LocusMCPServer

agent = Agent(model="{{ locus_example_model }}", tools=[lookup_order, issue_refund])
server = LocusMCPServer(agent=agent, name="billing-tools")
```

### 2. Pick a transport

```python
server.run(transport="stdio")         # for desktop clients (default)
server.run(transport="http")          # for HTTP MCP clients
```

`run(transport="stdio")` is what Claude Desktop, Cline, and most MCP
clients expect. `run(transport="http")` runs an HTTP MCP server
(transport + JSON-RPC) that any HTTP MCP client can reach. The supported
transports are `"stdio"`, `"http"`, `"sse"`, and `"streamable-http"`;
the last three are the HTTP family — `"http"` and `"streamable-http"`
select the same streamable HTTP transport, and `"sse"` selects the older
HTTP+SSE transport (see [Transports](#transports)).

### 3. Point a client at it

For Claude Desktop, add the server to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`;
Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "billing-tools": {
      "command": "python",
      "args": ["-m", "billing_tools.mcp_server"]
    }
  }
}
```

Restart Claude Desktop. Your `lookup_order` and `issue_refund` tools
appear in the model's tool list.

## What you get out of the box

### Schema preservation

`@tool`'s name carries across unchanged (hyphens included); its
docstring becomes the MCP description, with the `Args:` block becoming
the per-parameter descriptions. The JSON schema is not copied byte for
byte: FastMCP rebuilds it from a synthetic signature that types every
parameter as a string and gives optional parameters a `null` default,
so an `int` or `bool` parameter is advertised as a string. The tool
body still receives the declared Python types — `Tool.execute` coerces
arguments (`"10"` → `10`, `"true"` → `True`). `locus_tool_to_mcp(tool)`
returns the SDK schema unchanged when you need it as-is.

### Transports

| Transport | Use case |
|---|---|
| **stdio** — process pipes (`transport="stdio"`, the default) | Desktop clients (Claude Desktop, Cline). The MCP server is spawned as a subprocess. |
| **HTTP** — JSON-RPC over POST (`transport="http"`; `"streamable-http"` is the same streamable HTTP transport, `"sse"` the older HTTP+SSE one) | Networked clients. Good for a shared tool server the whole team reaches. |

### Idempotency is per-run, not cross-client

The `idempotent=True` dedup is a property of a single agent's ReAct loop
— it suppresses a *repeated* call within one run (see
[idempotency](idempotency.md)). It does **not** carry across MCP: tools
the SDK consumes from a remote MCP server are wrapped with
`idempotent=False`, and the MCP server wrapper invokes each tool
directly, so two separate clients firing the same call will each run it.
If you need exactly-once across clients, enforce it in the tool body
(e.g. an idempotency key checked against your own store).

## Round-trip example

A common shape: a support agent A consumes an external billing
MCP server, *and* exposes its own refund tools as MCP for a
case-management agent B to consume:

```python
# Agent A — consumes billing data, exposes its own refund tools
billing = MCPClient(server_command=[...])     # consumer side
await billing.connect()

agent_a = Agent(
    model="{{ locus_example_model }}",
    tools=[*billing.to_locus_tools(await billing.list_tools()), lookup_order, issue_refund],
)

# Producer side — publish agent_a's own tools back over MCP.
# server.run(transport="http") blocks, so run it on its own thread/process.
refunds = LocusMCPServer(agent=agent_a, name="refunds")
refunds.run(transport="http")
```

Same `MCPClient` API on the consumer side, same `LocusMCPServer` on
the producer side, same tool definitions. The transport is an
implementation detail.

## Governed MCP — hand the capability, not the authority

Exposing a tool over MCP gives a remote agent — a Claude or GPT client, a separate
orchestrator — the ability to call it. What keeps that safe: **the admission gate
lives inside the action, not the transport.** Build the tool so its body routes the
write through `admit()` / `ctx.actions.execute`, and a remote agent gets the
*capability* to ask without the *authority* to skip your `ControlPolicy` — the
write still clears policy (and, in production, a human) and still lands in your
audit trail.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `MCPConnectionError: timed out after Ns connecting to MCP server …` / `… is unreachable` | The MCP server subprocess crashed or never completed the handshake. Run the command manually to see the error. |
| `MCP tool 'X' is not allowed on …`, or the tool is missing from `await ti.list_tools()` | The server exposes a different name than you expected, or `allowed_tools` / `tool_filter` excludes it. Print `[t["name"] for t in await ti.list_tools()]` to see the actual list. |
| The tool call returns an error about argument types | The MCP tool returned an arg type that doesn't match its declared schema. Common with hand-written MCP servers. |
| Claude Desktop doesn't show your SDK tools | `claude_desktop_config.json` not picked up — check the file lives at the right path and Claude has been restarted. |
| Hangs on `await ti.connect()` | The MCP subprocess is waiting for input on stdin (some servers expect a handshake) or never finished its init. Run the `server_command` manually to see what it prints, and wrap `connect()` in `asyncio.wait_for(..., timeout=...)`. |

## Source and notebook

- [`locus.integrations.fastmcp`](https://github.com/oracle/locus/blob/main/src/locus/integrations/fastmcp.py) — built on FastMCP.
- [`notebook_46_mcp_integration.py`](https://github.com/oracle/locus/blob/main/examples/notebook_46_mcp_integration.py) — consumer + producer end-to-end.

## See also

- [Tools](tools.md) — the `@tool` decorator MCP wraps.
- [A2A](multi-agent/a2a.md) — purpose-built protocol for cross-process SDK-to-SDK agent meshes.
