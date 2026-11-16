# MCP — Model Context Protocol

Build a tool once — issue a refund, roll out a deploy, enrich an
IOC (indicator of compromise — an IP, domain, or file hash) — and let
any agent call it. [MCP](https://modelcontextprotocol.io)
is the protocol for that: wrap a function like `lookup_order` or `issue_refund`
in a MCP server, and any MCP client (Claude Desktop, Cline, an agent, your own tooling) invokes it without bespoke glue. The same SDK
also *consumes* existing MCP servers — a billing service, a
case-management server, a threat-intel feed — so your agent can reach
tools it didn't ship.

**The SDK speaks MCP both ways**. `MCPClient` turns a remote
server's tools into SDK tools, and `LocusMCPServer` publishes every
tool on an agent (plus `run_agent` / `run_agent_stream`) over stdio
or HTTP (streamable HTTP, plus the older SSE transport). Round-trip
means an SDK-built agent can be either side: pull data from a
billing server *and* serve its own refund tools back to the
operator's desktop.

Production use adds four things, each covered below: [signing in to a
remote server with OAuth 2.1](#signing-in-to-a-remote-server-oauth-21),
[shaping the imported toolset](#shaping-the-toolset-include-exclude-prefix)
so two servers mount side by side, [a server that asks the person a
question mid-call](#when-the-server-asks-a-question-elicitation), and
[trace context across the hop](#tracing-across-the-hop). Exposing an
agent is [one command](#locus-mcp-serve).

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
[`AuditTrail`](control-layer.md) (each entry commits to the one
before it, so editing any record breaks verification).

## Getting started — expose your tools as MCP

### 1. Wrap an agent in `LocusMCPServer`

`LocusMCPServer` exposes an **agent** (and the tools registered on
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

## Signing in to a remote server (OAuth 2.1)

A Streamable HTTP server that answers `401` is asking the client to sign
in. Give the client an `MCPOAuth` and it runs the OAuth 2.1
authorization-code flow with PKCE on the first challenge:

```python
from locus.integrations import FileTokenStore, MCPClient, MCPOAuth

billing = MCPClient(
    base_url="https://billing.example.com/mcp",
    oauth=MCPOAuth(
        client_name="Support agent",
        store=FileTokenStore(server_url="https://billing.example.com/mcp"),
    ),
)
```

What happens on that first `401`:

1. The client reads the protected-resource metadata the challenge points
   at (`WWW-Authenticate: Bearer resource_metadata="…"`, falling back to
   the server's `/.well-known/oauth-protected-resource` paths), which
   names the authorization server.
2. It reads the authorization server's metadata (`/.well-known/
   oauth-authorization-server`, with the OpenID Connect discovery
   document as a fallback) for the authorize, token and registration
   endpoints.
3. It registers itself dynamically as a public client, unless
   `client_metadata_url=` names a hosted client-metadata document to
   identify with instead.
4. It sends the person to the authorization URL — by default the
   browser opens and a loopback listener on `127.0.0.1` receives the
   redirect — and trades the returned code and the PKCE verifier for
   tokens.
5. It retries the request with the token. Later requests reuse it; an
   expired token is refreshed without a person, and a refresh the
   server rejects starts the flow again.

The person is only ever involved at step 4. A headless process supplies
its own `redirect_handler` (show or forward the URL) and
`callback_handler` (return `(code, state)`) together with the
`redirect_uri` the authorization server will call; `scopes=` asks for
specific scopes instead of what the server advertises; `timeout=` bounds
how long the sign-in may take (five minutes by default).

**Where tokens live** is the `store`:

| Store | Lifetime | Use it when |
|---|---|---|
| `InMemoryTokenStore()` (default) | the process | a service that signs in at start-up |
| `FileTokenStore(path=…)` or `FileTokenStore(server_url=…)` | across runs; a JSON file created with mode `0600` under `$XDG_CONFIG_HOME/locus/mcp-tokens` (or `~/.config/locus/mcp-tokens`) | a CLI or a desktop tool on one person's machine |
| `KeyringTokenStore(server_url)` | the OS keyring, through the optional `keyring` package | the same, where a keyring is available |

Any object with `get_tokens` / `set_tokens` / `get_client_info` /
`set_client_info` coroutines is a store, so a vault-backed one is a
small class. No token value is written to a log by the SDK.

Precedence with the other ways of authenticating: an `Authorization`
header from `headers=`, `headers_provider` or the run's `mcp_headers`
metadata is sent as given and wins over everything; a static
`access_token=` wins over `oauth=`. OAuth is for the agent's own identity
against the server; per-user headers remain the way to carry an end
user's token through a shared agent.

## Shaping the toolset: include, exclude, prefix

A server publishes what it publishes; the agent should see what it
needs, under names that do not collide with another server's.

```python
billing = MCPClient(
    base_url="https://billing.example.com/mcp",
    include=["lookup_*", "issue_refund"],   # names or globs, server-side names
    exclude=["*_internal"],                 # wins over include
    prefix="billing_",                      # the agent sees billing_lookup_order
)
travel = MCPClient(base_url="https://travel.example.com/mcp", prefix="travel_")
agent = Agent(model="{{ locus_example_model }}", mcp_servers=[billing, travel])
```

Precedence, in order:

1. `allowed_tools` (exact names) and `include` (names or globs) **narrow**:
   when both are given a tool must satisfy both.
2. `exclude` (names or globs) **removes**, and wins over `include`.
3. `tool_filter` (a predicate over the tool's schema) decides last, and
   only sees the tools that survived.
4. `prefix` **renames** what is left, verbatim: `prefix="billing_"` turns
   `lookup_order` into `billing_lookup_order`. It is applied after the
   matching, which always sees the server's own names — `exclude=
   ["billing_*"]` matches nothing.

`list_tools()` returns the server's names after steps 1–3; calls through
either the prefixed or the server name land on the server, and a call to
a tool the shaping left out raises `MCPToolNotAllowedError`.

## When the server asks a question (elicitation)

An MCP server can ask the person something in the middle of a call —
"Refund order 4821?", "Which region?" — as an *elicitation request*
with a message and a small schema for the answer. The SDK routes it
through the same interrupt mechanism `ask_user` uses, so the run pauses
instead of failing:

```python
from locus.core.events import InterruptEvent

async for event in agent.run("refund order 4821"):
    if isinstance(event, InterruptEvent):
        print(event.question)                 # "Refund order 4821?"
        print(event.fields)                   # [{"name": "approve", "type": "boolean", ...}]
        answer = input("> ")                  # "yes"
        billing.answer_elicitation(event, answer)
        async for later in agent.resume(answer, perform_dangling=True):
            ...
```

What happens: the client tells the server the question is unanswered
for now (`cancel`), the tool call returns the interrupt marker, and the
run pauses on an `InterruptEvent` whose `fields` describe the answer the
schema asks for (one field with an `enum` also fills `options`) and
whose `metadata["mcp_elicitation"]` carries the server, tool and schema.
`answer_elicitation(event, answer)` parks the answer on the client;
`agent.resume(answer, perform_dangling=True)` re-issues the paused call,
the server asks again, and this time it gets the answer. The answer is a
mapping matching the schema, or — for a one-field schema — a string,
bool or number coerced to the field's type (`"yes"` → `True`); `None`
declines. Because the call is re-issued, a server should ask before it
acts, which is what elicitation is for.

Outside an agent run — or inside one, when you would rather not pause —
an `elicitation_handler` answers:

```python
def approve_small_refunds(question):
    amount = question.schema["properties"].get("amount", {}).get("default", 0)
    return {"approve": amount < 50}

billing = MCPClient(base_url=..., elicitation_handler=approve_small_refunds)
```

The handler (sync or async) gets an `MCPElicitation` (`message`,
`schema`, `mode`, `url`, `server`, `tool_name`, `tool_call_id`) and
returns the same shapes as above, or an `ElicitationAnswer` to say
exactly what to send. Without a handler, a question that arrives
outside a run is declined and logged.

The other way round: an agent served by `LocusMCPServer` that pauses on
`ask_user` has its question relayed to the MCP client as an elicitation
(through `run_agent_stream`), and resumes with the reply. A client that
cannot be asked ends the call with the question in its result.

## Tracing across the hop

With the [telemetry hook](hooks.md) installed, every `tools/call` the
agent makes carries W3C trace context — `traceparent` and `tracestate`
— in the request's `_meta`, parented to the agent's span for that tool
call. A server that reads it files its own spans under that call, so one
trace shows the model's decision, the tool call and the remote work. The
server side is symmetric: a call served by `LocusMCPServer` runs under
the context its `_meta` carries, so an agent exposed over MCP files its
`agent.invocation` under the caller's span.

```python
from locus.hooks.builtin.telemetry import TelemetryHook

agent = Agent(model="{{ locus_example_model }}", mcp_servers=[billing], hooks=[TelemetryHook()])
```

Nothing is configured on the client: it is on by default and does
nothing without OpenTelemetry installed. `propagate_trace=False` keeps
the agent's trace inside the process. `locus.integrations.mcp_tracing`
has the pieces — `trace_meta()`, `trace_scope(meta)` — for a server
built on something other than `LocusMCPServer`.

## `locus mcp serve`

One command exposes an agent, or a module of tools, as an MCP server:

```bash
locus mcp serve app.py:agent                       # stdio, for desktop clients
locus mcp serve app.py:make_agent --name support   # a factory works too
locus mcp serve billing_tools --transport http --port 8000 --path /mcp
```

The target is `MODULE:ATTR` or `FILE.py:ATTR` naming an `Agent`, a
function returning one, a `Tool` or a list of tools — or a bare module or
file, in which case every module-level `@tool` is served. An agent goes
up through `LocusMCPServer` (its tools plus `run_agent` and
`run_agent_stream`); tools go up as themselves through
`create_tool_server(tools)`. Point a desktop client at the command:

```json
{
  "mcpServers": {
    "support": {"command": "locus", "args": ["mcp", "serve", "/srv/app.py:agent"]}
  }
}
```

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
| `MCP tool 'X' is not allowed on …`, or the tool is missing from `await ti.list_tools()` | The server exposes a different name than you expected, or `allowed_tools` / `include` / `exclude` / `tool_filter` leaves it out. Print `[t["name"] for t in await ti.list_tools()]` to see the actual list; patterns match the server's names, not the prefixed ones. |
| The run pauses on a question you did not expect | The server asked for input mid-call. Answer it with `client.answer_elicitation(event, …)` and `agent.resume(…, perform_dangling=True)`, or set `elicitation_handler` to answer without pausing. |
| The browser opens for sign-in on every start | Tokens are in memory by default. Give `MCPOAuth` a `FileTokenStore` or `KeyringTokenStore`. |
| The tool call returns an error about argument types | The MCP tool returned an arg type that doesn't match its declared schema. Common with hand-written MCP servers. |
| Claude Desktop doesn't show your SDK tools | `claude_desktop_config.json` not picked up — check the file lives at the right path and Claude has been restarted. |
| Hangs on `await ti.connect()` | The MCP subprocess is waiting for input on stdin (some servers expect a handshake) or never finished its init. Run the `server_command` manually to see what it prints, and wrap `connect()` in `asyncio.wait_for(..., timeout=...)`. |

## Source and notebook

- [`locus.integrations.fastmcp`](https://github.com/oracle/locus/blob/main/src/locus/integrations/fastmcp.py) — built on FastMCP.
- [`notebook_46_mcp_integration.py`](https://github.com/oracle/locus/blob/main/examples/notebook_46_mcp_integration.py) — consumer + producer end-to-end.

## See also

- [Tools](tools.md) — the `@tool` decorator MCP wraps.
- [A2A](multi-agent/a2a.md) — purpose-built protocol for cross-process SDK-to-SDK agent meshes.
