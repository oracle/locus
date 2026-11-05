---
title: Tools
---

# Tools

Tools are how an agent
affects the world. The model decides *"call `lookup_order` with order_id='ord-4821'"*;
the SDK runs your `lookup_order` function, captures the return value, and
feeds it back. From your side, a tool is **a regular Python function
with an `@tool` decorator** — the SDK introspects the signature and
docstring to build the schema the model sees. The drawing follows that
round trip: `@tool` reads the signature and docstring into a `Tool` with
a JSON schema, the `ToolRegistry` offers it to the model, and a call
comes back through the executor, your function and a `ToolResult`,
cut to size when it is too long, to the observation the model reads
next.

{{ locus_diagram tool-pipeline }}

Two parts of the drawing have no section of their own on this page. A
function that declares a `ctx: ToolContext` parameter receives the
call's `run_id`, `iteration`, `state` and invocation metadata, and that
parameter is left out of the schema the model sees; see
[Accessing context](../how-to/custom-tools.md#accessing-context) and
the [`ToolContext` reference](../api/tools.md#tool-context). A result
longer than [`max_tool_result_length`](../api/agent.md#agentconfig) is
cut to its head and tail around a marker that says how much was removed.
With a `tool_result_store` configured it is stored whole and a reference
key is inlined instead; with the ObservationPack on it goes whole up to
the pack's inline limit and above that is archived whole and sent cut
around a pointer to the archive.

This is the seam most production code touches. Get tools right and
the rest of the framework gets out of your way.

## When to write a tool

| You want… | Write a tool |
|---|---|
| The model to call your billing / CRM / infra APIs | ✓ |
| Side-effecting actions the model should be able to invoke (refunds, deploys) | ✓ |
| Read-only lookups (order status, deploy health checks) | ✓ |
| To mutate the agent's *internal* state (system prompt, config) | use a [hook](hooks.md), not a tool |
| To intercept *every* tool call (logging, retry) | use a [hook](hooks.md) |

## Getting started

### 1. Decorate a function

```python
from locus.tools import tool
@tool
def lookup_order(order_id: str, include_history: bool = False) -> dict:
    """Return the order record for ``order_id``, with past events when ``include_history``."""
    return billing.get_order(order_id, include_history)
```

The docstring becomes the tool description the model reads. Type
hints (`str`, `bool`, `dict`) build the JSON schema. Defaults
mark optional parameters.

### 2. Pass to the agent

{{ locus_model_tabs }}

```python
agent = Agent(model="{{ locus_model }}", tools=[lookup_order])
```

That's the wiring. The model now sees `lookup_order` in its tool list and
can call it whenever it decides to.

### 3. Run it

```python
result = agent.run_sync("What's the status of order ord-4821?")
```

If the model decides to call `lookup_order("ord-4821")`, the SDK
invokes your function with that argument, captures the return value, and
feeds it into the next model turn. You write Python; the SDK handles the
schema marshalling.

## What you get out of the box

### Idempotent tools — the model can retry; the side effect can't

Some side-effecting tools should not repeat for an identical call
within a run — issuing a refund, paging on-call, rolling out a deploy.
Mark them `idempotent=True`:

```python
@tool(idempotent=True)
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund ``amount`` against the order. Re-issuing the same
    (order_id, amount) within a single run returns the prior
    result; the body is not re-executed."""
    return billing.refund(order_id, amount)
```

When the model re-issues a tool call with the same
`(name, arguments)` tuple that already ran in this agent run, the
ReAct loop **reuses the prior result instead of invoking the
function again**. Defends against:

- Models that re-emit the same call after seeing the result.
- Network glitches where a call appears failed but actually succeeded.
- Users re-prompting "do X" when X has already been done.
- Replays after a checkpoint resume.

Read the [idempotency concept page](idempotency.md) for the full
picture and the matching notebook.

### Sync and async bodies

Both shapes are supported. Async bodies run on the agent's event
loop directly; sync bodies run in a thread-pool executor so the loop
is never blocked.

```python
@tool
def score_priority(amount: float, age_days: int) -> float:
    return amount / max(age_days, 1)    # sync — runs in thread pool

@tool
async def query_siem(query: str) -> str:
    """Search the SIEM (a security team's log platform) for matching events."""
    async with httpx.AsyncClient() as c:
        # async — runs on the loop
        return (await c.get(f"https://siem.example/search?q={query}")).text
```

### Parallel by default — fast when the model wants multiple things

```python
agent = Agent(
    model=...,
    tools=[lookup_order, lookup_customer, fetch_policy],
    tool_execution="concurrent",   # default
)
```

When the model emits multiple tool calls in one turn, the SDK runs
them concurrently via `asyncio.gather`. Three independent lookups
finish in `max(t1, t2, t3)`, not `t1+t2+t3`.

If your tools have side effects that must be ordered, switch to
`tool_execution="sequential"`.

### Error handling — tool failures don't crash the agent

If a tool raises, the loop catches the exception and turns it into a
`ToolResult` whose `error` is `f"{type(exc).__name__}: {message}"` —
first line only, with common secret patterns (`postgresql://`, `redis://`,
`mongodb://` and `mysql://` URLs, `password=`/`api_key=`/`token=`
assignments, well-known vendor key prefixes, `Authorization: Bearer`
values, JWTs, home-directory paths) redacted or masked on a best-effort
basis — then feeds that back into the next model turn. The redaction is
pattern-based: a `postgres://` URL or an unrecognised key format passes
through unchanged, so keep secrets out of exception messages.
(`ToolResult.success` is a read-only property derived from
`error` — it's `True` when `error is None`, not something you set.) The
model sees the failure and can react: retry, try a different tool, or
report to the user.

```python
@tool
def lookup_order(order_id: str) -> dict:
    record = billing.get_order(order_id)
    if record is None:
        raise ValueError(f"no order with id={order_id}")
    return record
```

The model sees `"Error: ValueError: no order with id=ord-4821"` and decides what to do.
Behind the scenes, the loop captures the exception type plus its sanitized
first line into `ToolResult.error`, and that string is folded into the tool
message content (prefixed with `Error:`) when the tool produced no content;
the traceback and any later lines of the message never reach the model.

### Custom names and descriptions

Override the auto-derived defaults when the function name doesn't
read well to the model:

```python
@tool(
    name="lookup_customer",
    description="Return the account record and open orders for a customer id.",
)
async def _lookup_customer_internal(customer_id: str) -> dict:
    ...
```

The model sees `lookup_customer`; your code keeps the internal name.

## Practical recipes

### Read-only lookups

```python
@tool
def get_deploy_status(deploy_id: str) -> dict:
    """Return the current rollout status and health for a deploy."""
    return deploys.get(deploy_id)
```

No need for `idempotent=True` — read-only calls are safe to repeat.

### Idempotent writes

```python
@tool(idempotent=True)
def deploy_service(service: str, version: str) -> dict:
    """Roll ``service`` out at ``version``. Re-fires return the cached deploy id."""
    return infra.deploy(service, version)
```

### A tool that's also exposed via MCP

If you've built a tool you want other agents to reach, expose it
through `LocusMCPServer` — same `@tool`, no rewrite. See
[MCP](mcp.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Model never calls the tool | Description / docstring isn't telling the model when to use it. Be explicit: *"Use this tool when the user asks about X."* |
| Tool fires twice on the same input | You're seeing the model retry. Add `idempotent=True` (a refund goes out once, not twice). |
| `TypeError: lookup_order was called with the wrong arguments — missing required argument(s): …` in the tool result | Function signature has a parameter without a default that you didn't surface in the docstring; the model omitted it, and `sig.bind()` rejects the call before your function runs. Add a default or explain the parameter. |
| Tool returns Python objects but the model echoes `<__main__.X object at 0x…>` | Tool return value isn't JSON-serialisable. Return a dict / Pydantic model / list of strings, not arbitrary objects. |
| Async tool blocks the event loop | The "async" body is calling sync I/O. Wrap the blocking call in `asyncio.to_thread(...)` or use an async client. |

## Source

- [`@tool` decorator and `Tool` class](https://github.com/oracle/locus/blob/main/src/locus/tools/decorator.py)
- [`ToolRegistry`](https://github.com/oracle/locus/blob/main/src/locus/tools/registry.py)
- [Built-in tools](https://github.com/oracle/locus/blob/main/src/locus/tools/builtins.py) — `get_today_date`. The agent also auto-injects `task_complete` and `ask_user` in explicit-completion mode.

## See also

- [Idempotency](idempotency.md) — the full story on `idempotent=True`.
- [Hooks](hooks.md) — for cross-cutting concerns (logging, retry, guardrails).
- [Executors](executors.md) — how concurrent vs sequential tool execution works.
- [MCP](mcp.md) — expose your tools to other agents over the Model Context Protocol.
- [Errors](errors.md) — how tool failures surface in the event stream.
