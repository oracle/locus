# Tools

Tools are how an Tulip agent
affects the world. The model decides *"call `search` with query='hnsw'"*;
the SDK runs your `search` function, captures the return value, and
feeds it back. From your side, a tool is **a regular Python function
with a `@tool` decorator** — the SDK introspects the signature and
docstring to build the schema the model sees.

This is the seam most production code touches. Get tools right and
the rest of the framework gets out of your way.

## When to write a tool

| You want… | Write a tool |
|---|---|
| The model to call your API / database / file system | ✓ |
| Side-effecting actions the model should be able to invoke | ✓ |
| Read-only lookups (catalogue search, status checks) | ✓ |
| To mutate the agent's *internal* state (system prompt, config) | use a [hook](hooks.md), not a tool |
| To intercept *every* tool call (logging, retry) | use a [hook](hooks.md) |

## Getting started

### 1. Decorate a function

```python
from tulip.tools import tool
@tool
def search(query: str, limit: int = 10) -> list[str]:
    """Search the knowledge base for ``query``, up to ``limit`` results."""
    return backend.search(query, limit)
```

The docstring becomes the tool description the model reads. Type
hints (`str`, `int`, `list[str]`) build the JSON schema. Defaults
mark optional parameters.

### 2. Pass to the agent

```python
agent = Agent(model="anthropic:claude-sonnet-4-6", tools=[search])
```

That's the wiring. The model now sees `search` in its tool list and
can call it whenever it decides to.

### 3. Run it

```python
result = agent.run_sync("Find documents about HNSW.")
```

If the model decides to call `search("hnsw")`, the SDK invokes your
function with that argument, captures the return value, and feeds it
into the next model turn. You write Python; the SDK handles the
schema marshalling.

## What you get out of the box

### Idempotent tools — the model can retry; the side effect can't

This is the SDK's flagship tool primitive. Some side-effecting tools
must run *exactly once* per logical request — bookings, charges,
emails, paging. Mark them `idempotent=True`:

```python
@tool(idempotent=True)
def book_flight(flight_id: str, customer_id: str) -> dict:
    """Book the flight. Re-issuing the same (flight_id, customer_id)
    within a single run returns the prior result; the body is not
    re-executed."""
    return billing.charge_and_book(flight_id, customer_id)
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
def add(a: int, b: int) -> int:
    return a + b                        # sync — runs in thread pool

@tool
async def fetch(url: str) -> str:
    async with httpx.AsyncClient() as c:
        return (await c.get(url)).text   # async — runs on the loop
```

### Parallel by default — fast when the model wants multiple things

```python
agent = Agent(
    model=...,
    tools=[search_a, search_b, search_c],
    tool_execution="concurrent",   # default
)
```

When the model emits multiple tool calls in one turn, the SDK runs
them concurrently via `asyncio.gather`. Three independent searches
finish in `max(t1, t2, t3)`, not `t1+t2+t3`.

If your tools have side effects that must be ordered, switch to
`tool_execution="sequential"`.

### Error handling — tool failures don't crash the agent

If a tool raises, the executor catches the exception, wraps it as a
`ToolResult(success=False, error=...)`, and feeds it back into the
next model turn. The model sees the failure and can react: retry,
try a different tool, or report to the user.

```python
@tool
def lookup_by_id(id: str) -> dict:
    record = db.get(id)
    if record is None:
        raise ValueError(f"no record with id={id}")
    return record
```

The model sees `"no record with id=42"` and decides what to do.
Behind the scenes, the SDK chains the original exception as the cause
on a `ToolExecutionError` for your structured logs.

### Custom names and descriptions

Override the auto-derived defaults when the function name doesn't
read well to the model:

```python
@tool(name="find_customer", description="Look up a customer by email address.")
async def _find_customer_internal(email: str) -> Customer:
    ...
```

The model sees `find_customer`; your code keeps the internal name.

## Practical recipes

### Read-only lookups

```python
@tool
def get_order_status(order_id: str) -> dict:
    """Return the current status and shipment info for an order."""
    return orders.get(order_id)
```

No need for `idempotent=True` — read-only calls are safe to repeat.

### Idempotent writes

```python
@tool(idempotent=True)
def submit_po(vendor_id: str, line_items: list[dict]) -> dict:
    """Submit a purchase order. Re-fires return the cached PO id."""
    return procurement.submit(vendor_id, line_items)
```

### A tool that's also exposed via MCP

If you've built a tool you want other agents to reach, expose it
through `TulipMCPServer` — same `@tool`, no rewrite. See
[MCP](mcp.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Model never calls the tool | Description / docstring isn't telling the model when to use it. Be explicit: *"Use this tool when the user asks about X."* |
| Tool fires twice on the same input | You're seeing the model retry. Add `idempotent=True`. |
| `TypeError: missing 1 required positional argument` at call time | Function signature has a parameter without a default that you didn't surface in the docstring; the model omitted it. Add a default or explain the parameter. |
| Tool returns Python objects but the model echoes `<__main__.X object at 0x…>` | Tool return value isn't JSON-serialisable. Return a dict / Pydantic model / list of strings, not arbitrary objects. |
| Async tool blocks the event loop | The "async" body is calling sync I/O. Wrap the blocking call in `asyncio.to_thread(...)` or use an async client. |

## Source

- [`@tool` decorator and `Tool` class](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/tools/decorator.py)
- [`ToolRegistry`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/tools/registry.py)
- [Built-in tools](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/tools/builtins) — `get_today_date`, `task_complete`, `ask_user`, `use_oci`, `describe_oci`

## See also

- [Idempotency](idempotency.md) — the full story on `idempotent=True`.
- [Hooks](hooks.md) — for cross-cutting concerns (logging, retry, guardrails).
- [Executors](executors.md) — how concurrent vs sequential tool execution works.
- [MCP](mcp.md) — expose your tools to other agents over the Model Context Protocol.
- [Errors](errors.md) — how tool failures surface in the event stream.
