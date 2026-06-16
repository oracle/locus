# Build a custom tool

Write a Python function, decorate it, pass it to the agent. The
decorator inspects the signature and docstring to build the JSON
schema the model will see.

```python
from locus.tools import tool
@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by ID.

    Args:
        order_id: The order identifier (e.g. "ORD-12345").

    Returns:
        Dict with keys: status, items, total.
    """
    return db.get_order(order_id)
```

## Sync or async

Both are supported. Sync bodies run in a thread-pool executor so the
event loop isn't blocked.

```python
@tool
async def search_docs(query: str, limit: int = 10) -> list[str]:
    """..."""
    return await vectorstore.search(query, limit)
```

## Idempotency for side-effecting tools

If your tool writes, books, transfers, or otherwise has a side
effect you never want duplicated, mark it idempotent:

```python
@tool(idempotent=True)
def transfer_points(from_user: str, to_partner: str, amount: int) -> dict:
    """Transfer points — must be charged exactly once per (user, partner, amount)."""
    return loyalty.transfer(from_user, to_partner, amount)
```

The ReAct loop dedupes calls with identical `(tool_name, arguments)`
within a run.

## Custom names and descriptions

```python
@tool(name="find_customer", description="Look up a customer by email.")
async def _impl(email: str) -> Customer:
    ...
```

The model sees `find_customer`; your Python name stays `_impl`.

## Accessing context

If your tool needs the current run id, iteration counter, agent state,
or per-invocation metadata passed via `agent.run(..., metadata=...)`,
accept a `ctx: ToolContext` parameter:

```python
from locus.tools import tool
from locus.tools.context import ToolContext

@tool
def with_context(message: str, ctx: ToolContext) -> str:
    """..."""
    return f"run={ctx.run_id} iter={ctx.iteration}: {message}"
```

`ToolContext` exposes:
`tool_call_id`, `tool_name`, `agent_id`, `run_id`, `iteration`, `state`,
`invocation_metadata`, `tool_config`, `messages`, `confidence`. Use
`ctx.invocation_metadata.get("thread_id")` if you persisted the
thread id there at agent-run time.

## Error handling

Your tool can raise anything. The agent catches at the executor
boundary and surfaces the error to the model via
`ToolResult(success=False, error=...)`. The original exception is
preserved as the `__cause__` of a
[`ToolExecutionError`](../concepts/errors.md).
