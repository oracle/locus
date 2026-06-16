# Tool execution

The `Execute` node is where tool calls actually fire. The agent's
`tool_execution` mode controls whether tool calls returned in a
single Think turn run **concurrently** (the default) or **one at a
time**:

```python
agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=[search_flights, search_hotels, search_restaurants],
    tool_execution="concurrent",   # default — fan out
    # tool_execution="sequential", # opt-in — one at a time
)
```

## Concurrent execution (default)

When Think returns multiple tool calls, Execute dispatches all of
them at once and gathers their results before the next Think:

```python
# Think emits this:
[search_flights(...), search_hotels(...), search_restaurants(...)]

# Execute fires all three concurrently
# Each emits its own ToolStartEvent / ToolCompleteEvent
# State accumulates all three results before the next Think
```

When parallelism helps:

- **Multi-source RAG** — fetch from a vector store, a keyword index,
  and a knowledge graph in parallel, then merge.
- **Independent reads** — flights, hotels, and weather have no
  dependency on each other; do them all at once.
- **Tool fan-out** — the model called `search_X` for ten X's; run
  them all instead of ten round-trips.

## Sequential execution (opt-in)

Some workloads must run **one tool at a time** — a write that depends
on a read, an external service that rate-limits to one request at a
time, or any flow where ordering matters. Set `tool_execution="sequential"`
on the `Agent`:

```python
agent = Agent(
    ...,
    tool_execution="sequential",
)
```

Tools then fire in the order Think returned them. This is global per
agent.

## Idempotent dedup runs *before* dispatch

Whichever mode you pick, dedup happens first. When Execute receives
a list of tool calls, the **first** thing it does — before launching
any coroutines — is hash each `(tool_name, arguments)` and walk
`state.tool_executions` for matches. For tools tagged
`@tool(idempotent=True)`, matched calls short-circuit to the cached
receipt and never enter the executor at all.

So a model that re-emits `book_flight(flight_id="AA-181", ...)` in
iteration 5 — when the same call already fired in iteration 2 — gets
the cached receipt without a network round-trip and without
charging again. See [Idempotency](idempotency.md).

## Errors don't kill the group (concurrent mode)

If one tool raises while three are running, the other two finish
normally. The error becomes a `ToolErrorEvent` and a tool-error
message in state; the next Think sees:

> *Tool `lookup_inventory` failed with: ConnectionTimeout(after 30s).*

…and decides what to do (retry, try a different tool, give up). The
agent loop never sees an exception unless the whole run errors.

## Tool implementation patterns

Within the tool body, you choose how cooperative to be:

- **Sync function.** The executor wraps it and runs it on a
  worker thread so it doesn't block the event loop.
- **Async function (`async def my_tool`).** Awaited directly by the
  executor.
- **Long-running tool that needs a stream of partial results.** Pair
  with the streaming events — emit progress via the agent's hook
  registry rather than blocking until the whole job finishes.

## Per-tool retry inside the body

When a tool's failure modes are transient (HTTP 429, occasional
timeouts), it's often cleaner to retry inside the tool body than to
let the loop see the error and replan. A common pattern:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@tool
@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=0.5))
def lookup_inventory(sku: str) -> dict:
    return inventory.get(sku)
```

For non-transient errors, raise — the loop will see a
`ToolErrorEvent` and the model will decide what to do.

## See also

- [Agent Loop](agent-loop.md) — where the Execute node lives in the
  larger picture.
- [Idempotency](idempotency.md) — the dedup pass before dispatch.
- [Tools](tools.md) — defining the tools the executor runs.
- [Retry Strategies](retry.md) — when to retry inside a tool vs. let
  the loop handle it.
