# Tool execution

The `Execute` node is where tool calls actually fire. The agent's
`tool_execution` mode controls whether tool calls returned in a
single Think turn run **concurrently** (the default) or **one at a
time**:

```python
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, fetch_customer_history, shipping_status],
    tool_execution="concurrent",   # default — fan out
    # tool_execution="sequential", # opt-in — one at a time
)
```

## Concurrent execution (default)

When Think returns multiple tool calls, Execute dispatches all of
them at once — each emitting its own `ToolStartEvent` /
`ToolCompleteEvent` — and gathers every result into state before the
next Think.

When parallelism helps:

- **Independent reads** — fetch the order, the customer's history, and
  the shipping status in parallel; none depends on the others.
- **Tool fan-out** — the model called `lookup_order` for ten orders;
  run them all instead of ten round-trips.
- **Multi-source enrichment** (security variant) — look up an IOC
  (indicator of compromise — an IP, domain, or file hash) in a
  threat-intel feed, a passive-DNS index, and a malware sandbox in
  parallel, then merge.

## Sequential execution (opt-in)

Some workloads must run **one tool at a time** — a write that depends
on a read, an external service that rate-limits to one request at a
time, or any flow where ordering matters. Set `tool_execution="sequential"`
on the `Agent` (as in the block above). Tools then fire in the order Think returned them. This is global per
agent.

## Idempotent dedup runs *before* dispatch

Whichever mode you pick, dedup happens before dispatch. When Execute
receives a list of tool calls, before it launches any coroutines it
checks each `(tool_name, arguments)` pair — after
[`on_before_tool_call`](hooks.md) hooks have run, so the check uses the
arguments as they reach the tool — and walks `state.tool_executions`
for matches. For tools tagged `@tool(idempotent=True)`, matched calls
short-circuit to the cached receipt and never enter the executor at all.

So a model that re-emits `issue_refund(order_id="ord-4821", ...)` in
iteration 5 — when the same call already fired in iteration 2 — gets
the cached receipt without a network round-trip and without refunding
the customer twice. See [Idempotency](idempotency.md).

## Errors don't kill the group (concurrent mode)

If one tool raises while three are running, the other two finish
normally. The failure surfaces as a `ToolCompleteEvent` with its
`error` field set (there is no separate `ToolErrorEvent`), plus a
tool-error message in state; the next Think sees:

> *Tool `shipping_status` failed with: ConnectionTimeout(after 30s).*

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
def shipping_status(order_id: str) -> dict:
    return carrier.track(order_id)
```

For non-transient errors, raise — the loop will see a
`ToolCompleteEvent` with its `error` set and the model will decide what
to do.

## See also

- [Agent Loop](agent-loop.md) — where the Execute node lives in the
  larger picture.
- [Idempotency](idempotency.md) — the dedup pass before dispatch.
- [Tools](tools.md) — defining the tools the executor runs.
- [Retry Strategies](retry.md) — when to retry inside a tool vs. let
  the loop handle it.
