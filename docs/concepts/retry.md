# Retry strategies

Production model calls fail. Rate limits, gateway timeouts, transient
5xx, occasional content-policy refusals on retryable inputs.
Tulip's retry posture is:
**automate what's transient; surface what's not.**

## The default behaviour

Out of the box, an `Agent(...)` with no retry hook still survives:

- **Network errors** raised by the provider client are caught at the
  Think node and retried once with exponential jitter.
- **Rate-limit errors** (HTTP 429) honour the provider's
  `Retry-After` header.
- **Persistent failures** propagate as a `ModelError` and the loop
  exits with `TerminateEvent(reason="ModelError")`.

This minimum keeps a happy-path agent from falling over on a single
flaky request without making you opt in.

## Configurable retry — `ModelRetryHook`

For production agents you usually want explicit policy:

```python
from tulip.hooks.builtin.retry import ModelRetryHook
from tulip.agent import Agent
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    hooks=[
        ModelRetryHook(
            max_attempts=3,
            backoff="exponential",
            initial_delay=0.5,         # seconds
            max_delay=8.0,
            retry_on=("rate_limit", "server_error", "timeout"),
        ),
    ],
)
```

The hook listens for `ModelErrorEvent` and returns `Retry()` from its
handler if the policy says to. The router observes the directive and
re-runs the Think node — same state, same messages, fresh model call.

## Tool-level retry

Tools fail too, and the failure mode is usually different — a
downstream HTTP call, a transient DB error, a JSON-decode glitch.
Three options:

```python
@tool
def lookup_inventory(sku: str) -> dict:
    """Look up inventory for a SKU."""
    return inventory.get(sku)
```

1. **Let the loop handle it.** When `lookup_inventory` raises, the SDK
   captures the exception, returns a `ToolErrorEvent` to state, and
   feeds the error message to the next Think. The model can then
   *decide* whether to retry the call, try a different tool, or
   give up — same as a human would.

2. **Retry inside the tool.** For idempotent operations,
   wrap with `tenacity` or the like and retry transparently:

   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @tool
   @retry(stop=stop_after_attempt(3),
          wait=wait_exponential(multiplier=0.5))
   def lookup_inventory(sku: str) -> dict: ...
   ```

3. **Cooperative cancellation.** Long-running tools should poll for
   cancellation from their async context (or check a shared flag) so
   the agent can give up cleanly when the user cancels or a budget
   hook fires.

## Idempotent retry

This is the SDK-distinctive bit. If a tool is tagged
`@tool(idempotent=True)` and the model retries the same call, the
**Execute node** dedupes inside the loop — the body never runs the
second time, and the cached receipt is returned.

```python
@tool(idempotent=True)
def submit_po(vendor_id: str, amount_usd: float) -> dict: ...
```

This means you can let the model loop, panic, and retry without
charging the customer twice. The Execute hash is `(tool_name,
kwargs)`, so semantically-different calls aren't accidentally
deduped.

See [Idempotency](idempotency.md) for the full contract.

## Termination interactions

Retries don't bypass `termination=`. The retry hook re-runs Think;
the router checks the termination algebra after every node. If
your composite includes `MaxIterations(10)`, ten iterations is
ten iterations whether or not Think retried inside one of them.

For wall-clock budgets, use `TimeLimit(seconds=60)`. The clock
includes retry waits.

## When to widen the retry net

| Scenario | Strategy |
|---|---|
| Flaky single calls | default `Agent(...)` retry is enough |
| Predictable rate limits | `ModelRetryHook(max_attempts=5, retry_on=("rate_limit",))` |
| Multi-region failover | `OCIChatCompletionsModel(endpoints=[primary, secondary])` |
| Customer-facing agents | wrap the *whole agent* in your own outer retry; the inner agent treats one client request = one run |

## See also

- [Hooks](hooks.md) — full hook system, including `ModelRetryHook`.
- [Idempotency](idempotency.md) — why marking tools idempotent is a
  retry safety valve.
- [Termination](termination.md) — how retries interact with stop
  conditions.
- [Models](models.md) — provider-specific retry semantics.
