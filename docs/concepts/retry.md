# Retry strategies

Production model calls fail. Rate limits, gateway timeouts, transient
5xx, occasional content-policy refusals on retryable inputs.
Locus's retry posture is:
**automate what's transient; surface what's not.**

## The default behaviour

Out of the box, an `Agent(...)` with **no retry hook does not retry
model calls**. Model retries are hook-driven (a hook sets
`event.retry = True` on the after-model-call event to trigger a
re-invocation), so without `ModelRetryHook` a provider exception
propagates: the loop emits `TerminateEvent(reason="error")` and
re-raises, so `arun()` / `run_sync()` raise instead of returning a
result.

So if you want resilience to transient model failures, add the hook
below — it's opt-in, not automatic.

## Configurable retry — `ModelRetryHook`

For production agents you usually want explicit policy:

```python
from locus.hooks.builtin.retry import ModelRetryHook
from locus.agent import Agent
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[...],
    hooks=[
        ModelRetryHook(
            max_retries=3,
            initial_delay=0.5,         # seconds, first retry delay
            max_delay=8.0,
            backoff_factor=2.0,        # exponential multiplier per attempt
            retry_on_empty=True,       # retry when the model returns no content
        ),
    ],
)
```

The hook implements `on_after_model_call`: it inspects the response and,
when the model came back **empty** (no content *and* no tool calls),
sets `event.retry = True` to trigger a re-invocation after an
exponential-backoff delay. The runtime observes `event.retry` and
re-runs the model call — same state, same messages, fresh request — up
to `max_retries` times.

!!! note "Scope: empty responses"
    `ModelRetryHook` retries **empty model responses** — it is not a
    generic HTTP-error/rate-limit retrier. For provider rate limits or
    5xx, retry inside your model client or wrap the whole run (see
    [When to widen the retry net](#when-to-widen-the-retry-net)).

## Tool-level retry

Tools fail too, and the failure mode is usually different — a
downstream HTTP call, a transient DB error, a JSON-decode glitch.
Three options for a tool such as `fetch_alert(alert_id: str) -> dict`:

1. **Let the loop handle it.** When `fetch_alert` raises, the SDK
   captures the exception, emits a `ToolCompleteEvent` with its `error`
   field set (no separate error-event type), records the failure in
   state, and feeds the error message to the next Think. The model can then
   *decide* whether to retry the call, try a different tool, or
   give up — same as a human would.

2. **Retry inside the tool.** For idempotent operations,
   wrap with `tenacity` or the like and retry transparently:

   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @tool
   @retry(stop=stop_after_attempt(3),
          wait=wait_exponential(multiplier=0.5))
   def fetch_alert(alert_id: str) -> dict: ...
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
def issue_refund(order_id: str, amount: float) -> dict: ...
```

This means you can let the model loop, panic, and retry without
refunding the same order twice (or paging the on-call twice). The
Execute hash is `(tool_name,
kwargs)`, so semantically-different calls aren't accidentally
deduped.

See [Idempotency](idempotency.md) for the full contract.

## Termination interactions

Retries don't bypass `termination=`. The retry hook re-runs Think;
the loop checks the termination algebra once per iteration, before
Think. If
your composite includes `MaxIterations(10)`, ten iterations is
ten iterations whether or not Think retried inside one of them.

For wall-clock budgets, use `TimeLimit(seconds=60)`. The clock
includes retry waits.

## When to widen the retry net

| Scenario | Strategy |
|---|---|
| Flaky empty completions | `ModelRetryHook(max_retries=5)` (retries empty responses) |
| Predictable rate limits / 5xx | retry inside your model client, or wrap the run; `ModelRetryHook` does not cover these |
| Provider / API-key failover | `CredentialPoolModel(pool=CredentialPool([...]), build_model=…)` rotates keys/endpoints on rate-limit or outage |
| Customer-facing agents | wrap the *whole agent* in your own outer retry; the inner agent treats one client request = one run |

## See also

- [Hooks](hooks.md) — full hook system, including `ModelRetryHook`.
- [Idempotency](idempotency.md) — why marking tools idempotent is a
  retry safety valve.
- [Termination](termination.md) — how retries interact with stop
  conditions.
- [Models](models.md) — provider-specific retry semantics.
