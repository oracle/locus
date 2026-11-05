# Retry strategies

Production model calls fail. Rate limits, gateway timeouts, transient
5xx, occasional content-policy refusals on retryable inputs.
The SDK's retry posture is:
**automate what's transient; surface what's not.**

## The default behaviour

Every model call the loop makes is wrapped in a retry governed by
[`AgentConfig.model_retry`](../api/agent.md#agentconfig), a
`ModelRetryConfig` that is on by default. When a call fails with a rate
limit (429), an overload or server error (5xx), a dropped connection or
a timeout — as the failover classifier
(`locus.models.failover.classify`) names them — the loop waits and
re-issues the same request. The wait is what the provider's
`retry-after` / `retry-after-ms` asks for when it sends one; otherwise
an exponential backoff with full jitter: a ceiling of 1 s that doubles
each retry up to 60 s, with the delay drawn uniformly below the ceiling
so parallel agents sharing one rate limit do not retry in lockstep. By
default the loop makes up to 6 retries within a 300 s budget counted
from the first attempt; a retry whose wait would end past the budget is
not started.

Failures a retry cannot fix fail at once: a context-length overflow, a
malformed request, a bad key, a billing failure, an unknown model. An
exception the classifier cannot place is not retried either, so a bug
in a hook or a provider binding surfaces immediately
(`retry_unclassified=True` includes those). Two further cases are never
retried: a streamed call once a chunk has reached the caller, and a
cancelled run.

Each retry emits a `ModelRetryEvent` (`attempt`, `delay_seconds`,
`reason`, `status_code`, `error`, `from_retry_after`) before the backoff
sleep — live between chunks with `stream_tokens=True`, otherwise once
the call returns or fails. When the retries are spent, or the failure is
not retryable, the exception propagates: the loop emits
`TerminateEvent(reason="error")` and re-raises, so `arun()` /
`run_sync()` raise instead of returning a result.

This retry sits outside any retry the provider client does itself; the
loop starts a new attempt only after the client has given up.

Tune the policy, or turn it off, on the agent:

```python
from locus.agent import Agent, ModelRetryConfig

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[...],
    model_retry=ModelRetryConfig(max_retries=10, total_budget_seconds=900),
)

no_retry = Agent(model="{{ locus_example_model }}", model_retry=False)
```

`model_retry=True` restores the defaults.

## Empty responses — `ModelRetryHook`

The loop-level retry acts on a call that fails. A call that returns
successfully but **empty** — no content and no tool calls — is a
different case; `ModelRetryHook` is the opt-in layer for it:

{{ locus_model_tabs }}

```python
from locus.hooks.builtin.retry import ModelRetryHook
from locus.agent import Agent
agent = Agent(
    model="{{ locus_model }}",
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
    `ModelRetryHook` retries **empty model responses** only; it is not
    an HTTP-error or rate-limit retrier, because `on_after_model_call`
    runs only once a call has returned. Provider rate limits, 5xx
    responses, dropped connections and timeouts are retried by the
    loop-level `model_retry` described above; adjust that layer with
    `model_retry=ModelRetryConfig(...)` (see
    [When to widen the retry net](#when-to-widen-the-retry-net)).

## Tool-level retry

Tools fail too, and the failure mode is usually different — a
downstream HTTP call, a transient DB error, a JSON-decode error.
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

This is where the SDK differs from a plain retry loop. If a tool is tagged
`@tool(idempotent=True)` and the model retries the same call, the
**Execute node** dedupes inside the loop — the body never runs the
second time, and the cached receipt is returned.

```python
@tool(idempotent=True)
def issue_refund(order_id: str, amount: float) -> dict: ...
```

This means you can let the model loop and retry without
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
| Predictable rate limits / 5xx | retried by the default `model_retry`; raise `max_retries`, `max_delay` or `total_budget_seconds` with `model_retry=ModelRetryConfig(...)` |
| Provider / API-key failover | `CredentialPoolModel(pool=CredentialPool([...]), build_model=…)` rotates keys/endpoints on rate-limit or outage |
| Customer-facing agents | wrap the *whole agent* in your own outer retry; the inner agent treats one client request = one run |

## See also

- [Hooks](hooks.md) — full hook system, including `ModelRetryHook`.
- [Idempotency](idempotency.md) — why marking tools idempotent is a
  retry safety valve.
- [Termination](termination.md) — how retries interact with stop
  conditions.
- [Events](events.md#other-events-on-the-stream) — `ModelRetryEvent`, the
  notice the loop sends before each retry.
- [Resilience](providers/resilience.md) — failover classification,
  fallback chains and credential pools.
