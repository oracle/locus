# Errors

Every exception class in `locus.core.errors` subclasses a single root — `LocusError`. One handler catches
the whole hierarchy; a stable `kind` attribute on each class
keeps your structured logs and metrics dashboards independent of
class names.

```python
from locus.core.errors import LocusError

try:
    result = agent.run_sync(prompt, thread_id=thread_id)
except LocusError as exc:
    logger.exception(
        "agent run failed",
        extra={"kind": exc.kind, "thread_id": thread_id},
    )
    raise
```

## When you'll catch which

| Situation | Catch |
|---|---|
| Anything from this hierarchy — single sweep handler at your service boundary | `LocusError` |
| `GSARConfig.fail_on_low_score` is set and the GSAR decision for a `run_sync` / `arun` result is not `proceed` | `GSARValidationError` (carries `.decision` and `.score`) |
| Every credential in a `CredentialPoolModel` is cooling down | `ModelAuthError` (its `kind` is `model_pool_exhausted`) |
| A safety check rejected input — `safe_resolve` (path), `validate_url` (URL), or `MCPClient.connect()` (server URL or launch package) | `ValidationError` |
| Provider auth, quota or rate-limit failure | The provider SDK's own exception — it is not wrapped. [`classify`](providers/resilience.md#classifying-an-error) maps it to a `FailoverReason` |
| A tool raised | Nothing — the loop returns the error to the model as the tool's result; `result.metrics.tool_errors` counts them |

`LocusError` does not cover everything a run can throw. Provider
exceptions pass through unwrapped, much of the SDK raises built-in
exceptions such as `ValueError`, and several modules define their own
exceptions outside this hierarchy — so a boundary handler that must see
every failure catches `Exception` after `LocusError`. The SDK does not
itself raise the tool, checkpoint and RAG families, `ConfigError`, or the
other model errors; they are public, so your own model adapters and
checkpointers can raise them under the same root.

## Hierarchy

{{ locus_diagram error-hierarchy }}

Class names may evolve; `kind` strings are part of the stable contract.
Key your dashboards on `kind`.

## Idiomatic patterns

### One handler, structured logs

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = agent.run_sync(prompt)
except LocusError as exc:
    logger.exception("agent failed", extra={"kind": exc.kind})
    return error_response(exc.kind)
```

### Metric on `kind`

```python
from locus.core.errors import LocusError

try:
    result = agent.run_sync(prompt)
except LocusError as exc:
    metrics.counter("agent.errors", tags={"kind": exc.kind}).increment()
    raise
```

Use `kind` instead of the class name — it is the documented stable key,
and it can be finer than the class: a `ModelAuthError` from an exhausted
credential pool carries `kind="model_pool_exhausted"`.

### Differentiated retry policy

Provider errors are not `LocusError`s, so route them on what
[`classify`](providers/resilience.md#classifying-an-error) says rather
than on their class:

```python
import time

from locus.core.errors import GSARValidationError, LocusError
from locus.models.failover import FailoverReason, classify

def run_with_backoff(prompt: str):
    for attempt in range(3):
        try:
            return agent.run_sync(prompt)
        except GSARValidationError:
            return fallback_path(prompt)     # answer not grounded — degrade gracefully
        except LocusError:
            raise                            # everything else from the hierarchy: no retry
        except Exception as exc:             # provider errors arrive unwrapped
            if classify(exc).reason is not FailoverReason.RATE_LIMIT:
                raise                        # anything but a rate limit: no retry
            time.sleep(2 ** attempt)         # rate limit — exponential back-off
    raise RuntimeError("still rate-limited after 3 attempts")
```

### Chained causes

Every constructor accepts a `cause=` keyword so the original exception
is preserved as `__cause__`:

```python
from locus.core.errors import CheckpointSerializationError

try:
    blob = json.dumps(state)
except (TypeError, ValueError) as exc:
    raise CheckpointSerializationError(
        f"failed to serialize state for {thread_id}",
        cause=exc,
    )
```

The full chain shows up in `traceback.format_exc()` and structured-
log adapters — you don't lose context.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| A provider error slips past `except LocusError` | Provider exceptions are not wrapped. Catch them after `LocusError` and route them with `classify`. |
| Rate-limit retries run forever | Cap the loop with a max attempt count or a deadline; don't rely on the provider giving up. |
| The same tool call keeps failing | The error goes back to the model as the tool's result, and the model isn't acting on it. Tighten the system prompt or reduce the tool's surface. |
| Cause chain lost in logs | Use `logger.exception(...)`, not `logger.error(str(exc))`. |

## Source

- [`locus.core.errors`](https://github.com/oracle/locus/blob/main/src/locus/core/errors.py) — every class in the hierarchy.

## See also

- [Retry](retry.md) — the opt-in `ModelRetryHook` (retries empty model responses).
- [Hooks](hooks.md) — `AfterToolCallEvent` carries the error message when a tool body raises.
- [Tools](tools.md) — how a tool that raises becomes an error result for the model.
