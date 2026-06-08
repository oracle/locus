# Errors

Every exception raised from inside Tulip subclasses a single root — `TulipError`. One handler catches
any SDK-originated failure; a stable `kind` attribute on each subclass
keeps your structured logs and metrics dashboards portable across
releases.

```python
from tulip.core.errors import TulipError

try:
    result = agent.run_sync(prompt, thread_id=thread_id)
except TulipError as exc:
    logger.exception(
        "agent run failed",
        extra={"kind": exc.kind, "thread_id": thread_id},
    )
    raise
```

## When you'll catch which

| Situation | Catch |
|---|---|
| Anything from the SDK — single sweep handler at your service boundary | `TulipError` |
| A specific tool blew up; want to retry / skip / re-route | `ToolError` (or one of its three subtypes) |
| Provider auth or quota issue; want to escalate or back off | `ModelError` (or `ModelAuthError` / `ModelThrottledError`) |
| Checkpoint resume failed; thread is corrupt or missing | `CheckpointError` |
| Vector store / embeddings call failed | `RAGError` |
| Bad config or invalid input at the public-API boundary | `ConfigError` / `ValidationError` |

Outside this hierarchy, nothing the SDK emits will leak through —
unwrapped third-party exceptions are wrapped at the boundary.

## Hierarchy

```
TulipError                       kind="tulip_error"
├── ToolError                    kind="tool_error"
│   ├── ToolNotFoundError        kind="tool_not_found"
│   ├── ToolValidationError      kind="tool_validation"
│   └── ToolExecutionError       kind="tool_execution"
├── ModelError                   kind="model_error"
│   ├── ModelAuthError           kind="model_auth"
│   ├── ModelThrottledError      kind="model_throttled"
│   └── ModelResponseError       kind="model_response"
├── CheckpointError              kind="checkpoint_error"
│   ├── CheckpointNotFoundError  kind="checkpoint_not_found"
│   └── CheckpointSerializationError  kind="checkpoint_serialization"
├── RAGError                     kind="rag_error"
│   ├── EmbeddingError           kind="embedding"
│   └── VectorStoreError         kind="vector_store"
├── ValidationError              kind="validation"     (public-API input)
└── ConfigError                  kind="config"         (invalid/missing config)
```

Class names may evolve; `kind` strings are part of the stable contract.
Key your dashboards on `kind`.

## Idiomatic patterns

### One handler, structured logs

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = agent.run_sync(prompt)
except TulipError as exc:
    logger.exception("agent failed", extra={"kind": exc.kind})
    return error_response(exc.kind)
```

### Metric on `kind`

```python
from tulip.core.errors import TulipError

try:
    result = agent.run_sync(prompt)
except TulipError as exc:
    metrics.counter("agent.errors", tags={"kind": exc.kind}).increment()
    raise
```

Use `kind` instead of the class name — the string never changes; the
class name might.

### Differentiated retry policy

```python
from tulip.core.errors import (
    ModelThrottledError, ModelAuthError, ToolExecutionError, TulipError,
)

for attempt in range(3):
    try:
        return agent.run_sync(prompt)
    except ModelThrottledError:
        time.sleep(2 ** attempt)         # 429 — exponential back-off
    except ModelAuthError:
        raise                            # auth issues never recover with retry
    except ToolExecutionError:
        return fallback_path(prompt)     # tool went south — degrade gracefully
    except TulipError:
        raise                            # everything else: no retry
```

### Chained causes

Every constructor accepts a `cause=` keyword so the original exception
is preserved as `__cause__`:

```python
from tulip.core.errors import CheckpointSerializationError

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
| Catching `Exception` instead of `TulipError` | You'll silently swallow `KeyboardInterrupt` and provider SDK bugs. Catch the concrete SDK base. |
| `ModelThrottledError` retries forever | Cap the loop with a max attempt count or a deadline; don't rely on the provider giving up. |
| `ToolValidationError` keeps firing for the same call | The model isn't reading the schema error. Tighten the system prompt or reduce the tool's surface. |
| Cause chain lost in logs | Use `logger.exception(...)`, not `logger.error(str(exc))`. |

## Source

- [`tulip.core.errors`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/core/errors.py) — every exception class.

## See also

- [Retry](retry.md) — built-in retry hook keyed on `ModelThrottledError`.
- [Hooks](hooks.md) — `AfterToolCallEvent` carries any exception raised by the body.
- [Tools](tools.md) — when `ToolValidationError` and `ToolExecutionError` fire.
