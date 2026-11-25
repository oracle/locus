# Resilience: failover, credential pools, rate limits

Model calls fail: rate limits, exhausted quota, revoked keys, requests too large
for the context window, transient 5xx. Retrying all of them the same way wastes
time on the ones a retry cannot fix. The SDK ships provider-neutral pieces that
let a caller decide *what to do* about a failure rather than only *what went
wrong*: a classifier that names the failure, a fallback chain that moves the
call to another model, a credential pool that rotates keys, and a parser for
rate-limit headers.

## Classifying an error

`locus.models.failover.classify` takes any exception raised by a model provider
and returns a frozen `FailoverDecision`:

| Field | Meaning |
|---|---|
| `reason` | A stable `FailoverReason` enum value, safe to log and alert on |
| `status_code` | The HTTP status, when there was one |
| `retryable` | Retrying the same request can succeed |
| `should_rotate_credential` | This key is the problem — try another |
| `should_compress` | The request is too large — compact the context and retry |
| `should_fallback` | This model or provider will not serve it — use another |

The classifier works through evidence in order of confidence: the HTTP status
(telling billing apart from a transient usage limit on 402, and context overflow
apart from a format error on 400), structured error codes in the response body,
message patterns when there is no status, a disconnect heuristic for large
sessions, and transport and timeout exception types. Anything unrecognised is
`UNKNOWN` and retryable with backoff.

It is deliberately provider-neutral and does not encode aggregator-specific
quirks; post-process the decision when you need them.

## Failing over between models

`locus.models.fallback.FallbackChain` builds on the classifier: an ordered list
of models, each behind a circuit breaker, that moves a call to the next tier
when the classifier blames the provider. The diagram follows one call through
it: a tier that fails for a health reason hands the call on, a tier whose
breaker is open is skipped, and the call ends served, exhausted, or raised at
once when no other tier would help.

{{ locus_diagram fallback-chain }}

A chain is a model, so an `Agent` takes it where it takes any other model. This
one puts an OCI model first and an Anthropic model behind it, so a failure at
one provider moves the call to the other:

```python
import logging

from locus.agent import Agent
from locus.models import get_model
from locus.models.fallback import FallbackChain

log = logging.getLogger(__name__)

model = FallbackChain(
    [
        get_model("{{ locus_example_model }}"),
        get_model("anthropic:claude-sonnet-5-5"),
    ],
    first_chunk_timeout=20.0,
    on_event=lambda event: log.info("model fallback: %s", event),
)
agent = Agent(model=model)
```

Tiers are tried in order, one attempt each. What a failure does depends on its reason:

- A failure whose reason is in `HEALTH_REASONS` (`rate_limit`, `overloaded`, `server_error`, `timeout`, `auth_transient`, `auth_permanent`, `billing`, `model_not_found`) says the provider is unwell: the call moves to the next tier and the failure counts against the tier's breaker.
- A `format_error` also moves the call on, because the classifier sets `should_fallback` for it, but does not count against the breaker.
- Any other reason (`context_too_long`, `payload_too_large`, `unknown`) is re-raised at once without trying another tier: the same request would fail the same way elsewhere, or the classifier could not place the failure.

Pass `should_fallback`, a callable `(exc, decision) -> bool`, to change the policy.

Each tier's breaker opens after `failure_threshold` health failures within
`window_s` seconds, skips the tier for `cooldown_s` seconds, then goes
half-open and admits a single probe call: success closes it, failure re-opens
it. When every breaker is open the primary is tried anyway. `stream()` fails
over only before a tier's first chunk; a failure after that is re-raised rather
than spliced with another model's output. `first_chunk_timeout` and
`complete_timeout` bound how long a tier may take to start or to answer.

`complete()` attaches the tier that answered to the response as
`response.metadata["fallback"]`, a dict with `tier`, `model` and `attempts`;
both `complete()` and `stream()` publish the same `ServedTier` to the calling
context, read with `locus.models.fallback.served_tier()`. When every tier
fails, `FallbackExhaustedError` is raised with each tier's error in `errors`,
chained from the last one. Every tier switch and breaker transition is reported
as a `FallbackEvent` to `on_event`, as `model.fallback` and
`model.fallback.breaker` events on the telemetry bus when a run is active, and
in the counters on `FallbackChain.metrics`.

### Retries around the chain

Retries wrap the chain, not a tier. The agent loop wraps every model call in `call_with_retry`, governed by [`AgentConfig.model_retry`](../../api/agent.md#agentconfig), a `ModelRetryConfig` that is on by default.

When the error that leaves the chain classifies as `rate_limit`, `overloaded`, `server_error` or `timeout`, the loop waits (the provider's `retry-after` when it sends one, otherwise an exponential backoff with full jitter) and calls the chain again from the top, up to `max_retries` times within `total_budget_seconds` (by default six retries within 300 seconds).

An `unknown` failure is not retried unless `retry_unclassified=True`, and a streamed attempt that has already forwarded a chunk is never retried. Each retry is reported as a [`ModelRetryEvent`](../events.md#other-events-on-the-stream).

[`ModelRetryHook`](../retry.md) is a different layer: it re-calls the model, which is the whole chain, when a reply comes back empty. A provider SDK's own retries (`max_retries` on `OpenAIConfig` and `AnthropicConfig`) run inside one tier, before the chain sees the error.

## Rotating credentials

`CredentialPoolModel` wraps any model and rotates through a pool of credentials
when the classifier says rotation should help. Other errors propagate unchanged.

```python
import os

from pydantic import SecretStr

from locus.agent import Agent
from locus.models.credentials import Credential, CredentialPool
from locus.models import AnthropicModel
from locus.models.pooled import CredentialPoolModel

pool = CredentialPool(
    [
        Credential(label="primary", api_key=SecretStr(os.environ["KEY_A"])),
        Credential(label="backup", api_key=SecretStr(os.environ["KEY_B"])),
    ]
)


def build(credential: Credential) -> AnthropicModel:
    return AnthropicModel(
        model="claude-sonnet-5-5",
        api_key=credential.api_key.get_secret_value(),
    )


agent = Agent(model=CredentialPoolModel(pool=pool, build_model=build, max_attempts=3))
```

You supply `build_model`, which receives a `Credential` and returns a configured
model, so the wrapper works with any provider. Models are cached per credential
label; a credential the classifier blames is marked bad with a cooldown, and the
next one is tried, up to `max_attempts`.

## Reading rate-limit headroom

`locus.models.rate_limits.parse_rate_limit_headers` turns `x-ratelimit-*`
response headers into frozen `RateLimitState` / `RateLimitBucket` models —
requests and tokens, per-minute and per-hour windows — so hooks and pools can
reason about headroom without re-parsing strings. Reset values may be seconds
(`58`) or OpenAI's duration format (`1m60s`, `200ms`).

It understands the OpenAI / OpenRouter header convention. For headers named
differently (Anthropic's `anthropic-ratelimit-*`) it
returns `None` rather than a half-filled state, so you can attach your own parser.

## Source

- [`locus/models/failover.py`](https://github.com/oracle/locus/blob/main/src/locus/models/failover.py)
- [`locus/models/fallback.py`](https://github.com/oracle/locus/blob/main/src/locus/models/fallback.py)
- [`locus/models/pooled.py`](https://github.com/oracle/locus/blob/main/src/locus/models/pooled.py)
- [`locus/models/rate_limits.py`](https://github.com/oracle/locus/blob/main/src/locus/models/rate_limits.py)
- [`locus/agent/model_retry.py`](https://github.com/oracle/locus/blob/main/src/locus/agent/model_retry.py)

## See also

- [Retry](../retry.md)
- [Events](../events.md#other-events-on-the-stream) — `ModelRetryEvent`, the
  notice the loop sends before each retry.
