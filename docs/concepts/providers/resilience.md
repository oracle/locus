# Resilience: failover, credential pools, rate limits

Model calls fail: rate limits, exhausted quota, revoked keys, requests too large
for the context window, transient 5xx. Retrying all of them the same way wastes
time on the ones a retry cannot fix. The SDK ships three provider-neutral pieces
that let a caller decide *what to do* about a failure rather than only *what went
wrong*.

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
- [`locus/models/pooled.py`](https://github.com/oracle/locus/blob/main/src/locus/models/pooled.py)
- [`locus/models/rate_limits.py`](https://github.com/oracle/locus/blob/main/src/locus/models/rate_limits.py)

## See also

- [Retry](../retry.md)
