# Deprecation policy

Locus follows [Semantic Versioning](https://semver.org) (since 1.0,
tagged 2026-06-09). This file explains how breaking changes and
deprecations work.

## Semantic Versioning

- **Major** version bumps can remove deprecated API. Breaking changes
  are called out in [`CHANGELOG.md`](CHANGELOG.md) under the version
  that ships them, in a `### Removed` or `### Changed` section, with a
  one-line migration note.
- **Minor** version bumps can add deprecations but not remove API.
- **Patch** version bumps are bug fixes only.

Consumers should pin a major (`locus-sdk>=5,<6`) and read the
CHANGELOG before bumping across majors.

A deprecated API will:

1. Still work for at least one full minor version.
2. Emit `LocusDeprecationWarning` on use.
3. Be listed in `CHANGELOG.md` with its planned removal version and a
   migration snippet.

### Changed in 6.7

- An OCI Generative AI model id (`oci:<vendor>.<model>`) no longer carries
  the served model's list price. `cost_usd` is `None` and `max_cost_usd` is
  refused on such an id until a rate is registered under the full id with
  `locus.models.metadata.register_metadata`. Migration: register the rate
  you are billed before constructing the agent (see the observability page).

### Deprecated in 6.2

- `locus.observability.emit.EV_MEMORY_CONVERSATION_ADDED`
  (`memory.conversation.added`): declared since the event registry was
  introduced and never emitted; nothing in the SDK adds to a conversation
  outside the agent loop. Importing it warns and returns the string until
  7.0.0, when the name is removed. No replacement: a subscriber never
  received it.

### Removed in 6.0

- The `locus.security` compatibility shim; `locus.control` is the
  import path.
- The `locus.loop` package (`ReActLoop`, `LoopRunner`, `BatchRunner`,
  `StreamingCollector`, the nodes and the router); `Agent` is the
  supported loop.
- `FallbackChain.last_tier`; read `response.metadata["fallback"]` or
  `locus.models.fallback.served_tier()`.
- The `tool` key on the `tool.integrity.refused` payload; read
  `tool_name`.

### Changed in 6.0

- `Agent(tool_call_integrity=...)` defaults to `True`; pass `False` to
  turn the check off. A hook that rewrites `event.arguments` now calls
  `event.resign(by=...)` afterwards, or the executor refuses the call.

## Using `LocusDeprecationWarning`

Internal callers emit deprecation warnings like this:

```python
from locus.core.warnings import LocusDeprecationWarning
import warnings

def old_api(...):
    warnings.warn(
        "old_api() is deprecated; use new_api() instead. "
        "old_api() will be removed in Locus 1.1.",
        LocusDeprecationWarning,
        stacklevel=2,
    )
    return new_api(...)
```

Consumers can opt into failing on deprecations during their own test
suites:

```python
import warnings
from locus.core.warnings import LocusDeprecationWarning

warnings.simplefilter("error", LocusDeprecationWarning)
```

That turns every deprecated call into a test failure, so you find out
before the removal release — not after.

## What counts as "public"

Only names in a module's `__all__` and in the top-level `locus`
namespace are public. Anything under a leading underscore
(`_private`), or inside a submodule that's not re-exported, is
implementation detail and can change in any release without
deprecation.

If you're importing from `locus.core.reducers`,
`locus.tools.executor._internal`, `locus.agent.agent._parse_*`, or
anything similar — that is your risk to carry.
