# Deprecation policy

Tulip follows [Semantic Versioning](https://semver.org) (since 1.0,
tagged 2026-06-09). This file explains how breaking changes and
deprecations work.

## Semantic Versioning

- **Major** version bumps can remove deprecated API. Breaking changes
  are called out in [`CHANGELOG.md`](CHANGELOG.md) under the version
  that ships them, in a `### Removed` or `### Changed` section, with a
  one-line migration note.
- **Minor** version bumps can add deprecations but not remove API.
- **Patch** version bumps are bug fixes only.

Consumers should pin a major (`tulip-agents>=2,<3`) and read the
CHANGELOG before bumping across majors.

A deprecated API will:

1. Still work for at least one full minor version.
2. Emit `TulipDeprecationWarning` on use.
3. Be listed in `CHANGELOG.md` with its planned removal version and a
   migration snippet.

## Using `TulipDeprecationWarning`

Internal callers emit deprecation warnings like this:

```python
from tulip.core.warnings import TulipDeprecationWarning
import warnings

def old_api(...):
    warnings.warn(
        "old_api() is deprecated; use new_api() instead. "
        "old_api() will be removed in Tulip 1.1.",
        TulipDeprecationWarning,
        stacklevel=2,
    )
    return new_api(...)
```

Consumers can opt into failing on deprecations during their own test
suites:

```python
import warnings
from tulip.core.warnings import TulipDeprecationWarning

warnings.simplefilter("error", TulipDeprecationWarning)
```

That turns every deprecated call into a test failure, so you find out
before the removal release — not after.

## What counts as "public"

Only names in a module's `__all__` and in the top-level `tulip`
namespace are public. Anything under a leading underscore
(`_private`), or inside a submodule that's not re-exported, is
implementation detail and can change in any release without
deprecation.

If you're importing from `tulip.core.reducers`,
`tulip.loop.nodes._internal`, `tulip.agent.agent._parse_*`, or
anything similar — that is your risk to carry.
