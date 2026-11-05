---
title: Compatibility and release policy
description: Supported Python versions, documentation versioning, API stability, deprecations, and integration status for the SDK.
---

# Compatibility and release policy

## Current documented line

| Surface | Policy |
|---|---|
| SDK | This build targets `locus-sdk` {{ locus_sdk_version }} |
| Python | 3.11, 3.12, 3.13, and 3.14 |
| Documentation examples | Tested against this repository's SDK in the documentation build |
| Provider and infrastructure dependencies | Install through the documented optional extra; consult each guide for credentials and limitations |

Current guides use an unpinned install so new readers receive the current
release:

```bash
python -m pip install "locus-sdk[openai]"
```

For production, resolve and lock dependencies in your application after testing
them. Historical research pages keep exact versions because they describe a
specific experiment rather than the current documentation.

## Changes and deprecations

The SDK uses semantic versioning as the intended public contract. Deprecations
are announced in the [changelog](https://github.com/oracle/locus/blob/main/CHANGELOG.md)
and [deprecation policy](https://github.com/oracle/locus/blob/main/DEPRECATION.md).
Before upgrading a production deployment, read both and run your policy,
approval, idempotency, and persistence tests against the new version.

## Capability status labels

| Label | Meaning |
|---|---|
| Supported | Public SDK surface covered by normal compatibility expectations |
| Experimental | Usable but may change outside the normal deprecation window |
| Template | Example adapter or skeleton requiring deployment-specific implementation and validation |
| Offline simulation | Deterministic local behavior; no claim about a live external system |
| Live | Contacts the named provider or infrastructure and needs its credentials/configuration |
| Supported interfaces | The SDK-side adapter interface is supported; the backing service (database, vector store) is yours to operate and test |
| Research-only | A published research or evaluation artifact describing a specific experiment; not a supported SDK surface, and the exact evaluated configuration is preserved |

Examples should state their execution mode, requirements, tested SDK line, and
limitations near the top. If a page is missing those details, treat it as
documentation debt rather than an implied production guarantee.
