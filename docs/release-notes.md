---
title: Release notes
description: What changed in each release of locus-sdk, newest first, generated from the repository's changelog when the site is built.
---

# Release notes

What changed in each release of `locus-sdk`, newest first. The page is
generated from the repository's
[`CHANGELOG.md`](https://github.com/oracle/locus/blob/main/CHANGELOG.md) when
the site is built, so it follows every release. Changes merged since the latest
release appear under *Unreleased* and ship with the next version.

Upgrade with:

```bash
python -m pip install -U locus-sdk
```

Before upgrading a production deployment, read the
[compatibility and release policy](compatibility.md) and the
[deprecation policy](https://github.com/oracle/locus/blob/main/DEPRECATION.md),
and run your policy, approval, idempotency and persistence tests against the
new version. Releases are tagged on
[GitHub](https://github.com/oracle/locus/releases) and published to
[PyPI](https://pypi.org/project/locus-sdk/).

{{ locus_release_notes }}
