# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — remote-API timing fingerprint graduated into the SDK.

The measurement is now first-class in :mod:`tulip.security.fingerprint`.
Importing from here still works; prefer::

    from tulip.security import (
        measure_endpoint_timing,
        FEATURE_KEYS,
        default_classifier,
    )
"""

from __future__ import annotations

from tulip.security.fingerprint import FEATURE_KEYS, default_classifier, measure_endpoint_timing


__all__ = ["FEATURE_KEYS", "default_classifier", "measure_endpoint_timing"]
