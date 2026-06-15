# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — GPU-probe dispatch graduated into the SDK.

The co-located timing-probe lifecycle is now first-class in
:mod:`tulip.security.fingerprint`. Importing from here still works; prefer::

    from tulip.security import dispatch_timing_probe, FEATURE_KEYS
"""

from __future__ import annotations

from tulip.security.fingerprint import FEATURE_KEYS, dispatch_timing_probe


__all__ = ["FEATURE_KEYS", "dispatch_timing_probe"]
