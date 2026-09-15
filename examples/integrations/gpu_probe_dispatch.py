# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — GPU-probe dispatch graduated into the SDK.

The co-located timing-probe lifecycle is now first-class in
:mod:`tulip.security.fingerprint`. Core ships the *offline reference*
(``dispatch_timing_probe_reference``); a live probe on a GPU cloud is an
integration you write. This shim re-exports the offline reference under the example's local name::

    from tulip.security import dispatch_timing_probe_reference, FEATURE_KEYS
"""

from __future__ import annotations

from tulip.security.fingerprint import FEATURE_KEYS
from tulip.security.fingerprint import dispatch_timing_probe_reference as dispatch_timing_probe


__all__ = ["FEATURE_KEYS", "dispatch_timing_probe"]
