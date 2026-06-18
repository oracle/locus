# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — GPU-probe dispatch graduated into the SDK.

The co-located timing-probe lifecycle is now first-class in
:mod:`tulip.security.fingerprint`. Core ships the *offline reference*
(``dispatch_timing_probe_reference``); the live probe is split by provider in
``tulip-integrations`` — RunPod (``tulip_integrations.compute.runpod``) and
Lambda Cloud (``tulip_integrations.compute.lambda_cloud``), routed by
``tulip_integrations.compute.dispatch_timing_probe(endpoint, provider=…)``.
This shim re-exports the offline reference under the example's local name::

    from tulip.security import dispatch_timing_probe_reference, FEATURE_KEYS
"""

from __future__ import annotations

from tulip.security.fingerprint import FEATURE_KEYS
from tulip.security.fingerprint import dispatch_timing_probe_reference as dispatch_timing_probe


__all__ = ["FEATURE_KEYS", "dispatch_timing_probe"]
