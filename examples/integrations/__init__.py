# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Worked vendor integrations for the cookbook.

Each module is a real integration with an external system a security agent
uses — a GPU cloud (RunPod/Lambda) for inference-fingerprint probes, a
threat-intel feed for IOC enrichment, a SIEM for log/alert queries. Every
one follows the same convention: read the vendor credential from the
environment and call the live API when it's set; otherwise return a
deterministic, benign sample so the example runs offline with no account.

See ``README.md`` in this directory for the bring-your-own-credentials
contract and the probe-image / result-sink details.
"""

from __future__ import annotations

from integrations.gpu_probe_dispatch import FEATURE_KEYS, dispatch_timing_probe
from integrations.siem_query import query_siem, siem_query_tool
from integrations.threat_intel import enrich_indicator, enrich_indicator_tool


__all__ = [
    "FEATURE_KEYS",
    "dispatch_timing_probe",
    "enrich_indicator",
    "enrich_indicator_tool",
    "query_siem",
    "siem_query_tool",
]
