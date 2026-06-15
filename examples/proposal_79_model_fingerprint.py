#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""PROPOSAL — Notebook 79: Model & hardware fingerprinting via timing side-channels.

STATUS: PROPOSAL — not yet promoted to a full notebook.

Scenario
────────
A CISO is about to sign a $2 M/year contract with a vendor claiming to serve
"GPT-4o on H100 clusters."  Before signing, she wants to verify:
  - Is the model actually GPT-4o (or a smaller substitute)?
  - Is it running on H100 (or shared consumer-grade GPU)?
  - How confident is the classifier, and can this be grounded in evidence?

Timing side-channels are the only non-intrusive signal available to a buyer
with black-box API access: TTFT (time-to-first-token), inter-token latency,
and tail percentiles differ significantly across model families and hardware
classes because of differences in FLOPs, KV-cache behaviour, and memory
bandwidth.

How it works
────────────
``measure_endpoint_timing(model=..., samples=N)`` collects N timed token
streams and extracts a feature vector over ``FEATURE_KEYS``
(ttft_ms_p50, itl_ms_mean, itl_cv, tps_mean).

With ``OPENAI_API_KEY`` (and optional ``TIMING_BASE_URL`` for any
OpenAI-compatible endpoint), the live streaming path is used.  Without it,
a deterministic offline sample is returned — enough to exercise the full
classify → ground pipeline.

``fingerprint_to_finding(features, asset=...)`` classifies the feature
vector, then calls ``ground_fingerprint`` internally: if feature coverage
clears the threshold, a grounded ``FingerprintFinding`` is returned; if too
few features were observed (e.g., the endpoint rate-limited or timed out),
the result is an ``Abstention`` — nothing is asserted.

The ``AuditTrail`` records the probe and its grounding decision, exportable
as JSONL for vendor SLA disputes or procurement audit evidence.

Run (offline, no credentials):
    python examples/proposal_79_model_fingerprint.py

Run against a real OpenAI-compatible endpoint:
    OPENAI_API_KEY=sk-... TIMING_BASE_URL=https://vendor-api.example.com/v1 \\
    python examples/proposal_79_model_fingerprint.py
"""

from __future__ import annotations

import asyncio
import os

from tulip.security import (
    AuditTrail,
    FEATURE_KEYS,
    fingerprint_to_finding,
    is_finding,
    measure_endpoint_timing,
)
from tulip.security.taxonomy import AtlasTechnique


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------

# Model name sent to the OpenAI-compatible API.
# Override with FINGERPRINT_MODEL env var to probe a different model name.
_MODEL = os.environ.get("FINGERPRINT_MODEL", "gpt-4o-mini")

# Number of timed streaming samples to collect.
# More samples = better feature coverage but more API cost and latency.
_SAMPLES = int(os.environ.get("FINGERPRINT_SAMPLES", "10"))

# Human-readable asset name for the Finding (e.g., the vendor + contract ref).
_ASSET = os.environ.get("FINGERPRINT_ASSET", "vendor-inference-api")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fingerprint() -> None:
    """
    Measure timing features, classify, and ground.

    Live path:   set OPENAI_API_KEY (+ TIMING_BASE_URL for non-OpenAI endpoints).
    Offline path: no credentials needed — deterministic sample used instead.
    """
    using_live = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"== Model / hardware fingerprinting via timing side-channels ==")
    print(f"   model  : {_MODEL!r}")
    print(f"   samples: {_SAMPLES}")
    print(f"   path   : {'live API stream' if using_live else 'offline deterministic sample'}")
    print()

    trail = AuditTrail()
    trail.record(
        "fingerprint-session-start",
        {"model": _MODEL, "samples": _SAMPLES, "asset": _ASSET, "live": using_live},
    )

    # ------------------------------------------------------------------ #
    # Step 1: Collect timing features                                      #
    # ------------------------------------------------------------------ #
    print(f"Collecting {_SAMPLES} timing samples...")
    features = measure_endpoint_timing(model=_MODEL, samples=_SAMPLES)

    print("\nExtracted timing features:")
    for key in FEATURE_KEYS:
        val = features.get(key)
        if val is not None:
            print(f"  {key:<20} {val:.4f}")
        else:
            print(f"  {key:<20} (not observed)")

    observed = sum(1 for k in FEATURE_KEYS if features.get(k) is not None)
    coverage = observed / len(FEATURE_KEYS)
    print(f"\nFeature coverage: {coverage:.0%} ({observed}/{len(FEATURE_KEYS)} keys)")

    # ------------------------------------------------------------------ #
    # Step 2: Classify + ground (fingerprint_to_finding does both)         #
    # ------------------------------------------------------------------ #
    print("\nClassifying and grounding...")
    result = fingerprint_to_finding(features, asset=_ASSET)

    trail.record_event(result)

    print()
    if is_finding(result):
        v = result.verdict
        print(f"  [FINDING ] Grounded FingerprintFinding")
        print(f"             gsar_score : {result.gsar_score:.2f}")
        print(f"             model      : {v.model}")
        print(f"             engine     : {v.engine}")
        print(f"             hardware   : {v.hardware}")
        print(f"             confidence : {v.confidence:.2f}")
        print(f"             coverage   : {v.feature_coverage:.0%}")
        print(f"             evidence   : {result.evidence_refs}")
        taxonomy_str = ", ".join(str(t) for t in result.taxonomy)
        print(f"             taxonomy   : {taxonomy_str}")
        print()
        print("  Interpretation:")
        print(f"    Vendor claims 'GPT-4o on H100 clusters'.")
        print(f"    Timing fingerprint identifies: {v.model} / {v.engine} / {v.hardware}.")
        if "H100" in v.hardware or "A100" in v.hardware:
            print("    Hardware class CONSISTENT with H100 claim.")
        else:
            print(f"    Hardware class {v.hardware!r} may DIVERGE from H100 claim — investigate.")
    else:
        print(f"  [ABSTAIN ] Insufficient timing evidence — cannot ground a finding")
        print(f"             reason     : {result.reason}")
        print(f"             gsar_score : {result.gsar_score:.2f}")
        print()
        print("  Action: Collect more samples (increase FINGERPRINT_SAMPLES) or")
        print("  check that the endpoint isn't rate-limiting streaming responses.")

    # ------------------------------------------------------------------ #
    # Step 3: Audit trail                                                  #
    # ------------------------------------------------------------------ #
    trail.record("fingerprint-session-end", {"grounded": is_finding(result)})
    assert trail.verify(), "Audit trail integrity check failed"

    jsonl = trail.export_jsonl()
    lines = jsonl.strip().split("\n")
    print(f"\nAudit trail: {len(lines)} records, integrity OK")
    print("(Export JSONL for vendor SLA dispute or procurement audit evidence.)")

    print()
    if not using_live:
        print(
            "To probe a real vendor endpoint, run:\n"
            f"  OPENAI_API_KEY=sk-... \\\n"
            f"  TIMING_BASE_URL=https://vendor-api.example.com/v1 \\\n"
            f"  FINGERPRINT_MODEL=gpt-4o \\\n"
            f"  FINGERPRINT_SAMPLES=20 \\\n"
            f"  FINGERPRINT_ASSET='VendorCo-contract-2026' \\\n"
            f"  python examples/proposal_79_model_fingerprint.py"
        )


if __name__ == "__main__":
    run_fingerprint()
