# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Live inference-fingerprint probe against a real GPU-served endpoint.

The headline AI-security capability on real silicon: stream completions from a
self-hosted vLLM endpoint (RunPod / Lambda / anywhere OpenAI-compatible), time
the token arrivals, and ground the resulting timing feature vector into a
:class:`~tulip.security.FingerprintFinding`.

Point it at the endpoint via env and it runs; absent that, it skips:

    TIMING_BASE_URL   the OpenAI-compatible base URL of the vLLM endpoint
                      (e.g. https://api.runpod.ai/v2/<id>/openai/v1)
    TIMING_MODEL      the model id the endpoint serves
    OPENAI_API_KEY    bearer token for the endpoint (the RunPod key works)

The *measurement* and the *grounding* are real. The classifier that maps the
feature vector to a model identity is a deterministic stand-in here — that is
the piece a trained model (the Clusiana program) replaces.

Maps to MITRE ATLAS AML.T0040 (inference-API access) / AML.T0024
(exfiltration via inference API): the same measurement an attacker uses for
model-extraction recon, run defensively against your own endpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import (
    FingerprintVerdict,
    Severity,
    ground_fingerprint,
    is_finding,
)


# The streaming-timing probe lives in the cookbook's vendor integrations.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from integrations.remote_timing import (  # noqa: E402
    _SAMPLE_FEATURES,
    FEATURE_KEYS,
    measure_endpoint_timing,
)


skip_without_endpoint = pytest.mark.skipif(
    not os.environ.get("TIMING_BASE_URL"),
    reason="set TIMING_BASE_URL (+ TIMING_MODEL, OPENAI_API_KEY) to a live vLLM endpoint",
)


def _classify(features: dict[str, float]) -> FingerprintVerdict:
    """Deterministic stand-in classifier (the trained model is private)."""
    coverage = sum(1 for k in FEATURE_KEYS if k in features) / len(FEATURE_KEYS)
    # An endpoint we stood up ourselves: we expect a vLLM/open-weights profile.
    return FingerprintVerdict(
        model="open-weights",
        engine="vllm",
        hardware="datacenter-gpu",
        confidence=0.9,
        feature_coverage=coverage,
    )


@skip_without_endpoint
def test_fingerprint_probe_grounds_real_endpoint() -> None:
    model = os.environ.get("TIMING_MODEL", "gpt-4o-mini")
    features = measure_endpoint_timing(model=model, samples=5)

    # Real silicon was probed — the measurement is not the offline fallback.
    assert features != _SAMPLE_FEATURES, "got the offline sample, not a live measurement"
    assert features["tps_mean"] > 0
    assert features["ttft_ms_p50"] > 0
    assert set(features) == set(FEATURE_KEYS)  # full coverage

    verdict = _classify(features)
    finding = ground_fingerprint(
        verdict=verdict,
        asset=os.environ["TIMING_BASE_URL"],
        severity=Severity.MEDIUM,
        partition=Partition(
            grounded=[
                Claim(
                    text=f"TTFT p50 {features['ttft_ms_p50']}ms / ITL mean "
                    f"{features['itl_ms_mean']}ms / {features['tps_mean']} tok/s "
                    "matches a vLLM open-weights profile",
                    type=EvidenceType.TOOL_MATCH,
                    evidence_refs=[f"probe:timing:{k}={features[k]}" for k in FEATURE_KEYS],
                ),
            ],
        ),
    )

    # Full-coverage timing evidence grounds a real fingerprint finding.
    assert is_finding(finding)
    assert finding.verdict.engine == "vllm"
    assert finding.gsar_score >= 0.5
    assert any("probe:timing:" in ref for ref in finding.evidence_refs)


@skip_without_endpoint
def test_fingerprint_abstains_on_sparse_coverage() -> None:
    # One brief sample is too thin to confidently identify the endpoint — so it
    # is an *inference*, not a tool-grounded match. ground_fingerprint scores
    # the partition, and an ungrounded-only partition abstains.
    ttft = measure_endpoint_timing(samples=2)["ttft_ms_p50"]
    verdict = FingerprintVerdict(
        model="unknown",
        engine="unknown",
        hardware="unknown",
        confidence=0.4,
        feature_coverage=1 / len(FEATURE_KEYS),
    )
    result = ground_fingerprint(
        verdict=verdict,
        asset=os.environ["TIMING_BASE_URL"],
        partition=Partition(
            ungrounded=[
                Claim(
                    text="single TTFT sample — insufficient to identify the endpoint",
                    type=EvidenceType.INFERENCE,
                    evidence_refs=[f"probe:timing:ttft_ms_p50={ttft}"],
                ),
            ],
        ),
    )
    assert not is_finding(result)  # abstains on thin evidence
