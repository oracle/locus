# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Model extraction / inference-API fingerprinting.

Threat: an adversary probes an inference endpoint to identify the model,
engine, and hardware behind it (reconnaissance for extraction), or to
extract behavior through high-volume crafted queries. The timing of
streamed tokens alone leaks the fingerprint — no privileges, no exploit.

Defense (built-in SDK primitives): measure your own endpoint
(integrations.remote_timing — a real streaming probe, offline sample with
no key), classify the feature vector, and ground the verdict with
tulip.security.ground_fingerprint — a low-coverage probe abstains instead
of asserting. Pair with rate limiting / unbounded-consumption caps to blunt
high-volume extraction.

Taxonomy: MITRE ATLAS AML.T0040 (Inference API Access) · AML.T0024
(Exfiltration via Inference API) · AML.T0043 (Craft Adversarial Data) ·
OWASP LLM10 (Unbounded Consumption).
"""

from __future__ import annotations

import sys
from pathlib import Path

from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import FingerprintVerdict, Severity, ground_fingerprint, is_finding


# Make the sibling examples/integrations/ package importable when this gist
# is run directly from examples/scenarios/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.remote_timing import FEATURE_KEYS, measure_endpoint_timing  # noqa: E402


_COVERAGE_FLOOR = 0.60


def classify(features: dict[str, float]) -> FingerprintVerdict:
    """Deterministic mock classifier — a real service swaps in behind this."""
    coverage = sum(1 for k in FEATURE_KEYS if k in features) / len(FEATURE_KEYS)
    if coverage < _COVERAGE_FLOOR:
        return FingerprintVerdict(
            model="unknown",
            engine="unknown",
            hardware="unknown",
            confidence=0.30,
            feature_coverage=coverage,
        )
    return FingerprintVerdict(
        model="open-weights-7b",
        engine="vLLM",
        hardware="datacenter-gpu",
        confidence=0.91,
        feature_coverage=coverage,
    )


def main() -> None:
    print("Scenario: model extraction / fingerprinting  [AML.T0040 · T0024 · T0043 · LLM10]\n")

    # Measure your own endpoint (offline sample with no OPENAI_API_KEY).
    features = measure_endpoint_timing()
    print(f"timing features ({len(features)}/{len(FEATURE_KEYS)} observed): {features}")
    verdict = classify(features)
    print(
        f"verdict: {verdict.model} on {verdict.engine} @ {verdict.confidence:.0%} "
        f"(coverage {verdict.feature_coverage:.0%})"
    )

    result = ground_fingerprint(
        verdict=verdict,
        asset="203.0.113.10:443",
        severity=Severity.MEDIUM,
        partition=Partition(
            grounded=[
                Claim(
                    text=f"TTFT p50 {features.get('ttft_ms_p50')}ms matches the vLLM profile",
                    type=EvidenceType.TOOL_MATCH,
                    evidence_refs=["probe:ttft_ms_p50"],
                ),
                Claim(
                    text=f"inter-token latency {features.get('itl_ms_mean')}ms fits the 7B class",
                    type=EvidenceType.SPECIFIC_DATA,
                    evidence_refs=["probe:itl_ms_mean"],
                ),
            ],
        ),
    )
    if is_finding(result):
        print(f"  [SHIPS]   fingerprint finding: gsar={result.gsar_score:.2f} — {result.title}")
    else:
        print(f"  [ABSTAIN] {result.reason}")

    # A sparse probe (1/4 features) abstains rather than guessing.
    sparse = classify({"ttft_ms_p50": 41.0})
    weak = ground_fingerprint(
        verdict=sparse,
        asset="203.0.113.20:443",
        partition=Partition(
            ungrounded=[
                Claim(
                    text="a single TTFT sample loosely resembles a batched server",
                    type=EvidenceType.INFERENCE,
                    evidence_refs=["probe:ttft_ms_p50=41.0"],
                )
            ],
        ),
    )
    tag = "SHIPS" if is_finding(weak) else "ABSTAIN"
    print(f"  [{tag}]   under-observed endpoint (coverage {sparse.feature_coverage:.0%})")

    print("\nDefenders run this on their own endpoints; sparse evidence abstains, not guesses.")


if __name__ == "__main__":
    main()
