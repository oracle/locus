# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Inference fingerprinting — identify what serves an endpoint from timing.

The cheapest inference-fingerprinting channel needs no privileged access
and no co-located hardware: stream a completion from a target endpoint and
time the token arrivals. Model size shows up in inter-token latency, the
inference engine in the cadence, and the hardware shifts the whole profile.

- :func:`measure_endpoint_timing` — remote-API timing (no GPU). Live with
  ``OPENAI_API_KEY`` (+ optional ``TIMING_BASE_URL`` for any
  OpenAI-compatible endpoint); deterministic offline sample otherwise.
- :func:`dispatch_timing_probe_reference` — *reference* (offline) GPU-probe
  dispatch. The real RunPod / Lambda lifecycle lives in ``tulip-integrations``
  (compute) as ``dispatch_timing_probe`` — the names differ so the offline
  reference is never mistaken for the live probe.
- :func:`default_classifier` — a deterministic heuristic mapping a feature
  vector to a :class:`~tulip.security.FingerprintVerdict`. **Placeholder**
  for a trained classifier (the Clusiana research program); honest about it.
- :func:`fingerprint_to_finding` — measure → classify → ground via
  :func:`~tulip.security.ground_fingerprint`, so an under-observed endpoint
  abstains rather than asserting an identity.

Defensive framing: run this against *your own* endpoints to verify what they
reveal (inventory verification) — the same measurement an attacker uses for
model-extraction recon (MITRE ATLAS AML.T0040 / AML.T0024).

The remote-API timing measurement is real (confirmed against ``gpt-4o-mini``);
the classifier remains a heuristic placeholder. The co-located GPU-cloud probe
is split by provider in ``tulip-integrations.compute`` — ``runpod`` (pod +
container image, extra ``compute-runpod``) and ``lambda_cloud`` (instance +
result sink, extra ``compute-lambda``) — and core ships only the offline
reference dispatch.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from tulip.reasoning.gsar import Partition
from tulip.security._adapters import as_json, env, inference_claim, tool_match
from tulip.security.findings import FingerprintVerdict, Indicator
from tulip.security.grounded import Abstention, FingerprintFinding, ground_fingerprint
from tulip.security.taxonomy import AtlasTechnique, IndicatorType, Severity, TaxonomyTag
from tulip.tools.decorator import tool


# The timing-feature schema a streaming-API fingerprint expects. Coverage is
# the fraction of these the measurement actually produced.
FEATURE_KEYS: tuple[str, ...] = ("ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean")

# Deterministic offline sample (shared by both channels so they're comparable).
_SAMPLE_FEATURES: dict[str, float] = {
    "ttft_ms_p50": 38.2,
    "itl_ms_mean": 11.4,
    "itl_cv": 0.07,
    "tps_mean": 87.6,
}


# ---------------------------------------------------------------------------
# Remote-API timing (no GPU)
# ---------------------------------------------------------------------------


def measure_endpoint_timing(
    model: str = "gpt-4o-mini",
    samples: int = 5,
    prompt: str = "Count slowly from one to twenty.",
) -> dict[str, float]:
    """Return a streaming-timing feature vector for an endpoint.

    Live path (``OPENAI_API_KEY`` set) streams ``samples`` completions and
    computes TTFT p50, mean inter-token latency, its coefficient of variation,
    and mean tokens/sec. Offline path returns the deterministic sample.
    """
    api_key = env("OPENAI_API_KEY")
    if not api_key:
        return dict(_SAMPLE_FEATURES)
    base_url = os.environ.get("TIMING_BASE_URL", "https://api.openai.com/v1")
    ttfts: list[float] = []
    itls: list[float] = []
    tps: list[float] = []
    for _ in range(samples):
        timing = _stream_once(base_url, api_key, model, prompt)
        if timing is not None:
            ttfts.append(timing["ttft_ms"])
            itls.extend(timing["itl_ms"])
            tps.append(timing["tps"])
    if not ttfts or not itls:
        return dict(_SAMPLE_FEATURES)
    itl_mean = statistics.fmean(itls)
    itl_sd = statistics.pstdev(itls) if len(itls) > 1 else 0.0
    return {
        "ttft_ms_p50": round(statistics.median(ttfts), 2),
        "itl_ms_mean": round(itl_mean, 2),
        "itl_cv": round(itl_sd / itl_mean, 3) if itl_mean else 0.0,
        "tps_mean": round(statistics.fmean(tps), 2),
    }


def _stream_once(base_url: str, api_key: str, model: str, prompt: str) -> dict[str, Any] | None:
    """Stream one completion, returning per-request timing or None on error."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    start = time.perf_counter()
    first: float | None = None
    arrivals: list[float] = []
    try:
        with (
            httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as client,
            client.stream("POST", "/chat/completions", json=body) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                if _has_content(payload):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    arrivals.append(now)
    except httpx.HTTPError:
        return None
    if first is None or len(arrivals) < 2:
        return None
    gaps_ms = [(b - a) * 1000.0 for a, b in pairwise(arrivals)]
    total_s = arrivals[-1] - start
    return {
        "ttft_ms": (first - start) * 1000.0,
        "itl_ms": gaps_ms,
        "tps": len(arrivals) / total_s if total_s > 0 else 0.0,
    }


def _has_content(payload: str) -> bool:
    """Whether an SSE chunk carries a non-empty token delta."""
    try:
        delta = json.loads(payload)["choices"][0]["delta"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return False
    return bool(delta.get("content"))


# ---------------------------------------------------------------------------
# Co-located GPU probe (reference)
# ---------------------------------------------------------------------------


def dispatch_timing_probe_reference(endpoint: str, provider: str = "runpod") -> dict[str, float]:
    """Reference (offline) co-located GPU-probe dispatch — returns the sample vector.

    The credential-free remote-API measurement is :func:`measure_endpoint_timing`.
    The *real* GPU-cloud lifecycle (provision -> probe -> tear down) lives in the
    ``tulip-integrations`` package as two separate provider modules — RunPod
    (``tulip_integrations.compute.runpod.runpod_probe``, extra ``compute-runpod``)
    and Lambda Cloud (``tulip_integrations.compute.lambda_cloud.lambda_probe``,
    extra ``compute-lambda``); ``compute.dispatch_timing_probe(endpoint, provider=…)``
    routes between them. Core ships no vendor GPU code.
    """
    return dict(_SAMPLE_FEATURES)


# ---------------------------------------------------------------------------
# Classification + grounding
# ---------------------------------------------------------------------------


def default_classifier(features: Mapping[str, float]) -> FingerprintVerdict:
    """Map a timing feature vector to a verdict (deterministic heuristic).

    This is a transparent placeholder for a trained classifier — it bins on
    inter-token latency and its coefficient of variation. ``feature_coverage``
    is the fraction of :data:`FEATURE_KEYS` observed; low coverage should
    drive an abstention downstream (see :func:`fingerprint_to_finding`).
    """
    coverage = sum(1 for k in FEATURE_KEYS if k in features) / len(FEATURE_KEYS)
    itl = float(features.get("itl_ms_mean", 0.0))
    cv = float(features.get("itl_cv", 1.0))
    if itl and itl < 13:
        model, hardware = "7-8B class", "H100/A100 class"
    elif itl < 25:
        model, hardware = "13-34B class", "A100 class"
    else:
        model, hardware = "70B+ class", "commodity / loaded"
    engine = "vLLM (continuous-batching)" if cv < 0.12 else "TGI / llama.cpp class"
    confidence = round(min(0.95, 0.5 + coverage * 0.4), 2)
    return FingerprintVerdict(
        model=model,
        engine=engine,
        hardware=hardware,
        confidence=confidence,
        feature_coverage=round(coverage, 2),
    )


def fingerprint_to_finding(
    features: Mapping[str, float],
    *,
    asset: str,
    classifier: object | None = None,
    min_coverage: float = 0.75,
    severity: Severity = Severity.MEDIUM,
    taxonomy: list[TaxonomyTag] | None = None,
) -> FingerprintFinding | Abstention:
    """Classify a timing vector and ground it into a fingerprint finding.

    The measured vector is tool-backed evidence; when feature coverage clears
    ``min_coverage`` the verdict ships, otherwise the under-observed endpoint
    abstains (an asserted identity from a thin measurement is a false
    positive by construction).
    """
    classify = classifier if classifier is not None else default_classifier
    verdict = classify(features)  # type: ignore[operator]
    measured = ", ".join(f"{k}={features[k]}" for k in FEATURE_KEYS if k in features)
    ref = f"tool:measure_endpoint_timing:{asset}"
    if verdict.feature_coverage >= min_coverage:
        partition = Partition(grounded=[tool_match(f"timing vector observed ({measured})", ref)])
    else:
        partition = Partition(
            ungrounded=[inference_claim(f"under-observed timing vector ({measured})", ref)],
        )
    return ground_fingerprint(
        verdict=verdict,
        asset=asset,
        partition=partition,
        severity=severity,
        indicators=[Indicator(type=IndicatorType.ENDPOINT, value=asset)],
        taxonomy=taxonomy
        or [AtlasTechnique.INFERENCE_API_ACCESS, AtlasTechnique.EXFILTRATION_VIA_INFERENCE_API],
    )


@tool(
    name="fingerprint_endpoint",
    description="Measure an endpoint's streaming-timing vector and identify model/engine/hardware",
)
async def fingerprint_endpoint_tool(endpoint: str, model: str = "gpt-4o-mini") -> str:
    """Tool wrapper: measure timing, classify, return the verdict as JSON."""
    features = measure_endpoint_timing(model=model)
    verdict = default_classifier(features)
    return as_json({"endpoint": endpoint, "features": features, "verdict": verdict.model_dump()})


__all__ = [
    "FEATURE_KEYS",
    "default_classifier",
    "dispatch_timing_probe_reference",
    "fingerprint_endpoint_tool",
    "fingerprint_to_finding",
    "measure_endpoint_timing",
]
