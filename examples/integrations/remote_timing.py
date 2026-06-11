# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Remote-API timing fingerprint — measure an endpoint with no GPU.

The cheapest inference-fingerprinting channel needs no privileged access
and no co-located hardware: stream a completion from the target endpoint
and time the token arrivals. Model size shows up in inter-token latency,
the inference engine shows up in the cadence (vLLM continuous-batching vs
TGI vs llama.cpp have distinct ITL distributions), and the hardware shifts
the whole profile.

With ``OPENAI_API_KEY`` (and optionally ``TIMING_BASE_URL`` pointing at any
OpenAI-compatible endpoint — a hosted API or your own vLLM) set, this
streams real completions and computes the feature vector. With none set it
returns a deterministic offline sample so the cookbook runs with no
credentials. Feed the result to ``classify_fingerprint`` +
``ground_fingerprint`` (see scenarios/model_extraction.py).

Defensive framing: run this against *your own* endpoints to verify what
they reveal (inventory verification) — the same measurement an attacker
would use for model-extraction reconnaissance (MITRE ATLAS AML.T0040 /
AML.T0024).

VERIFIED LIVE: the streaming parse + timing math were confirmed against
OpenAI ``gpt-4o-mini`` (measured ttft≈750ms / itl≈15-20ms / tps≈31 over the
public internet — real numbers, not the sample). The *measurement* is real.
Note the classifier that maps the feature vector to a model identity (see
``scenarios/model_extraction.py``) is still a deterministic mock — that is
the piece a trained model (the Clusiana program) replaces. CV is inflated
here by public-internet jitter; a co-located probe reads much lower.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from itertools import pairwise


FEATURE_KEYS: tuple[str, ...] = ("ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean")

# Deterministic offline sample (matches the GPU-probe sample so the two
# channels are comparable in the cookbook).
_SAMPLE_FEATURES: dict[str, float] = {
    "ttft_ms_p50": 38.2,
    "itl_ms_mean": 11.4,
    "itl_cv": 0.07,
    "tps_mean": 87.6,
}


def measure_endpoint_timing(
    model: str = "gpt-4o-mini",
    samples: int = 5,
    prompt: str = "Count slowly from one to twenty.",
) -> dict[str, float]:
    """Return a streaming-timing feature vector for an endpoint.

    Live path (``OPENAI_API_KEY`` set) streams ``samples`` completions and
    computes TTFT p50, mean inter-token latency, its coefficient of
    variation, and mean tokens/sec. Offline path returns the sample.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return dict(_SAMPLE_FEATURES)
    base_url = os.environ.get("TIMING_BASE_URL", "https://api.openai.com/v1")
    ttfts: list[float] = []
    itls: list[float] = []
    tps: list[float] = []
    for _ in range(samples):
        t = _stream_once(base_url, api_key, model, prompt)
        if t is not None:
            ttfts.append(t["ttft_ms"])
            itls.extend(t["itl_ms"])
            tps.append(t["tps"])
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


def _stream_once(base_url: str, api_key: str, model: str, prompt: str) -> dict | None:
    """Stream one completion, returning per-request timing or None on error."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    start = time.perf_counter()
    first: float | None = None
    arrivals: list[float] = []
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


if __name__ == "__main__":
    feats = measure_endpoint_timing()
    print(f"timing features: {feats}")
