# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Dispatch a timing side-channel probe to a GPU cloud (RunPod / Lambda).

Inference fingerprinting measures *where the hardware is*: a small probe
image/script measures streaming-timing features against a target endpoint
and emits a JSON feature vector. This module orchestrates the pod/instance
lifecycle and returns that vector for grounding with
:func:`tulip.security.ground_fingerprint`.

Bring your own credentials. With ``RUNPOD_API_KEY`` (or ``LAMBDA_API_KEY``)
set, the matching live path runs; with neither set, a deterministic offline
sample is returned so the cookbook stays runnable on a clean machine — the
same bring-your-own-credentials pattern the rest of the examples use.

Probe contract: the probe emits a JSON object of timing features, e.g.::

    {"ttft_ms_p50": 38.2, "itl_ms_mean": 11.4, "itl_cv": 0.07, "tps_mean": 87.6}

UNVERIFIED LIVE PATH: the RunPod/Lambda branches are real lifecycle code
but depend on a probe artifact that does NOT ship — ``_PROBE_IMAGE`` is a
placeholder, and no container/result-sink exists yet. So the live path is
currently a no-op: with credentials set it would launch hardware but has
nothing to measure with. Building that probe (the CUDA kernels) is research
work that lives in the clusiana repo, not here. Only the offline sample
path runs today; the co-located GPU side-channel is not wired end-to-end.
For a probe that genuinely measures with no extra artifact, see
``remote_timing.py`` (remote-API timing, no GPU).
"""

from __future__ import annotations

import os
import time


# The timing-feature schema a streaming-API fingerprint expects. Coverage
# is the fraction of these the probe actually measured.
FEATURE_KEYS: tuple[str, ...] = ("ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean")

# Deterministic offline sample so the workflow runs with no credentials.
_SAMPLE_FEATURES: dict[str, float] = {
    "ttft_ms_p50": 38.2,
    "itl_ms_mean": 11.4,
    "itl_cv": 0.07,
    "tps_mean": 87.6,
}

# The probe container image (RunPod) — supply your own; see the README.
_PROBE_IMAGE = "tuliplabs/timing-probe:latest"


def dispatch_timing_probe(endpoint: str, provider: str = "runpod") -> dict[str, float]:
    """Run a timing probe against ``endpoint`` on a GPU cloud; return features.

    ``provider`` selects the GPU cloud — ``"runpod"`` or ``"lambda"``. With
    the matching API key set, the live path provisions hardware, runs the
    probe, collects the feature vector, and tears the hardware down. With no
    key set, returns the deterministic offline sample.
    """
    if provider == "runpod" and os.environ.get("RUNPOD_API_KEY"):
        return _runpod_probe(endpoint)
    if provider == "lambda" and os.environ.get("LAMBDA_API_KEY"):
        return _lambda_probe(endpoint)
    return dict(_SAMPLE_FEATURES)  # offline / no-credential fallback


def _select_features(raw: dict[str, object]) -> dict[str, float]:
    """Keep only the known feature keys, coerced to float."""
    return {k: float(raw[k]) for k in FEATURE_KEYS if k in raw}  # type: ignore[arg-type]


def _runpod_probe(endpoint: str) -> dict[str, float]:
    """Provision a RunPod GPU pod, run the probe image, collect, tear down.

    The probe image reads ``TARGET_ENDPOINT`` from its environment and emits
    ``{"features": {...}}`` as its output. Requires the ``runpod`` package
    and ``RUNPOD_API_KEY``.
    """
    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod = runpod.create_pod(
        name="tulip-timing-probe",
        image_name=_PROBE_IMAGE,
        gpu_type_id="NVIDIA H100",
        env={"TARGET_ENDPOINT": endpoint},
    )
    try:
        output = runpod.wait_for_output(pod["id"])
        features = output.get("features", output) if isinstance(output, dict) else {}
        return _select_features(features)
    finally:
        runpod.terminate_pod(pod["id"])


def _lambda_probe(endpoint: str) -> dict[str, float]:
    """Launch a Lambda Cloud GPU instance, poll the probe result, terminate.

    Lambda has no built-in output channel, so the probe uploads its feature
    JSON to a result sink (S3 object, small HTTP endpoint, …) whose URL is
    given in ``LAMBDA_PROBE_RESULT_URL``. Requires ``LAMBDA_API_KEY``.
    """
    import httpx

    key = os.environ["LAMBDA_API_KEY"]
    region = os.environ.get("LAMBDA_REGION", "us-east-1")
    with httpx.Client(
        base_url="https://cloud.lambdalabs.com/api/v1",
        headers={"Authorization": f"Bearer {key}"},
        timeout=60.0,
    ) as client:
        launched = client.post(
            "/instance-operations/launch",
            json={
                "instance_type_name": "gpu_1x_h100_pcie",
                "name": "tulip-timing-probe",
                "region_name": region,
            },
        )
        instance_id = launched.json()["data"]["instance_ids"][0]
        try:
            return _poll_probe_result(endpoint)
        finally:
            client.post("/instance-operations/terminate", json={"instance_ids": [instance_id]})


def _poll_probe_result(
    endpoint: str, attempts: int = 30, delay_s: float = 10.0
) -> dict[str, float]:
    """Poll the result sink the probe uploads its feature JSON to."""
    import httpx

    result_url = os.environ.get("LAMBDA_PROBE_RESULT_URL")
    if not result_url:
        msg = "set LAMBDA_PROBE_RESULT_URL to where the probe uploads its feature JSON"
        raise RuntimeError(msg)
    for _ in range(attempts):
        resp = httpx.get(result_url, params={"endpoint": endpoint}, timeout=30.0)
        if resp.status_code == 200:
            return _select_features(resp.json())
        time.sleep(delay_s)
    msg = "probe result not available within the polling window"
    raise TimeoutError(msg)


if __name__ == "__main__":
    # Offline demo: no credentials → the deterministic sample.
    feats = dispatch_timing_probe("203.0.113.10:443")
    observed = sum(1 for k in FEATURE_KEYS if k in feats)
    print(f"probe features ({observed}/{len(FEATURE_KEYS)} observed): {feats}")
