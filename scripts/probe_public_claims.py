#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Probe the public claims on locusagents.ai against the code.

Every check maps to a concrete, verbatim claim made on the marketing or docs
pages. Offline-only: uses the bundled MockModel, no credentials, no cost. Run:

    .venv/bin/python scripts/probe_public_claims.py

Exit code is non-zero if any claim fails, so this can gate a publish. Claims
that require credentials (live LLM, live timing, live vendor APIs) or billable
GPU (RunPod/Lambda) are reported as SKIPPED with what they'd need.
"""

from __future__ import annotations

import sys


sys.path.insert(0, "examples")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(claim: str, status: str, detail: str) -> None:
    results.append((claim, status, detail))


def main() -> None:
    # ── CLAIM 2 — abstain by construction: an ungrounded answer yields an ────
    # Abstention, never a Finding. (open-source.html: "an ungrounded finding is
    # a false positive by construction. There's no public path that builds one.")
    try:
        from locus.control import Abstention, Severity, ground_finding
        from locus.reasoning.gsar import Claim, EvidenceType, Partition

        out = ground_finding(
            title="suspicious IAM role",
            description="role looks over-privileged",
            severity=Severity.MEDIUM,
            asset="arn:aws:iam::000000000000:role/x",
            remediation="review the trust policy",
            # only an ungrounded inference, no tool-backed evidence -> must abstain
            partition=Partition(
                ungrounded=[
                    Claim(
                        text="looks risky",
                        type=EvidenceType.INFERENCE,
                        evidence_refs=["model:guess"],
                    )
                ]
            ),
        )
        ok = isinstance(out, Abstention)
        record(
            "abstain-by-construction: ungrounded -> Abstention",
            PASS if ok else FAIL,
            f"returned {type(out).__name__}"
            + (f" (gsar={out.gsar_score:.2f})" if hasattr(out, "gsar_score") else ""),
        )
    except Exception as exc:  # noqa: BLE001
        record("abstain-by-construction", FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 3 — "40+ canonical events" (open-source.html) ──────────────────
    try:
        import pathlib
        import re

        vals: set[str] = set()
        for p in pathlib.Path("src/locus").rglob("*.py"):
            for m in re.finditer(r'EV_[A-Z0-9_]+\s*=\s*"([^"]+)"', p.read_text()):
                vals.add(m.group(1))
        n = len(vals)
        record(
            '"40+ canonical events"',
            PASS if n >= 40 else FAIL,
            f"{n} unique EV_* canonical event-type strings across locus.*",
        )
    except Exception as exc:  # noqa: BLE001
        record('"40+ canonical events"', FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 4 — "Eight shapes, one class" coordination ─────────────────────
    try:
        shapes = []
        from locus.agent import Agent  # 1 single

        shapes.append("single")
        from locus.agent.composition import (  # 2,3 pipeline/loop
            LoopAgent,
            SequentialPipeline,
        )

        shapes += ["pipeline", "loop"]
        import importlib

        for name, target in [
            ("parallel", "locus.agent.composition:ParallelPipeline"),  # 4 parallel
            ("orchestrator", "locus.multiagent.orchestrator:Orchestrator"),  # 5 orchestrator
            ("swarm", "locus.multiagent.swarm:Swarm"),  # 6 swarm
            ("handoff", "locus.multiagent.handoff:Handoff"),  # 7 hand-off
            ("a2a", "locus.a2a:"),  # 8 cross-process A2A
        ]:
            mod, _, attr = target.partition(":")
            obj = importlib.import_module(mod)
            if attr and not hasattr(obj, attr):
                raise AttributeError(f"{mod}.{attr} missing")
            shapes.append(name)
        _ = (Agent, SequentialPipeline, LoopAgent)
        record(
            '"eight shapes, one class"',
            PASS if len(shapes) >= 8 else FAIL,
            f"{len(shapes)} importable shapes: {', '.join(shapes)}",
        )
    except Exception as exc:  # noqa: BLE001
        record('"eight shapes, one class"', FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 6 — "provider-agnostic: OpenAI, Anthropic, self-hosted vLLM, ───
    # or any OpenAI-compatible endpoint via base_url". vLLM is reached through
    # the OpenAI provider + base_url (not a dedicated prefix), so verify both
    # native providers are registered AND the OpenAI provider accepts base_url.
    try:
        from locus.models.registry import get_model, list_providers

        provs = set(list_providers())
        natives = {"openai", "anthropic"} <= provs
        m = get_model("openai:any-model", base_url="http://localhost:8000/v1", api_key="x")
        vllm_path = getattr(m.config, "base_url", None) == "http://localhost:8000/v1"
        record(
            "provider-agnostic: openai+anthropic native, vLLM via base_url",
            PASS if (natives and vllm_path) else FAIL,
            f"providers={sorted(provs)}; openai base_url override={'ok' if vllm_path else 'MISSING'}",
        )
    except Exception as exc:  # noqa: BLE001
        record("provider-agnostic", FAIL, f"{type(exc).__name__}: {exc}")

    # ── Credential / billable claims — reported, not run ─────────────────────
    record(
        "Clusiana: 94% model-ID, 1.2s probe, ~25% mitigation cost",
        SKIP,
        "private clusiana repo; trace to eval_report.json + experiments/0007,0017,0018",
    )
    record(
        "compute: live RunPod/Lambda GPU probe",
        SKIP,
        "BILLABLE H100 spin-up — needs explicit auth + RUNPOD_API_KEY / LAMBDA_API_KEY",
    )
    record(
        "integrations: live VirusTotal / Auth0 / SIEM / EDR",
        SKIP,
        "needs vendor keys (source ~/.profile); VT + Auth0 previously live-verified",
    )

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PUBLIC-CLAIM PROBE  —  locusagents.ai  vs  the code")
    print("=" * 80)
    width = max(len(c) for c, _, _ in results)
    for claim, status, detail in results:
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}[status]
        print(f"{mark} {status:4} {claim:<{width}}  {detail}")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print("-" * 80)
    print(f"{n_pass} pass · {n_fail} fail · {n_skip} skipped (need keys / billable)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
