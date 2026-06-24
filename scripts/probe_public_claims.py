#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Probe the public claims on tuliplabs.ai / tulipagents.ai against the code.

Every check maps to a concrete, verbatim claim made on the marketing or docs
pages. Offline-only: uses the bundled MockModel, no credentials, no cost. Run:

    .venv/bin/python scripts/probe_public_claims.py

Exit code is non-zero if any claim fails, so this can gate a publish. Claims
that require credentials (live LLM, live timing, live vendor APIs) or billable
GPU (RunPod/Lambda) are reported as SKIPPED with what they'd need.
"""

from __future__ import annotations

import inspect
import os
import sys


sys.path.insert(0, "examples")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(claim: str, status: str, detail: str) -> None:
    results.append((claim, status, detail))


def main() -> None:
    # ── CLAIM 1 — the homepage hero snippet's API contract is valid ──────────
    # The published snippet (now corrected) is:
    #     analyst = create_soc_analyst(model="anthropic:claude-sonnet-4-6")
    #     result  = analyst.run_sync("Audit this account's IAM posture.")
    #     for f in ground_report(result.parsed):
    #         print(f.severity, f.title, f.gsar_score)  /  print("withheld:", f.reason)
    # We assert the exact contract: run_sync -> AgentResult with .parsed (NOT
    # .output), and that .run is an async generator (so the OLD `await
    # analyst.run(...)` form was invalid Python — the bug this guards against).
    try:
        from config import get_model

        from tulip.agent.agent import Agent
        from tulip.agent.result import AgentResult
        from tulip.security import create_soc_analyst, ground_report, is_finding  # noqa: F401

        analyst = create_soc_analyst(model=get_model())
        result = analyst.run_sync("Audit this account's IAM posture.")
        contract = (
            isinstance(result, AgentResult)
            and hasattr(result, "parsed")
            and not hasattr(result, "output")  # the published bug
            and inspect.isasyncgenfunction(Agent.run)  # why `await .run()` was invalid
        )
        record(
            "hero snippet: run_sync -> AgentResult.parsed (not .output / not await .run)",
            PASS if contract else FAIL,
            f"{type(result).__name__}.parsed ok; Agent.run is async-gen ({inspect.isasyncgenfunction(Agent.run)})",
        )
    except Exception as exc:  # noqa: BLE001
        record("hero snippet: API contract", FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 2 — abstain by construction: an ungrounded answer yields an ────
    # Abstention, never a Finding. (open-source.html: "an ungrounded finding is
    # a false positive by construction. There's no public path that builds one.")
    try:
        from tulip.reasoning.gsar import Partition
        from tulip.security import Abstention, ground_finding
        from tulip.security._adapters import inference_claim
        from tulip.security.taxonomy import Severity

        out = ground_finding(
            title="suspicious IAM role",
            description="role looks over-privileged",
            severity=Severity.MEDIUM,
            asset="arn:aws:iam::000000000000:role/x",
            remediation="review the trust policy",
            # only an ungrounded inference, no tool-backed evidence -> must abstain
            partition=Partition(ungrounded=[inference_claim("looks risky", "model:guess")]),
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
        for p in pathlib.Path("src/tulip").rglob("*.py"):
            for m in re.finditer(r'EV_[A-Z0-9_]+\s*=\s*"([^"]+)"', p.read_text()):
                vals.add(m.group(1))
        n = len(vals)
        record(
            '"40+ canonical events"',
            PASS if n >= 40 else FAIL,
            f"{n} unique EV_* canonical event-type strings across tulip.*",
        )
    except Exception as exc:  # noqa: BLE001
        record('"40+ canonical events"', FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 4 — "Eight shapes, one class" coordination ─────────────────────
    try:
        shapes = []
        from tulip.agent import Agent  # 1 single

        shapes.append("single")
        from tulip.agent.composition import (  # 2,3 pipeline/loop
            LoopAgent,
            SequentialPipeline,
        )

        shapes += ["pipeline", "loop"]
        import importlib

        for name, target in [
            ("parallel", "tulip.agent.composition:ParallelPipeline"),  # 4 parallel
            ("orchestrator", "tulip.multiagent.orchestrator:Orchestrator"),  # 5 orchestrator
            ("swarm", "tulip.multiagent.swarm:Swarm"),  # 6 swarm
            ("handoff", "tulip.multiagent.handoff:Handoff"),  # 7 hand-off
            ("a2a", "tulip.a2a:"),  # 8 cross-process A2A
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

    # ── CLAIM 5 — read-only cloud-posture auditor ("its tools admit only ─────
    # describe/list/get calls")
    try:
        from tulip.security import cloud_posture_audit

        tools = cloud_posture_audit()
        names = [getattr(t, "name", str(t)) for t in tools]
        bad = [
            n
            for n in names
            if any(
                w in n.lower() for w in ("create", "delete", "put", "update", "attach", "remove")
            )
            and "posture" not in n.lower()  # submit_posture is the output channel, not an AWS write
        ]
        record(
            "read-only auditor: no mutating AWS tools",
            PASS if not bad else FAIL,
            f"{len(names)} tools; mutating={bad or 'none'}",
        )
    except Exception as exc:  # noqa: BLE001
        record("read-only auditor", FAIL, f"{type(exc).__name__}: {exc}")

    # ── CLAIM 6 — "provider-agnostic: OpenAI, Anthropic, self-hosted vLLM, ───
    # or any OpenAI-compatible endpoint via base_url". vLLM is reached through
    # the OpenAI provider + base_url (not a dedicated prefix), so verify both
    # native providers are registered AND the OpenAI provider accepts base_url.
    try:
        from tulip.models.registry import get_model, list_providers

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

    # ── CLAIM 7 — "fingerprint AI infrastructure": timing measurement ────────
    try:
        from tulip.security.fingerprint import FEATURE_KEYS, measure_endpoint_timing

        feats = measure_endpoint_timing()  # offline sample without OPENAI_API_KEY
        have = set(feats) >= set(FEATURE_KEYS)
        live = bool(os.environ.get("OPENAI_API_KEY"))
        record(
            "fingerprint: measure_endpoint_timing returns feature vector",
            PASS if have else FAIL,
            ("LIVE vs OpenAI" if live else "offline sample") + f"; keys={sorted(feats)}",
        )
        if not live:
            record(
                'fingerprint: "verified live vs gpt-4o-mini"',
                SKIP,
                "needs OPENAI_API_KEY (source ~/.profile) — measurement is real, rerun to re-verify",
            )
    except Exception as exc:  # noqa: BLE001
        record("fingerprint: measure_endpoint_timing", FAIL, f"{type(exc).__name__}: {exc}")

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
    print("PUBLIC-CLAIM PROBE  —  tuliplabs.ai / tulipagents.ai  vs  the code")
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
