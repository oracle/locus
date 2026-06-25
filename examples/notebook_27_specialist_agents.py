# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 27: CURATOR — the inference-fingerprinting specialist.

Notebook 26 introduced the Specialist as the worker MARSHAL hands tasks
to. This notebook dives into the Specialist itself, using CURATOR — the
SOC's inference-forensics specialist — as the running case: how to narrow
a model's failure surface with a focused system prompt, a hand-picked
tool set, optional playbooks (your runbooks, encoded), and a confidence
threshold.

CURATOR's job is timing side-channel inference fingerprinting: identifying
which model, inference engine, and accelerator are serving an endpoint
purely from observable streaming-timing features (TTFT, inter-token
latency, cadence variance) — no privileges, no exploit. The defensive use
is inventory verification and detecting model-extraction reconnaissance
against the org's own inference endpoints (MITRE ATLAS AML.T0040 AI Model
Inference API Access, AML.T0024 Exfiltration via AI Inference API).

The capability is demonstrated with ``tulip.security``: a deterministic
mock classifier maps a timing feature vector to a ``FingerprintVerdict``,
and ``ground_fingerprint`` turns that verdict into a finding **only** when
the evidence clears the GSAR grounding threshold. Low feature coverage
yields a weak evidence partition, so an under-observed endpoint abstains
rather than asserting a fingerprint — a false fingerprint is worse than
none.

- A ``Specialist`` is a Tulip ``Agent`` with role metadata
  (``specialist_type``, ``description``), a tool list, and a
  ``confidence_threshold`` that gates "good enough" answers.
- ``Playbook`` and ``PlaybookStep`` encode standard procedures —
  preconditions, ordered steps with required tools and expected
  outputs, plus failure handling.
- A specialist can carry multiple playbooks; ``select_playbook(task)``
  picks one based on task description matching.
- Pre-built helpers (``create_log_analyst``, ``create_metrics_analyst``,
  ``create_trace_analyst``, ``create_code_analyst``) ship with sensible
  defaults for evidence domains every investigation touches.

Run it:
    .venv/bin/python examples/notebook_27_specialist_agents.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 06 (Agent basics).
- Notebook 26 (Orchestrator) — Specialists are the workers MARSHAL routes to.
"""

import asyncio
import time
from collections.abc import Mapping

from config import get_model, print_config
from integrations.gpu_probe_dispatch import dispatch_timing_probe

from tulip.agent import Agent
from tulip.multiagent.specialist import (
    Playbook,
    PlaybookStep,
    Specialist,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_trace_analyst,
)
from tulip.reasoning.gsar import Claim, EvidenceType, Partition
from tulip.security import (
    AtlasTechnique,
    FingerprintVerdict,
    Indicator,
    IndicatorType,
    Severity,
    ground_fingerprint,
    is_finding,
)
from tulip.tools.decorator import tool


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner — used for commentary."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# The expected timing-feature schema for a streaming-API fingerprint.
# Coverage is the fraction of these the probe actually measured; below
# this floor the classifier returns low confidence and grounding abstains.
_FINGERPRINT_FEATURES = ("ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean")
_COVERAGE_FLOOR = 0.60


def classify_fingerprint(features: Mapping[str, float]) -> FingerprintVerdict:
    """Deterministic mock inference-fingerprint classifier.

    Maps a timing feature vector to a ``(model, engine, hardware)`` verdict
    over a fixed lookup — no model file, no scikit, no network. A real
    deployment swaps in a fingerprinting service behind this same
    signature. Coverage below :data:`_COVERAGE_FLOOR` returns a
    low-confidence "unknown" verdict so the caller abstains under GSAR.

    Args:
        features: Observed timing features keyed by name.

    Returns:
        A :class:`~tulip.security.FingerprintVerdict`.
    """
    observed = [f for f in _FINGERPRINT_FEATURES if f in features]
    coverage = len(observed) / len(_FINGERPRINT_FEATURES)
    if coverage < _COVERAGE_FLOOR:
        return FingerprintVerdict(
            model="unknown",
            engine="unknown",
            hardware="unknown",
            confidence=0.30,
            feature_coverage=coverage,
        )
    # Fixed lookup: continuous-batching cadence + 7B-class inter-token
    # latency reads as an open-weights model behind vLLM on a datacenter GPU.
    itl = features.get("itl_ms_mean", 0.0)
    if itl <= 15.0:
        return FingerprintVerdict(
            model="open-weights-7b",
            engine="vLLM",
            hardware="datacenter-gpu",
            confidence=0.91,
            feature_coverage=coverage,
        )
    return FingerprintVerdict(
        model="open-weights-70b",
        engine="TGI",
        hardware="datacenter-gpu",
        confidence=0.78,
        feature_coverage=coverage,
    )


async def main():
    print("=" * 60)
    print("Notebook 27: CURATOR — the inference-fingerprinting specialist")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: anatomy of a Specialist
    # =========================================================================
    print("\n=== Part 1: Specialist Anatomy ===\n")

    # A specialist pairs a focused system prompt with domain tools,
    # optional playbooks, and a confidence threshold. CURATOR is the SOC's
    # inference-forensics specialist.
    specialist = Specialist(
        name="CURATOR",
        specialist_type="inference_forensics",
        description="Fingerprints inference endpoints from timing side-channels",
        system_prompt="""You are CURATOR, an inference-forensics specialist. Your expertise:
1. Measuring streaming-API timing features (TTFT, inter-token latency, cadence variance)
2. Mapping a timing feature vector to a (model, engine, hardware) verdict
3. Verifying served endpoints against the approved inventory
4. Flagging model-extraction reconnaissance against the org's own endpoints

When fingerprinting:
- Treat a single sample as inference, not evidence — collect a stable feature vector
- Report feature coverage; abstain when too little of the schema is observed
- Never assert a fingerprint you cannot ground — a wrong model attribution is worse than none""",
        max_iterations=10,
        confidence_threshold=0.85,
        model=model,
    )

    print(f"Specialist: {specialist.name}")
    print(f"  Type: {specialist.specialist_type}")
    print(f"  Max iterations: {specialist.max_iterations}")
    print(f"  Confidence threshold: {specialist.confidence_threshold}")
    t0 = time.perf_counter()
    p1 = await specialist.execute(task="In one sentence, what is your specialty?")
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · specialist.execute()]")
    if p1.output:
        print(f"  Smoke output: {p1.output[:160]}")

    # =========================================================================
    # Part 2: hand-pick tools for the domain
    # =========================================================================
    print("\n=== Part 2: Domain Tools ===\n")

    @tool(name="probe_timing", description="Measure streaming-timing features for an endpoint")
    async def probe_timing(endpoint: str, provider: str = "runpod") -> str:
        # Dispatch the probe to a dedicated GPU cluster (RunPod / Lambda),
        # where the measurement actually runs; falls back to a deterministic
        # sample offline. See examples/integrations/gpu_probe_dispatch.py.
        feats = dispatch_timing_probe(endpoint, provider=provider)
        observed = sum(1 for f in _FINGERPRINT_FEATURES if f in feats)
        body = " ".join(f"{k}={v}" for k, v in feats.items())
        return (
            f"Probe {endpoint} via {provider}: {body} "
            f"({observed}/{len(_FINGERPRINT_FEATURES)} features observed)"
        )

    @tool(name="classify_endpoint", description="Map the timing feature vector to a model verdict")
    async def classify_endpoint() -> str:
        return "VerificationResult: open-weights-7b on vLLM / datacenter-gpu, confidence 0.91, coverage 1.00"

    @tool(name="check_inventory", description="Compare a verdict against the approved inventory")
    async def check_inventory() -> str:
        return "Inventory: endpoint 203.0.113.10:443 approved to serve open-weights-7b on vLLM"

    specialist = Specialist(
        name="CURATOR",
        specialist_type="inference_forensics",
        description="Fingerprints inference endpoints from timing side-channels",
        system_prompt="You fingerprint inference endpoints from streaming-timing features.",
        tools=[probe_timing, classify_endpoint, check_inventory],
        model=model,
    )

    print(f"Tools available: {[t.name for t in specialist.tools]}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why does giving a Specialist domain-specific security tools dramatically narrow its failure surface?')}"
    )

    # =========================================================================
    # Part 3: encode a runbook as a Playbook
    # =========================================================================
    print("\n=== Part 3: Specialist Playbooks ===\n")

    fingerprint_playbook = Playbook(
        name="Inference Fingerprint Procedure",
        description="Standard procedure for fingerprinting an inference endpoint",
        preconditions=[
            "Endpoint is in scope for inventory verification",
            "Probing is rate-limited and authorized against our own endpoint",
        ],
        steps=[
            PlaybookStep(
                instruction="Measure the streaming-timing feature vector",
                required_tools=["probe_timing"],
                expected_output="TTFT, inter-token latency, cadence variance, and coverage",
            ),
            PlaybookStep(
                instruction="Classify the feature vector to a model/engine/hardware verdict",
                required_tools=["classify_endpoint"],
                expected_output="VerificationResult with confidence and feature coverage",
                on_failure="Abstain if feature coverage is below the schema floor",
            ),
            PlaybookStep(
                instruction="Compare the verdict against the approved inventory",
                required_tools=["check_inventory"],
                expected_output="Match / mismatch against what the endpoint is approved to serve",
            ),
        ],
        success_criteria="Grounded verdict reconciled against inventory, or a logged abstention",
    )

    specialist.playbooks.append(fingerprint_playbook)

    print(f"Playbook: {fingerprint_playbook.name}")
    print(f"  Preconditions: {fingerprint_playbook.preconditions}")
    print(f"  Steps: {len(fingerprint_playbook.steps)}")
    print(f"  Success criteria: {fingerprint_playbook.success_criteria}")

    # ``to_prompt()`` renders the playbook as the text block injected
    # into the system prompt when the playbook is selected.
    playbook_prompt = fingerprint_playbook.to_prompt()
    print("\nPlaybook prompt:")
    print("-" * 40)
    print(playbook_prompt[:500] + "...")
    print(
        f"AI commentary: {_llm_call('In one sentence, when does attaching a fixed Playbook to a security Specialist matter most?')}"
    )

    # =========================================================================
    # Part 4: pick the right playbook from a pool
    # =========================================================================
    print("\n=== Part 4: Playbook Selection ===\n")

    extraction_recon_playbook = Playbook(
        name="Model Extraction Recon Triage",
        description="Procedure for investigating suspected model-extraction probing",
        steps=[
            PlaybookStep(instruction="Profile request volume and prompt diversity per client"),
            PlaybookStep(
                instruction="Check whether timing-probe patterns target a single endpoint"
            ),
            PlaybookStep(instruction="Correlate the source against the partner-mesh allowlist"),
        ],
    )

    inventory_drift_playbook = Playbook(
        name="Inventory Drift Investigation",
        description="Procedure for investigating an endpoint serving an unexpected model",
        steps=[
            PlaybookStep(instruction="Re-fingerprint the endpoint to confirm the served model"),
            PlaybookStep(instruction="Diff the verdict against the approved inventory record"),
            PlaybookStep(instruction="Escalate a confirmed mismatch to the change-control owner"),
        ],
    )

    specialist.playbooks.extend([extraction_recon_playbook, inventory_drift_playbook])

    # select_playbook matches the task description against each playbook's
    # name and description; it returns None if nothing fits.
    tasks = [
        "Fingerprint the inference endpoint at 203.0.113.10 for case IR-2026-017",
        "Investigate suspected model-extraction recon against the public API",
        "Investigate an endpoint that appears to be serving an unexpected model",
    ]

    for task in tasks:
        selected = specialist.select_playbook(task)
        if selected:
            print(f"Task: '{task[:40]}...'")
            print(f"  Selected playbook: {selected.name}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is automatic playbook selection by task description risky in a SOC and how do you mitigate it?')}"
    )

    # =========================================================================
    # Part 4b: ground the fingerprint — emit a finding only with evidence
    # =========================================================================
    # CURATOR's verdict is not a finding until it clears the GSAR grounding
    # threshold. ``ground_fingerprint`` (tulip.security) admits a
    # FingerprintFinding only when the evidence partition proceeds; a weak
    # partition abstains. This is the inference-fingerprinting capability
    # the mythos calls "the Probe" — MITRE ATLAS AML.T0040 / AML.T0024.
    print("\n=== Part 4b: Grounding the Fingerprint ===\n")

    # Full-coverage probe: 4/4 timing features observed. The classifier
    # returns a high-confidence verdict; the timing feature vector is the
    # finding's evidence, partitioned by GSAR evidence type.
    full_features = {"ttft_ms_p50": 38.2, "itl_ms_mean": 11.4, "itl_cv": 0.07, "tps_mean": 87.6}
    verdict = classify_fingerprint(full_features)
    print(
        f"VerificationResult (coverage {verdict.feature_coverage:.0%}): "
        f"{verdict.model} on {verdict.engine} / {verdict.hardware} "
        f"@ confidence {verdict.confidence:.0%}"
    )

    grounded_partition = Partition(
        grounded=[
            Claim(
                text="TTFT p50 38.2ms matches the vLLM continuous-batching profile",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["probe:ttft_ms_p50=38.2"],
            ),
            Claim(
                text="inter-token latency 11.4ms/token is consistent with the 7B class",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["probe:itl_ms_mean=11.4"],
            ),
            Claim(
                text="cadence variance 0.07 sits in the expected datacenter-GPU timing band",
                type=EvidenceType.SIGNAL_MATCH,
                evidence_refs=["probe:itl_cv=0.07"],
            ),
        ],
    )
    fp_result = ground_fingerprint(
        verdict=verdict,
        asset="203.0.113.10:443",
        partition=grounded_partition,
        severity=Severity.MEDIUM,
        indicators=[Indicator(type=IndicatorType.ENDPOINT, value="203.0.113.10:443")],
        taxonomy=[
            AtlasTechnique.INFERENCE_API_ACCESS,
            AtlasTechnique.EXFILTRATION_VIA_INFERENCE_API,
        ],
    )
    if is_finding(fp_result):
        print(f"  SHIPPED finding: {fp_result.title}")
        print(f"    gsar_score={fp_result.gsar_score:.2f} severity={fp_result.severity.value}")
        print(f"    taxonomy={[t.value for t in fp_result.taxonomy]}")
    else:
        print(f"  abstained: {fp_result.reason}")

    # Low-coverage probe: 1/4 features. The classifier returns "unknown"
    # at low confidence; the partition is a lone inference claim, so GSAR
    # abstains rather than asserting a fingerprint.
    sparse_features = {"ttft_ms_p50": 41.0}
    weak_verdict = classify_fingerprint(sparse_features)
    print(
        f"\nUnder-observed endpoint (coverage {weak_verdict.feature_coverage:.0%}): "
        f"classifier confidence {weak_verdict.confidence:.0%}"
    )
    weak_partition = Partition(
        ungrounded=[
            Claim(
                text="a single TTFT sample loosely resembles a batched server",
                type=EvidenceType.INFERENCE,
                evidence_refs=["probe:ttft_ms_p50=41.0"],
            ),
        ],
    )
    weak_result = ground_fingerprint(
        verdict=weak_verdict,
        asset="203.0.113.20:443",
        partition=weak_partition,
    )
    if is_finding(weak_result):
        print(f"  SHIPPED finding: {weak_result.title}")
    else:
        print(f"  abstained ({weak_result.decision.value}): {weak_result.reason}")

    # =========================================================================
    # Part 5: drive the specialist end-to-end
    # =========================================================================
    print("\n=== Part 5: Executing Specialists ===\n")

    result = await specialist.execute(
        task="Fingerprint the inference endpoint at 203.0.113.10:443 and reconcile the verdict "
        "against the approved inventory.",
        context={
            "case_id": "IR-2026-017",
            "endpoint": "203.0.113.10:443",
            "submitted_by": "inventory-verification sweep",
        },
    )

    print("Execution Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"  Output: {result.output[:300]}...")
    if result.error:
        print(f"  Error: {result.error}")

    # =========================================================================
    # Part 6: pre-built specialists for common evidence domains
    # =========================================================================
    print("\n=== Part 6: Pre-built Specialists ===\n")

    log_analyst = create_log_analyst(model=model)
    metrics_analyst = create_metrics_analyst(model=model)
    trace_analyst = create_trace_analyst(model=model)
    code_analyst = create_code_analyst(model=model)

    specialists = [log_analyst, metrics_analyst, trace_analyst, code_analyst]

    print("Pre-built Specialists:")
    for spec in specialists:
        print(f"\n  {spec.name}")
        print(f"    Type: {spec.specialist_type}")
        prompt_preview = spec.system_prompt.split("\n")[0]
        print(f"    Focus: {prompt_preview[:60]}...")
    t0 = time.perf_counter()
    p6 = await metrics_analyst.execute(
        task="In one sentence, what does a metrics analyst contribute to incident response?"
    )
    dt = time.perf_counter() - t0
    print(f"\n  [model call: {dt:.2f}s · metrics_analyst.execute()]")
    if p6.output:
        print(f"  Output: {p6.output[:160]}")

    # =========================================================================
    # Part 7: extend a pre-built specialist with your own tools
    # =========================================================================
    print("\n=== Part 7: Custom Tools Integration ===\n")

    @tool(name="search_auth_logs", description="Search authentication logs for patterns")
    async def search_auth_logs(pattern: str, timerange: str = "1h") -> str:
        return f"Found 42 auth-log matches for '{pattern}' in last {timerange}"

    @tool(name="get_alert_history", description="Get recent security alerts for a host")
    async def get_alert_history(limit: int = 10) -> str:
        return f"Retrieved {limit} most recent security alerts"

    custom_log_analyst = create_log_analyst(
        model=model,
        tools=[search_auth_logs, get_alert_history],
    )

    print(f"Custom log analyst tools: {[t.name for t in custom_log_analyst.tools]}")

    log_result = await custom_log_analyst.execute(
        task="Search for failed admin logins from 192.0.2.55 in the last hour",
    )

    print("Log analysis result:")
    print(f"  Confidence: {log_result.confidence:.0%}")
    if log_result.output:
        print(f"  Output: {log_result.output[:200]}...")

    # =========================================================================
    # Part 8: how confidence is estimated
    # =========================================================================
    print("\n=== Part 8: Confidence Estimation ===\n")

    # The built-in estimator scans the response for hedging vs.
    # certainty markers — a rough proxy that you'd typically replace
    # with a domain-specific scorer in production.
    responses = [
        ("definitely vLLM — timing profile is unambiguous", "High confidence markers"),
        ("might be a batched server, hard to say from one sample", "Low confidence markers"),
        ("confirmed by the full timing feature vector", "Verification markers"),
        ("unclear which engine produced this cadence", "Uncertainty markers"),
    ]

    print("Confidence markers in responses:")
    for response, description in responses:
        confidence = specialist._estimate_confidence(response)
        print(f"  '{response}' -> {confidence:.0%} ({description})")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is keyword-based confidence estimation only a rough proxy for a security verdict?')}"
    )

    # =========================================================================
    # Part 9: common specialist shapes
    # =========================================================================
    print("\n=== Part 9: Specialist Patterns ===\n")

    print("Pattern 1: Domain Expert")
    print("  Focused prompt + domain tools + high confidence threshold.")
    print()

    print("Pattern 2: Procedure Follower")
    print("  Runbook-driven; each step validated against expected output.")
    print()

    print("Pattern 3: Adaptive Analyst")
    print("  Multiple playbooks; the right one selected per alert.")
    print()

    print("Pattern 4: Pipeline Stage")
    print("  Drops into a larger IR workflow; structured output, context in/out.")
    print(
        f"AI suggestion: {_llm_call('Suggest one extra security Specialist pattern not in the four listed above. One short sentence.')}"
    )

    # =========================================================================
    # Part 10: assemble an incident-response team
    # =========================================================================
    print("\n=== Part 10: Specialist Teams ===\n")

    def create_incident_response_team(model):
        """One triage specialist plus three pre-built analysts."""
        return {
            "triage": Specialist(
                name="Triage Specialist",
                specialist_type="triage",
                description="Initial alert assessment and severity classification",
                system_prompt="Assess security alerts and determine severity and routing.",
                model=model,
            ),
            "logs": create_log_analyst(model=model),
            "metrics": create_metrics_analyst(model=model),
            "code": create_code_analyst(model=model),
        }

    team = create_incident_response_team(model)
    print("Incident Response Team:")
    for role, spec in team.items():
        print(f"  {role}: {spec.name}")
    t0 = time.perf_counter()
    p10 = await team["triage"].execute(
        task="In one sentence, classify this alert: 'EDR flagged encoded PowerShell spawned by winword.exe on WS-0142.'",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · triage.execute()]")
    if p10.output:
        print(f"  Triage verdict: {p10.output[:160]}")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Next: Notebook 28 — A2A Protocol")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
