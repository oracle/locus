# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 27: RIGHTSIZER — the cloud-rightsizing specialist.

Notebook 26 introduced the Specialist as the worker MARSHAL hands tasks
to. This notebook dives into the Specialist itself, using RIGHTSIZER —
the platform team's cloud cost-and-capacity specialist — as the running
case: how to narrow a model's failure surface with a focused system
prompt, a hand-picked tool set, optional playbooks (your runbooks,
encoded), and a confidence threshold.

RIGHTSIZER's job is workload fingerprinting for rightsizing: identifying
which instance family and size a compute workload should run on purely
from observable utilization telemetry (CPU p50, memory mean, IOPS,
network throughput) — no agent on the box, no guesswork. The payoff is
cutting waste on over-provisioned instances and catching capacity risk
on hot ones before it pages someone.

The capability is demonstrated with the domain-neutral GSAR grounding
core (``tulip.reasoning.gsar``): a deterministic classifier maps a
utilization feature vector to a ``WorkloadVerdict``, and the verdict
becomes a recommendation **only** when the evidence partition clears the
grounding threshold. Low feature coverage yields a weak evidence
partition, so an under-observed instance abstains rather than asserting a
rightsizing — a wrong instance-type call is worse than none.

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
from dataclasses import dataclass

from config import get_model, print_config

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
from tulip.reasoning.gsar import (
    Claim,
    Decision,
    EvidenceType,
    Partition,
    decide,
    gsar_score,
)
from tulip.tools.decorator import tool


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """One model call with a timing/token banner — used for commentary."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# The expected utilization-feature schema for a rightsizing verdict.
# Coverage is the fraction of these the telemetry pull actually returned;
# below this floor the classifier returns low confidence and grounding abstains.
_WORKLOAD_FEATURES = ("cpu_p50", "mem_mean", "iops_mean", "net_mbps")
_COVERAGE_FLOOR = 0.60


@dataclass
class WorkloadVerdict:
    """A rightsizing call: which family/size a workload fits, and how sure."""

    family: str
    size: str
    action: str
    confidence: float
    feature_coverage: float


def _sample_utilization(instance: str) -> dict[str, float]:
    """Deterministic offline utilization sample for an instance.

    A real deployment swaps in a CloudWatch / Stackdriver query behind
    this same signature; offline we return a fixed feature vector so the
    notebook runs end-to-end with no cloud credentials.
    """
    return {"cpu_p50": 18.4, "mem_mean": 22.1, "iops_mean": 140.0, "net_mbps": 31.5}


def classify_workload(features: Mapping[str, float]) -> WorkloadVerdict:
    """Deterministic mock rightsizing classifier.

    Maps a utilization feature vector to a ``(family, size, action)``
    verdict over a fixed lookup — no model file, no scikit, no network.
    A real deployment swaps in a rightsizing service behind this same
    signature. Coverage below :data:`_COVERAGE_FLOOR` returns a
    low-confidence "unknown" verdict so the caller abstains.

    Args:
        features: Observed utilization features keyed by name.

    Returns:
        A :class:`WorkloadVerdict`.
    """
    observed = [f for f in _WORKLOAD_FEATURES if f in features]
    coverage = len(observed) / len(_WORKLOAD_FEATURES)
    if coverage < _COVERAGE_FLOOR:
        return WorkloadVerdict(
            family="unknown",
            size="unknown",
            action="unknown",
            confidence=0.30,
            feature_coverage=coverage,
        )
    # Fixed lookup: sustained low CPU p50 with memory headroom reads as an
    # over-provisioned workload that fits a smaller general-purpose size.
    cpu = features.get("cpu_p50", 100.0)
    if cpu <= 25.0:
        return WorkloadVerdict(
            family="general-purpose",
            size="m6i.large",
            action="downsize",
            confidence=0.91,
            feature_coverage=coverage,
        )
    return WorkloadVerdict(
        family="compute-optimized",
        size="c6i.2xlarge",
        action="upsize",
        confidence=0.78,
        feature_coverage=coverage,
    )


async def main():
    print("=" * 60)
    print("Notebook 27: RIGHTSIZER — the cloud-rightsizing specialist")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: anatomy of a Specialist
    # =========================================================================
    print("\n=== Part 1: Specialist Anatomy ===\n")

    # A specialist pairs a focused system prompt with domain tools,
    # optional playbooks, and a confidence threshold. RIGHTSIZER is the
    # platform team's cost-and-capacity specialist.
    specialist = Specialist(
        name="RIGHTSIZER",
        specialist_type="cloud_rightsizing",
        description="Recommends instance rightsizing from utilization telemetry",
        system_prompt="""You are RIGHTSIZER, a cloud cost-and-capacity specialist. Your expertise:
1. Pulling utilization telemetry (CPU p50, memory mean, IOPS, network throughput)
2. Mapping a utilization feature vector to a (family, size, action) verdict
3. Reconciling a recommendation against the approved instance catalog
4. Flagging over-provisioned waste and capacity risk on hot fleets

When rightsizing:
- Treat a single data point as a hint, not evidence — collect a stable feature vector
- Report feature coverage; abstain when too little of the schema is observed
- Never assert a rightsizing you cannot ground — a wrong instance-type call is worse than none""",
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

    @tool(name="pull_utilization", description="Pull utilization telemetry for an instance")
    async def pull_utilization(instance: str, source: str = "cloudwatch") -> str:
        # Query the metrics backend (CloudWatch / Stackdriver) where the
        # measurement actually lives; falls back to a deterministic sample
        # offline so the notebook runs with no cloud credentials.
        feats = _sample_utilization(instance)
        observed = sum(1 for f in _WORKLOAD_FEATURES if f in feats)
        body = " ".join(f"{k}={v}" for k, v in feats.items())
        return (
            f"Telemetry {instance} via {source}: {body} "
            f"({observed}/{len(_WORKLOAD_FEATURES)} features observed)"
        )

    @tool(
        name="classify_instance", description="Map the utilization vector to a rightsizing verdict"
    )
    async def classify_instance() -> str:
        return "WorkloadVerdict: downsize to m6i.large (general-purpose), confidence 0.91, coverage 1.00"

    @tool(
        name="check_catalog", description="Compare a verdict against the approved instance catalog"
    )
    async def check_catalog() -> str:
        return "Catalog: instance i-0abc123def currently on c6i.2xlarge; m6i.large is an approved target"

    specialist = Specialist(
        name="RIGHTSIZER",
        specialist_type="cloud_rightsizing",
        description="Recommends instance rightsizing from utilization telemetry",
        system_prompt="You recommend instance rightsizing from utilization telemetry.",
        tools=[pull_utilization, classify_instance, check_catalog],
        model=model,
    )

    print(f"Tools available: {[t.name for t in specialist.tools]}")
    _ai_note = await _llm_call("In one sentence, why does giving a Specialist domain-specific cloud tools dramatically narrow its failure surface?")
    print(f"AI commentary: {_ai_note}")

    # =========================================================================
    # Part 3: encode a runbook as a Playbook
    # =========================================================================
    print("\n=== Part 3: Specialist Playbooks ===\n")

    rightsizing_playbook = Playbook(
        name="Rightsizing Procedure",
        description="Standard procedure for rightsizing a compute instance",
        preconditions=[
            "Instance is in scope for the cost-optimization review",
            "Telemetry covers a representative window, not a quiet weekend",
        ],
        steps=[
            PlaybookStep(
                instruction="Pull the utilization feature vector",
                required_tools=["pull_utilization"],
                expected_output="CPU p50, memory mean, IOPS, network throughput, and coverage",
            ),
            PlaybookStep(
                instruction="Classify the feature vector to a family/size/action verdict",
                required_tools=["classify_instance"],
                expected_output="WorkloadVerdict with confidence and feature coverage",
                on_failure="Abstain if feature coverage is below the schema floor",
            ),
            PlaybookStep(
                instruction="Compare the verdict against the approved instance catalog",
                required_tools=["check_catalog"],
                expected_output="Match / mismatch against what the catalog allows",
            ),
        ],
        success_criteria="Grounded verdict reconciled against the catalog, or a logged abstention",
    )

    specialist.playbooks.append(rightsizing_playbook)

    print(f"Playbook: {rightsizing_playbook.name}")
    print(f"  Preconditions: {rightsizing_playbook.preconditions}")
    print(f"  Steps: {len(rightsizing_playbook.steps)}")
    print(f"  Success criteria: {rightsizing_playbook.success_criteria}")

    # ``to_prompt()`` renders the playbook as the text block injected
    # into the system prompt when the playbook is selected.
    playbook_prompt = rightsizing_playbook.to_prompt()
    print("\nPlaybook prompt:")
    print("-" * 40)
    print(playbook_prompt[:500] + "...")
    _ai_note = await _llm_call("In one sentence, when does attaching a fixed Playbook to a cloud Specialist matter most?")
    print(f"AI commentary: {_ai_note}")

    # =========================================================================
    # Part 4: pick the right playbook from a pool
    # =========================================================================
    print("\n=== Part 4: Playbook Selection ===\n")

    cost_anomaly_playbook = Playbook(
        name="Cost Anomaly Triage",
        description="Procedure for investigating a sudden cloud-spend spike",
        steps=[
            PlaybookStep(instruction="Break the spend spike down by service and account"),
            PlaybookStep(instruction="Check whether new resources were launched in the window"),
            PlaybookStep(instruction="Correlate the spike against the deploy and scaling history"),
        ],
    )

    config_drift_playbook = Playbook(
        name="Configuration Drift Investigation",
        description="Procedure for investigating an instance running an unexpected type",
        steps=[
            PlaybookStep(instruction="Re-pull telemetry to confirm the running instance type"),
            PlaybookStep(instruction="Diff the running type against the approved catalog record"),
            PlaybookStep(instruction="Escalate a confirmed mismatch to the change-control owner"),
        ],
    )

    specialist.playbooks.extend([cost_anomaly_playbook, config_drift_playbook])

    # select_playbook matches the task description against each playbook's
    # name and description; it returns None if nothing fits.
    tasks = [
        "Recommend rightsizing for instance i-0abc123def for ticket FIN-2026-017",
        "Investigate a sudden cloud-spend spike on the public-api fleet",
        "Investigate an instance that appears to be running an unexpected type",
    ]

    for task in tasks:
        selected = specialist.select_playbook(task)
        if selected:
            print(f"Task: '{task[:40]}...'")
            print(f"  Selected playbook: {selected.name}")
    _ai_note = await _llm_call("In one sentence, why is automatic playbook selection by task description risky for cloud ops and how do you mitigate it?")
    print(f"AI commentary: {_ai_note}")

    # =========================================================================
    # Part 4b: ground the verdict — emit a recommendation only with evidence
    # =========================================================================
    # RIGHTSIZER's verdict is not a recommendation until it clears the GSAR
    # grounding threshold. We partition the evidence, score it with
    # ``gsar_score``, and ``decide`` whether to ship or abstain; a weak
    # partition abstains. This is the domain-neutral grounding core that
    # keeps a confident-sounding model from acting on thin evidence.
    print("\n=== Part 4b: Grounding the Verdict ===\n")

    # Full-coverage telemetry: 4/4 utilization features observed. The
    # classifier returns a high-confidence verdict; the utilization feature
    # vector is the recommendation's evidence, partitioned by GSAR type.
    full_features = {"cpu_p50": 18.4, "mem_mean": 22.1, "iops_mean": 140.0, "net_mbps": 31.5}
    verdict = classify_workload(full_features)
    print(
        f"WorkloadVerdict (coverage {verdict.feature_coverage:.0%}): "
        f"{verdict.action} to {verdict.size} ({verdict.family}) "
        f"@ confidence {verdict.confidence:.0%}"
    )

    grounded_partition = Partition(
        grounded=[
            Claim(
                text="CPU p50 18.4% sits well below the 40% rightsizing band",
                type=EvidenceType.TOOL_MATCH,
                evidence_refs=["telemetry:cpu_p50=18.4"],
            ),
            Claim(
                text="memory mean 22.1% confirms ample headroom on the current size",
                type=EvidenceType.SPECIFIC_DATA,
                evidence_refs=["telemetry:mem_mean=22.1"],
            ),
            Claim(
                text="IOPS mean 140 fits comfortably inside the m6i.large envelope",
                type=EvidenceType.SIGNAL_MATCH,
                evidence_refs=["telemetry:iops_mean=140.0"],
            ),
        ],
    )
    score = gsar_score(grounded_partition)
    decision = decide(score)
    if decision == Decision.PROCEED:
        print(f"  SHIPPED recommendation: {verdict.action} i-0abc123def to {verdict.size}")
        print(f"    gsar_score={score:.2f} decision={decision.value}")
    else:
        print(
            f"  abstained ({decision.value}): grounding score {score:.2f} below proceed threshold"
        )

    # Low-coverage telemetry: 1/4 features. The classifier returns "unknown"
    # at low confidence; the partition is a lone inference claim, so GSAR
    # decides replan rather than asserting a rightsizing.
    sparse_features = {"cpu_p50": 41.0}
    weak_verdict = classify_workload(sparse_features)
    print(
        f"\nUnder-observed instance (coverage {weak_verdict.feature_coverage:.0%}): "
        f"classifier confidence {weak_verdict.confidence:.0%}"
    )
    weak_partition = Partition(
        ungrounded=[
            Claim(
                text="a single CPU sample loosely resembles a busy workload",
                type=EvidenceType.INFERENCE,
                evidence_refs=["telemetry:cpu_p50=41.0"],
            ),
        ],
    )
    weak_score = gsar_score(weak_partition)
    weak_decision = decide(weak_score)
    if weak_decision == Decision.PROCEED:
        print(f"  SHIPPED recommendation: {weak_verdict.action} to {weak_verdict.size}")
    else:
        print(f"  abstained ({weak_decision.value}): grounding score {weak_score:.2f}")

    # =========================================================================
    # Part 5: drive the specialist end-to-end
    # =========================================================================
    print("\n=== Part 5: Executing Specialists ===\n")

    result = await specialist.execute(
        task="Recommend rightsizing for instance i-0abc123def and reconcile the verdict "
        "against the approved instance catalog.",
        context={
            "ticket_id": "FIN-2026-017",
            "instance": "i-0abc123def",
            "submitted_by": "quarterly cost-optimization review",
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
        task="In one sentence, what does a metrics analyst contribute to capacity planning?"
    )
    dt = time.perf_counter() - t0
    print(f"\n  [model call: {dt:.2f}s · metrics_analyst.execute()]")
    if p6.output:
        print(f"  Output: {p6.output[:160]}")

    # =========================================================================
    # Part 7: extend a pre-built specialist with your own tools
    # =========================================================================
    print("\n=== Part 7: Custom Tools Integration ===\n")

    @tool(name="search_app_logs", description="Search application logs for patterns")
    async def search_app_logs(pattern: str, timerange: str = "1h") -> str:
        return f"Found 42 app-log matches for '{pattern}' in last {timerange}"

    @tool(name="get_scaling_history", description="Get recent autoscaling events for a service")
    async def get_scaling_history(limit: int = 10) -> str:
        return f"Retrieved {limit} most recent autoscaling events"

    custom_log_analyst = create_log_analyst(
        model=model,
        tools=[search_app_logs, get_scaling_history],
    )

    print(f"Custom log analyst tools: {[t.name for t in custom_log_analyst.tools]}")

    log_result = await custom_log_analyst.execute(
        task="Search for OOMKilled events on the checkout service in the last hour",
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
        (
            "definitely over-provisioned — utilization profile is unambiguous",
            "High confidence markers",
        ),
        ("might fit a smaller size, hard to say from one data point", "Low confidence markers"),
        ("confirmed by the full utilization feature vector", "Verification markers"),
        ("unclear which family this workload belongs on", "Uncertainty markers"),
    ]

    print("Confidence markers in responses:")
    for response, description in responses:
        confidence = specialist._estimate_confidence(response)
        print(f"  '{response}' -> {confidence:.0%} ({description})")
    _ai_note = await _llm_call("In one sentence, why is keyword-based confidence estimation only a rough proxy for a rightsizing decision?")
    print(f"AI commentary: {_ai_note}")

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
    print("  Multiple playbooks; the right one selected per request.")
    print()

    print("Pattern 4: Pipeline Stage")
    print("  Drops into a larger FinOps workflow; structured output, context in/out.")
    _ai_note = await _llm_call("Suggest one extra cloud Specialist pattern not in the four listed above. One short sentence.")
    print(f"AI suggestion: {_ai_note}")

    # =========================================================================
    # Part 10: assemble a capacity-review team
    # =========================================================================
    print("\n=== Part 10: Specialist Teams ===\n")

    def create_capacity_review_team(model):
        """One intake specialist plus three pre-built analysts."""
        return {
            "intake": Specialist(
                name="Intake Specialist",
                specialist_type="intake",
                description="Initial request assessment and priority classification",
                system_prompt="Assess rightsizing requests and determine priority and routing.",
                model=model,
            ),
            "logs": create_log_analyst(model=model),
            "metrics": create_metrics_analyst(model=model),
            "code": create_code_analyst(model=model),
        }

    team = create_capacity_review_team(model)
    print("Capacity Review Team:")
    for role, spec in team.items():
        print(f"  {role}: {spec.name}")
    t0 = time.perf_counter()
    p10 = await team["intake"].execute(
        task="In one sentence, classify this request: 'p95 CPU on the checkout fleet held above 85% for 6 hours.'",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · intake.execute()]")
    if p10.output:
        print(f"  Intake verdict: {p10.output[:160]}")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Next: Notebook 28 — A2A Protocol")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
