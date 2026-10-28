# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 23: RIGHTSIZER — the cloud-rightsizing specialist.

Notebook 22 introduced the Specialist as the worker STEWARD hands tasks
to. This notebook dives into the Specialist itself, using RIGHTSIZER —
the platform team's cloud cost-and-capacity specialist — as the running
case: how to narrow a model's failure surface with a focused system
prompt, a hand-picked tool set, optional playbooks (your runbooks,
encoded), and a confidence threshold.

RIGHTSIZER's job is workload fingerprinting for rightsizing: identifying
which compute shape a workload should run on purely from observable
utilization telemetry (CPU p50, memory mean, IOPS, network throughput) —
no agent on the box, no guesswork. The payoff is cutting waste on
over-provisioned instances and catching capacity risk on hot ones before
it pages someone. Notebooks 21 and 55 show how a specialist's verdict is
gated by GSAR grounding before it ships; here the subject is the
Specialist API itself.

Key ideas:
- A ``Specialist`` is a Locus ``Agent`` with role metadata
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
    .venv/bin/python examples/notebook_23_specialist_agents.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose. Set
``LOCUS_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 02 (Agent basics).
- Notebook 22 (Orchestrator) — Specialists are the workers STEWARD routes to.
"""

import asyncio
import time
from typing import Any

from config import get_model, print_config

from locus.multiagent.specialist import (
    Playbook,
    PlaybookStep,
    Specialist,
    create_code_analyst,
    create_log_analyst,
    create_metrics_analyst,
    create_trace_analyst,
)
from locus.tools.decorator import tool


# The utilization-feature schema a rightsizing verdict needs. Coverage is
# the fraction of these the telemetry pull returned; the tools report it so
# the specialist can abstain on an under-observed instance.
_WORKLOAD_FEATURES = ("cpu_p50", "mem_mean", "iops_mean", "net_mbps")


# =============================================================================
# RIGHTSIZER's domain tools
# =============================================================================
# Three hand-picked tools, each a thin wrapper over something the platform
# team already runs. Offline they return fixed strings so the notebook runs
# with no cloud credentials; a real deployment swaps in the OCI Monitoring
# query, the rightsizing service, and the shape catalog behind the same
# signatures.


@tool(name="pull_utilization", description="Pull utilization telemetry for an instance")
async def pull_utilization(instance: str, source: str = "oci-monitoring") -> str:
    feats = {"cpu_p50": 18.4, "mem_mean": 22.1, "iops_mean": 140.0, "net_mbps": 31.5}
    body = " ".join(f"{k}={v}" for k, v in feats.items())
    return (
        f"Telemetry {instance} via {source}: {body} "
        f"({len(feats)}/{len(_WORKLOAD_FEATURES)} features observed)"
    )


@tool(name="classify_instance", description="Map the utilization vector to a rightsizing verdict")
async def classify_instance() -> str:
    return (
        "WorkloadVerdict: downsize to VM.Standard.E4.Flex (2 OCPU, 16 GB), "
        "confidence 0.91, coverage 1.00"
    )


@tool(name="check_catalog", description="Compare a verdict against the approved shape catalog")
async def check_catalog() -> str:
    return (
        "Catalog: instance web-07 currently on VM.Standard.E4.Flex (8 OCPU, 64 GB); "
        "VM.Standard.E4.Flex (2 OCPU, 16 GB) is an approved target"
    )


# =============================================================================
# Part 1: anatomy of a Specialist
# =============================================================================


async def example_anatomy(model: Any) -> Specialist:
    """Build RIGHTSIZER once — prompt, tools, iteration cap, confidence threshold."""
    print("\n=== Part 1: Specialist Anatomy ===\n")

    # A specialist pairs a focused system prompt with domain tools,
    # optional playbooks, and a confidence threshold. RIGHTSIZER is built
    # here and carried through every part that follows.
    specialist = Specialist(
        name="RIGHTSIZER",
        specialist_type="cloud_rightsizing",
        description="Recommends instance rightsizing from utilization telemetry",
        system_prompt="""You are RIGHTSIZER, a cloud cost-and-capacity specialist. Your expertise:
1. Pulling utilization telemetry (CPU p50, memory mean, IOPS, network throughput)
2. Mapping a utilization feature vector to a (shape, action) verdict
3. Reconciling a recommendation against the approved shape catalog
4. Flagging over-provisioned waste and capacity risk on hot fleets

When rightsizing:
- Treat a single data point as a hint, not evidence — collect a stable feature vector
- Report feature coverage; abstain when too little of the schema is observed
- Never assert a rightsizing you cannot ground — a wrong shape call is worse than none""",
        tools=[pull_utilization, classify_instance, check_catalog],
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
    return specialist


# =============================================================================
# Part 2: hand-pick tools for the domain
# =============================================================================


def example_domain_tools(specialist: Specialist) -> None:
    """The tool list is the failure surface — RIGHTSIZER can do three things and nothing else."""
    print("\n=== Part 2: Domain Tools ===\n")

    # Each tool's name and description are what the model sees when
    # deciding whether to call it, so they are written for the model.
    print("Tools available:")
    for t in specialist.tools:
        print(f"  - {t.name}: {t.description}")


# =============================================================================
# Part 3: encode a runbook as a Playbook
# =============================================================================


def example_playbooks(specialist: Specialist) -> None:
    """A Playbook is the team's rightsizing runbook, typed."""
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
                instruction="Classify the feature vector to a shape/action verdict",
                required_tools=["classify_instance"],
                expected_output="WorkloadVerdict with confidence and feature coverage",
                on_failure="Abstain if feature coverage is below the schema floor",
            ),
            PlaybookStep(
                instruction="Compare the verdict against the approved shape catalog",
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


# =============================================================================
# Part 4: pick the right playbook from a pool
# =============================================================================


def example_playbook_selection(specialist: Specialist) -> None:
    """select_playbook(task) matches the task against each playbook's name and description."""
    print("\n=== Part 4: Playbook Selection ===\n")

    cost_anomaly_playbook = Playbook(
        name="Cost Anomaly Triage",
        description="Procedure for investigating a sudden cloud-spend spike",
        steps=[
            PlaybookStep(instruction="Break the spend spike down by service and compartment"),
            PlaybookStep(instruction="Check whether new resources were launched in the window"),
            PlaybookStep(instruction="Correlate the spike against the deploy and scaling history"),
        ],
    )

    config_drift_playbook = Playbook(
        name="Configuration Drift Investigation",
        description="Procedure for investigating an instance running an unexpected shape",
        steps=[
            PlaybookStep(instruction="Re-pull telemetry to confirm the running shape"),
            PlaybookStep(instruction="Diff the running shape against the approved catalog record"),
            PlaybookStep(instruction="Escalate a confirmed mismatch to the change-control owner"),
        ],
    )

    specialist.playbooks.extend([cost_anomaly_playbook, config_drift_playbook])

    # select_playbook returns None if nothing fits.
    tasks = [
        "Recommend rightsizing for instance web-07 for ticket FIN-2026-017",
        "Investigate a sudden cloud-spend spike on the public-api fleet",
        "Investigate an instance that appears to be running an unexpected shape",
    ]

    for task in tasks:
        selected = specialist.select_playbook(task)
        if selected:
            print(f"Task: '{task[:40]}...'")
            print(f"  Selected playbook: {selected.name}")


# =============================================================================
# Part 5: drive the specialist end-to-end
# =============================================================================


async def example_execute(specialist: Specialist) -> None:
    """One real task through RIGHTSIZER, with the confidence read against its threshold."""
    print("\n=== Part 5: Executing Specialists ===\n")

    result = await specialist.execute(
        task="Recommend rightsizing for instance web-07 and reconcile the verdict "
        "against the approved shape catalog.",
        context={
            "ticket_id": "FIN-2026-017",
            "instance": "web-07",
            "submitted_by": "quarterly cost-optimization review",
        },
    )

    # ``confidence`` comes from the built-in estimator, which scans the
    # reply for hedging vs certainty markers — a rough proxy you would
    # replace with a domain scorer in production. The specialist's
    # ``confidence_threshold`` is what that number is compared against.
    print("Execution Result:")
    print(f"  Success: {result.success}")
    print(
        f"  Confidence: {result.confidence:.0%} (threshold {specialist.confidence_threshold:.0%})"
    )
    print(f"  Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"  Output: {result.output[:300]}...")
    if result.error:
        print(f"  Error: {result.error}")


# =============================================================================
# Part 6: pre-built specialists for common evidence domains
# =============================================================================


async def example_prebuilt(model: Any) -> None:
    """The four shipped specialists, plus one call to show they run like any other."""
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


# =============================================================================
# Part 7: extend a pre-built specialist with your own tools
# =============================================================================


async def example_custom_tools(model: Any) -> None:
    """create_log_analyst(tools=...) keeps the shipped prompt and adds your tools."""
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


# =============================================================================
# Part 8: assemble a capacity-review team
# =============================================================================


async def example_team(model: Any) -> None:
    """One intake specialist plus three pre-built analysts, keyed by role."""
    print("\n=== Part 8: Specialist Teams ===\n")

    team = {
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

    print("Capacity Review Team:")
    for role, spec in team.items():
        print(f"  {role}: {spec.name}")
    t0 = time.perf_counter()
    p8 = await team["intake"].execute(
        task="In one sentence, classify this request: 'p95 CPU on the checkout fleet held above 85% for 6 hours.'",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · intake.execute()]")
    if p8.output:
        print(f"  Intake verdict: {p8.output[:160]}")


async def main():
    print("=" * 60)
    print("Notebook 23: RIGHTSIZER — the cloud-rightsizing specialist")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    specialist = await example_anatomy(model)
    example_domain_tools(specialist)
    example_playbooks(specialist)
    example_playbook_selection(specialist)
    await example_execute(specialist)
    await example_prebuilt(model)
    await example_custom_tools(model)
    await example_team(model)

    print("\n" + "=" * 60)
    print(
        "Next: Notebook 24 — Outage war room — mitigation, diagnostics, and comms agents in a swarm"
    )
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
