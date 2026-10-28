# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 49: playbooks — a GDPR data-subject-access runbook the agent must follow.

A playbook is a typed, ordered sequence of steps with declared
``expected_tools``. Wire it into an agent and the agent is constrained
to walk the steps in order, calling only the tools each step allows.
That is exactly what privacy operations demand: a GDPR-style data
subject access request (DSAR) procedure where no step is skipped,
every tool call is attributable to a step, and the whole run is
auditable after the fact.

- ``PlaybookStep`` — id, description, expected tools, hints,
  validation rules.
- ``Playbook`` — a collection of steps with ordering, max-iteration,
  and tagging.
- ``PlaybookPlan`` and ``StepExecution`` — runtime tracking, progress,
  status (``PENDING`` / ``IN_PROGRESS`` / ``COMPLETED`` / ``FAILED`` /
  ``SKIPPED``).
- ``Agent(playbook=...)`` — bind a playbook to an agent and watch it
  execute against real tools.

Parts 1–8 are the data model and are pure Python (Part 3 asks the model
for one next-step suggestion from the plan's state); Part 9 drives a
live agent through a playbook. Design notes for production playbooks
are in ``docs/concepts/playbooks.md``.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_49_playbooks.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_49_playbooks.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

import asyncio
import time
from datetime import UTC, datetime

from config import get_model

from locus.agent import Agent
from locus.playbooks import (
    Playbook,
    PlaybookPlan,
    PlaybookStep,
    StepExecution,
    StepStatus,
)
from locus.tools import tool


def _banner(result) -> None:
    m = result.metrics
    print(
        f"  [model call: {m.duration_ms / 1000.0:.2f}s · "
        f"{m.prompt_tokens}→{m.completion_tokens} tokens · iters={m.iterations}]"
    )


# =============================================================================
# Part 1: PlaybookStep — each step declares the tools it expects.
# =============================================================================


def part1_steps() -> list[PlaybookStep]:
    print("\n=== Part 1: Creating Playbook Steps ===\n")
    steps = [
        PlaybookStep(
            id="verify_identity",
            description="Verify the requester's identity before processing the access request",
            expected_tools=["lookup_subject", "validate_id_document"],
            hints=["Match against the subject registry", "Confirm the ID document is current"],
            required=True,
            max_tool_calls=5,
        ),
        PlaybookStep(
            id="locate_records",
            description="Find every personal-data record held for the verified subject",
            expected_tools=["search_data_inventory", "query_systems"],
            hints=["Group records by system", "Note collection timestamps"],
            required=True,
        ),
        PlaybookStep(
            id="review_scope",
            description="Determine which records are in scope and which exemptions apply",
            expected_tools=["check_legal_basis", "list_data_categories"],
            hints=["Flag third-party PII that must be excluded"],
            required=False,
        ),
        PlaybookStep(
            id="compile_response",
            description="Assemble the access-request response package for the subject",
            expected_tools=[],
            hints=["Include the data categories disclosed and the legal basis"],
            required=True,
        ),
    ]
    for step in steps:
        flag = "required" if step.required else "optional"
        print(f"  Step: {step.id:18s} {flag:8s} expected_tools={step.expected_tools}")
    return steps


# =============================================================================
# Part 2: Assemble steps into a Playbook with ordering and tags.
# =============================================================================


def part2_playbook(steps: list[PlaybookStep]) -> Playbook:
    print("\n=== Part 2: Creating a Playbook ===\n")
    playbook = Playbook(
        id="dsar_right_of_access",
        name="DSAR Playbook — Right of Access",
        description="GDPR Article 15 right-of-access fulfillment procedure for verified subjects",
        version="1.0.0",
        steps=steps,
        strict_sequence=True,
        allow_extra_tools=True,
        max_iterations=20,
        tags=["data-privacy", "gdpr", "dsar"],
    )
    print(f"  Playbook: {playbook.name} v{playbook.version} steps={len(playbook.steps)}")
    print(f"  strict_sequence={playbook.strict_sequence}  tags={playbook.tags}")
    return playbook


# =============================================================================
# Part 3: PlaybookPlan — runtime state. Track step executions and progress,
#         then ask the model what the next step should call.
# =============================================================================


async def part3_plan(playbook: Playbook) -> PlaybookPlan:
    print("\n=== Part 3: Execution Plans ===\n")
    plan = PlaybookPlan(playbook=playbook)
    plan.step_executions["verify_identity"] = StepExecution(
        step_id="verify_identity",
        status=StepStatus.COMPLETED,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        tool_calls=["lookup_subject", "validate_id_document", "lookup_subject"],
        tool_call_count=3,
        result="Identity verified for subject S-4821 via passport match",
    )
    plan.current_step_index = 1
    print(f"  Progress: {plan.progress:.0%}  current_step={plan.current_step.id}")

    # The plan is what an agent would hand the model between steps: what is
    # done, what comes next, and which tools that step allows.
    advisor = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one short sentence.")
    res = await advisor.arun(
        f"The previous step '{plan.completed_steps[0]}' completed and verified the "
        f"subject's identity. The next step is '{plan.current_step.id}' and may use "
        f"{plan.current_step.expected_tools}. Suggest one specific tool call for that step."
    )
    _banner(res)
    print(f"  AI next-step suggestion: {res.message.strip()}")
    return plan


# =============================================================================
# Part 4: StepStatus values — what each one means in practice.
# =============================================================================


def part4_status(plan: PlaybookPlan) -> None:
    print("\n=== Part 4: Step Status Tracking ===\n")
    for status in StepStatus:
        print(f"  - {status.value}")
    print(f"  is_step_complete('verify_identity') = {plan.is_step_complete('verify_identity')}")
    print(f"  is_step_complete('locate_records')  = {plan.is_step_complete('locate_records')}")


# =============================================================================
# Part 5: Validation rules — min_tool_calls and keyword checks per step.
# =============================================================================


def part5_validation() -> None:
    print("\n=== Part 5: Playbook Validation ===\n")
    validated_step = PlaybookStep(
        id="validate_redaction",
        description="Verify third-party PII has been redacted before disclosure",
        expected_tools=["scan_pii", "apply_redactions"],
        validation={
            "min_tool_calls": 1,
            "required_result_keywords": ["redacted", "no third-party"],
        },
        required=True,
    )
    print(f"  Step: {validated_step.id}  validation={validated_step.validation}")


# =============================================================================
# Part 6: Step metadata — arbitrary fields for severity, channels, etc.
# =============================================================================


def part6_metadata() -> None:
    print("\n=== Part 6: Playbook Metadata ===\n")
    step_with_meta = PlaybookStep(
        id="escalate",
        description="Escalate if the request is complex or high-risk for the data controller",
        expected_tools=["notify_dpo", "open_review_ticket"],
        metadata={
            "risk_threshold": "high",
            "statutory_deadline_days": 30,
            "notify_channels": ["#privacy-ops", "#dpo-oncall"],
        },
    )
    print(f"  metadata: {step_with_meta.metadata}")


# =============================================================================
# Part 7: Build playbooks programmatically — one factory function,
#         parameterised by environment and target-system list.
# =============================================================================


def erasure_playbook(env: str, systems: list[str]) -> Playbook:
    steps = [
        PlaybookStep(
            id="pre_check",
            description=f"Confirm erasure is lawful and in scope for {env}",
            expected_tools=["check_legal_basis", "verify_scope"],
            required=True,
        )
    ]
    steps += [
        PlaybookStep(
            id=f"erase_{s}",
            description=f"Delete the subject's personal data in {s} ({env})",
            expected_tools=["delete_records", "wait_confirmed"],
            metadata={"system": s},
            required=True,
        )
        for s in systems
    ]
    steps.append(
        PlaybookStep(
            id="post_validate",
            description="Validate erasure completed across every target system",
            expected_tools=["scan_residual_pii", "check_backups"],
            required=True,
        )
    )
    return Playbook(
        id=f"erase_{env}",
        name=f"{env.title()} Right-to-Erasure",
        steps=steps,
        tags=["erasure", env],
    )


def part7_factory() -> None:
    print("\n=== Part 7: Building Playbooks Programmatically ===\n")
    prod_playbook = erasure_playbook("production", ["crm", "billing", "analytics"])
    print(f"  Generated: {prod_playbook.name}  steps={[s.id for s in prod_playbook.steps]}")


# =============================================================================
# Part 8: Render progress — a quick ASCII bar from PlaybookPlan.progress.
# =============================================================================


def part8_progress(playbook: Playbook) -> None:
    print("\n=== Part 8: Progress Visualization ===\n")
    demo_plan = PlaybookPlan(playbook=playbook)
    demo_plan.step_executions["verify_identity"] = StepExecution(
        step_id="verify_identity", status=StepStatus.COMPLETED
    )
    demo_plan.step_executions["locate_records"] = StepExecution(
        step_id="locate_records", status=StepStatus.IN_PROGRESS
    )
    demo_plan.current_step_index = 1
    filled = int(demo_plan.progress * 20)
    bar = "#" * filled + "-" * (20 - filled)
    print(f"  [{bar}] {demo_plan.progress:.0%}")


# =============================================================================
# Part 9: Agent(playbook=...) — real tools, real agent, the playbook
#         enforces order. All request data below is mock (fictional IDs).
# =============================================================================


@tool
def fetch_request(request_id: str) -> str:
    """Pull a data-subject request by id."""
    return (
        f"[{request_id}] 2026-06-02T03:14:09Z DSAR right_of_access subject 'S-4821' "
        "spanning crm, billing, analytics\n[DSAR-7741] "
        "2026-06-02T03:14:41Z note: subject record includes a third-party contact"
    )


@tool
def classify_request(snippet: str) -> str:
    """Classify a request as 'complex' or 'standard' from its text."""
    return "complex" if "right_of_access" in snippet and "third-party" in snippet else "standard"


@tool
def notify_dpo(complexity: str, request_id: str) -> str:
    """Notify the data protection officer about a request."""
    return f"notified DPO for {request_id} at complexity {complexity}"


async def part9_live_agent() -> None:
    print("\n=== Part 9: Live Agent driving a Playbook ===\n")

    triage = Playbook(
        id="dsar_triage",
        name="DSAR triage",
        steps=[
            PlaybookStep(
                id="gather", description="Pull the request", expected_tools=["fetch_request"]
            ),
            PlaybookStep(
                id="classify", description="Decide complexity", expected_tools=["classify_request"]
            ),
            PlaybookStep(
                id="notify", description="Notify DPO if complex", expected_tools=["notify_dpo"]
            ),
        ],
    )

    triage_agent = Agent(
        model=get_model(max_tokens=400),
        tools=[fetch_request, classify_request, notify_dpo],
        playbook=triage,
        system_prompt=(
            "You are a privacy analyst handling data-subject requests. Follow the "
            "playbook steps in order: fetch the request, classify its complexity, "
            "then notify the DPO if it is complex."
        ),
    )
    t0 = time.perf_counter()
    triage_result = await triage_agent.arun("Triage request DSAR-7741.")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{triage_result.metrics.prompt_tokens}→{triage_result.metrics.completion_tokens} tokens · "
        f"iters={triage_result.metrics.iterations}]"
    )
    print(f"  Tools called: {[te.tool_name for te in triage_result.tool_executions]}")
    print(f"Triage outcome: {triage_result.message[:300]}")


async def main():
    print("=" * 60)
    print("Notebook 49: playbooks")
    print("=" * 60)

    steps = part1_steps()
    playbook = part2_playbook(steps)
    plan = await part3_plan(playbook)
    part4_status(plan)
    part5_validation()
    part6_metadata()
    part7_factory()
    part8_progress(playbook)
    await part9_live_agent()

    print("\n" + "=" * 60)
    print("Done. Next: notebook 50 — checkpoint backends.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
