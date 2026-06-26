# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 46: playbooks — a GDPR data-subject-access runbook the agent must follow.

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

Each part fires a real model call so you can see live behaviour next
to the structured execution mechanics — every section prints
``[model call: X.XXs · prompt→completion tokens]``.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_46_playbooks.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_46_playbooks.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

import time
from datetime import UTC, datetime

from config import get_model

from tulip.agent import Agent
from tulip.playbooks import (
    Playbook,
    PlaybookPlan,
    PlaybookStep,
    StepExecution,
    StepStatus,
)
from tulip.tools import tool


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 100
) -> str:
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


def main():
    print("=" * 60)
    print("Notebook 46: playbooks")
    print("=" * 60)

    # =========================================================================
    # Part 1: PlaybookStep — each step declares the tools it expects.
    # =========================================================================
    print("\n=== Part 1: Creating Playbook Steps ===\n")
    step1 = PlaybookStep(
        id="verify_identity",
        description="Verify the requester's identity before processing the access request",
        expected_tools=["lookup_subject", "validate_id_document"],
        hints=["Match against the subject registry", "Confirm the ID document is current"],
        required=True,
        max_tool_calls=5,
    )
    step2 = PlaybookStep(
        id="locate_records",
        description="Find every personal-data record held for the verified subject",
        expected_tools=["search_data_inventory", "query_systems"],
        hints=["Group records by system", "Note collection timestamps"],
        required=True,
    )
    step3 = PlaybookStep(
        id="review_scope",
        description="Determine which records are in scope and which exemptions apply",
        expected_tools=["check_legal_basis", "list_data_categories"],
        hints=["Flag third-party PII that must be excluded"],
        required=False,
    )
    step4 = PlaybookStep(
        id="compile_response",
        description="Assemble the access-request response package for the subject",
        expected_tools=[],
        hints=["Include the data categories disclosed and the legal basis"],
        required=True,
    )
    print(f"  Step: {step1.id} ({len(step1.expected_tools)} expected tools)")
    rationale = _llm_call(
        "In one sentence, why does a GDPR data-subject-access runbook benefit from "
        "having `expected_tools` declared per step?",
    )
    print(f"AI rationale: {rationale}")

    # =========================================================================
    # Part 2: Assemble steps into a Playbook with ordering and tags.
    # =========================================================================
    print("\n=== Part 2: Creating a Playbook ===\n")
    playbook = Playbook(
        id="dsar_right_of_access",
        name="DSAR Playbook — Right of Access",
        description="GDPR Article 15 right-of-access fulfilment procedure for verified subjects",
        version="1.0.0",
        steps=[step1, step2, step3, step4],
        strict_sequence=True,
        allow_extra_tools=True,
        max_iterations=20,
        tags=["data-privacy", "gdpr", "dsar"],
    )
    print(f"  Playbook: {playbook.name} v{playbook.version} steps={len(playbook.steps)}")
    described = _llm_call(
        f"Describe this playbook in two sentences: {playbook.description}. "
        f"Steps: {[s.id for s in playbook.steps]}.",
        max_tokens=160,
    )
    print(f"AI description: {described}")

    # =========================================================================
    # Part 3: PlaybookPlan — runtime state. Track step executions and progress.
    # =========================================================================
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
    next_step = _llm_call(
        f"The previous step '{plan.completed_steps[0]}' completed and verified the "
        f"subject's identity. The next step is "
        f"'{plan.current_step.id}'. Suggest one specific tool call for that step.",
        max_tokens=80,
    )
    print(f"AI next-step suggestion: {next_step}")

    # =========================================================================
    # Part 4: StepStatus values — what each one means in practice.
    # =========================================================================
    print("\n=== Part 4: Step Status Tracking ===\n")
    for status in StepStatus:
        print(f"  - {status.value}")
    print(f"  is_step_complete('verify_identity') = {plan.is_step_complete('verify_identity')}")
    summary = _llm_call(
        "In one sentence, when should a privacy analyst mark a DSAR runbook step "
        "as SKIPPED rather than FAILED?",
        max_tokens=80,
    )
    print(f"AI summary: {summary}")

    # =========================================================================
    # Part 5: Validation rules — min_tool_calls and keyword checks per step.
    # =========================================================================
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
    judge = _llm_call(
        f"This step requires the result to contain {validated_step.validation['required_result_keywords']}. "
        "If the actual result is 'two contact records redacted, no third-party PII remains', does it satisfy the validation? Reply YES or NO with one-word reason.",
        max_tokens=40,
    )
    print(f"AI judgment: {judge}")

    # =========================================================================
    # Part 6: Step metadata — arbitrary fields for severity, channels, etc.
    # =========================================================================
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
    suggestion = _llm_call(
        "Suggest one extra metadata field a production-grade GDPR data-subject-access "
        "playbook step should carry, with a one-line rationale.",
        max_tokens=80,
    )
    print(f"AI suggestion: {suggestion}")

    # =========================================================================
    # Part 7: Build playbooks programmatically — one factory function,
    #         parameterised by environment and target-system list.
    # =========================================================================
    print("\n=== Part 7: Building Playbooks Programmatically ===\n")

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

    prod_playbook = erasure_playbook("production", ["crm", "billing", "analytics"])
    print(f"  Generated: {prod_playbook.name}  steps={[s.id for s in prod_playbook.steps]}")
    review = _llm_call(
        f"Review this generated right-to-erasure playbook: {[s.id for s in prod_playbook.steps]}. "
        "Spot one weakness in one short sentence.",
        max_tokens=100,
    )
    print(f"AI review: {review}")

    # =========================================================================
    # Part 8: Render progress — a quick ASCII bar from PlaybookPlan.progress.
    # =========================================================================
    print("\n=== Part 8: Progress Visualization ===\n")
    demo_plan = PlaybookPlan(playbook=playbook)
    demo_plan.step_executions["verify_identity"] = StepExecution(
        step_id="verify_identity", status=StepStatus.COMPLETED
    )
    demo_plan.step_executions["locate_records"] = StepExecution(
        step_id="locate_records", status=StepStatus.IN_PROGRESS
    )
    demo_plan.current_step_index = 1
    bar = "#" * int(demo_plan.progress * 20) + "-" * (20 - int(demo_plan.progress * 20))
    print(f"  [{bar}] {demo_plan.progress:.0%}")
    eta = _llm_call(
        "A data-subject-access playbook has 4 steps. One is done, one is "
        "in progress, two are pending. Roughly how long should we expect the "
        "remaining work to take? Answer in one short sentence.",
        max_tokens=80,
    )
    print(f"AI ETA: {eta}")

    # =========================================================================
    # Part 9: The model writes a short best-practices cheatsheet.
    # =========================================================================
    print("\n=== Part 9: Best Practices ===\n")
    practices = _llm_call(
        "Write five terse best-practice bullets for designing reliable Tulip "
        "data-privacy request playbooks. Five bullets only.",
        max_tokens=240,
    )
    print(practices)

    # =========================================================================
    # Part 10: Agent(playbook=...) — real tools, real agent, the playbook
    #          enforces order. All request data below is mock (fictional IDs).
    # =========================================================================
    print("\n=== Part 10: Live Agent driving a Playbook ===\n")

    @tool
    def fetch_request(request_id: str) -> str:
        return (
            f"[{request_id}] 2026-06-02T03:14:09Z DSAR right_of_access subject 'S-4821' "
            "spanning crm, billing, analytics\n[DSAR-7741] "
            "2026-06-02T03:14:41Z note: subject record includes a third-party contact"
        )

    @tool
    def classify_request(snippet: str) -> str:
        return (
            "complex" if "right_of_access" in snippet and "third-party" in snippet else "standard"
        )

    @tool
    def notify_dpo(complexity: str, request_id: str) -> str:
        return f"notified DPO for {request_id} at complexity {complexity}"

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
    triage_result = triage_agent.run_sync("Triage request DSAR-7741.")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{triage_result.metrics.prompt_tokens}→{triage_result.metrics.completion_tokens} tokens · "
        f"iters={triage_result.metrics.iterations}]"
    )
    print(f"Triage outcome: {triage_result.message[:300]}")

    print("\n" + "=" * 60)
    print("Done. Next: notebook 47 — plugins.")
    print("=" * 60)


if __name__ == "__main__":
    main()
