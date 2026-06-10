# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 46: playbooks — a NIST 800-61 IR runbook the agent must follow.

A playbook is a typed, ordered sequence of steps with declared
``expected_tools``. Wire it into an agent and the agent is constrained
to walk the steps in order, calling only the tools each step allows.
That is exactly what incident response demands: a NIST 800-61-style
runbook where no step is skipped, every tool call is attributable to a
step, and the whole run is auditable after the fact.

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
        id="gather_telemetry",
        description="Collect SIEM alerts and endpoint telemetry for the affected hosts",
        expected_tools=["query_siem", "fetch_edr_events"],
        hints=["Start with the alert window", "Pull both network and endpoint telemetry"],
        required=True,
        max_tool_calls=5,
    )
    step2 = PlaybookStep(
        id="analyze_indicators",
        description="Extract and analyze indicators of compromise from the telemetry",
        expected_tools=["extract_iocs", "lookup_reputation"],
        hints=["Group IOCs by type", "Note first-seen timestamps"],
        required=True,
    )
    step3 = PlaybookStep(
        id="scope_impact",
        description="Determine which hosts and accounts are affected",
        expected_tools=["query_asset_inventory", "list_account_activity"],
        hints=["Focus on possible lateral-movement paths"],
        required=False,
    )
    step4 = PlaybookStep(
        id="summarize_findings",
        description="Write the detection-and-analysis summary with recommended containment",
        expected_tools=[],
        hints=["Include suspected root cause and confidence level"],
        required=True,
    )
    print(f"  Step: {step1.id} ({len(step1.expected_tools)} expected tools)")
    rationale = _llm_call(
        "In one sentence, why does an incident-response runbook benefit from "
        "having `expected_tools` declared per step?",
    )
    print(f"AI rationale: {rationale}")

    # =========================================================================
    # Part 2: Assemble steps into a Playbook with ordering and tags.
    # =========================================================================
    print("\n=== Part 2: Creating a Playbook ===\n")
    playbook = Playbook(
        id="ir_detection_analysis",
        name="IR Playbook — Detection & Analysis",
        description="NIST 800-61 detection-and-analysis procedure for suspected intrusions",
        version="1.0.0",
        steps=[step1, step2, step3, step4],
        strict_sequence=True,
        allow_extra_tools=True,
        max_iterations=20,
        tags=["incident-response", "nist-800-61", "soc"],
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
    plan.step_executions["gather_telemetry"] = StepExecution(
        step_id="gather_telemetry",
        status=StepStatus.COMPLETED,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        tool_calls=["query_siem", "fetch_edr_events", "query_siem"],
        tool_call_count=3,
        result="Found 14 failed logins then a success from 198.51.100.7",
    )
    plan.current_step_index = 1
    print(f"  Progress: {plan.progress:.0%}  current_step={plan.current_step.id}")
    next_step = _llm_call(
        f"The previous step '{plan.completed_steps[0]}' completed and found 14 "
        f"failed logins followed by a success. The next step is "
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
    print(f"  is_step_complete('gather_telemetry') = {plan.is_step_complete('gather_telemetry')}")
    summary = _llm_call(
        "In one sentence, when should an incident responder mark a runbook step "
        "as SKIPPED rather than FAILED?",
        max_tokens=80,
    )
    print(f"AI summary: {summary}")

    # =========================================================================
    # Part 5: Validation rules — min_tool_calls and keyword checks per step.
    # =========================================================================
    print("\n=== Part 5: Playbook Validation ===\n")
    validated_step = PlaybookStep(
        id="validate_containment",
        description="Verify the containment action took effect",
        expected_tools=["check_isolation", "run_edr_sweep"],
        validation={"min_tool_calls": 1, "required_result_keywords": ["isolated", "clean"]},
        required=True,
    )
    print(f"  Step: {validated_step.id}  validation={validated_step.validation}")
    judge = _llm_call(
        f"This step requires the result to contain {validated_step.validation['required_result_keywords']}. "
        "If the actual result is 'host isolated: web-prod-03, follow-up sweep clean', does it satisfy the validation? Reply YES or NO with one-word reason.",
        max_tokens=40,
    )
    print(f"AI judgment: {judge}")

    # =========================================================================
    # Part 6: Step metadata — arbitrary fields for severity, channels, etc.
    # =========================================================================
    print("\n=== Part 6: Playbook Metadata ===\n")
    step_with_meta = PlaybookStep(
        id="escalate",
        description="Escalate if the intrusion is confirmed or spreading",
        expected_tools=["send_alert", "page_oncall"],
        metadata={
            "severity_threshold": "high",
            "escalation_timeout_minutes": 30,
            "notify_channels": ["#soc-incidents", "#ir-oncall"],
        },
    )
    print(f"  metadata: {step_with_meta.metadata}")
    suggestion = _llm_call(
        "Suggest one extra metadata field a production-grade incident-response "
        "playbook step should carry, with a one-line rationale.",
        max_tokens=80,
    )
    print(f"AI suggestion: {suggestion}")

    # =========================================================================
    # Part 7: Build playbooks programmatically — one factory function,
    #         parameterised by environment and affected-host list.
    # =========================================================================
    print("\n=== Part 7: Building Playbooks Programmatically ===\n")

    def containment_playbook(env: str, hosts: list[str]) -> Playbook:
        steps = [
            PlaybookStep(
                id="pre_check",
                description=f"Verify containment approval and scope for {env}",
                expected_tools=["check_approval", "verify_scope"],
                required=True,
            )
        ]
        steps += [
            PlaybookStep(
                id=f"contain_{h}",
                description=f"Isolate {h} in {env}",
                expected_tools=["isolate_host", "wait_confirmed"],
                metadata={"host": h},
                required=True,
            )
            for h in hosts
        ]
        steps.append(
            PlaybookStep(
                id="post_validate",
                description="Validate containment success across the scope",
                expected_tools=["run_edr_sweep", "check_alert_volume"],
                required=True,
            )
        )
        return Playbook(
            id=f"contain_{env}",
            name=f"{env.title()} Containment",
            steps=steps,
            tags=["containment", env],
        )

    prod_playbook = containment_playbook("production", ["web-01", "db-02", "jump-03"])
    print(f"  Generated: {prod_playbook.name}  steps={[s.id for s in prod_playbook.steps]}")
    review = _llm_call(
        f"Review this generated containment playbook: {[s.id for s in prod_playbook.steps]}. "
        "Spot one weakness in one short sentence.",
        max_tokens=100,
    )
    print(f"AI review: {review}")

    # =========================================================================
    # Part 8: Render progress — a quick ASCII bar from PlaybookPlan.progress.
    # =========================================================================
    print("\n=== Part 8: Progress Visualization ===\n")
    demo_plan = PlaybookPlan(playbook=playbook)
    demo_plan.step_executions["gather_telemetry"] = StepExecution(
        step_id="gather_telemetry", status=StepStatus.COMPLETED
    )
    demo_plan.step_executions["analyze_indicators"] = StepExecution(
        step_id="analyze_indicators", status=StepStatus.IN_PROGRESS
    )
    demo_plan.current_step_index = 1
    bar = "#" * int(demo_plan.progress * 20) + "-" * (20 - int(demo_plan.progress * 20))
    print(f"  [{bar}] {demo_plan.progress:.0%}")
    eta = _llm_call(
        "An incident-response playbook has 4 steps. One is done, one is "
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
        "incident-response playbooks. Five bullets only.",
        max_tokens=240,
    )
    print(practices)

    # =========================================================================
    # Part 10: Agent(playbook=...) — real tools, real agent, the playbook
    #          enforces order. All alert data below is mock (RFC 5737 IPs).
    # =========================================================================
    print("\n=== Part 10: Live Agent driving a Playbook ===\n")

    @tool
    def fetch_alerts(incident_id: str) -> str:
        return (
            f"[{incident_id}] 2026-06-02T03:14:09Z ALERT auth.bruteforce 14 failed "
            "logins for 'svc-backup' from 198.51.100.7\n[INC-7741] "
            "2026-06-02T03:14:41Z ALERT auth.success 'svc-backup' login from 198.51.100.7"
        )

    @tool
    def classify_severity(snippet: str) -> str:
        return "P1" if "bruteforce" in snippet and "auth.success" in snippet else "P3"

    @tool
    def page_oncall(severity: str, incident_id: str) -> str:
        return f"paged IR oncall for {incident_id} at severity {severity}"

    triage = Playbook(
        id="alert_triage",
        name="Alert triage",
        steps=[
            PlaybookStep(id="gather", description="Pull alerts", expected_tools=["fetch_alerts"]),
            PlaybookStep(
                id="classify", description="Decide severity", expected_tools=["classify_severity"]
            ),
            PlaybookStep(
                id="page", description="Page oncall if P1", expected_tools=["page_oncall"]
            ),
        ],
    )

    triage_agent = Agent(
        model=get_model(max_tokens=400),
        tools=[fetch_alerts, classify_severity, page_oncall],
        playbook=triage,
        system_prompt=(
            "You are a SOC analyst on call. Follow the playbook steps in order: "
            "fetch alerts, classify severity, then page the IR oncall if it is P1."
        ),
    )
    t0 = time.perf_counter()
    triage_result = triage_agent.run_sync("Triage incident INC-7741.")
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
