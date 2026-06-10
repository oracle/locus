# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 43: playbooks — typed step-by-step procedures the agent must follow.

A playbook is a typed, ordered sequence of steps with declared
``expected_tools``. Wire it into an agent and the agent is constrained
to walk the steps in order, calling only the tools each step allows.
Useful for incident response, deployments, and any procedure where
you want auditability over agent freedom.

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
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_48_playbooks.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_48_playbooks.py

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
    print("Notebook 43: playbooks")
    print("=" * 60)

    # =========================================================================
    # Part 1: PlaybookStep — each step declares the tools it expects.
    # =========================================================================
    print("\n=== Part 1: Creating Playbook Steps ===\n")
    step1 = PlaybookStep(
        id="gather_logs",
        description="Collect relevant log files from the affected services",
        expected_tools=["read_file", "search_logs"],
        hints=["Start with the most recent logs", "Look for ERROR and WARN levels"],
        required=True,
        max_tool_calls=5,
    )
    step2 = PlaybookStep(
        id="analyze_errors",
        description="Analyze the collected logs for error patterns",
        expected_tools=["analyze_logs", "count_errors"],
        hints=["Group errors by type", "Note timestamps"],
        required=True,
    )
    step3 = PlaybookStep(
        id="check_metrics",
        description="Review system metrics during the incident window",
        expected_tools=["query_metrics", "get_dashboard"],
        hints=["Focus on CPU, memory, and network"],
        required=False,
    )
    step4 = PlaybookStep(
        id="summarize_findings",
        description="Create a summary of findings and recommendations",
        expected_tools=[],
        hints=["Include root cause if identified"],
        required=True,
    )
    print(f"  Step: {step1.id} ({len(step1.expected_tools)} expected tools)")
    rationale = _llm_call(
        "In one sentence, why does an incident-response playbook benefit from "
        "having `expected_tools` declared per step?",
    )
    print(f"AI rationale: {rationale}")

    # =========================================================================
    # Part 2: Assemble steps into a Playbook with ordering and tags.
    # =========================================================================
    print("\n=== Part 2: Creating a Playbook ===\n")
    playbook = Playbook(
        id="incident_investigation",
        name="Incident Investigation Playbook",
        description="Standard procedure for investigating production incidents",
        version="1.0.0",
        steps=[step1, step2, step3, step4],
        strict_sequence=True,
        allow_extra_tools=True,
        max_iterations=20,
        tags=["incident", "investigation", "production"],
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
    plan.step_executions["gather_logs"] = StepExecution(
        step_id="gather_logs",
        status=StepStatus.COMPLETED,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        tool_calls=["read_file", "search_logs", "read_file"],
        tool_call_count=3,
        result="Found 15 error entries in app.log",
    )
    plan.current_step_index = 1
    print(f"  Progress: {plan.progress:.0%}  current_step={plan.current_step.id}")
    next_step = _llm_call(
        f"The previous step '{plan.completed_steps[0]}' completed and found 15 "
        f"error entries. The next step is '{plan.current_step.id}'. Suggest one "
        "specific tool call for that step.",
        max_tokens=80,
    )
    print(f"AI next-step suggestion: {next_step}")

    # =========================================================================
    # Part 4: StepStatus values — what each one means in practice.
    # =========================================================================
    print("\n=== Part 4: Step Status Tracking ===\n")
    for status in StepStatus:
        print(f"  - {status.value}")
    print(f"  is_step_complete('gather_logs') = {plan.is_step_complete('gather_logs')}")
    summary = _llm_call(
        "In one sentence, when should an SRE mark a playbook step as SKIPPED rather than FAILED?",
        max_tokens=80,
    )
    print(f"AI summary: {summary}")

    # =========================================================================
    # Part 5: Validation rules — min_tool_calls and keyword checks per step.
    # =========================================================================
    print("\n=== Part 5: Playbook Validation ===\n")
    validated_step = PlaybookStep(
        id="validate_fix",
        description="Verify the fix is working",
        expected_tools=["run_tests", "check_health"],
        validation={"min_tool_calls": 1, "required_result_keywords": ["passed", "healthy"]},
        required=True,
    )
    print(f"  Step: {validated_step.id}  validation={validated_step.validation}")
    judge = _llm_call(
        f"This step requires the result to contain {validated_step.validation['required_result_keywords']}. "
        "If the actual result is 'tests passed: 12, services healthy', does it satisfy the validation? Reply YES or NO with one-word reason.",
        max_tokens=40,
    )
    print(f"AI judgment: {judge}")

    # =========================================================================
    # Part 6: Step metadata — arbitrary fields for severity, channels, etc.
    # =========================================================================
    print("\n=== Part 6: Playbook Metadata ===\n")
    step_with_meta = PlaybookStep(
        id="escalate",
        description="Escalate if issue persists",
        expected_tools=["send_alert", "page_oncall"],
        metadata={
            "severity_threshold": "high",
            "escalation_timeout_minutes": 30,
            "notify_channels": ["#incidents", "#oncall"],
        },
    )
    print(f"  metadata: {step_with_meta.metadata}")
    suggestion = _llm_call(
        "Suggest one extra metadata field a production-grade incident playbook "
        "step should carry, with a one-line rationale.",
        max_tokens=80,
    )
    print(f"AI suggestion: {suggestion}")

    # =========================================================================
    # Part 7: Build playbooks programmatically — one factory function,
    #         parameterised by environment and service list.
    # =========================================================================
    print("\n=== Part 7: Building Playbooks Programmatically ===\n")

    def deployment_playbook(env: str, services: list[str]) -> Playbook:
        steps = [
            PlaybookStep(
                id="pre_check",
                description=f"Verify {env} environment is ready",
                expected_tools=["check_health", "verify_deps"],
                required=True,
            )
        ]
        steps += [
            PlaybookStep(
                id=f"deploy_{s}",
                description=f"Deploy {s} to {env}",
                expected_tools=["deploy", "wait_healthy"],
                metadata={"service": s},
                required=True,
            )
            for s in services
        ]
        steps.append(
            PlaybookStep(
                id="post_validate",
                description="Validate deployment success",
                expected_tools=["run_smoke_tests", "check_metrics"],
                required=True,
            )
        )
        return Playbook(
            id=f"deploy_{env}",
            name=f"{env.title()} Deployment",
            steps=steps,
            tags=["deployment", env],
        )

    prod_playbook = deployment_playbook("production", ["api", "web", "worker"])
    print(f"  Generated: {prod_playbook.name}  steps={[s.id for s in prod_playbook.steps]}")
    review = _llm_call(
        f"Review this generated deployment playbook: {[s.id for s in prod_playbook.steps]}. "
        "Spot one weakness in one short sentence.",
        max_tokens=100,
    )
    print(f"AI review: {review}")

    # =========================================================================
    # Part 8: Render progress — a quick ASCII bar from PlaybookPlan.progress.
    # =========================================================================
    print("\n=== Part 8: Progress Visualization ===\n")
    demo_plan = PlaybookPlan(playbook=playbook)
    demo_plan.step_executions["gather_logs"] = StepExecution(
        step_id="gather_logs", status=StepStatus.COMPLETED
    )
    demo_plan.step_executions["analyze_errors"] = StepExecution(
        step_id="analyze_errors", status=StepStatus.IN_PROGRESS
    )
    demo_plan.current_step_index = 1
    bar = "#" * int(demo_plan.progress * 20) + "-" * (20 - int(demo_plan.progress * 20))
    print(f"  [{bar}] {demo_plan.progress:.0%}")
    eta = _llm_call(
        "An incident-investigation playbook has 4 steps. One is done, one is "
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
        "playbooks. Five bullets only.",
        max_tokens=240,
    )
    print(practices)

    # =========================================================================
    # Part 10: Agent(playbook=...) — real tools, real agent, the playbook
    #          enforces order.
    # =========================================================================
    print("\n=== Part 10: Live Agent driving a Playbook ===\n")

    @tool
    def fetch_logs(incident_id: str) -> str:
        return (
            f"[{incident_id}] 2026-05-03T19:01:14Z ERROR db.pool exhausted "
            "(50/50 conns)\n[INC-42] 2026-05-03T19:01:18Z ERROR api.handler "
            "timeout calling /v1/orders"
        )

    @tool
    def classify_severity(snippet: str) -> str:
        return "P1" if "ERROR" in snippet and "exhausted" in snippet else "P3"

    @tool
    def page_oncall(severity: str, incident_id: str) -> str:
        return f"paged oncall for {incident_id} at severity {severity}"

    triage = Playbook(
        id="incident_triage",
        name="Incident triage",
        steps=[
            PlaybookStep(id="gather", description="Pull logs", expected_tools=["fetch_logs"]),
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
        tools=[fetch_logs, classify_severity, page_oncall],
        playbook=triage,
        system_prompt=(
            "You are an SRE on call. Follow the playbook steps in order: "
            "fetch logs, classify severity, then page oncall if it is P1."
        ),
    )
    t0 = time.perf_counter()
    triage_result = triage_agent.run_sync("Triage incident INC-42.")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{triage_result.metrics.prompt_tokens}→{triage_result.metrics.completion_tokens} tokens · "
        f"iters={triage_result.metrics.iterations}]"
    )
    print(f"Triage outcome: {triage_result.message[:300]}")

    print("\n" + "=" * 60)
    print("Done. Next: notebook 43 — plugins.")
    print("=" * 60)


if __name__ == "__main__":
    main()
