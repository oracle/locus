# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 28: specialist agents — domain focus, tools, and playbooks.

Notebook 27 introduced the Specialist as the worker an orchestrator
hands tasks to. This notebook dives into the Specialist itself: how to
narrow a model's failure surface with a focused system prompt, a
hand-picked tool set, optional playbooks, and a confidence threshold.

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
  defaults for common observability domains.

Run it:
    .venv/bin/python examples/notebook_33_specialist_agents.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- Notebook 08 (Agent basics).
- Notebook 27 (Orchestrator) — Specialists are the workers it routes to.
"""

import asyncio
import time

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


async def main():
    print("=" * 60)
    print("Notebook 28: specialist agents — domain focus, tools, playbooks")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: anatomy of a Specialist
    # =========================================================================
    print("\n=== Part 1: Specialist Anatomy ===\n")

    # A specialist pairs a focused system prompt with domain tools,
    # optional playbooks, and a confidence threshold.
    specialist = Specialist(
        name="API Specialist",
        specialist_type="api_analyst",
        description="Analyzes API performance, errors, and patterns",
        system_prompt="""You are an API analysis specialist. Your expertise:
1. Analyzing HTTP status codes and error rates
2. Identifying slow endpoints
3. Detecting anomalous traffic patterns
4. Recommending API optimizations

When analyzing:
- Check error rates by endpoint
- Look for latency outliers
- Identify authentication issues
- Note rate limiting triggers""",
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

    @tool(name="get_endpoint_stats", description="Get statistics for an API endpoint")
    async def get_endpoint_stats(endpoint: str) -> str:
        return f"Endpoint {endpoint}: 1000 req/min, 2.5% error rate, p99=450ms"

    @tool(name="get_error_breakdown", description="Get error breakdown by status code")
    async def get_error_breakdown() -> str:
        return "Errors: 400=15%, 401=5%, 403=2%, 500=75%, 503=3%"

    @tool(name="get_top_slow_endpoints", description="Get slowest API endpoints")
    async def get_top_slow_endpoints() -> str:
        return "Slowest: /api/users (800ms), /api/search (650ms), /api/reports (500ms)"

    specialist = Specialist(
        name="API Specialist",
        specialist_type="api_analyst",
        description="Analyzes API performance, errors, and patterns",
        system_prompt="You analyze API behavior and performance.",
        tools=[get_endpoint_stats, get_error_breakdown, get_top_slow_endpoints],
        model=model,
    )

    print(f"Tools available: {[t.name for t in specialist.tools]}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why does giving a Specialist domain-specific tools dramatically narrow its failure surface?')}"
    )

    # =========================================================================
    # Part 3: encode a procedure as a Playbook
    # =========================================================================
    print("\n=== Part 3: Specialist Playbooks ===\n")

    api_debug_playbook = Playbook(
        name="API Debug Procedure",
        description="Standard procedure for debugging API issues",
        preconditions=[
            "Incident ticket exists",
            "Basic metrics are accessible",
        ],
        steps=[
            PlaybookStep(
                instruction="Check overall API health metrics",
                required_tools=["get_endpoint_stats"],
                expected_output="Current request rate and error percentages",
            ),
            PlaybookStep(
                instruction="Analyze error distribution",
                required_tools=["get_error_breakdown"],
                expected_output="Breakdown of errors by type",
                on_failure="Escalate if unable to get error data",
            ),
            PlaybookStep(
                instruction="Identify slow endpoints",
                required_tools=["get_top_slow_endpoints"],
                expected_output="List of endpoints exceeding latency threshold",
            ),
        ],
        success_criteria="Root cause identified or escalation path determined",
    )

    specialist.playbooks.append(api_debug_playbook)

    print(f"Playbook: {api_debug_playbook.name}")
    print(f"  Preconditions: {api_debug_playbook.preconditions}")
    print(f"  Steps: {len(api_debug_playbook.steps)}")
    print(f"  Success criteria: {api_debug_playbook.success_criteria}")

    # ``to_prompt()`` renders the playbook as the text block injected
    # into the system prompt when the playbook is selected.
    playbook_prompt = api_debug_playbook.to_prompt()
    print("\nPlaybook prompt:")
    print("-" * 40)
    print(playbook_prompt[:500] + "...")
    print(
        f"AI commentary: {_llm_call('In one sentence, when does attaching a Playbook to a Specialist matter most?')}"
    )

    # =========================================================================
    # Part 4: pick the right playbook from a pool
    # =========================================================================
    print("\n=== Part 4: Playbook Selection ===\n")

    performance_playbook = Playbook(
        name="Performance Optimization",
        description="Procedure for optimizing API performance",
        steps=[
            PlaybookStep(instruction="Profile slow endpoints"),
            PlaybookStep(instruction="Identify bottlenecks"),
            PlaybookStep(instruction="Recommend optimizations"),
        ],
    )

    security_playbook = Playbook(
        name="Security Analysis",
        description="Procedure for analyzing security issues",
        steps=[
            PlaybookStep(instruction="Check authentication failures"),
            PlaybookStep(instruction="Review access patterns"),
            PlaybookStep(instruction="Identify suspicious activity"),
        ],
    )

    specialist.playbooks.extend([performance_playbook, security_playbook])

    # select_playbook matches the task description against each playbook's
    # name and description; it returns None if nothing fits.
    tasks = [
        "Debug the API errors we're seeing",
        "Optimize the slow /api/search endpoint",
        "Check for unauthorized access attempts",
    ]

    for task in tasks:
        selected = specialist.select_playbook(task)
        if selected:
            print(f"Task: '{task[:40]}...'")
            print(f"  Selected playbook: {selected.name}")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is automatic playbook selection by task description risky and how do you mitigate it?')}"
    )

    # =========================================================================
    # Part 5: drive the specialist end-to-end
    # =========================================================================
    print("\n=== Part 5: Executing Specialists ===\n")

    result = await specialist.execute(
        task="API error rates have spiked in the last hour. Investigate and identify the cause.",
        context={
            "incident_id": "INC-2024-001",
            "affected_services": ["api-gateway"],
            "start_time": "2024-01-15T10:00:00Z",
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
    # Part 6: pre-built specialists for common domains
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
    p6 = await metrics_analyst.execute(task="In one sentence, what does a metrics analyst do?")
    dt = time.perf_counter() - t0
    print(f"\n  [model call: {dt:.2f}s · metrics_analyst.execute()]")
    if p6.output:
        print(f"  Output: {p6.output[:160]}")

    # =========================================================================
    # Part 7: extend a pre-built specialist with your own tools
    # =========================================================================
    print("\n=== Part 7: Custom Tools Integration ===\n")

    @tool(name="search_logs", description="Search logs for patterns")
    async def search_logs(pattern: str, timerange: str = "1h") -> str:
        return f"Found 42 matches for '{pattern}' in last {timerange}"

    @tool(name="get_error_logs", description="Get recent error logs")
    async def get_error_logs(limit: int = 10) -> str:
        return f"Retrieved {limit} most recent error logs"

    custom_log_analyst = create_log_analyst(
        model=model,
        tools=[search_logs, get_error_logs],
    )

    print(f"Custom log analyst tools: {[t.name for t in custom_log_analyst.tools]}")

    log_result = await custom_log_analyst.execute(
        task="Search for NullPointerException errors in the last hour",
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
        ("definitely the root cause", "High confidence markers"),
        ("might be related to", "Low confidence markers"),
        ("confirmed by the logs", "Verification markers"),
        ("unclear what is causing", "Uncertainty markers"),
    ]

    print("Confidence markers in responses:")
    for response, description in responses:
        confidence = specialist._estimate_confidence(response)
        print(f"  '{response}' -> {confidence:.0%} ({description})")
    print(
        f"AI commentary: {_llm_call('In one sentence, why is keyword-based confidence estimation only a rough proxy?')}"
    )

    # =========================================================================
    # Part 9: common specialist shapes
    # =========================================================================
    print("\n=== Part 9: Specialist Patterns ===\n")

    print("Pattern 1: Domain Expert")
    print("  Focused prompt + domain tools + high confidence threshold.")
    print()

    print("Pattern 2: Procedure Follower")
    print("  Playbook-driven; each step validated against expected output.")
    print()

    print("Pattern 3: Adaptive Analyst")
    print("  Multiple playbooks; the right one selected per task.")
    print()

    print("Pattern 4: Pipeline Stage")
    print("  Drops into a larger workflow; structured output, context in/out.")
    print(
        f"AI suggestion: {_llm_call('Suggest one extra Specialist pattern not in the four listed above. One short sentence.')}"
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
                description="Initial incident assessment and severity classification",
                system_prompt="Assess incidents and determine severity and routing.",
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
        task="In one sentence, classify this incident: 'API p99 jumped from 30ms to 800ms after a deploy.'",
    )
    dt = time.perf_counter() - t0
    print(f"  [model call: {dt:.2f}s · triage.execute()]")
    if p10.output:
        print(f"  Triage verdict: {p10.output[:160]}")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Next: Notebook 29 — A2A Protocol")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
