#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 58: Incident-response runbook (SRE workflow).

Models the loop a real on-call engineer runs when a page fires::

    Page fires
      │
      └──> Triage  ──>  scatter to 3 parallel investigators
                          ├── log analyst
                          ├── metric analyst
                          └── trace analyst
                          ▼
                   Synthesizer (root-cause hypothesis)
                          │
                          ▼
            Severity gate ─── critical? ──> page humans (interrupt)
                          │                     │
                          │                  approve mitigation? yes/no
                          │                     │
                          ▼                     ▼
                       Mitigator <──────────────┘
                          │
                          ▼
                       Postmortem (structured)

- Send: fan out to 3 investigator Agents in parallel.
- add_conditional_edges: severity-based routing decides auto-mitigate
  vs escalate to a human.
- interrupt(): critical severity pauses for explicit human approval
  before any mitigation runs.
- output_schema=Postmortem: the final report is a typed Pydantic
  instance, ready to file into a runbook database.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_63_incident_response.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_63_incident_response.py

    # Pin a strong-enough model for the structured postmortem schema:
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_63_incident_response.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import get_model
from pydantic import BaseModel, Field

from tulip.agent import Agent, AgentConfig
from tulip.core import Command, interrupt
from tulip.core.events import TerminateEvent
from tulip.core.send import Send
from tulip.multiagent.graph import END, START, StateGraph


# Data shapes for the typed workflow state.


class Postmortem(BaseModel):
    """Structured incident postmortem the workflow emits."""

    incident_id: str
    severity: str = Field(description="info | warn | critical")
    root_cause_hypothesis: str
    contributing_factors: list[str]
    mitigation_applied: str
    follow_up_actions: list[str]


class InvestigatorReport(BaseModel):
    source: str  # "logs" | "metrics" | "traces"
    findings: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# Specialist prompts — one role per workflow node.


PROMPTS = {
    "triage": (
        "You are an SRE triage bot. Given an incident description, classify "
        "severity as one of: info, warn, critical. Reply with one word."
    ),
    "logs": (
        "You are a log-analysis bot. List 1–3 concrete error patterns or "
        "warning sequences a human should investigate. Be specific. Bullets only."
    ),
    "metrics": (
        "You are a metrics-analysis bot. List 1–3 specific anomalies (latency "
        "spikes, error rate, saturation, traffic) that match the symptom. Bullets only."
    ),
    "traces": (
        "You are a distributed-trace bot. Identify 1–3 specific code paths or "
        "downstream calls likely involved. Bullets only."
    ),
    "synthesize": (
        "You are an incident commander. Given investigator reports, name the "
        "single most likely root-cause hypothesis. One sentence."
    ),
    "mitigate": (
        "You are an automated remediator. Given a root-cause hypothesis, "
        "propose ONE specific mitigation step. One sentence."
    ),
}


def _make_agent(role: str, model: Any) -> Agent:
    return Agent(
        config=AgentConfig(
            agent_id=f"sre-{role}",
            model=model,
            system_prompt=PROMPTS[role],
            max_iterations=2,
            max_tokens=300,
        )
    )


async def _run(agent: Agent, prompt: str) -> str:
    final = ""
    async for event in agent.run(prompt):
        if isinstance(event, TerminateEvent):
            final = event.final_message or ""
    return final.strip()


# Graph nodes.


async def triage(state: dict[str, Any]) -> dict[str, Any]:
    """Classify severity by asking the triage agent."""
    symptom = state.get("symptom", "")
    agent = _make_agent("triage", state["__model__"])
    raw = await _run(agent, f"Incident: {symptom}")
    sev = (raw.lower().split() or ["warn"])[0].strip(".,!?:;")
    severity = sev if sev in {"info", "warn", "critical"} else "warn"
    print(f"  [triage] agent classified severity={severity!r} (raw={raw!r})")
    return {"severity": severity}


async def scatter_investigators(state: dict[str, Any]) -> list[Send]:
    """Fan out to three investigators in parallel."""
    return [
        Send(node="investigate", payload={"source": s}, metadata={"source": s})
        for s in ("logs", "metrics", "traces")
    ]


async def investigate(state: dict[str, Any]) -> dict[str, Any]:
    source = state["source"]
    agent = _make_agent(source, state["__model__"])
    findings_text = await _run(
        agent, f"Symptom: {state['symptom']}\n\nFrom {source}, what do you see?"
    )
    findings = [line.lstrip("- *•").strip() for line in findings_text.splitlines() if line.strip()]
    return {
        "report": InvestigatorReport(
            source=source,
            findings=findings or [findings_text or f"(no {source} signal)"],
            confidence=0.7,
        )
    }


async def synthesize(state: dict[str, Any]) -> dict[str, Any]:
    reports = [v["report"] for v in state.values() if isinstance(v, dict) and "report" in v]
    bullets = "\n".join(f"[{r.source}] " + "; ".join(r.findings) for r in reports)
    agent = _make_agent("synthesize", state["__model__"])
    hypothesis = await _run(
        agent, f"Symptom: {state['symptom']}\n\nInvestigator findings:\n{bullets}"
    )
    return {"hypothesis": hypothesis, "reports": reports}


def severity_gate(state: dict[str, Any]) -> str:
    return "page_human" if state.get("severity") == "critical" else "auto_mitigate"


async def page_human(state: dict[str, Any]) -> dict[str, Any]:
    """Critical incidents pause for human approval before applying mitigation."""
    response = interrupt(
        {
            "type": "approval",
            "question": (
                f"CRITICAL incident — apply auto-mitigation for: "
                f"{state.get('hypothesis', '(no hypothesis)')!r}?"
            ),
            "options": ["yes", "no"],
            "incident": state.get("symptom"),
        }
    )
    return {"human_approved": response == "yes"}


async def mitigate(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("severity") == "critical" and not state.get("human_approved"):
        return {"mitigation": "(skipped — human declined)"}
    agent = _make_agent("mitigate", state["__model__"])
    mitigation = await _run(
        agent, f"Hypothesis: {state.get('hypothesis', '')}\n\nProposed mitigation?"
    )
    return {"mitigation": mitigation}


async def write_postmortem(state: dict[str, Any]) -> dict[str, Any]:
    """Emit the structured Postmortem via ``Agent.output_schema=Postmortem``.

    The agent reads the workflow's accumulated state (severity, hypothesis,
    investigator reports, mitigation) and emits a Pydantic ``Postmortem``.
    ``result.parsed`` must be populated — if the model can't honor the JSON
    schema we surface that as a hard error rather than fabricating a record.
    """
    import asyncio as _asyncio

    reports: list[InvestigatorReport] = state.get("reports", [])
    bullets = "\n".join(f"- [{r.source}] {'; '.join(r.findings)}" for r in reports)
    agent = Agent(
        config=AgentConfig(
            agent_id="postmortem-writer",
            model=state["__model__"],
            system_prompt=(
                "You are an SRE writing a postmortem. Produce a Postmortem "
                "object. Be terse and factual."
            ),
            output_schema=Postmortem,
            max_iterations=2,
            max_tokens=400,
        )
    )
    prompt = (
        f"Incident:   {state.get('incident_id', 'INC-0001')}\n"
        f"Severity:   {state.get('severity', 'warn')}\n"
        f"Symptom:    {state.get('symptom', '')}\n"
        f"Hypothesis: {state.get('hypothesis', '(unknown)')}\n"
        f"Findings:\n{bullets}\n"
        f"Mitigation: {state.get('mitigation', '(none)')}\n\n"
        "Emit the Postmortem now."
    )
    last_exc: BaseException | None = None
    result = None
    for attempt in range(3):
        try:
            result = await _asyncio.to_thread(agent.run_sync, prompt)
            break
        except Exception as exc:  # noqa: BLE001 — retry transient provider flakiness
            last_exc = exc
            await _asyncio.sleep(0.5 * (attempt + 1))
    if result is None:
        raise RuntimeError(
            f"Postmortem writer failed after 3 attempts. Last error: {last_exc!r}"
        ) from last_exc
    pm = result.parsed
    if pm is None:
        raise RuntimeError(
            "Postmortem writer returned no parsed Postmortem. The configured "
            "model could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 57. Raw output: {result.message!r}"
        )
    return {"postmortem": pm}


# Build the runbook graph.


def build_runbook() -> StateGraph:
    g = StateGraph(name="incident-runbook")
    g.add_node("triage", triage)
    g.add_node("scatter", scatter_investigators)
    g.add_node("investigate", investigate)
    g.add_node("synthesize", synthesize)
    g.add_node("page_human", page_human)
    g.add_node("auto_mitigate", mitigate)
    g.add_node("postmortem", write_postmortem)

    g.add_edge(START, "triage")
    g.add_edge("triage", "scatter")
    g.add_edge("scatter", "synthesize")
    g.add_conditional_edges(
        "synthesize",
        severity_gate,
        targets={"page_human": "page_human", "auto_mitigate": "auto_mitigate"},
    )
    g.add_edge("page_human", "auto_mitigate")
    g.add_edge("auto_mitigate", "postmortem")
    g.add_edge("postmortem", END)
    return g


# Driver.


def _print_postmortem(pm: Postmortem | None) -> None:
    print("\nPostmortem:")
    print("-" * 60)
    if pm is None:
        print("(missing)")
        return
    print(f"  Incident:        {pm.incident_id}")
    print(f"  Severity:        {pm.severity}")
    print(f"  Root cause:      {pm.root_cause_hypothesis}")
    print(f"  Mitigation:      {pm.mitigation_applied}")
    print(
        f"  Contributing:    " + ("\n                   ".join(pm.contributing_factors) or "(none)")
    )
    print(f"  Follow-ups:      " + "\n                   ".join(pm.follow_up_actions))


async def main() -> None:
    print("Notebook 58: Incident response runbook")
    print("=" * 60)

    model = get_model()
    graph = build_runbook()

    # No keyword pre-filter — severity comes entirely from the triage
    # agent's classification, so the human-approval branch only fires
    # when the model itself calls the incident critical.
    initial = {
        "incident_id": "INC-2026-0517",
        "symptom": (
            "API p99 latency is 8x its baseline and approximately 12% of "
            "requests return 503. The regression started at 04:12 UTC and is "
            "ongoing. Customer-facing checkout is degraded for paying users "
            "across multiple regions; on-call has been paged."
        ),
        "__model__": model,
    }

    print(f"\nIncident: {initial['symptom']}\n")

    result = await graph.execute(initial)

    if result.interrupt:
        # Critical path — workflow paused for human approval. Resume
        # with the operator's decision.
        payload = result.interrupt.interrupt.payload
        print(f"  ⏸  PAGED: {payload.get('question')}")
        print("  ▶  Operator responds: 'yes'")
        result = await graph.execute(
            Command(resume="yes", update={**result.final_state, "__model__": model})
        )

    print(
        f"\nWorkflow finished in {result.duration_ms:.0f} ms across "
        f"{result.iterations} graph iterations"
    )
    _print_postmortem(result.final_state.get("postmortem"))


if __name__ == "__main__":
    asyncio.run(main())
