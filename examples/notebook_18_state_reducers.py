# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 18: Merging multi-scanner findings with state reducers.

By default, when two nodes write to the same state field, the second
one wins — which means a SAST scanner's findings vanish the moment the
dependency scanner reports. A reducer is a function attached to a field
that says how to merge an incoming update with the current value, so
every scanner's findings land in one investigation state. The findings
here span SQL injection (CWE-89), a hardcoded secret (CWE-798), and a
vulnerable dependency.

- Annotated[type, reducer] on a Pydantic state schema declares the rule.
- Built-in reducers: add_messages, add_numbers, merge_dict, append_list, last_value.
- @reducer turns any (current, new) -> merged function into a custom reducer.
- Multiple reducers on one schema — each field merges independently.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_18_state_reducers.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time
from typing import Annotated

from config import get_model
from pydantic import BaseModel

from tulip.agent import Agent
from tulip.core import (
    Message,
    add_messages,
    add_numbers,
    append_list,
    last_value,
    merge_dict,
)
from tulip.core.reducers import reducer
from tulip.multiagent import END, START, StateGraph


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 60
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: Why reducers exist
# =============================================================================


async def example_without_reducers():
    """Without a reducer the last write wins — earlier findings are lost."""
    print("=== Part 1: Why reducers exist ===\n")
    note = _llm_call(
        "In one sentence, explain why a triage graph that overwrites findings loses evidence."
    )
    print(f"AI note: {note}")

    graph = StateGraph()

    async def sast_scan(inputs):
        return {"findings": ["SQL injection in login.py"]}

    async def deps_scan(inputs):
        # No reducer on "findings" — this overwrites the SAST finding entirely.
        return {"findings": ["CVE-2024-99999 in libfoo 1.2"]}

    async def secrets_scan(inputs):
        return {"findings": ["hardcoded API key in config.py"]}

    graph.add_node("sast", sast_scan)
    graph.add_node("deps", deps_scan)
    graph.add_node("secrets", secrets_scan)

    graph.add_edge(START, "sast")
    graph.add_edge("sast", "deps")
    graph.add_edge("deps", "secrets")
    graph.add_edge("secrets", END)

    result = await graph.execute({})
    print(f"Without reducers: findings = {result.final_state.get('findings')}")
    print("  (Only the secrets finding - we lost the SAST and dependency findings!)")
    print()


async def example_with_reducers():
    """Same graph, but `findings` is annotated with the append_list reducer."""
    print("=== Part 1b: Same graph, with append_list ===\n")
    note = _llm_call("In one sentence, explain what a reducer does in an investigation graph.")
    print(f"AI note: {note}")

    class ScanState(BaseModel):
        findings: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=ScanState)

    async def sast_scan(inputs):
        return {"findings": ["SQL injection in login.py"]}

    async def deps_scan(inputs):
        return {"findings": ["CVE-2024-99999 in libfoo 1.2"]}

    async def secrets_scan(inputs):
        return {"findings": ["hardcoded API key in config.py"]}

    graph.add_node("sast", sast_scan)
    graph.add_node("deps", deps_scan)
    graph.add_node("secrets", secrets_scan)

    graph.add_edge(START, "sast")
    graph.add_edge("sast", "deps")
    graph.add_edge("deps", "secrets")
    graph.add_edge("secrets", END)

    result = await graph.execute({})
    print(f"With append_list reducer: findings = {result.final_state.get('findings')}")
    print("  (All three scanners' findings preserved!)")
    print()


# =============================================================================
# Part 2: Built-in reducers
# =============================================================================


async def example_builtin_reducers():
    """add_messages appends case-log turns, add_numbers sums numeric fields."""
    print("=== Part 2: Built-in reducers ===\n")
    note = _llm_call(
        "In one sentence, when would you use add_messages vs add_numbers in a Tulip triage graph?"
    )
    print(f"AI note: {note}")

    class CaseLogState(BaseModel):
        messages: Annotated[list, add_messages] = []
        ioc_count: Annotated[int, add_numbers] = 0

    graph = StateGraph(state_schema=CaseLogState)

    async def analyst_turn(inputs):
        return {
            "messages": [Message.user("New alert on host WS-204 — anything related?")],
            "ioc_count": 2,
        }

    async def copilot_turn(inputs):
        return {
            "messages": [Message.assistant("Correlated: same sender domain as case 1042.")],
            "ioc_count": 3,
        }

    graph.add_node("analyst", analyst_turn)
    graph.add_node("copilot", copilot_turn)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "copilot")
    graph.add_edge("copilot", END)

    result = await graph.execute({})

    print("add_messages reducer:")
    messages = result.final_state.get("messages", [])
    for msg in messages:
        print(f"  [{msg.role.value}] {msg.content}")

    print("\nadd_numbers reducer:")
    print(f"  ioc_count = {result.final_state.get('ioc_count')}")
    print()


async def example_merge_dict():
    """merge_dict shallow-merges incoming keys into the existing dict."""
    print("=== Part 2b: merge_dict ===\n")
    note = _llm_call("In one sentence, give a security use-case for merge_dict.")
    print(f"AI note: {note}")

    class ScanConfigState(BaseModel):
        scan_config: Annotated[dict, merge_dict] = {}

    graph = StateGraph(state_schema=ScanConfigState)

    async def set_defaults(inputs):
        return {"scan_config": {"deep_scan": False, "timeout": 30, "retries": 3}}

    async def enable_deep_scan(inputs):
        return {"scan_config": {"deep_scan": True}}

    async def extend_timeout(inputs):
        return {"scan_config": {"timeout": 60}}

    graph.add_node("defaults", set_defaults)
    graph.add_node("deep_scan", enable_deep_scan)
    graph.add_node("timeout", extend_timeout)

    graph.add_edge(START, "defaults")
    graph.add_edge("defaults", "deep_scan")
    graph.add_edge("deep_scan", "timeout")
    graph.add_edge("timeout", END)

    result = await graph.execute({})
    print(f"Final scan config: {result.final_state.get('scan_config')}")
    print("  (All settings merged together)")
    print()


# =============================================================================
# Part 3: Custom reducers
# =============================================================================


async def example_custom_reducer():
    """The @reducer decorator wraps any (current, new) -> merged function."""
    print("=== Part 3: Custom reducers ===\n")
    note = _llm_call("In one sentence, name two cases where a custom reducer beats append_list.")
    print(f"AI note: {note}")

    @reducer
    def max_value(current: int, new: int) -> int:
        """Keep the larger of the two values."""
        return max(current or 0, new or 0)

    @reducer
    def unique_append(current: list, new: list) -> list:
        """Append items from `new` that aren't already in `current`."""
        result = list(current or [])
        for item in new or []:
            if item not in result:
                result.append(item)
        return result

    class InvestigationState(BaseModel):
        peak_risk_score: Annotated[int, max_value] = 0
        observed_iocs: Annotated[list, unique_append] = []

    graph = StateGraph(state_schema=InvestigationState)

    async def sast_scanner(inputs):
        return {"peak_risk_score": 70, "observed_iocs": ["192.0.2.10"]}

    async def edr_scanner(inputs):
        return {"peak_risk_score": 45, "observed_iocs": ["192.0.2.10", "192.0.2.22"]}

    async def intel_scanner(inputs):
        return {"peak_risk_score": 90, "observed_iocs": ["192.0.2.22", "198.51.100.5"]}

    graph.add_node("sast", sast_scanner)
    graph.add_node("edr", edr_scanner)
    graph.add_node("intel", intel_scanner)

    graph.add_edge(START, "sast")
    graph.add_edge("sast", "edr")
    graph.add_edge("edr", "intel")
    graph.add_edge("intel", END)

    result = await graph.execute({})
    print(f"Peak risk score (max): {result.final_state.get('peak_risk_score')}")
    print(f"Observed IOCs (unique): {result.final_state.get('observed_iocs')}")
    print()


# =============================================================================
# Part 4: last_value
# =============================================================================


async def example_last_value():
    """last_value spells out the default behaviour: take the latest write."""
    print("=== Part 4: last_value ===\n")
    note = _llm_call(
        "In one sentence, what kind of investigation field fits the last_value reducer?"
    )
    print(f"AI note: {note}")

    class CaseState(BaseModel):
        status: Annotated[str, last_value] = "new"
        timeline: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=CaseState)

    async def collect(inputs):
        return {"status": "collecting", "timeline": ["Evidence collection complete"]}

    async def analyze(inputs):
        return {"status": "analyzing", "timeline": ["Scanner correlation complete"]}

    async def contain(inputs):
        return {"status": "contained", "timeline": ["Containment actions logged"]}

    graph.add_node("collect", collect)
    graph.add_node("analyze", analyze)
    graph.add_node("contain", contain)

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "analyze")
    graph.add_edge("analyze", "contain")
    graph.add_edge("contain", END)

    result = await graph.execute({})
    print(f"Status (last value): {result.final_state.get('status')}")
    print(f"Timeline (accumulated): {result.final_state.get('timeline')}")
    print()


# =============================================================================
# Part 5: Mixing reducers on one schema
# =============================================================================


async def example_complex_state():
    """An investigation schema where each field merges differently."""
    print("=== Part 5: Mixing reducers on one schema ===\n")
    note = _llm_call(
        "In one sentence, explain why combining append_list, add_numbers, and "
        "merge_dict reducers is useful when merging multi-scanner output."
    )
    print(f"AI note: {note}")

    class InvestigationState(BaseModel):
        findings: Annotated[list, append_list] = []
        risk_score: Annotated[float, add_numbers] = 0.0
        adjustments: Annotated[dict, merge_dict] = {}
        status: Annotated[str, last_value] = "new"
        messages: Annotated[list, add_messages] = []

    graph = StateGraph(state_schema=InvestigationState)

    async def record_sast(inputs):
        return {
            "findings": [{"source": "SAST", "title": "SQL injection in login.py"}],
            "risk_score": 40.0,
            "status": "findings_recorded",
            "messages": [Message.system("Finding added: SQL injection in login.py")],
        }

    async def record_deps(inputs):
        return {
            "findings": [{"source": "deps", "title": "CVE-2024-99999 in libfoo 1.2"}],
            "risk_score": 25.0,
            "status": "findings_recorded",
            "messages": [Message.system("Finding added: CVE-2024-99999 in libfoo 1.2")],
        }

    async def review_false_positives(inputs):
        deduction = inputs.get("risk_score", 0) * 0.1
        return {
            "adjustments": {"fp_review": deduction},
            # add_numbers will sum this in — a negative delta acts like a subtraction.
            "risk_score": -deduction,
            "status": "fp_reviewed",
            "messages": [Message.system(f"10% false-positive deduction: -{deduction:.2f}")],
        }

    async def finalize(inputs):
        return {
            "status": "finalized",
            "messages": [Message.system(f"Case risk score: {inputs.get('risk_score', 0):.2f}")],
        }

    graph.add_node("record_sast", record_sast)
    graph.add_node("record_deps", record_deps)
    graph.add_node("fp_review", review_false_positives)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "record_sast")
    graph.add_edge("record_sast", "record_deps")
    graph.add_edge("record_deps", "fp_review")
    graph.add_edge("fp_review", "finalize")
    graph.add_edge("finalize", END)

    result = await graph.execute({})

    print("Final Investigation State:")
    print(f"  Findings: {len(result.final_state.get('findings', []))} findings")
    print(f"  Risk score: {result.final_state.get('risk_score', 0):.2f}")
    print(f"  Adjustments: {result.final_state.get('adjustments')}")
    print(f"  Status: {result.final_state.get('status')}")
    print(f"  Messages: {len(result.final_state.get('messages', []))} audit entries")
    print()


# =============================================================================
# Part 6: add_messages with two LLM-producing nodes
# =============================================================================


async def example_reducer_with_llm():
    """Both nodes generate text and append a Message — add_messages keeps both."""
    print("=== Part 6: add_messages with two LLM-producing nodes ===\n")

    class CaseLog(BaseModel):
        messages: Annotated[list, add_messages] = []

    graph = StateGraph(state_schema=CaseLog)

    import time as _t

    async def summary(_inputs):
        agent = Agent(
            model=get_model(max_tokens=40),
            system_prompt="You write punchy one-line incident summaries.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync("Summarize an incident merging SAST, EDR, and intel findings.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call (summary): {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"messages": [Message.assistant(f"[summary] {result.message.strip()}")]}

    async def recommendation(_inputs):
        agent = Agent(
            model=get_model(max_tokens=40),
            system_prompt="You write 6-word containment recommendations.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync("Recommendation for a multi-scanner credential-theft case.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call (recommendation):  {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"messages": [Message.assistant(f"[recommendation] {result.message.strip()}")]}

    graph.add_node("summary", summary)
    graph.add_node("recommendation", recommendation)
    graph.add_edge(START, "summary")
    graph.add_edge("summary", "recommendation")
    graph.add_edge("recommendation", END)

    result = await graph.execute({})
    for msg in result.final_state.get("messages", []):
        print(f"  {msg.content}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 18: Merging scanner findings with state reducers")
    print("=" * 60)
    print()

    await example_without_reducers()
    await example_with_reducers()
    await example_builtin_reducers()
    await example_merge_dict()
    await example_custom_reducer()
    await example_last_value()
    await example_complex_state()
    await example_reducer_with_llm()

    print("=" * 60)
    print("Next: Notebook 19 — Containment approval gates (human-in-the-loop)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
