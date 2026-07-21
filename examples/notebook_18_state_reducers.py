# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 18: Merging multi-check payment signals with state reducers.

By default, when two nodes write to the same state field, the second
one wins — which means the fraud screen's signals vanish the moment the
AVS check reports. A reducer is a function attached to a field that says
how to merge an incoming update with the current value, so every check's
signals land in one authorization state. The signals here span a
velocity spike, an AVS mismatch, and a failed 3-D Secure challenge.

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


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 60
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: Why reducers exist
# =============================================================================


async def example_without_reducers():
    """Without a reducer the last write wins — earlier signals are lost."""
    print("=== Part 1: Why reducers exist ===\n")
    note = await _llm_call(
        "In one sentence, explain why a payment screening graph that overwrites signals loses evidence."
    )
    print(f"AI note: {note}")

    graph = StateGraph()

    async def fraud_check(inputs):
        return {"signals": ["Velocity: 8 charges in 2 minutes"]}

    async def avs_check(inputs):
        # No reducer on "signals" — this overwrites the fraud signal entirely.
        return {"signals": ["AVS mismatch: billing ZIP does not match"]}

    async def threeds_check(inputs):
        return {"signals": ["3-D Secure authentication failed"]}

    graph.add_node("fraud", fraud_check)
    graph.add_node("avs", avs_check)
    graph.add_node("threeds", threeds_check)

    graph.add_edge(START, "fraud")
    graph.add_edge("fraud", "avs")
    graph.add_edge("avs", "threeds")
    graph.add_edge("threeds", END)

    result = await graph.execute({})
    print(f"Without reducers: signals = {result.final_state.get('signals')}")
    print("  (Only the 3-D Secure signal - we lost the fraud and AVS signals!)")
    print()


async def example_with_reducers():
    """Same graph, but `signals` is annotated with the append_list reducer."""
    print("=== Part 1b: Same graph, with append_list ===\n")
    note = await _llm_call(
        "In one sentence, explain what a reducer does in a payment authorization graph."
    )
    print(f"AI note: {note}")

    class ScreenState(BaseModel):
        signals: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=ScreenState)

    async def fraud_check(inputs):
        return {"signals": ["Velocity: 8 charges in 2 minutes"]}

    async def avs_check(inputs):
        return {"signals": ["AVS mismatch: billing ZIP does not match"]}

    async def threeds_check(inputs):
        return {"signals": ["3-D Secure authentication failed"]}

    graph.add_node("fraud", fraud_check)
    graph.add_node("avs", avs_check)
    graph.add_node("threeds", threeds_check)

    graph.add_edge(START, "fraud")
    graph.add_edge("fraud", "avs")
    graph.add_edge("avs", "threeds")
    graph.add_edge("threeds", END)

    result = await graph.execute({})
    print(f"With append_list reducer: signals = {result.final_state.get('signals')}")
    print("  (All three checks' signals preserved!)")
    print()


# =============================================================================
# Part 2: Built-in reducers
# =============================================================================


async def example_builtin_reducers():
    """add_messages appends case-log turns, add_numbers sums numeric fields."""
    print("=== Part 2: Built-in reducers ===\n")
    note = await _llm_call(
        "In one sentence, when would you use add_messages vs add_numbers in a Tulip payment graph?"
    )
    print(f"AI note: {note}")

    class CaseLogState(BaseModel):
        messages: Annotated[list, add_messages] = []
        flagged_txn_count: Annotated[int, add_numbers] = 0

    graph = StateGraph(state_schema=CaseLogState)

    async def analyst_turn(inputs):
        return {
            "messages": [Message.user("New chargeback on account A-204 — anything related?")],
            "flagged_txn_count": 2,
        }

    async def reviewer_turn(inputs):
        return {
            "messages": [Message.assistant("Correlated: same card BIN as case 1042.")],
            "flagged_txn_count": 3,
        }

    graph.add_node("analyst", analyst_turn)
    graph.add_node("reviewer", reviewer_turn)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "reviewer")
    graph.add_edge("reviewer", END)

    result = await graph.execute({})

    print("add_messages reducer:")
    messages = result.final_state.get("messages", [])
    for msg in messages:
        print(f"  [{msg.role.value}] {msg.content}")

    print("\nadd_numbers reducer:")
    print(f"  flagged_txn_count = {result.final_state.get('flagged_txn_count')}")
    print()


async def example_merge_dict():
    """merge_dict shallow-merges incoming keys into the existing dict."""
    print("=== Part 2b: merge_dict ===\n")
    note = await _llm_call("In one sentence, give a payments use-case for merge_dict.")
    print(f"AI note: {note}")

    class GatewayConfigState(BaseModel):
        gateway_config: Annotated[dict, merge_dict] = {}

    graph = StateGraph(state_schema=GatewayConfigState)

    async def set_defaults(inputs):
        return {"gateway_config": {"step_up_3ds": False, "timeout": 30, "retries": 3}}

    async def enable_step_up(inputs):
        return {"gateway_config": {"step_up_3ds": True}}

    async def extend_timeout(inputs):
        return {"gateway_config": {"timeout": 60}}

    graph.add_node("defaults", set_defaults)
    graph.add_node("step_up", enable_step_up)
    graph.add_node("timeout", extend_timeout)

    graph.add_edge(START, "defaults")
    graph.add_edge("defaults", "step_up")
    graph.add_edge("step_up", "timeout")
    graph.add_edge("timeout", END)

    result = await graph.execute({})
    print(f"Final gateway config: {result.final_state.get('gateway_config')}")
    print("  (All settings merged together)")
    print()


# =============================================================================
# Part 3: Custom reducers
# =============================================================================


async def example_custom_reducer():
    """The @reducer decorator wraps any (current, new) -> merged function."""
    print("=== Part 3: Custom reducers ===\n")
    note = await _llm_call("In one sentence, name two cases where a custom reducer beats append_list.")
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

    class AuthorizationState(BaseModel):
        peak_fraud_score: Annotated[int, max_value] = 0
        flagged_cards: Annotated[list, unique_append] = []

    graph = StateGraph(state_schema=AuthorizationState)

    async def fraud_engine(inputs):
        return {"peak_fraud_score": 70, "flagged_cards": ["411111******1111"]}

    async def device_check(inputs):
        return {"peak_fraud_score": 45, "flagged_cards": ["411111******1111", "511111******2222"]}

    async def network_intel(inputs):
        return {"peak_fraud_score": 90, "flagged_cards": ["511111******2222", "601111******3333"]}

    graph.add_node("fraud", fraud_engine)
    graph.add_node("device", device_check)
    graph.add_node("intel", network_intel)

    graph.add_edge(START, "fraud")
    graph.add_edge("fraud", "device")
    graph.add_edge("device", "intel")
    graph.add_edge("intel", END)

    result = await graph.execute({})
    print(f"Peak fraud score (max): {result.final_state.get('peak_fraud_score')}")
    print(f"Flagged cards (unique): {result.final_state.get('flagged_cards')}")
    print()


# =============================================================================
# Part 4: last_value
# =============================================================================


async def example_last_value():
    """last_value spells out the default behaviour: take the latest write."""
    print("=== Part 4: last_value ===\n")
    note = await _llm_call(
        "In one sentence, what kind of authorization field fits the last_value reducer?"
    )
    print(f"AI note: {note}")

    class CaseState(BaseModel):
        status: Annotated[str, last_value] = "new"
        timeline: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=CaseState)

    async def screen(inputs):
        return {"status": "screening", "timeline": ["Risk screening complete"]}

    async def review(inputs):
        return {"status": "reviewing", "timeline": ["Manual review complete"]}

    async def decide(inputs):
        return {"status": "declined", "timeline": ["Decline decision logged"]}

    graph.add_node("screen", screen)
    graph.add_node("review", review)
    graph.add_node("decide", decide)

    graph.add_edge(START, "screen")
    graph.add_edge("screen", "review")
    graph.add_edge("review", "decide")
    graph.add_edge("decide", END)

    result = await graph.execute({})
    print(f"Status (last value): {result.final_state.get('status')}")
    print(f"Timeline (accumulated): {result.final_state.get('timeline')}")
    print()


# =============================================================================
# Part 5: Mixing reducers on one schema
# =============================================================================


async def example_complex_state():
    """An authorization schema where each field merges differently."""
    print("=== Part 5: Mixing reducers on one schema ===\n")
    note = await _llm_call(
        "In one sentence, explain why combining append_list, add_numbers, and "
        "merge_dict reducers is useful when merging multi-check payment output."
    )
    print(f"AI note: {note}")

    class AuthorizationState(BaseModel):
        signals: Annotated[list, append_list] = []
        risk_score: Annotated[float, add_numbers] = 0.0
        adjustments: Annotated[dict, merge_dict] = {}
        status: Annotated[str, last_value] = "new"
        messages: Annotated[list, add_messages] = []

    graph = StateGraph(state_schema=AuthorizationState)

    async def record_velocity(inputs):
        return {
            "signals": [{"source": "fraud", "title": "Velocity: 8 charges in 2 minutes"}],
            "risk_score": 40.0,
            "status": "signals_recorded",
            "messages": [Message.system("Evidence added: velocity spike on account A-204")],
        }

    async def record_avs(inputs):
        return {
            "signals": [{"source": "avs", "title": "AVS mismatch: billing ZIP"}],
            "risk_score": 25.0,
            "status": "signals_recorded",
            "messages": [Message.system("Evidence added: AVS mismatch on billing ZIP")],
        }

    async def review_false_positives(inputs):
        deduction = inputs.get("risk_score", 0) * 0.1
        return {
            "adjustments": {"allowlist_review": deduction},
            # add_numbers will sum this in — a negative delta acts like a subtraction.
            "risk_score": -deduction,
            "status": "allowlist_reviewed",
            "messages": [Message.system(f"10% allowlist deduction: -{deduction:.2f}")],
        }

    async def finalize(inputs):
        return {
            "status": "finalized",
            "messages": [Message.system(f"Case risk score: {inputs.get('risk_score', 0):.2f}")],
        }

    graph.add_node("record_velocity", record_velocity)
    graph.add_node("record_avs", record_avs)
    graph.add_node("allowlist_review", review_false_positives)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "record_velocity")
    graph.add_edge("record_velocity", "record_avs")
    graph.add_edge("record_avs", "allowlist_review")
    graph.add_edge("allowlist_review", "finalize")
    graph.add_edge("finalize", END)

    result = await graph.execute({})

    print("Final Authorization State:")
    print(f"  Signals: {len(result.final_state.get('signals', []))} signals")
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
            system_prompt="You write punchy one-line payment-fraud case summaries.",
        )
        t0 = _t.perf_counter()
        result = await agent.arun("Summarize a case merging velocity, AVS, and 3-D Secure signals.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call (summary): {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"messages": [Message.assistant(f"[summary] {result.message.strip()}")]}

    async def recommendation(_inputs):
        agent = Agent(
            model=get_model(max_tokens=40),
            system_prompt="You write 6-word transaction-disposition recommendations.",
        )
        t0 = _t.perf_counter()
        result = await agent.arun("Recommendation for a multi-check suspected-fraud transaction.")
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
    print("Notebook 18: Merging payment check signals with state reducers")
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
    print("Next: Notebook 19 — Payment hold approval gates (human-in-the-loop)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
