# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Control how state updates combine instead of overwriting each other.

By default, when two nodes write to the same state field, the second
one wins. A reducer is a function attached to a field that says how to
merge an incoming update with the current value — append to a list,
sum numbers, merge dicts, keep the max, and so on.

- Annotated[type, reducer] on a Pydantic state schema declares the rule.
- Built-in reducers: add_messages, add_numbers, merge_dict, append_list, last_value.
- @reducer turns any (current, new) -> merged function into a custom reducer.
- Multiple reducers on one schema — each field merges independently.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_24_state_reducers.py

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
    """Without a reducer the last write wins — earlier values are lost."""
    print("=== Part 1: Why reducers exist ===\n")
    note = _llm_call("In one sentence, explain why a graph that overwrites state can lose data.")
    print(f"AI note: {note}")

    graph = StateGraph()

    async def node_a(inputs):
        return {"items": ["apple"]}

    async def node_b(inputs):
        # No reducer on "items" — this overwrites ["apple"] entirely.
        return {"items": ["banana"]}

    async def node_c(inputs):
        return {"items": ["cherry"]}

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)

    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)

    result = await graph.execute({})
    print(f"Without reducers: items = {result.final_state.get('items')}")
    print("  (Only 'cherry' - we lost 'apple' and 'banana'!)")
    print()


async def example_with_reducers():
    """Same graph, but `items` is annotated with the append_list reducer."""
    print("=== Part 1b: Same graph, with append_list ===\n")
    note = _llm_call("In one sentence, explain what a reducer does in a state graph.")
    print(f"AI note: {note}")

    class AppState(BaseModel):
        items: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=AppState)

    async def node_a(inputs):
        return {"items": ["apple"]}

    async def node_b(inputs):
        return {"items": ["banana"]}

    async def node_c(inputs):
        return {"items": ["cherry"]}

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)

    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)

    result = await graph.execute({})
    print(f"With append_list reducer: items = {result.final_state.get('items')}")
    print("  (All three items preserved!)")
    print()


# =============================================================================
# Part 2: Built-in reducers
# =============================================================================


async def example_builtin_reducers():
    """add_messages appends conversation turns, add_numbers sums numeric fields."""
    print("=== Part 2: Built-in reducers ===\n")
    note = _llm_call(
        "In one sentence, when would you use add_messages vs add_numbers in a Tulip graph?"
    )
    print(f"AI note: {note}")

    class ChatState(BaseModel):
        messages: Annotated[list, add_messages] = []
        total_tokens: Annotated[int, add_numbers] = 0

    graph = StateGraph(state_schema=ChatState)

    async def user_turn(inputs):
        return {
            "messages": [Message.user("Hello!")],
            "total_tokens": 5,
        }

    async def assistant_turn(inputs):
        return {
            "messages": [Message.assistant("Hi there!")],
            "total_tokens": 8,
        }

    graph.add_node("user", user_turn)
    graph.add_node("assistant", assistant_turn)
    graph.add_edge(START, "user")
    graph.add_edge("user", "assistant")
    graph.add_edge("assistant", END)

    result = await graph.execute({})

    print("add_messages reducer:")
    messages = result.final_state.get("messages", [])
    for msg in messages:
        print(f"  [{msg.role.value}] {msg.content}")

    print("\nadd_numbers reducer:")
    print(f"  total_tokens = {result.final_state.get('total_tokens')}")
    print()


async def example_merge_dict():
    """merge_dict shallow-merges incoming keys into the existing dict."""
    print("=== Part 2b: merge_dict ===\n")
    note = _llm_call("In one sentence, give an example use-case for merge_dict.")
    print(f"AI note: {note}")

    class ConfigState(BaseModel):
        config: Annotated[dict, merge_dict] = {}

    graph = StateGraph(state_schema=ConfigState)

    async def set_defaults(inputs):
        return {"config": {"debug": False, "timeout": 30, "retries": 3}}

    async def override_debug(inputs):
        return {"config": {"debug": True}}

    async def override_timeout(inputs):
        return {"config": {"timeout": 60}}

    graph.add_node("defaults", set_defaults)
    graph.add_node("debug", override_debug)
    graph.add_node("timeout", override_timeout)

    graph.add_edge(START, "defaults")
    graph.add_edge("defaults", "debug")
    graph.add_edge("debug", "timeout")
    graph.add_edge("timeout", END)

    result = await graph.execute({})
    print(f"Final config: {result.final_state.get('config')}")
    print("  (All settings merged together)")
    print()


# =============================================================================
# Part 3: Custom reducers
# =============================================================================


async def example_custom_reducer():
    """The @reducer decorator wraps any (current, new) -> merged function."""
    print("=== Part 3: Custom reducers ===\n")
    note = _llm_call("In one sentence, name two cases where a custom reducer beats add_messages.")
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

    class GameState(BaseModel):
        high_score: Annotated[int, max_value] = 0
        achievements: Annotated[list, unique_append] = []

    graph = StateGraph(state_schema=GameState)

    async def level_1(inputs):
        return {"high_score": 100, "achievements": ["first_step"]}

    async def level_2(inputs):
        return {"high_score": 50, "achievements": ["first_step", "speedrun"]}

    async def level_3(inputs):
        return {"high_score": 200, "achievements": ["speedrun", "perfectionist"]}

    graph.add_node("level1", level_1)
    graph.add_node("level2", level_2)
    graph.add_node("level3", level_3)

    graph.add_edge(START, "level1")
    graph.add_edge("level1", "level2")
    graph.add_edge("level2", "level3")
    graph.add_edge("level3", END)

    result = await graph.execute({})
    print(f"High score (max): {result.final_state.get('high_score')}")
    print(f"Achievements (unique): {result.final_state.get('achievements')}")
    print()


# =============================================================================
# Part 4: last_value
# =============================================================================


async def example_last_value():
    """last_value spells out the default behaviour: take the latest write."""
    print("=== Part 4: last_value ===\n")
    note = _llm_call(
        "In one sentence, what kind of field is a good fit for the last_value reducer?"
    )
    print(f"AI note: {note}")

    class ProcessState(BaseModel):
        status: Annotated[str, last_value] = "pending"
        log: Annotated[list, append_list] = []

    graph = StateGraph(state_schema=ProcessState)

    async def step1(inputs):
        return {"status": "processing", "log": ["Step 1 complete"]}

    async def step2(inputs):
        return {"status": "validating", "log": ["Step 2 complete"]}

    async def step3(inputs):
        return {"status": "done", "log": ["Step 3 complete"]}

    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    result = await graph.execute({})
    print(f"Status (last value): {result.final_state.get('status')}")
    print(f"Log (accumulated): {result.final_state.get('log')}")
    print()


# =============================================================================
# Part 5: Mixing reducers on one schema
# =============================================================================


async def example_complex_state():
    """An order schema where each field merges differently."""
    print("=== Part 5: Mixing reducers on one schema ===\n")
    note = _llm_call(
        "In one sentence, explain why combining append_list, add_numbers, and "
        "merge_dict reducers is useful for an order-processing graph."
    )
    print(f"AI note: {note}")

    class OrderState(BaseModel):
        items: Annotated[list, append_list] = []
        total: Annotated[float, add_numbers] = 0.0
        discounts: Annotated[dict, merge_dict] = {}
        status: Annotated[str, last_value] = "new"
        messages: Annotated[list, add_messages] = []

    graph = StateGraph(state_schema=OrderState)

    async def add_item(inputs):
        return {
            "items": [{"name": "Laptop", "price": 999.99}],
            "total": 999.99,
            "status": "items_added",
            "messages": [Message.system("Item added: Laptop")],
        }

    async def add_another(inputs):
        return {
            "items": [{"name": "Mouse", "price": 49.99}],
            "total": 49.99,
            "status": "items_added",
            "messages": [Message.system("Item added: Mouse")],
        }

    async def apply_discount(inputs):
        discount_amount = inputs.get("total", 0) * 0.1
        return {
            "discounts": {"loyalty": discount_amount},
            # add_numbers will sum this in — a negative delta acts like a subtraction.
            "total": -discount_amount,
            "status": "discount_applied",
            "messages": [Message.system(f"10% loyalty discount: -${discount_amount:.2f}")],
        }

    async def finalize(inputs):
        return {
            "status": "finalized",
            "messages": [Message.system(f"Order total: ${inputs.get('total', 0):.2f}")],
        }

    graph.add_node("add_item", add_item)
    graph.add_node("add_another", add_another)
    graph.add_node("discount", apply_discount)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "add_item")
    graph.add_edge("add_item", "add_another")
    graph.add_edge("add_another", "discount")
    graph.add_edge("discount", "finalize")
    graph.add_edge("finalize", END)

    result = await graph.execute({})

    print("Final Order State:")
    print(f"  Items: {len(result.final_state.get('items', []))} items")
    print(f"  Total: ${result.final_state.get('total', 0):.2f}")
    print(f"  Discounts: {result.final_state.get('discounts')}")
    print(f"  Status: {result.final_state.get('status')}")
    print(f"  Messages: {len(result.final_state.get('messages', []))} entries")
    print()


# =============================================================================
# Part 6: add_messages with two LLM-producing nodes
# =============================================================================


async def example_reducer_with_llm():
    """Both nodes generate text and append a Message — add_messages keeps both."""
    print("=== Part 6: add_messages with two LLM-producing nodes ===\n")

    class ChatLog(BaseModel):
        messages: Annotated[list, add_messages] = []

    graph = StateGraph(state_schema=ChatLog)

    import time as _t

    async def headline(_inputs):
        agent = Agent(
            model=get_model(max_tokens=40),
            system_prompt="You write punchy one-line product headlines.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync("Write a headline for an SDK that orchestrates AI agents.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call (headline): {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"messages": [Message.assistant(f"[headline] {result.message.strip()}")]}

    async def tagline(_inputs):
        agent = Agent(
            model=get_model(max_tokens=40),
            system_prompt="You write 6-word taglines.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync("Tagline for a multi-agent reasoning SDK.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call (tagline):  {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"messages": [Message.assistant(f"[tagline] {result.message.strip()}")]}

    graph.add_node("headline", headline)
    graph.add_node("tagline", tagline)
    graph.add_edge(START, "headline")
    graph.add_edge("headline", "tagline")
    graph.add_edge("tagline", END)

    result = await graph.execute({})
    for msg in result.final_state.get("messages", []):
        print(f"  {msg.content}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 19: State reducers")
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
    print("Next: Notebook 20 — Human-in-the-loop")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
