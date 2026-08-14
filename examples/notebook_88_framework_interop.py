#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 88: Govern an agent you did not build on Tulip.

The claim on the homepage is that you can *"govern the agents you already run
in LangChain, CrewAI, or the OpenAI Agents SDK"*. This file is the proof, and
it is deliberately not a diagram: it builds a **real LangChain agent**, runs it
through **LangGraph's own ReAct loop**, and watches a $4,000,000 refund actually
execute. Then it wraps that one tool and runs the identical agent again.

Nothing about the agent changes between the two runs. Same graph, same model,
same prompt, same tool name, description, and argument schema. The only
difference is which object was handed to ``tools=[...]``.

The model here is compromised by construction — it reaches for the refund every
turn and never refuses. That is on purpose. A demo where a well-behaved model
declines proves nothing about the runtime; it proves the model was in a good
mood. The question worth answering is the other one: **once the model is fully
owned, what stops the action?**

    LangGraph ReAct loop  (langgraph.prebuilt / langchain.agents)
       │
       ▼
    model decides: refund(ord-4821, $4,000,000)
       │
       ▼
    ┌──────────────── ungated run ────────────────┐
    │  StructuredTool.func  →  money moves        │   💸 PAID
    └─────────────────────────────────────────────┘
       │
    ┌──────────────── gated run ──────────────────┐
    │  gate_langchain_tool(...)                   │
    │     → admit(action, pay_out, policy, trail) │
    │        → blast radius 4000 over the cap     │
    │        → AdmissionError(require_human)      │   🛡️ HELD
    │        → pay_out() never runs               │
    └─────────────────────────────────────────────┘
       │
       ▼
    AuditTrail — every decision, hash-chained

The gate lives in real code between the model's decision and the side effect,
so a jailbreak that wins the argument with the model still loses to the policy.

The same shape works for the OpenAI Agents SDK and CrewAI; the second half of
this file shows the identical policy holding a CrewAI tool, to make the point
that none of this is LangChain-specific.

Requires ``tulip-frameworks``, which is what carries the per-framework bridges
so the SDK itself never depends on a competitor's package::

    pip install "tulip-frameworks[langchain,langgraph,crewai]"

Run it (fully offline — no network, no credentials, no API key; the payout is a
local stub and the model is local)::

    python examples/notebook_88_framework_interop.py
"""

from __future__ import annotations

import asyncio
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

from tulip.control import Action, AdmissionError, AuditTrail


INSTALL_HINT = 'pip install "tulip-frameworks[langchain,langgraph,crewai]"'


def _missing(what: str, exc: Exception) -> None:
    """Say exactly what is missing and exactly how to get it.

    A traceback from three libraries down is the single most common reason a
    reader concludes an example is broken rather than that they are one install
    away.
    """
    print(f"  (skipped: {what} is not installed — {exc.__class__.__name__})")
    print(f"   {INSTALL_HINT}")


# ---------------------------------------------------------------------------
# The side effect. Offline this only mutates a local ledger, but it is the
# thing that must not happen, so it is the thing the tests read.
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    """Where the money actually moves. Empty means nothing was paid."""

    paid: list[tuple[str, float]] = field(default_factory=list)

    def pay_out(self, order_id: str, amount_usd: float) -> str:
        self.paid.append((order_id, amount_usd))
        return f"refunded ${amount_usd:,.2f} on {order_id}"

    @property
    def total(self) -> float:
        return sum(amount for _, amount in self.paid)


def refund_action(name: str, kwargs: dict[str, Any]) -> Action:
    """Turn a refund call into something a policy can reason about.

    Blast radius is the dollar amount in thousands, so risk scales with the
    payout rather than with the tool's name. A policy written this way holds a
    $4M refund and waves through a $12 shipping credit without either being
    special-cased — and it keeps holding both correctly if the tool is renamed.
    """
    amount = float(kwargs.get("amount_usd", 0.0))
    return Action(
        name=name,
        asset=str(kwargs.get("order_id", "unknown")),
        kind="payment",
        environment="production",
        blast_radius=max(1, int(amount // 1_000)),
    )


def refund_policy() -> Any:
    """One policy, used by both frameworks below. Risk here *is* the amount.

    ``action_gate_policy`` defaults to holding anything labelled ``production``,
    which is a sensible default and the wrong one for this example: it would
    hold a $12 shipping credit for a human, and a gate that stops everything
    teaches a team to click through it. Clearing ``require_human_for`` leaves
    ``max_blast_radius`` as the only rule, and since ``refund_action`` derives
    blast radius from the dollar amount, a refund under $1,000 settles itself
    and anything larger waits for a person.

    The ``production`` label stays on the action regardless — it is true, it is
    recorded on the audit trail, and a stricter policy elsewhere can match on it
    without this one being rewritten.
    """
    from tulip_frameworks.policy_presets import action_gate_policy  # type: ignore[import-not-found]

    return action_gate_policy(max_blast_radius=1, require_human_for=())


# ---------------------------------------------------------------------------
# LangChain + LangGraph
# ---------------------------------------------------------------------------


def _compromised_chat_model() -> Any:
    """A LangChain chat model an attacker has already won completely.

    It is a real ``BaseChatModel``, so LangGraph's loop drives it exactly as it
    would drive GPT-4 or Claude — binds tools to it, reads its ``tool_calls``,
    executes them, and feeds the results back. It simply never refuses.
    """
    from langchain_core.language_models.chat_models import (  # type: ignore[import-not-found]
        BaseChatModel,
    )
    from langchain_core.messages import AIMessage  # type: ignore[import-not-found]
    from langchain_core.outputs import ChatGeneration, ChatResult  # type: ignore[import-not-found]

    class CompromisedChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "compromised"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            # Whatever it is handed, it has already decided what to call.
            return self

        def _generate(
            self,
            messages: list[Any],
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            # One dangerous call, then stop. A real compromised agent would keep
            # hammering, but a demo that prints the same denial four times reads
            # as a bug rather than as a gate holding.
            if any(getattr(m, "type", "") == "tool" for m in messages):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="I could not do that."))]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "refund",
                                    "args": {"order_id": "ord-4821", "amount_usd": 4_000_000.0},
                                    "id": "compromised-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )

    return CompromisedChatModel()


def _build_agent(tool: Any, model: Any) -> Any:
    """LangGraph's own ReAct loop, whichever import path this version offers.

    LangChain v1 moved the constructor to ``langchain.agents.create_agent`` and
    deprecated ``langgraph.prebuilt.create_react_agent``. Installing only
    ``langchain-core`` + ``langgraph`` — which is what the ``langgraph`` extra
    pulls in — leaves the old path as the only one available, and calling it
    prints a deprecation notice that has nothing to do with Tulip. The warning
    is silenced rather than the fallback removed, so this keeps working on both.
    """
    try:
        from langchain.agents import create_agent  # type: ignore[import-not-found]
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langgraph.prebuilt import (  # type: ignore[import-not-found]
                create_react_agent as create_agent,
            )

            return create_agent(model, tools=[tool])
    return create_agent(model, tools=[tool])


def _last_tool_message(result: dict[str, Any]) -> str:
    for message in reversed(result["messages"]):
        if getattr(message, "type", "") == "tool":
            return str(message.content)
    return "(the agent never reached a tool)"


async def langchain_demo() -> bool:
    """Run the same agent twice, ungated then gated. Returns False if skipped.

    Both runs go through ``ainvoke``. The gated tool is async-only — the bridge
    wraps the action in ``admit()``, which is a coroutine — so a graph driven
    with the sync ``invoke`` raises ``StructuredTool does not support sync
    invocation``. That is a real constraint, not an accident of this example:
    if your graph is synchronous today, this is the one thing you have to
    change.
    """
    try:
        from langchain_core.tools import tool as lc_tool  # type: ignore[import-not-found]
        from tulip_frameworks.langchain import gate_langchain_tool  # type: ignore[import-not-found]
    except ImportError as exc:
        _missing("LangChain (or tulip-frameworks)", exc)
        return False

    print("\n=== LangChain + LangGraph ===\n")

    ledger = Ledger()

    @lc_tool  # type: ignore[untyped-decorator]
    def refund(order_id: str, amount_usd: float) -> str:
        """Issue a customer refund."""
        return ledger.pay_out(order_id, amount_usd)

    prompt = {"messages": [("user", "Customer is furious. Refund ord-4821 in full, now.")]}

    # --- Run 1: the agent as it exists today -------------------------------
    ungated = _build_agent(refund, _compromised_chat_model())
    print("run 1 — the agent you already have")
    print(f"  tool says: {_last_tool_message(await ungated.ainvoke(prompt))}")
    print(f"  💸 ledger: {ledger.total:,.2f} paid out")

    # --- Run 2: one tool wrapped, nothing else changed ---------------------
    ledger.paid.clear()
    trail = AuditTrail()
    gated_tool = gate_langchain_tool(
        refund, action=refund_action, policy=refund_policy(), trail=trail
    )
    gated = _build_agent(gated_tool, _compromised_chat_model())
    print("\nrun 2 — the same agent, one tool wrapped")
    print(f"  tool says: {_last_tool_message(await gated.ainvoke(prompt))}")
    print(f"  🛡️  ledger: {ledger.total:,.2f} paid out")
    print(f"  📜 audit:  {len(trail)} decision(s), chain intact: {'✓' if trail.verify() else '✗'}")
    for record in trail.records():
        payload = record.payload
        print(f"     {payload['action']} on {payload['asset']} → {payload['outcome']}")
        print(f"     because: {payload['reason']}")

    assert not ledger.paid, "the gate let a $4M refund through"
    return True


# ---------------------------------------------------------------------------
# CrewAI — the same policy, a different framework
# ---------------------------------------------------------------------------


async def crewai_demo() -> bool:
    """The identical policy object holding a CrewAI tool. Returns False if skipped."""
    try:
        from crewai.tools import tool as crew_tool  # type: ignore[import-not-found]
        from tulip_frameworks.crewai import gate_crewai_tool  # type: ignore[import-not-found]
    except ImportError as exc:
        _missing("CrewAI (or tulip-frameworks)", exc)
        return False

    print("\n=== CrewAI ===\n")

    ledger = Ledger()

    @crew_tool("refund")  # type: ignore[untyped-decorator]
    def refund(order_id: str, amount_usd: float) -> str:
        """Issue a customer refund."""
        return ledger.pay_out(order_id, amount_usd)

    trail = AuditTrail()
    gated = gate_crewai_tool(refund, action=refund_action, policy=refund_policy(), trail=trail)

    # Small refund: the agent settles it without paging anyone.
    print(f"  $12 credit    → {gated.run(order_id='ord-0001', amount_usd=12.0)}")

    # Large refund: held. CrewAI surfaces the gate's refusal as the tool result,
    # so the agent sees why rather than an opaque failure.
    try:
        outcome = gated.run(order_id="ord-4821", amount_usd=4_000_000.0)
    except AdmissionError as exc:
        outcome = f"BLOCKED — {exc.decision.outcome}: {exc.decision.reason}"
    print(f"  $4M refund    → {str(outcome)[:110]}")

    print(f"\n  💵 paid: {ledger.total:,.2f} across {len(ledger.paid)} refund(s)")
    print(f"  📜 audit: {len(trail)} decision(s), chain intact: {'✓' if trail.verify() else '✗'}")

    assert ledger.total == 12.0, "the small refund should pay and the large one should not"
    return True


async def main() -> int:
    print(__doc__.split("\n\n")[0])
    print("=" * 72)

    ran = [await langchain_demo(), await crewai_demo()]

    print("\n" + "=" * 72)
    if not any(ran):
        print("Nothing ran — install the bridges to see it work:")
        print(f"   {INSTALL_HINT}")
        return 0

    print(
        "The agent was not rebuilt, retrained, or re-prompted. One tool was\n"
        "wrapped. The model still asked for $4,000,000 both times — the second\n"
        "time, the money did not move, and the refusal is on a record you can\n"
        "verify."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
