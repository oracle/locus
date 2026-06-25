# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 15: bounded recon — deciding when the enumeration loop stops.

A recon agent that never stops is a liability: a subdomain-enumeration
sweep (ATT&CK T1595, Active Scanning) must end after a fixed budget, a
time limit, or as soon as the agent declares its findings complete —
both to bound cost and to keep the activity inside an authorized scope.
Tulip ships a handful of small predicates that you compose with ``|``
(OR) and ``&`` (AND) to describe exactly when the loop should end. This
notebook also covers two related features: ``output_key`` (auto-save the
final verdict into the agent's state metadata) and a callable
``system_prompt`` that picks its text from runtime context.

Key ideas:
- Termination predicates: ``MaxIterations``, ``TextMention``,
  ``TokenLimit``, ``TimeLimit``, ``ConfidenceMet``, plus
  ``CustomCondition(callable)`` for anything else.
- Combine them: ``MaxIterations(5) | TextMention("RECON_COMPLETE")``
  stops on either; ``MaxIterations(3) & TokenLimit(1000)`` stops only
  when both budgets are met.
- ``output_key="verdict"`` tells the agent to write ``result.message``
  into ``result.state.metadata["verdict"]`` — handy for handing data
  between agents without parsing prose.
- ``system_prompt`` can be a callable ``ctx -> str``; Tulip calls it
  with the runtime context (including ``metadata``) on every turn.

Run it:
    .venv/bin/python examples/notebook_15_termination.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs; OpenAI and Anthropic
also work.
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.core.termination import (
    ConfidenceMet,
    CustomCondition,
    MaxIterations,
    TextMention,
    TimeLimit,
    TokenLimit,
)


# =============================================================================
# Part 1: composing termination predicates with | and &
# =============================================================================


def example_termination():
    """Build OR / AND combinations of stop predicates and probe them by hand."""
    print("=== Part 1: Composable Termination ===\n")

    from tulip.core.messages import Message
    from tulip.core.state import AgentState

    # OR — stop the sweep when either predicate fires.
    condition = MaxIterations(5) | TextMention("RECON_COMPLETE")
    print("MaxIterations(5) | TextMention('RECON_COMPLETE')")

    state = AgentState(agent_id="recon").with_iteration(6)
    stop, reason = condition.check(state)
    print(f"  Sweep 6: stop={stop}, reason={reason}")

    state2 = AgentState(agent_id="recon").with_message(
        Message.assistant("Subdomain sweep finished. RECON_COMPLETE")
    )
    stop2, reason2 = condition.check(state2)
    print(f"  Message 'RECON_COMPLETE': stop={stop2}, reason={reason2}")

    # AND — stop only when both budget predicates fire.
    condition2 = MaxIterations(3) & TokenLimit(1000)
    print("\nMaxIterations(3) & TokenLimit(1000)")

    state3 = AgentState(agent_id="recon").with_iteration(4)
    stop3, _ = condition2.check(state3)
    print(f"  Iterations met, tokens not: stop={stop3}")

    state4 = state3.with_token_usage(prompt_tokens=600, completion_tokens=500)
    stop4, reason4 = condition2.check(state4)
    print(f"  Both met: stop={stop4}, reason={reason4}")

    # Roll your own predicate with CustomCondition.
    custom = CustomCondition(lambda state, **ctx: (state.iteration > 10, "scan_budget_exhausted"))
    print(f"\nCustomCondition: {custom.check(AgentState(agent_id='r').with_iteration(11))}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one short sentence.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "In one sentence, why should a subdomain-enumeration agent compose stop "
        "conditions (MaxIterations | TextMention) instead of hard-coding a single "
        "stop check inside the Agent?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


# =============================================================================
# Part 2: output_key — store the verdict at a known key
# =============================================================================


def example_output_key():
    """Set output_key='verdict' and the final message lands in state.metadata['verdict']."""
    print("\n=== Part 2: output_key ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a SOC triage assistant. Answer in one word.",
            max_iterations=3,
            model=model,
            output_key="verdict",
        )
    )

    result = agent.run_sync("VerificationResult for an EICAR test file found on a developer laptop?")
    print(f"Response: {result.message}")
    print(f"State metadata['verdict']: {result.state.metadata.get('verdict')}")
    print("Downstream agents read state.metadata['verdict'] directly — no parsing.")


# =============================================================================
# Part 3: a callable system prompt
# =============================================================================


def example_dynamic_prompt():
    """System prompt is a function of runtime context.metadata."""
    print("\n=== Part 3: Dynamic System Prompt ===\n")

    model = get_model()

    def my_prompt(context):
        role = context.get("metadata", {}).get("role", "SOC analyst")
        audience = context.get("metadata", {}).get("audience", "the security team")
        return f"You are a {role}. Write for {audience}. Be concise."

    agent = Agent(
        config=AgentConfig(
            system_prompt=my_prompt,
            max_iterations=3,
            model=model,
        )
    )

    # Different metadata → different system prompt → different behaviour.
    r1 = agent.run_sync(
        "Summarize: 3 failed logins then a success from 198.51.100.7.",
        metadata={"role": "tier-1 SOC analyst"},
    )
    print(f"Tier-1 analyst: {r1.message}")

    r2 = agent.run_sync(
        "What does 'lateral movement' mean in an incident report?",
        metadata={"role": "incident responder", "audience": "non-technical executives"},
    )
    print(f"Incident responder (for executives): {r2.message[:100]}")


if __name__ == "__main__":
    example_termination()
    example_output_key()
    example_dynamic_prompt()
