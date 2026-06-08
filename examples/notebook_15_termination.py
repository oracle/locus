# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 16: deciding when an agent stops.

Every agent loop needs a termination condition. Tulip ships a handful of
small predicates that you compose with ``|`` (OR) and ``&`` (AND) to
describe exactly when the loop should end. This notebook also covers
two related features: ``output_key`` (auto-save the final message into
the agent's state metadata) and a callable ``system_prompt`` that picks
its text from runtime context.

Key ideas:
- Termination predicates: ``MaxIterations``, ``TextMention``,
  ``TokenLimit``, ``TimeLimit``, ``ConfidenceMet``, plus
  ``CustomCondition(callable)`` for anything else.
- Combine them: ``MaxIterations(5) | TextMention("DONE")`` stops on
  either; ``MaxIterations(3) & TokenLimit(1000)`` stops only when both
  are met.
- ``output_key="answer"`` tells the agent to write ``result.message``
  into ``result.state.metadata["answer"]`` — handy for handing data
  between agents without parsing prose.
- ``system_prompt`` can be a callable ``ctx -> str``; Tulip calls it
  with the runtime context (including ``metadata``) on every turn.

Run it:
    .venv/bin/python examples/notebook_21_termination.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs; OpenAI, Anthropic, and
Anthropic also works.
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

    # OR — stop when either predicate fires.
    condition = MaxIterations(5) | TextMention("DONE")
    print("MaxIterations(5) | TextMention('DONE')")

    state = AgentState(agent_id="test").with_iteration(6)
    stop, reason = condition.check(state)
    print(f"  Iteration 6: stop={stop}, reason={reason}")

    state2 = AgentState(agent_id="test").with_message(Message.assistant("All DONE"))
    stop2, reason2 = condition.check(state2)
    print(f"  Message 'DONE': stop={stop2}, reason={reason2}")

    # AND — stop only when both predicates fire.
    condition2 = MaxIterations(3) & TokenLimit(1000)
    print(f"\nMaxIterations(3) & TokenLimit(1000)")

    state3 = AgentState(agent_id="test").with_iteration(4)
    stop3, _ = condition2.check(state3)
    print(f"  Iterations met, tokens not: stop={stop3}")

    state4 = state3.with_token_usage(prompt_tokens=600, completion_tokens=500)
    stop4, reason4 = condition2.check(state4)
    print(f"  Both met: stop={stop4}, reason={reason4}")

    # Roll your own predicate with CustomCondition.
    custom = CustomCondition(lambda state, **ctx: (state.iteration > 10, "too_many_iterations"))
    print(f"\nCustomCondition: {custom.check(AgentState(agent_id='t').with_iteration(11))}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one short sentence.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "In one sentence, why is composable termination (MaxIterations | TextMention) "
        "better than hard-coding a single stop check inside an Agent?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


# =============================================================================
# Part 2: output_key — store the answer at a known key
# =============================================================================


def example_output_key():
    """Set output_key='answer' and the final message lands in state.metadata['answer']."""
    print("\n=== Part 2: output_key ===\n")

    model = get_model()

    agent = Agent(
        config=AgentConfig(
            system_prompt="Answer in one word.",
            max_iterations=3,
            model=model,
            output_key="answer",
        )
    )

    result = agent.run_sync("Capital of France?")
    print(f"Response: {result.message}")
    print(f"State metadata['answer']: {result.state.metadata.get('answer')}")
    print("Downstream agents read state.metadata['answer'] directly — no parsing.")


# =============================================================================
# Part 3: a callable system prompt
# =============================================================================


def example_dynamic_prompt():
    """System prompt is a function of runtime context.metadata."""
    print("\n=== Part 3: Dynamic System Prompt ===\n")

    model = get_model()

    def my_prompt(context):
        role = context.get("metadata", {}).get("role", "assistant")
        language = context.get("metadata", {}).get("language", "English")
        return f"You are a {role}. Respond in {language}. Be concise."

    agent = Agent(
        config=AgentConfig(
            system_prompt=my_prompt,
            max_iterations=3,
            model=model,
        )
    )

    # Different metadata → different system prompt → different behaviour.
    r1 = agent.run_sync("What is 7*8?", metadata={"role": "math teacher"})
    print(f"Math teacher: {r1.message}")

    r2 = agent.run_sync("What is gravity?", metadata={"role": "physicist", "language": "Spanish"})
    print(f"Physicist (Spanish): {r2.message[:100]}")


if __name__ == "__main__":
    example_termination()
    example_output_key()
    example_dynamic_prompt()
