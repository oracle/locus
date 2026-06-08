# Reasoning

A model that loops without thinking just pays you to be wrong faster.
Tulip ships three reasoning
add-ons. Each catches a different class of mistake *before* the next
tool call:

- **Reflexion** catches *wrong premises* — the agent self-evaluates
  after each turn and re-plans if the last step was a dead end,
  instead of stacking another tool call on top.
- **Grounding** catches *hallucinations* — before the answer ships,
  every factual claim is checked against the tool results that
  produced it; unsupported claims are dropped or sent back.
- **Causal reasoning** catches *contradictions* — a running
  cause-effect graph surfaces the case where turn 3's "fix"
  contradicts turn 1's "root cause", which linear chat history hides.

Each is a single argument on `Agent(...)`. You can combine them.

## When to pick which

| Situation | Add-on |
|---|---|
| Agent loops endlessly or stacks tool calls on a wrong premise | `reflexion=True` |
| Customer-facing answers where hallucinated facts cost money (drug names, prices, account numbers) | `grounding=True` |
| Multi-step diagnosis or root-cause analysis where one bad assumption poisons the chain | `causal=True` |
| All three apply — production research agent, compliance-sensitive answer | turn them all on |
| Quick prototype, low-stakes Q&A | leave them off — extra model calls are wasted |

The cost is more model round-trips. The win is fewer wrong answers.
For short tasks the math doesn't pencil out. For runs of 5+ tool calls
or anything that ships to a customer, it almost always does.

## Getting started

### Reflexion

Self-evaluate per turn.

```python
from tulip.agent import Agent
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    reflexion=True,
)

result = agent.run_sync("Find Q3 revenue and explain the YoY change.")
print(result.metrics.reflexion_iterations)
```

After each tool result, the agent is asked: *given this, was the last
step right?* If the answer is "no", the next turn rewrites the plan
instead of stacking another tool call on top. Streamed as
`ReflectEvent` — render it in your UI and the user can literally watch
the agent change its mind.

### Grounding

Verify claims before answering.

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_pricing, lookup_inventory],
    grounding=True,
)

result = agent.run_sync("What's the cheapest GPU instance with 80GB?")
for claim in result.grounding_report.unsupported:
    print(f"DROPPED: {claim.text}")
```

Before the agent finalises an answer, every factual claim is checked
against the conversation's tool results. A second model — the judge —
reads each claim and the supporting tool output and emits *supported /
unsupported / partially supported*. Unsupported claims are dropped or
sent back for re-research. Streamed as `GroundingEvent`.

### Causal

Track cause-effect chains.

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[fetch_logs, query_metrics, traceback],
    causal=True,
)

result = agent.run_sync("Why is checkout p99 latency up 4x since 14:00?")
print(result.causal_chain.root_causes)
```

The agent maintains a running cause-effect graph — *X happened
because Y; Y because Z* — and validates new conclusions against it.
Cycles, contradictions, and unsupported jumps surface as the chain
grows. Particularly useful for incident triage where the linear chat
log doesn't show that turn 3's "fix" contradicts turn 1's "root
cause".

## Combining them

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    reflexion=True,
    grounding=True,
    causal=True,
)
```

The order is fixed: reflect first, build the causal graph as you go,
ground last. Reflect first because if the last step was wrong,
grounding on its claims is wasted judge work; ground last because
intermediate claims will be rewritten before they ship, so any tokens
spent verifying them are tokens spent verifying drafts. All three are
observable as their own event types.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Reflexion loops forever | The model can't agree with itself. Cap with `MaxIterations` in your termination condition. |
| Grounding flags everything as unsupported | The judge model is stricter than the answerer. Use the same model for both, or lower the threshold. |
| Causal graph has many disconnected nodes | The model isn't naming entities consistently across turns. Sharpen the system prompt to name entities the same way each time. |
| Reasoning add-ons feel slow | They're extra model calls — that's the trade. Keep them for runs that ship to a human, drop them for hot paths. |

## Source and notebook

- [`notebook_36_reasoning_patterns.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_36_reasoning_patterns.py) — all three add-ons end-to-end.
- [`tulip.reasoning.reflexion`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/reflexion.py)
- [`tulip.reasoning.grounding`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/grounding.py)
- [`tulip.reasoning.causal`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/reasoning/causal.py)
- [`ReflectNode`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/loop/nodes.py) in the ReAct loop — where reflection plugs in.

Reflexion: [Shinn et al., 2023](https://arxiv.org/abs/2303.11366).
Grounding-Stratified Adaptive Replanning: see [GSAR](gsar.md) for the
typed-evidence variant the SDK also ships.

## See also

- [GSAR](gsar.md) — typed-grounding layer with weighted scoring and tiered replanning.
- [Events](events.md) — `ReflectEvent`, `GroundingEvent`, causal node/edge events.
- [Termination](termination.md) — combine `ConfidenceMet` with reflexion to early-stop on high-confidence answers.
