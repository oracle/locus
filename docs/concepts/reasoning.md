# Reasoning

A model that loops without thinking just pays you to be wrong faster.
Locus ships three reasoning
add-ons. Each catches a different class of mistake *before* the next
tool call:

- **Reflexion** catches *wrong premises* — the agent self-evaluates
  after each turn and re-plans if the last step was a dead end,
  instead of stacking another tool call on top.
- **Grounding** catches *hallucinations* — before the answer ships,
  every factual claim is checked against the tool results that
  produced it; unsupported claims are dropped or sent back.
- **Causal reasoning** catches *contradictions* — a cause-effect graph
  surfaces the case where turn 3's "fix" contradicts turn 1's "root
  cause", which linear chat history hides.

Reflexion and grounding are each a single argument on `Agent(...)` and
combine freely. Causal reasoning is a standalone graph builder
(`build_causal_chain()`) you run over the events the agent surfaced —
see its section below.

## When to pick which

| Situation | Add-on |
|---|---|
| Agent loops endlessly or stacks tool calls on a wrong premise | `reflexion=True` |
| Analyst-facing findings where a hallucinated fact drives a wrong action (indicator values, hostnames, CVE ids) | `grounding=True` |
| Multi-step investigation or root-cause analysis where one bad assumption poisons the chain | `build_causal_chain(events)` |
| All three apply — production triage agent, audit-sensitive finding | reflexion + grounding on the agent, causal chain over its events |
| Quick prototype, low-stakes lookup | leave them off — extra model calls are wasted |

Grounding costs an extra judge call per answer; reflexion is an
in-process heuristic with no model call. The win is fewer wrong answers.
For short tasks the math doesn't pencil out. For runs of 5+ tool calls
or anything that ships to an analyst, it almost always does.

## Getting started

### Reflexion

Self-evaluate per turn.

```python
from locus.agent import Agent
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[query_siem, summarise],
    reflexion=True,
)

result = agent.run_sync("Find the lateral-movement events for INC-92 and explain the spread.")
print(result.metrics.reflexion_evaluations)
```

After each tool-calling iteration the `Reflector` scores the turn
without a model call: it looks for a repeating tool pattern, counts the
iteration's successful and failed calls, and nudges a confidence score.
A `stuck` or `loop_detected` assessment adds guidance to the next
Think, so the plan is rewritten instead of stacking another tool call
on top. Streamed as `ReflectEvent` — render it in your UI and the user
can literally watch the agent change its mind.

### Grounding

Verify claims before answering.

```python
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[enrich_indicator, lookup_hash],
    grounding=True,
)

result = agent.run_sync("Is the hash 44d88612... associated with known malware?")
print(f"grounding score: {result.grounding_score}")
for claim in result.ungrounded_claims:
    print(f"DROPPED: {claim}")
```

Before the agent finalises an answer, every factual claim is checked
against the conversation's tool results. A second model — the judge —
reads each claim and the supporting tool output and emits *supported /
unsupported / partially supported*. Unsupported claims are dropped or
sent back for re-research. Streamed as `GroundingEvent`.

### Causal

Track cause-effect chains. Unlike reflexion and grounding, causal
reasoning isn't an `Agent(...)` flag — it's a standalone builder,
`build_causal_chain()`, that you run over the events your agent or
tools surfaced. It returns a `CausalChain` you can summarise or query.

```python
from locus.reasoning.causal import build_causal_chain

# Events the agent extracted from the investigation, each optionally
# naming its cause(s) by label.
chain = build_causal_chain([
    {"label": "Deploy 412 shipped a config change"},
    {"label": "Cache hit-rate dropped",
     "causes": ["Deploy 412 shipped a config change"]},
    {"label": "Checkout latency doubled",
     "causes": ["Cache hit-rate dropped"]},
])

summary = chain.get_chain_summary()
print(summary["root_causes"])   # ['Deploy 412 shipped a config change']
print(summary["symptoms"])      # ['Checkout latency doubled']
print(summary["conflicts"])     # count of contradictory edges detected
```

The chain is a cause-effect graph — *X happened because Y; Y because
Z* — that auto-classifies each node as root cause, intermediate, or
symptom, and surfaces cycles and contradictions as the graph grows.
The same builder reconstructs an attack chain just as well — phished
credential → lateral movement → domain admin token.
Particularly useful for incident triage where the linear chat log
doesn't show that turn 3's "fix" contradicts turn 1's "root cause".
`build_causal_chain()` is plain Python with no model call — feed it
the events and read the structure back.

## Combining them

Turn on the two in-loop add-ons together, then build the causal graph
over the events the run surfaced:

```python
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[...],
    reflexion=True,
    grounding=True,
)

result = agent.run_sync("Why did checkout latency double after deploy 412?")

# Two events lifted from the run's tool trail; the second names the first as its cause.
from locus.reasoning.causal import build_causal_chain
deploy, cache = result.state.tool_executions[:2]
chain = build_causal_chain([
    {"label": deploy.result},                            # "deploy 412 changed the cache TTL"
    {"label": cache.result, "causes": [deploy.result]},  # "cache hit-rate dropped to 40%"
])
```

The order is fixed for the in-loop add-ons: reflect first, ground last.
Reflect first because if the last step was wrong, grounding on its
claims is wasted judge work; ground last because intermediate claims
will be rewritten before they ship, so any tokens spent verifying them
are tokens spent verifying drafts. Reflexion and grounding are
observable as their own event types (`ReflectEvent`, `GroundingEvent`);
the causal chain is built after the fact from the run's events.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Reflexion loops forever | The model can't agree with itself. Cap with `MaxIterations` in your termination condition. |
| Grounding flags everything as unsupported | The judge model is stricter than the answerer. Use the same model for both, or lower the threshold. |
| Causal graph has many disconnected nodes | The model isn't naming entities consistently across turns. Sharpen the system prompt to name entities the same way each time. |
| Grounding feels slow | It is an extra judge call per answer — that's the trade. Keep it for runs that ship to a human, drop it for hot paths. |

## Source and notebook

- [`notebook_54_reasoning_patterns.py`](https://github.com/oracle/locus/blob/main/examples/notebook_54_reasoning_patterns.py) — all three add-ons end-to-end.
- [`locus.reasoning.reflexion`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/reflexion.py)
- [`locus.reasoning.grounding`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/grounding.py)
- [`locus.reasoning.causal`](https://github.com/oracle/locus/blob/main/src/locus/reasoning/causal.py)
- [Reflexion in the ReAct loop](https://github.com/oracle/locus/blob/main/src/locus/agent/runtime_loop.py) — where the `Reflector` plugs in.

Reflexion: [Shinn et al., 2023](https://arxiv.org/abs/2303.11366).
Typed grounding: see [GSAR](gsar.md) for the
typed-evidence variant the SDK also ships.

## See also

- [GSAR](gsar.md) — typed-grounding layer with weighted scoring and tiered replanning.
- [Events](events.md) — `ReflectEvent`, `GroundingEvent`.
- [Termination](termination.md) — combine `ConfidenceMet` with reflexion to early-stop on high-confidence answers.
