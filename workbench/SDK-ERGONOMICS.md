# SDK ergonomics — workbench audit

How much code does a developer need to write to do something useful with
the Oracle Generative AI – Multi-Agent Locus SDK? This is a short,
honest snapshot from running the workbench against every model-only
notebook.

## The 90% case is one screen

```python
from config import get_model
from locus.agent import Agent

agent = Agent(model=get_model(), system_prompt="You are concise.")
result = agent.run_sync("Explain entanglement in one sentence.")
print(result.message)
```

Five lines. No transport selection, no auth wiring, no message-list
construction. `get_model()` (in `examples/config.py`) reads
`LOCUS_MODEL_PROVIDER` + per-provider env and returns the right Model
class — so the same code runs against OpenAI, Anthropic, OCI session,
or OCI api-key without changes. The workbench injects those env vars
per the user's Provider settings.

## Two-style API

For the advanced 10%, drop in `AgentConfig`:

```python
from locus.agent import Agent, AgentConfig
from pydantic import BaseModel

class Verdict(BaseModel):
    winner: str
    confidence: float

agent = Agent(config=AgentConfig(
    model=model,
    system_prompt="Pick a winner.",
    output_schema=Verdict,
    max_iterations=2,
    reflexion=True,
))
result = agent.run_sync("Python vs JS for backend in 2026.")
print(result.parsed)   # → Verdict(winner='...', confidence=0.83)
```

The split feels right: short form for "do the thing"; `AgentConfig` for
typed output, reflexion, grounding, hooks, retry policies, etc.

## Multi-agent shapes get convenience constructors

Patterns the SDK ships with (`from locus.multiagent import ...`):

```python
from locus.multiagent import Orchestrator, Specialist

orch = Orchestrator(
    coordinator_model=model,
    specialists=[
        Specialist(name="researcher", agent=research_agent,
                   description="Reads sources, summarises with citations."),
        Specialist(name="editor", agent=editor_agent,
                   description="Tightens prose, removes fluff."),
    ],
)
result = await orch.execute("Write a one-paragraph case for AI in healthcare.")
```

Same shape applies for `SequentialPipeline`, `ParallelPipeline`,
`LoopAgent`, `Swarm`, `Handoff`, `StateGraph`, plus a `@task` /
`@entrypoint` functional API. None of them require you to assemble more
than ~5 lines of plumbing for the standard case.

## Rough patches we hit

1. **OCI transport split.** Cohere R-series uses `OCIModel` (native SDK
   chat shape); openai/meta/llama/grok uses `OCIChatCompletionsModel` (the
   `/openai/v1` endpoint). Picking the wrong one = 404. The workbench
   papers over this with an `oci_transport: "auto"` rule that mirrors
   `examples/config.py:_pick_oci_transport`. Would be nicer if the SDK
   had a single `OCIGenAIModel(model_id=...)` that auto-routed.
2. **`output_schema=` is buried.** Notebook 14 demonstrates it but it's
   in a later section; first-time users tend to miss it and hand-parse
   JSON instead. Worth surfacing in `Agent`'s docstring.
3. **Inconsistent kwargs.** `max_tokens` (some providers) vs
   `max_completion_tokens` (others). The wrappers normalise some of
   this but it leaks at the edges.
4. **Hand-built graphs are still ~30 lines** for map-reduce (notebook
   42's `StateGraph` + `Send`). A `MapReduce(workers=[...], reduce=fn)`
   one-liner would eat the bottom 90% of those.

## Notebook categories — what runs in the workbench

| Category | Count | Status |
|---|---|---|
| GREEN — pure model + tools | 27 | runs to completion on real OCI gpt-5.5 |
| YELLOW — minor gotchas (tmp dirs, demo sleeps) | 5 | runs but a few seconds slower |
| RED — uses `locus.core.interrupt()` | 5 | needs human stdin, blocked in the workbench (run `python examples/notebook_NN_*.py` locally) |

So: **32/37** of the curated notebooks run end-to-end, fully automated,
against real GenAI through the workbench. The 5 stdin-dependent ones
are human-in-the-loop demos that need a real terminal — fundamental,
not a fix-tomorrow.

## Verdict

The SDK is friendly for first-time users. Two-style `Agent(...)` API +
sensible env-driven `get_model()` glue + convenience constructors for
every multi-agent pattern means the boilerplate floor is low and the
ceiling is reachable. The papercuts above are real but incremental —
none of them require rewrites, just polish.
