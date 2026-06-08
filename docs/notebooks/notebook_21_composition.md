# Composition Patterns

Chain agents, run them in parallel, or loop one until it's satisfied.
When the work decomposes cleanly into agent-shaped pieces, you don't
need a full `StateGraph`. The three pipeline classes here are
batteries-included composition primitives that take a list of `Agent`
instances and orchestrate them for you.

What you'll see:

- `SequentialPipeline` — each agent's output becomes the next agent's input.
- `ParallelPipeline` — run agents concurrently, then merge their results.
- `LoopAgent` — run one agent repeatedly until a stop condition fires.
- One-liner helpers: `sequential()`, `parallel()`, `loop()`.

Runs on the same default (mock) as the rest of the notebooks:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_21_composition.py
# or, fully offline:
TULIP_MODEL_PROVIDER=mock python examples/notebook_21_composition.py
```

## Source

```python
--8<-- "examples/notebook_21_composition.py"
```
