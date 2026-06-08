# Cognitive Router

`tulip.router` compiles a natural-language request onto existing Tulip
primitives. The LLM never picks topology — it fills a typed
`GoalFrame`; the router selects a protocol deterministically, and the
compiler emits a real `Agent` / `SequentialPipeline` /
`ParallelPipeline` / `LoopAgent` from a curated registry.

Pipeline::

    natural-language input
          │
          ▼
    Agent(output_schema=GoalFrame)     ← LLM fills typed schema only
          │ GoalFrame(primary_goal, domain, complexity, risk, …)
          ▼
    ProtocolRegistry.select(frame)     ← typed filter + ranking
          │ Protocol (e.g. "specialist_fanout")
          ▼
    PolicyGate.check(frame, protocol)  ← allow | require_approval | deny
          │
          ▼
    CognitiveCompiler.compile(…)       ← emits Runnable adapter
          │ wraps real Agent / Pipeline / Orchestrator
          ▼
    runnable.execute(task)
          │
          ▼
    RunnableResult(text, protocol_id, frame)

- Define a small capability set as annotated tools.
- Register all 8 built-in protocols.
- Load `SKILL.md` packages and tag them by domain so every emitted
  Agent gets the right catalog at runtime.
- Stand up a `Router` with a `GoalFrame` extractor and a
  `CognitiveCompiler`.
- Dispatch five distinct inputs that hit five different protocols
  (`direct_response` / `plan_execute_validate` / `specialist_fanout` /
  `debate` / `codegen_test_validate`) and print which protocol fired,
  the compiled runtime shape, and the result.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_58_cognitive_router.py

Offline (uses fallback frames):

    TULIP_MODEL_PROVIDER=mock python examples/notebook_58_cognitive_router.py

## Source

```python
--8<-- "examples/notebook_58_cognitive_router.py"
```
