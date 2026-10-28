# Notebook 29: Cognitive router — risk-gated routing for infra/devops requests

locus.router compiles a natural-language operations request onto existing
Locus primitives. The LLM never picks topology — it fills a typed
GoalFrame; the router selects a protocol deterministically, and the
compiler emits a real Agent / SequentialPipeline / ParallelPipeline /
LoopAgent from a curated registry. PolicyGate is the star, and it is the
control that keeps an autonomous router from over-reaching: the frame's
risk band, not the model's confidence, decides whether an action runs. A
LOW-risk "summarize this runbook" runs straight through to a direct
answer; a HIGH-risk "drain a prod node and restart the payments
deployment" compiles behind an approval interrupt and does not execute
until a human says so.

- Define a small infra/devops capability set as annotated tools.
- Register all 8 built-in protocols.
- Load SKILL.md packages from examples/skills/ and tag them by domain
  so the compiler attaches the right ones to every emitted Agent.
- Stand up a Router with an Agent(output_schema=GoalFrame) extractor
  plus a CognitiveCompiler.
- Dispatch six requests: five hit five different protocols (answer /
  plan / diagnose / debate / codegen) and print which protocol fired
  and the runtime shape; the sixth is HIGH risk and PolicyGate holds
  it for approval instead of executing.

Run it
    # OCI Generative AI with an OCI profile, otherwise the bundled mock model
    python examples/notebook_29_cognitive_router.py

    # Offline / no credentials (uses fallback frames):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_29_cognitive_router.py

## Source

````python
--8<-- "examples/notebook_29_cognitive_router.py"
````
