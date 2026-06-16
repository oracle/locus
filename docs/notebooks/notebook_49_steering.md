# Steering

`SteeringHook` runs a second LLM ("the steering model") in front of
every tool call. The steering model reads a natural-language policy
plus the agent's activity so far, then returns one of three actions:

- `PROCEED` — let the tool call go through.
- `GUIDE` — let it through but inject a note for the agent to read.
- `INTERRUPT` — block the tool call and return a refusal message.

The result is a real-time guardrail you can author in plain English —
no rules engine, no policy DSL.

- `SteeringHook(model=..., policy="...")` — attach it to any agent via
  the `hooks=` parameter.
- `steering.decisions` — every action with its reason, for audit.

OCI GenAI drives both the agent and the steering model by default.

## Run it

OCI GenAI is the default — the notebook harness picks it when `~/.oci/config` exists (set `OCI_PROFILE` to choose a non-default profile):

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_49_steering.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_49_steering.py
```

## Prerequisites

- An OCI profile with GenAI access, or `LOCUS_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.

## Source

```python
--8<-- "examples/notebook_49_steering.py"
```
