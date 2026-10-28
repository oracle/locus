# Steering

`SteeringHook` runs a second LLM ("the steering model") in front of
every tool call. The steering model reads a natural-language policy
plus the agent's activity so far, then returns one of three actions:

- `PROCEED` — let the tool call go through.
- `GUIDE` — cancel the tool call and return the guidance message in
  its place, so the agent reads the correction instead of the result.
- `INTERRUPT` — cancel the tool call and flag it as requiring human
  approval (returns a `REQUIRES APPROVAL` message).

The result is a real-time guardrail you can author in plain English —
no rules engine, no policy DSL. In an infra/devops setting this lets you
hold an on-call agent to a read-only diagnostics session: it can query
Prometheus and read logs, but a mutating operation — restarting a
service, scaling a deployment, terminating an instance — is blocked
unless a human approves it.

- `SteeringHook(model=..., policy="...")` — attach it to any agent via
  the `hooks=` parameter.
- `steering.decisions` — every action with its reason, for audit.

The configured provider drives both the agent and the steering model.

## Run it

OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_68_steering.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_68_steering.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.

## Source

````python
--8<-- "examples/notebook_68_steering.py"
````
