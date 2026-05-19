# A2A Protocol

A2A (Agent-to-Agent) is the public cross-framework protocol at
[a2aproject.github.io/A2A](https://a2aproject.github.io/A2A/). Locus
implements both sides; this notebook spins up a real Agent behind
`A2AServer`, drives every spec endpoint from `A2AClient`, and inspects
the typed task lifecycle.

This notebook covers:

- Agent Card at `/.well-known/agent-card.json` with typed `AgentSkill`
  entries — enough for any A2A client to discover and call the agent.
- JSON-RPC 2.0 endpoints: `message/send`, `tasks/get`, `tasks/cancel`,
  and `message/stream` (SSE lifecycle events).
- `TaskNotCancelable` (-32002) surfaced as a `RuntimeError` when you
  try to cancel a terminal task.
- `A2AClient.invoke` — backwards-compatible flat shape for non-spec
  peers.
- `A2AClient.as_tool(...)` — wrap a remote agent as a Locus `@tool` so
  a local agent can delegate to it.

## Prerequisites

- `pip install fastapi uvicorn` for the server side.
- Notebook 08 (Agent basics). The wire format is provider-agnostic.

## Run

```bash
python examples/notebook_34_a2a_protocol.py
```

The default provider is OCI Generative AI. With `~/.oci/config`
present the agent talks to a live OCI model; canonical picks are
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`. Set
`LOCUS_MODEL_PROVIDER=mock` for offline runs.

The notebook starts an in-process uvicorn server and drives a client
against it; expect a few seconds of warm-up before the first
``message/send`` returns.

## Source

```python
--8<-- "examples/notebook_34_a2a_protocol.py"
```
