# A2A Protocol

A2A (Agent-to-Agent) is the public cross-framework protocol at
[a2aproject.github.io/A2A](https://a2aproject.github.io/A2A/). Tulip
implements both sides on the A2A **v1.0** wire format; this notebook
spins up a real Agent behind `A2AServer`, drives every v1.0 method
from `A2AClient`, and inspects the typed task lifecycle.

This notebook covers:

- Agent Card at `/.well-known/agent-card.json` with typed `AgentSkill`
  entries, `protocolVersion`, and `supportedInterfaces` — enough for
  any A2A client to discover and call the agent.
- A2A v1.0 JSON-RPC methods over `POST /` with `A2A-Version: 1.0`:
  `SendMessage`, `GetTask`, `ListTasks`, `CancelTask`, and
  `SendStreamingMessage` (SSE `StreamResponse` envelopes).
- `TaskNotCancelable` (-32002) surfaced as a `RuntimeError` when you
  try to cancel a terminal task.
- Opting back into the legacy method names with `protocol_version=None`
  for pre-v1.0 peers.
- `A2AClient.invoke` — backwards-compatible flat shape for non-spec
  peers.
- `A2AClient.as_tool(...)` — wrap a remote agent as a Tulip `@tool` so
  a local agent can delegate to it.

## Prerequisites

- `pip install fastapi uvicorn` for the server side.
- Notebook 08 (Agent basics). The wire format is provider-agnostic.

## Run

```bash
python examples/notebook_28_a2a_protocol.py
```

The default provider is the bundled mock model. Set `TULIP_MODEL_PROVIDER`
(openai / anthropic) and credentials to use a live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

The notebook starts an in-process uvicorn server and drives a client
against it; expect a few seconds of warm-up before the first
`SendMessage` returns.

## Source

```python
--8<-- "examples/notebook_28_a2a_protocol.py"
```
