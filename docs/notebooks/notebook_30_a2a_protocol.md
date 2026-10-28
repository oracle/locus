# A2A Protocol

A2A (Agent-to-Agent) is the public cross-framework protocol at
[a2aproject.github.io/A2A](https://a2aproject.github.io/A2A/). Locus
implements both sides on the A2A **v1.0** wire format; this notebook
spins up a real Agent behind `A2AServer` — a payment network's risk
agent shared with a partner bank — drives every v1.0 method from
`A2AClient`, and inspects the typed task lifecycle.

This notebook covers:

- Agent Card at `/.well-known/agent-card.json` with typed `AgentSkill`
  entries, `protocolVersion`, and `supportedInterfaces` — enough for
  any partner bank's A2A client to discover and call the agent.
- A2A v1.0 JSON-RPC methods over `POST /` with `A2A-Version: 1.0`:
  `SendMessage`, `GetTask`, `ListTasks`, `CancelTask`, and
  `SendStreamingMessage` (SSE `StreamResponse` envelopes).
- `TaskNotCancelable` (-32002) surfaced as a `RuntimeError` when you
  try to cancel a terminal task.
- Opting back into the legacy method names with `protocol_version=None`
  for pre-v1.0 peers.
- `A2AClient.invoke` — backwards-compatible flat shape for non-spec
  peers.
- `A2AClient.as_tool(...)` — wrap a remote agent as a Locus `@tool` so
  a local payments agent can delegate merchant-risk questions to it.

A cross-org payment mesh spans organizational boundaries, so treat
every inbound message as untrusted: the bearer token authenticates the
peer and the published `AgentSkill` set bounds its reach, but ground
what a partner asserts about a merchant or a card BIN before you act
on it.

## Prerequisites

- `pip install fastapi uvicorn` for the server side.
- Agent basics. The wire format is provider-agnostic.

## Run

```bash
python examples/notebook_30_a2a_protocol.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

The notebook starts an in-process uvicorn server and drives a client
against it; expect a few seconds of warm-up before the first
`SendMessage` returns.

## Source

````python
--8<-- "examples/notebook_30_a2a_protocol.py"
````
