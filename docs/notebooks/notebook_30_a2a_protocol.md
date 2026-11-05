# A2A protocol

A2A (Agent-to-Agent) is the public cross-framework protocol at
[a2aproject.github.io/A2A](https://a2aproject.github.io/A2A/), and the SDK
implements both sides of its **v1.0** wire format (see the
[A2A concept page](../concepts/multi-agent/a2a.md)). This notebook
publishes an `Agent` behind `A2AServer` as a payment network's risk agent
shared with a partner bank, drives the five v1.0 methods `A2AClient`
exposes, and inspects the typed task lifecycle.

A payment mesh spans organizational boundaries, so treat every inbound
message as untrusted input: the bearer token authenticates the peer and
the published `AgentSkill` set declares its scope, but ground what a partner
asserts about a merchant or a card BIN (Bank Identification Number) before
acting on it.

What this notebook covers:

- The Agent Card at `/.well-known/agent-card.json`, with typed `AgentSkill`
  entries, `protocolVersion` and `supportedInterfaces`: enough for any
  partner bank's A2A client to discover and call the agent.
- The v1.0 JSON-RPC methods over `POST /`, sent with `A2A-Version: 1.0`:
  `SendMessage`, `GetTask`, `ListTasks`, `CancelTask` and
  `SendStreamingMessage`, the last delivered as server-sent events (SSE)
  whose JSON-RPC envelopes each carry a v1.0 `StreamResponse`.
- `TaskNotCancelable` (-32002) surfaced as a `RuntimeError` on a
  `CancelTask` against a task already in a terminal state.
- Opting back into the legacy method names with `protocol_version=None`
  for pre-v1.0 peers.
- `A2AClient.invoke`: the backwards-compatible flat shape for non-spec
  peers.
- `A2AClient.as_tool(...)`: wrap a remote agent as an `@tool`. The
  notebook builds and inspects the tool but does not call it: the
  generated function runs `asyncio.run` over the flat `invoke` path, so
  calling it directly inside `main`'s running event loop would raise. An
  agent that holds the tool runs it on a worker thread (`Tool.execute`),
  where no event loop is running.

## Prerequisites

- `pip install fastapi uvicorn` for the server side.
- [Agent basics](notebook_02_basic_agent.md).

The drawing is Parts 2 and 3 seen from both processes. In the partner
bank's process, `A2AClient` fetches the Agent Card from
`GET /.well-known/agent-card.json` (the risk agent's skills, capabilities
and URL) and then posts a JSON-RPC `SendMessage` to `/`; in the payment
network's process, `A2AServer` runs the risk agent on the message text and
returns a completed `Task` carrying the answer as an artifact. The two
notes under the lanes are Parts 7 and 1: `SendStreamingMessage` runs a
second message the same way but delivers the task, its status updates and
the artifact as server-sent events, and because the server was given an
`api_key`, every route, the card included, requires the bearer token:

{{ locus_diagram a2a }}

## Key lines

Every A2A call in this example is made inline in `main`, one numbered part
per step; the full source is collapsed below. The helper quoted here is
what lets one script act as both processes. `A2AServer` builds its FastAPI
app on first access to `server.app`; `_start_server` serves that app with
uvicorn on a daemon thread and polls uvicorn's `started` flag with an async
sleep rather than a blocking one, so the partner bank's `A2AClient` can
discover and call the agent from the same event loop:

{{ locus_source examples/notebook_30_a2a_protocol.py::_start_server }}

## Run it

```bash
python examples/notebook_30_a2a_protocol.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. The wire format
itself is provider-agnostic.

The script starts an in-process uvicorn server, waits for it to come up,
then drives the partner bank's client against it; each `SendMessage`
returns a `Task` whose last artifact is the model's reply.

## See also

- [A2A](../concepts/multi-agent/a2a.md) — the wire surface, authentication
  and TLS, and cross-process delegation.
- [A2A API reference](../api/a2a.md) — `A2AServer`, `A2AClient`, the Agent
  Card models and the v1 protocol types.

## Source

??? note "Full source — `examples/notebook_30_a2a_protocol.py`"

    ````python
    --8<-- "examples/notebook_30_a2a_protocol.py"
    ````
