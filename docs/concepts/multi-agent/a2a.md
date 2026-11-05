# A2A

A2A (Agent-to-Agent) is the **cross-process / cross-runtime** version
of multi-agent.
Each agent runs as its own service, advertises an Agent Card
(capabilities + skills + endpoint URL) at a well-known URL, and other
agents discover and call it over HTTP.

The SDK implements the public
[A2A protocol](https://a2aproject.github.io/A2A/) — an open wire
format that any conforming runtime can speak — so an
SDK-built agent can call a non-SDK A2A peer (or be called by one)
without an adapter.

{{ locus_diagram a2a }}

## Wire surface

`A2AServer` is v1.0-first while preserving the older pre-v1 surface. Clients that request v1.0 send `A2A-Version: 1.0`; clients
that use the old method names continue to hit the legacy dispatcher. A
JSON-RPC request to `POST /` that names any other `A2A-Version` is
rejected with a version-not-supported error.

`A2AServer` exposes:

| Endpoint | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | Public Agent Card — name, description, skills, capabilities, modes, `protocolVersion`, and `supportedInterfaces`. |
| `POST /` + `A2A-Version: 1.0` | A2A v1.0 JSON-RPC dispatch — `SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`. |
| `POST /` with old methods | Legacy JSON-RPC dispatch — `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`. |
| `GET /agent-card`, `POST /a2a/{invoke,stream}` | Backwards-compat aliases for peers using the original shape. |

The v1.0 Task lifecycle uses protocol enum values such as
`TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, and
`TASK_STATE_COMPLETED`. The SDK still maps those to the Python-facing
`TaskState` enum (`submitted`, `working`, `completed`, ...) for SDK
callers. Streaming responses on `SendStreamingMessage` arrive as SSE
JSON-RPC envelopes whose `result` is a v1.0 `StreamResponse`, containing
exactly one of `task`, `message`, `statusUpdate`, or `artifactUpdate`.
The old `final` streaming flag is not emitted on the v1.0 path.

The Python SDK keeps the familiar `TextPart`, `FilePart`, and `DataPart`
models. On the v1.0 wire they are converted to the v1.0 oneof part
shape (`text`, `raw`, `url`, `data`) and back.

## When to use it

- **Multi-process or multi-host** agent deployments.
- **Different teams own different agents** on different stacks.
- You need a **network boundary** for security or scaling.
- **Polyglot** — an SDK agent calling a non-SDK A2A peer (or
  vice versa) speaks the same protocol verbatim.
- **Capability-based discovery** — the caller reads the Agent
  Card and decides whether to delegate.

## When NOT to use it

- **Single-process** — use one of the in-process patterns; HTTP
  round-trips are pure overhead.
- **Tight latency requirements** — every A2A hop is an HTTP round-trip.
- The peer is **always the same agent** — just call it directly.

## Code

### Host side — expose an agent over A2A

{{ locus_model_tabs }}

```python
from locus.agent import Agent
from locus.a2a import A2AServer, AgentSkill

research_agent = Agent(
    model="{{ locus_model }}",
    tools=[web_search, fetch_page, cite],
    system_prompt="You research questions across sources and summarise the evidence.",
)

server = A2AServer(
    agent=research_agent,
    name="research",
    description="Researches a question across sources. Cites the source of each claim.",
    url="https://research.example.com",
    skills=[
        AgentSkill(
            id="topic_research",
            name="Topic research",
            description="Research a question across sources and summarise the answer.",
            tags=["research", "search"],
        ),
        AgentSkill(
            id="source_check",
            name="Source check",
            description="Verify a claim against its cited sources and their dates.",
            tags=["verification"],
        ),
    ],
    api_key="rotate-this-secret",
)
server.run(host="0.0.0.0", port=7421)
```

The Agent Card is now reachable at
`https://research.example.com/.well-known/agent-card.json` (with the
required bearer token).

### Client side — fetch the card and send a message

```python
import asyncio
from locus.a2a import A2AClient, Message, TextPart


async def main():
    client = A2AClient(url="https://research.example.com", api_key="rotate-this-secret")
    prompt = "Research why checkout conversion dropped last week; cite sources."

    # Read the public card to learn the agent's skills + capabilities.
    card = await client.get_agent_card()
    print(card.name, [s.id for s in card.skills])

    # Synchronous send — returns a Task in the `completed` state.
    task = await client.send_message(
        Message(role="user", parts=[TextPart(text=prompt)], messageId="m-1")
    )
    final_text = task.artifacts[-1].parts[0].text
    print(final_text)


asyncio.run(main())
```

By default `A2AClient` sends `A2A-Version: 1.0` and uses the v1.0 method
names. If it talks to an older peer that returns a legacy Task directly,
the client still accepts it and maps it back into the SDK `Task` model.
To force the older JSON-RPC method names from the client, pass
`protocol_version=None`: the same methods then send `message/send`,
`message/stream`, `tasks/get` and `tasks/cancel` with no `A2A-Version`
header, and `list_tasks()` raises, since it needs v1.0. Any other value
is sent as the `A2A-Version` header, and `A2AServer` rejects the
JSON-RPC calls that carry it.

### Streaming

```python
async for event in client.send_message_streaming(
    Message(role="user", parts=[TextPart(text=prompt)], messageId="m-2")
):
    if event.get("kind") == "status-update":
        print("status:", event["status"]["state"])
    elif event.get("kind") == "artifact-update":
        print("got artifact")
    elif event.get("kind") == "task":
        print("initial task:", event["id"])
```

`send_message_streaming()` returns SDK-shaped events to Python callers
even though the wire format is v1.0 when `A2A-Version: 1.0` is active.
This keeps existing router and application code stable while the HTTP
boundary remains protocol-correct.

### Task lifecycle

```python
task = await client.send_message(message)
# Long-running tasks: poll while still in working / input-required.
fresh = await client.get_task(task.id)
if fresh.status.state == "input-required":
    # ... gather input from the human, then send a follow-up message
    ...
# Or list tasks by context / status.
tasks, next_page = await client.list_tasks(
    context_id=task.contextId,
    status="completed",
    page_size=20,
)
# Or cancel.
await client.cancel_task(task.id)
```

`list_tasks()` calls the v1.0 `ListTasks` method and maps the response
back into SDK `Task` objects, returning `(tasks, next_page_token)`.

## Cross-process delegation

Delegate through `A2AClient.send_message()`, so calls to a remote agent use
the v1.0 path by default. `send_message()` does not fall back on its own: when
the peer answers with a JSON-RPC error, such as `SendMessage` not found, it
raises `RuntimeError`. To reach an older peer that answers `SendMessage` with
method not found (`-32601`) but still serves the flat `/a2a/invoke` endpoint,
catch that error and call `client.invoke()`. A peer with no JSON-RPC route at
`POST /` fails the HTTP request instead, and `send_message()` raises
`httpx.HTTPStatusError`, which the example below does not catch.

```python
import asyncio

from locus.a2a import A2AClient, Message, TextPart


async def main():
    client = A2AClient("https://research.example.com", api_key="rotate-this-secret")
    prompt = "Summarise the Q3 incident reports."
    try:
        task = await client.send_message(
            Message(role="user", parts=[TextPart(text=prompt)], messageId="m-3")
        )
        reply = task.artifacts[-1].parts[0].text
    except RuntimeError as exc:
        if "A2A error -32601" not in str(exc):  # -32601: method not found
            raise
        reply = await client.invoke(prompt)
    print(reply)


asyncio.run(main())
```

The remote agent still owns its tools and orchestration. The caller passes a
single user message and unwraps the final text from the returned Task artifact.

## Authentication and TLS

`A2AServer` ships with bearer-token auth: pass `api_key="..."` (or set
`LOCUS_A2A_API_KEY`) and every route — including
`/.well-known/agent-card.json` — requires `Authorization: Bearer ...`.
With no key the server refuses non-loopback bindings unless
`allow_unauthenticated=True` is passed (use that only behind an
upstream proxy that terminates auth). TLS is handled as for any FastAPI
application: terminate it at the load balancer, or pass uvicorn's
`ssl_keyfile` and `ssl_certfile` options to `server.run()`, which
forwards keyword arguments to `uvicorn.run()`.

## Backwards compatibility

The pre-spec endpoints are still served:

```python
# Legacy: flat string-in / string-out — bypass the JSON-RPC envelope.
reply = await client.invoke("Research why checkout conversion dropped last week...")
```

Anything that imported `A2AMessage` / `A2ARequest` / `A2AResponse` from
`locus.a2a.protocol` keeps working — those models are preserved as
aliases for the legacy `/a2a/invoke` shape. Spec-aware peers should
use `Message` + `client.send_message()` so they can read the full
`Task` (status, history, artifacts).

## Notebook

[`notebook_30_a2a_protocol.py`](https://github.com/oracle/locus/blob/main/examples/notebook_30_a2a_protocol.py)
— host + client + streaming.

## Source

[`a2a/spec.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/spec.py)
— typed Pydantic models for every spec object.

[`a2a/protocol.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/protocol.py)
— `A2AServer`, `A2AClient`, JSON-RPC dispatch, in-memory task store.

## See also

- [Multi-agent overview](../multi-agent.md) — pick a shape.
- [Agent Server](../server.md) — the FastAPI invoke/stream server.
  `A2AServer` is a separate server that wraps an agent through the same
  `run()` contract.
- [Conversation Management](../conversation-management.md) — history
  that survives across requests needs a `thread_id`. `A2AServer` runs
  the agent on the message text without one, so `contextId` groups
  tasks but does not carry the agent's conversation.
