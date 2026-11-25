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
request that names any other `A2A-Version` is rejected with a
version-not-supported error.

`A2AServer` exposes:

| Endpoint | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | Public Agent Card — name, description, skills, capabilities, modes, `protocolVersion`, and `supportedInterfaces`. |
| `POST /` + `A2A-Version: 1.0` | A2A v1.0 JSON-RPC dispatch — `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, the `*TaskPushNotificationConfig` methods. |
| `POST /message:send`, `GET /tasks/{id}`, ... | The HTTP+JSON binding: the same methods as resource paths (table below). |
| `lf.a2a.v1.A2AService` on `grpc_port` | The gRPC binding, when configured. |
| `GET /.well-known/jwks.json` | The public keys push notifications are signed with, when a signer is configured. |
| `POST /` with old methods | Legacy JSON-RPC dispatch — `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`. |
| `GET /agent-card`, `POST /a2a/{invoke,stream}` | Backwards-compat aliases for peers using the original shape. |

### Transports

The specification defines three protocol bindings. A server advertises
the ones it serves in the card's `supportedInterfaces`, one
`AgentInterface` per binding, and a client picks from that list.

| Binding | `protocolBinding` | Where | Needs |
|---|---|---|---|
| JSON-RPC 2.0 | `JSONRPC` | `POST /` on the FastAPI app; streaming over server-sent events whose `data` is a JSON-RPC response | the `server` extra |
| HTTP+JSON (REST) | `HTTP+JSON` | `POST /message:send`, `POST /message:stream`, `GET /tasks`, `GET /tasks/{id}`, `POST /tasks/{id}:cancel`, `POST /tasks/{id}:subscribe`, `/tasks/{id}/pushNotificationConfigs[/{configId}]`, `GET /extendedAgentCard`; bodies are the v1.0 JSON shapes as `application/a2a+json`, errors are `google.rpc.Status` with the HTTP status the specification assigns | the `server` extra |
| gRPC | `GRPC` | `lf.a2a.v1.A2AService`, all eleven RPCs, on `grpc_port` (plaintext, or TLS with `credentials=`); errors carry a `google.rpc.Status` in the trailing metadata | the `a2a-grpc` extra |

Server side, `transports=` names the bindings to serve and advertise; the
default is the two HTTP bindings, which share one app and one bearer
token. `grpc_port=` adds gRPC and `run()` serves both. Every binding
lands in the same handlers, so a task created over gRPC can be read over
HTTP+JSON and streamed over JSON-RPC. Whichever binding carried an error,
it names the same reason (`TASK_NOT_FOUND`, `UNSUPPORTED_OPERATION`, ...)
in a `google.rpc.ErrorInfo` detail.

```python
from locus.a2a import A2AServer

server = A2AServer(
    agent=research_agent,
    name="research",
    url="https://research.example.com",
    transports=["JSONRPC", "HTTP+JSON", "GRPC"],
    grpc_port=7422,
    api_key="rotate-this-secret",
)
server.run(host="0.0.0.0", port=7421)  # HTTP on 7421, gRPC on 7422
```

Client side, `A2AClient(url)` speaks JSON-RPC at `url`; `A2AClient(url, binding="http+json")` or `binding="grpc"` speaks one other binding. `A2AClient.connect(url, prefer=[...])` (or `A2AClient.from_card(card, prefer=[...])`) reads the card and orders the advertised interfaces by `prefer` — `["JSONRPC", "HTTP+JSON", "GRPC"]` when not given.

If the preferred interface proves unreachable (connection refused, 404, 405, 501, gRPC `UNAVAILABLE` or `UNIMPLEMENTED`), the client moves on to the next one and stays there; `client.binding` says which it is using. A card that advertises no usable interface falls back to JSON-RPC at the card's `url`. An interface whose binding needs a package that is not installed (gRPC without `grpcio`) is skipped.

```python
client = await A2AClient.connect(
    "https://research.example.com", prefer=["grpc", "http+json"], api_key="..."
)
print(client.binding)  # GRPC, if the card advertises it and grpcio is installed
```

### Push notifications

A task that outlives the request that created it can call the client back. The client registers a `TaskPushNotificationConfig` — a webhook `url`, a `token` the server echoes back, and optionally the `authentication` (`scheme` and `credentials`) the server must present — either in the `SendMessage` configuration or afterwards with `CreateTaskPushNotificationConfig`.

The server then `POST`s a `StreamResponse` (`statusUpdate` or `artifactUpdate`) to the URL on every change, as `application/a2a+json`, with `X-A2A-Notification-Token` and the configured `Authorization` header. Delivery is best effort: a failure is retried with back-off and logged by task id, config id and host, never with the token or the credentials.

Push notifications are off until the server is given
`push_notifications=PushNotificationSettings(...)` (or `True` for the
defaults); until then the config methods answer
`PushNotificationNotSupported` and the card says
`capabilities.pushNotifications: false`. By default the server refuses to
post to loopback, private and link-local addresses, as the specification
advises; `allow_private_urls=True` is for a receiver on the same machine.

`signer=PushNotificationSigner.generate()` (or `.from_pem(...)`) signs
every body: the server publishes the public key as a JWKS at
`/.well-known/jwks.json` and sends a detached JWS over the body in
`X-A2A-Signature`, whose claims bind the body hash, the task id and an
expiry. Signing needs the `cryptography` package. The private key is held
in memory and never appears in a log line or a `repr()`.

```python
from locus.a2a import A2AServer, PushNotificationSettings, PushNotificationSigner

server = A2AServer(
    agent=research_agent,
    url="https://research.example.com",
    push_notifications=PushNotificationSettings(signer=PushNotificationSigner.generate()),
    api_key="rotate-this-secret",
)
```

On the client, `PushNotificationReceiver` is the webhook: an ASGI
application you mount or `serve()`, which checks the token and, when
given the agent's JWKS, the signature, then hands each verified
notification to `events()`.

```python
import asyncio

from locus.a2a import A2AClient, Message, PushNotificationReceiver, TextPart


async def main():
    receiver = PushNotificationReceiver(token="one-per-task-token")
    asyncio.create_task(receiver.serve("0.0.0.0", 9000))

    client = A2AClient("https://research.example.com", api_key="...")
    task = await client.send_message(
        Message(role="user", parts=[TextPart(text="Research ...")], messageId="m-1"),
        push_notification_config={"url": "https://me.example.com:9000/a2a/push", "token": "one-per-task-token"},
        blocking=False,
    )
    async for note in receiver.events():
        print(note.kind, note.state)
        if note.state == "TASK_STATE_COMPLETED":
            break


asyncio.run(main())
```

`blocking=False` asks the server to return the task as soon as it is
created and run it in the background, which is the shape push
notifications are for. `client.set_push_notification_config()`,
`get_push_notification_config()`, `list_push_notification_configs()` and
`delete_push_notification_config()` manage the configs on a task.

### Interrupts and approvals: `input-required`

A Locus run that pauses for a person maps onto the specification's
interrupted state. When the agent calls `ask_user`, or a gated tool holds
for approval (`gate_tool(..., on_refusal="interrupt")`), the task moves
to `input-required` and the status message carries the question. Its
`metadata` says what kind of pause it is:

| Pause | `metadata.kind` | Also in `metadata` |
|---|---|---|
| A question (`ask_user`) | `question` | `interruptId`, `options`, `fields` when the question declares them |
| An approval hold | `approval` | `interruptId`, `approvalId`, `hold` (the action, asset, principal, reason, arguments and approvers so far), `replies: ["approve", "deny"]` |

The peer answers by sending a message on the same task (`taskId` set). The run resumes through `Agent.resume()` with the message's text:

- for a question, as the answer
- for a hold, the first word decides it: `approve` (also `yes`, `allow`, `ok`, ...) or `deny` (`reject`, `no`, `refuse`, ...)

Anything else is refused with `InvalidParams` and the task stays paused.

Give the server the agent's approval store (`A2AServer(..., approvals=store)`) and the peer's verdict is recorded on the approval first, under `metadata.decidedBy` on the reply when set (`a2a:<contextId>` otherwise), so the store's authority rules and the audit trail see who decided; the held call is then re-issued under that decision.

Without `approvals=`, the reply only resumes the run and the gate acts on whatever the store holds: a hold still undecided answers `UnsupportedOperation` and keeps the task paused until someone decides it (`locus approvals approve ...`).

An approved hold performs the call once and the task runs on to
`completed`. A denied hold refuses the call and ends the task in
`rejected`, the terminal state for work the agent will not do, with a
status message `Denied by <who>: <question>`; the agent's closing reply,
if any, is attached as an artifact. `CancelTask` on a paused task cancels
it, and a message to a finished task is `UnsupportedOperation`.

Streaming callers see the pause as the final `statusUpdate` of
`SendStreamingMessage`; sending the follow-up with `SendStreamingMessage`
streams the resumed run. `SubscribeToTask` on a paused task returns the
snapshot and closes.

```python
task = await client.send_message(message)
if task.status.state == TaskState.input_required:
    meta = task.status.message.metadata or {}
    print(task.status.message.parts[0].text, meta.get("options"))
    reply = "approve" if meta.get("kind") == "approval" else input("> ")
    task = await client.send_message(
        Message(role="user", parts=[TextPart(text=reply)], messageId="m-2", taskId=task.id,
                metadata={"decidedBy": "alice@example.com"})
    )
```

### Generating the card: `locus a2a card`

`card_from_agent()` builds a v1.0 card from what the agent already
declares: one `AgentSkill` per skill in `config.skills` (tagged `skill`)
and per tool (tagged `tool`, plus `read-only` when the tool says so; the
built-in `ask_user` and `task_complete` are left out), the first line of
the system prompt as the description, `capabilities.streaming` on, and
one `AgentInterface` per transport. The command line writes it:

```bash
locus a2a card app.py:agent --url https://research.example.com -o agent-card.json
locus a2a card app.py:agent --url https://research.example.com \
    --transport jsonrpc --transport http+json --transport grpc --grpc-port 7422 --push
locus a2a card app.py:agent --url http://127.0.0.1:7421 --serve --port 7421 --api-key "$KEY"
```

`--serve` runs an `A2AServer` that publishes exactly that card, with the
transports, push setting and bearer key named on the command line.

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
    # ... gather input from the human, then send a follow-up message on
    # the same taskId (see "Interrupts and approvals" above)
    ...
# Or follow a running task live: the snapshot, then its events.
async for event in client.subscribe(task.id):
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
— `A2AServer`, `A2AClient`, JSON-RPC dispatch, in-memory task store, the
interrupt mapping.

[`a2a/rest.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/rest.py)
— the HTTP+JSON binding.

[`a2a/grpc_binding.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/grpc_binding.py)
— the gRPC binding, on the protobuf definition in `a2a/proto/`.

[`a2a/push.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/push.py)
— push-notification store, dispatcher, signer and receiver.

[`a2a/transports.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/transports.py)
— the client transports and interface selection.

[`a2a/card.py`](https://github.com/oracle/locus/blob/main/src/locus/a2a/card.py)
— `card_from_agent`, behind `locus a2a card`.

## See also

- [Multi-agent overview](../multi-agent.md) — pick a shape.
- [Agent Server](../server.md) — the FastAPI invoke/stream server.
  `A2AServer` is a separate server that wraps an agent through the same
  `run()` contract.
- [Conversation Management](../conversation-management.md) — history
  that survives across requests needs a `thread_id`. `A2AServer` runs
  the agent on the message text without one, so `contextId` groups
  tasks but does not carry the agent's conversation.
