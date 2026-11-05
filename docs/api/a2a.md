# A2A

`locus.a2a` implements the Agent-to-Agent protocol: one agent calls another
over HTTP, across process and organisation boundaries, without either side
importing the other's code.

Two generations of the wire format ship side by side. The `A2AV1*` types are
the v1 spec — use these for anything new. The unprefixed types are the earlier
shape, kept because deployed peers still speak it; `A2AServer` accepts both.

For the concepts, start with [the A2A protocol](../concepts/multi-agent/a2a.md)
and the [walkthrough notebook](../notebooks/notebook_30_a2a_protocol.md).

## Client and server

`A2AServer` exposes an agent over the protocol; `A2AClient` calls one. Neither
requires the other side to be built with the SDK.

To publish a compiled `StateGraph` instead of an agent, wrap it in
`locus.server.GraphRunnable` and pass that as `agent=` — see
[Serving a graph](server.md#serving-a-graph).

::: locus.a2a.protocol.A2AServer
::: locus.a2a.protocol.A2AClient

## Agent cards — discovery

An agent card is what a peer publishes about itself: who runs it, what it can
do, and how to reach it. It is the only thing a caller needs before the first
request.

::: locus.a2a.spec.AgentCard
::: locus.a2a.spec.AgentSkill
::: locus.a2a.spec.AgentCapabilities
::: locus.a2a.spec.AgentProvider
::: locus.a2a.spec.AgentInterface

## v1 protocol

The current wire format. `A2A_V1_PROTOCOL_VERSION` is the version string sent
on the wire and the one a peer negotiates against.

::: locus.a2a.spec_v1.A2A_V1_PROTOCOL_VERSION

### Sending a message

::: locus.a2a.spec_v1.A2AV1SendMessageRequest
::: locus.a2a.spec_v1.A2AV1SendMessageResponse
::: locus.a2a.spec_v1.A2AV1SendMessageConfiguration
::: locus.a2a.spec_v1.A2AV1Message
::: locus.a2a.spec_v1.A2AV1Part
::: locus.a2a.spec_v1.A2AV1Role

### Tasks

A request that is not answered immediately becomes a task the caller polls or
streams. `A2AV1TaskState` is the state machine; the update events are what a
streaming caller receives.

::: locus.a2a.spec_v1.A2AV1Task
::: locus.a2a.spec_v1.A2AV1TaskState
::: locus.a2a.spec_v1.A2AV1TaskStatus
::: locus.a2a.spec_v1.A2AV1GetTaskRequest
::: locus.a2a.spec_v1.A2AV1CancelTaskRequest
::: locus.a2a.spec_v1.A2AV1ListTasksRequest
::: locus.a2a.spec_v1.A2AV1ListTasksResponse

### Streaming and artifacts

::: locus.a2a.spec_v1.A2AV1StreamResponse
::: locus.a2a.spec_v1.A2AV1TaskStatusUpdateEvent
::: locus.a2a.spec_v1.A2AV1TaskArtifactUpdateEvent
::: locus.a2a.spec_v1.A2AV1Artifact

## Messages and parts

A message is a list of parts. `Part` is the discriminated union — a part is
text, a file, or structured data, and the `kind` field decides which.

::: locus.a2a.spec.Message
::: locus.a2a.spec.Part
::: locus.a2a.spec.TextPart
::: locus.a2a.spec.DataPart
::: locus.a2a.spec.FilePart
::: locus.a2a.spec.FileWithBytes
::: locus.a2a.spec.FileWithUri
::: locus.a2a.spec.Artifact

## Tasks

::: locus.a2a.spec.Task
::: locus.a2a.spec.TaskState
::: locus.a2a.spec.TaskStatus
::: locus.a2a.spec.TaskIdParams
::: locus.a2a.spec.TaskQueryParams
::: locus.a2a.spec.TaskStatusUpdateEvent
::: locus.a2a.spec.TaskArtifactUpdateEvent
::: locus.a2a.spec.MessageSendParams
::: locus.a2a.spec.MessageSendConfiguration

## Push notifications

For work long enough that polling is the wrong shape: the peer calls you back
when the task changes.

::: locus.a2a.spec.PushNotificationConfig
::: locus.a2a.spec.TaskPushNotificationConfig
::: locus.a2a.spec.PushNotificationAuthenticationInfo

## JSON-RPC envelope

The transport shapes. You rarely construct these directly — `A2AClient` and
`A2AServer` do — but an error response is worth being able to read.

::: locus.a2a.spec.JsonRpcRequest
::: locus.a2a.spec.JsonRpcSuccessResponse
::: locus.a2a.spec.JsonRpcErrorResponse
::: locus.a2a.spec.JsonRpcError

## Legacy shapes

The pre-v1 request/response types. `A2AServer` still accepts them so deployed
peers keep working; do not build anything new on them.

::: locus.a2a.protocol.A2AMessage
::: locus.a2a.protocol.A2ARequest
::: locus.a2a.protocol.A2AResponse
