# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 28: A2A — cross-org payment-intel sharing between spec-compliant peers.

A2A (Agent-to-Agent) is the public cross-framework protocol at
https://a2aproject.github.io/A2A/. Tulip implements the server and
client sides on the A2A **v1.0** wire format; this notebook publishes one
payment network's risk agent behind ``A2AServer``, drives every v1.0
method from a partner bank's ``A2AClient``, and inspects the typed
task lifecycle — the plumbing of a cross-org payment-risk mesh.

Trust note: a cross-org payment mesh spans organizational boundaries,
so treat every inbound message as untrusted input. The wire here is
authenticated with a bearer token and the partner agent's reach is
bounded by its published ``AgentSkill`` set, but a partner's response
can still carry indirect prompt injection the same way any other tool
output can. Ground what a peer asserts about a merchant or a card BIN;
do not promote it to a decline or a chargeback unverified.

- Agent Card published at ``/.well-known/agent-card.json`` carries
  capabilities, typed ``AgentSkill`` entries, provider metadata,
  ``protocolVersion``, and ``supportedInterfaces`` — enough for any
  partner bank's A2A client to discover and call the agent.
- A2A v1.0 JSON-RPC methods over ``POST /`` with
  ``A2A-Version: 1.0``: ``SendMessage`` (synchronous round-trip),
  ``GetTask`` (poll by id), ``ListTasks`` (page by context / status),
  ``CancelTask`` (terminal-state errors surface as the spec's
  ``TaskNotCancelable`` code), and ``SendStreamingMessage`` (SSE
  lifecycle events as v1.0 ``StreamResponse`` envelopes).
- The Python surface (``Message``, ``TextPart``, ``Task``, ...) is
  unchanged; the v1.0 conversion happens at the HTTP boundary.
- ``A2AClient.invoke`` is the backwards-compatible flat shape for peers
  that haven't picked up the spec yet.
- ``A2AClient.as_tool(...)`` wraps a remote agent as a Tulip ``@tool``
  so a local payments agent can delegate risk questions transparently.

Run it:
    .venv/bin/python examples/notebook_28_a2a_protocol.py

The default provider is the bundled mock model. Set TULIP_MODEL_PROVIDER=openai
(or anthropic) and the matching credentials to use a live model. Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs.

Prerequisites:
- ``pip install fastapi uvicorn`` for the server side.
- Notebook 06 (Agent basics). The wire format is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

from config import get_model

from tulip.a2a import (
    A2AClient,
    A2AServer,
    AgentSkill,
    Message,
    TaskState,
    TextPart,
)
from tulip.agent import Agent, AgentConfig


def _free_port() -> int:
    # Bind-and-release to grab an unused port. Small TOCTOU window, fine
    # for a notebook.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _start_server(server: A2AServer, port: int) -> threading.Thread:
    """Run uvicorn in a daemon thread so the client below can drive it."""
    import uvicorn

    config = uvicorn.Config(
        app=server.app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True, name="a2a-server")
    thread.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not uv.started:
        time.sleep(0.05)
    if not uv.started:
        msg = "uvicorn did not start within deadline"
        raise RuntimeError(msg)
    return thread


async def main() -> None:
    print("=" * 60)
    print("Notebook 28: A2A — cross-org payment-risk sharing mesh")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Part 1: publish the Network's risk agent behind A2A
    # ---------------------------------------------------------------
    print("\n=== Part 1: A2AServer with typed skills ===\n")

    model = get_model()
    risk_agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You are a payment-risk analyst for Acme Pay network. Reply in one short sentence."
            ),
            max_iterations=2,
            model=model,
        )
    )

    port = _free_port()
    # Demo bearer token for the partner mesh, not a real credential.
    api_key = "partner-mesh-token"  # noqa: S105

    server = A2AServer(
        agent=risk_agent,
        name="acme-pay-risk",
        description="Shares merchant risk context and chargeback summaries with partner banks.",
        url=f"http://127.0.0.1:{port}",
        skills=[
            AgentSkill(
                id="merchant-risk",
                name="Merchant Risk",
                description="Summarise what Acme Pay knows about a merchant's risk profile.",
                tags=["payments", "risk"],
                examples=["What do we know about the merchant MID-4471?"],
            ),
        ],
        api_key=api_key,
    )
    _start_server(server, port)
    print(f"  A2AServer listening on http://127.0.0.1:{port}")

    base_url = f"http://127.0.0.1:{port}"
    # The partner bank's client — authenticated like any partner in the mesh.
    client = A2AClient(url=base_url, api_key=api_key)

    # ---------------------------------------------------------------
    # Part 2: discover via /.well-known/agent-card.json
    # ---------------------------------------------------------------
    print("\n=== Part 2: Agent Card discovery ===\n")
    card = await client.get_agent_card()
    print(f"  name:             {card.name}")
    print(f"  description:      {card.description}")
    print(f"  url:              {card.url}")
    print(f"  protocolVersion:  {card.protocolVersion}")
    print(
        f"  capabilities:     streaming={card.capabilities.streaming} "
        f"push={card.capabilities.pushNotifications}"
    )
    for iface in card.supportedInterfaces or []:
        print(f"  interface:        {iface.protocolBinding} v{iface.protocolVersion} @ {iface.url}")
    for skill in card.skills:
        print(f"  skill:            {skill.id} — {skill.name} (tags={skill.tags})")

    # ---------------------------------------------------------------
    # Part 3: SendMessage (v1.0) returns a typed Task
    # ---------------------------------------------------------------
    # A2AClient defaults to protocol_version="1.0", so send_message
    # dispatches the v1.0 SendMessage method with `A2A-Version: 1.0`.
    print("\n=== Part 3: SendMessage (v1.0) → Task ===\n")
    task = await client.send_message(
        Message(
            role="user",
            parts=[TextPart(text="What do we know about the merchant MID-4471?")],
            messageId="m-1",
        )
    )
    print(f"  task.id:           {task.id}")
    print(f"  task.contextId:    {task.contextId}")
    print(f"  task.status.state: {task.status.state.value}")
    if task.artifacts:
        first_part = task.artifacts[-1].parts[0]
        text = getattr(first_part, "text", "")
        print(f"  reply artifact:    {text[:120]}")

    # ---------------------------------------------------------------
    # Part 4: GetTask (v1.0) — poll the task by id
    # ---------------------------------------------------------------
    print("\n=== Part 4: GetTask (v1.0) ===\n")
    refetched = await client.get_task(task.id)
    print(
        f"  re-fetched task is in {refetched.status.state.value} state "
        f"(== completed: {refetched.status.state == TaskState.completed})"
    )

    # ---------------------------------------------------------------
    # Part 5: ListTasks (v1.0) — page tasks by context / status
    # ---------------------------------------------------------------
    print("\n=== Part 5: ListTasks (v1.0) ===\n")
    listed, next_page = await client.list_tasks(
        context_id=task.contextId,
        status=TaskState.completed,
        page_size=10,
    )
    print(f"  context {task.contextId} → {len(listed)} task(s); next_page={next_page!r}")
    for t in listed:
        print(f"    - {t.id}  state={t.status.state.value}")

    # ---------------------------------------------------------------
    # Part 6: CancelTask (v1.0) on a terminal task surfaces TaskNotCancelable
    # ---------------------------------------------------------------
    print("\n=== Part 6: CancelTask (v1.0) on a terminal task ===\n")
    try:
        await client.cancel_task(task.id)
    except RuntimeError as e:
        print(f"  spec error surfaced: {e}")

    # ---------------------------------------------------------------
    # Part 7: SendStreamingMessage (v1.0) — SSE lifecycle events
    # ---------------------------------------------------------------
    print("\n=== Part 7: SendStreamingMessage (v1.0) ===\n")
    seen: list[str] = []
    async for event in client.send_message_streaming(
        Message(
            role="user",
            parts=[TextPart(text="Stream a one-sentence assessment of the card BIN 411111.")],
            messageId="m-2",
        )
    ):
        kind = event.get("kind") or "?"
        seen.append(kind)
        if kind == "task":
            print(f"  initial task envelope: id={event.get('id')}")
        elif kind == "status-update":
            state = event.get("status", {}).get("state")
            print(f"  status-update: state={state}")
        elif kind == "artifact-update":
            artifact = event.get("artifact", {})
            parts = artifact.get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            print(f"  artifact-update: {text[:120]}")
    print(f"  total events: {len(seen)}")

    # ---------------------------------------------------------------
    # Part 8: opt out of v1.0 — legacy JSON-RPC method names
    # ---------------------------------------------------------------
    # Pass protocol_version=None to drop the A2A-Version header and use
    # the legacy methods (message/send, tasks/get, ...). Useful only
    # when a partner bank hasn't picked up v1.0 yet; A2AServer still
    # understands both.
    print("\n=== Part 8: legacy protocol_version=None ===\n")
    legacy = A2AClient(url=base_url, api_key=api_key, protocol_version=None)
    legacy_task = await legacy.send_message(
        Message(
            role="user",
            parts=[TextPart(text="One sentence: why share payment-risk signals across banks?")],
            messageId="m-3",
        )
    )
    print(f"  legacy task.status.state: {legacy_task.status.state.value}")

    # ---------------------------------------------------------------
    # Part 9: backwards-compat flat invoke for non-spec peers
    # ---------------------------------------------------------------
    print("\n=== Part 9: legacy /a2a/invoke (backwards-compat) ===\n")
    text = await client.invoke("Give me a one-line summary of the current card-testing fraud wave.")
    print(f"  flat reply: {text[:120]}")

    # ---------------------------------------------------------------
    # Part 10: wrap a partner bank's agent as a local @tool
    # ---------------------------------------------------------------
    print("\n=== Part 10: A2AClient.as_tool ===\n")
    tool = client.as_tool(
        name="ask_partner_risk", description="ask the partner network's payment-risk agent"
    )
    print(f"  tool.name = {tool.name}, tool.description = {tool.description}")
    # tool.fn calls asyncio.run() internally, so it can only be invoked
    # from sync code. We're already inside an async main, so just
    # inspect the tool object instead of calling it.

    print("\n" + "=" * 60)
    print("Next: Notebook 29 — DeepAgent")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
