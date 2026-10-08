# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The box runner's protocol with the gateway, against a fake gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from tests.unit.runner.conftest import RUN_ID, TOKEN, FakeGateway
from tulip import Agent, tool
from tulip.core.events import TerminateEvent
from tulip.core.state import AgentState
from tulip.hooks.provider import BeforeToolCallEvent, HookPriority
from tulip.runner import (
    ALLOWED_TYPES,
    AdmitResult,
    GatewayCheckpointer,
    GatewayClient,
    GatewayError,
    GatewayEvents,
    GatewayUnavailable,
    RemoteGate,
    RemoteTool,
    RemoteToolError,
    RunManifest,
    RunnerConfig,
    ToolEntry,
    canonical_digest,
    fetch_manifest,
    next_op,
)
from tulip.testing import ScriptedModel, text, tool_call


def _manifest(**over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": RUN_ID,
        "tenant": "acme",
        "agent": "refunds",
        "definition": {"name": "refunds", "instructions": "help"},
        "tools": [
            {"name": "lookup", "labels": ["read"], "runs": "box"},
            {"name": "request_secret", "runs": "gateway"},
            {"name": "search_docs", "runs": "mcp:docs"},
        ],
        "mcp": [{"id": "docs", "url": "https://docs.example/mcp"}],
        "model": {"model": "qwen", "base_url": "https://llm.example/v1", "api_key_env": "LLM_KEY"},
    }
    data.update(over)
    return data


# ── config and client ────────────────────────────────────────────────────────


def test_config_reads_the_box_environment_and_hides_the_token() -> None:
    config = RunnerConfig.from_env(
        {"TULIP_ADMIT_URL": "http://gw:8421/", "TULIP_ADMIT_TOKEN": TOKEN, "TULIP_RUN_ID": "r"}
    )
    assert config.url == "http://gw:8421"
    assert config.run_id == "r"
    assert TOKEN not in repr(config)
    assert TOKEN not in repr(GatewayClient(config))


@pytest.mark.parametrize("missing", ["TULIP_ADMIT_URL", "TULIP_ADMIT_TOKEN", "TULIP_RUN_ID"])
def test_config_names_a_missing_variable(missing: str) -> None:
    env = {"TULIP_ADMIT_URL": "http://gw", "TULIP_ADMIT_TOKEN": TOKEN, "TULIP_RUN_ID": "r"}
    env[missing] = " "
    with pytest.raises(ValueError, match=missing) as raised:
        RunnerConfig.from_env(env)
    assert TOKEN not in str(raised.value)


def test_config_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_ADMIT_URL", "http://gw")
    monkeypatch.setenv("TULIP_ADMIT_TOKEN", TOKEN)
    monkeypatch.setenv("TULIP_RUN_ID", "r9")
    assert RunnerConfig.from_env().run_id == "r9"


async def test_every_call_carries_the_workload_token(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    gateway.on("GET", "/internal/v1/runner/next", {"op": "start"})
    await next_op(client)
    assert gateway.requests[0].headers["authorization"] == f"Bearer {TOKEN}"


async def test_client_tells_refusals_from_outages(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    gateway.on("GET", "/a", lambda r: httpx.Response(403, json={"detail": "not yours"}))
    gateway.on("GET", "/b", lambda r: httpx.Response(503, text="busy"))
    gateway.on("GET", "/c", lambda r: httpx.Response(400, text="plain"))
    gateway.on("GET", "/d", lambda r: httpx.Response(422, json=["bad"]))
    gateway.on("GET", "/e", lambda r: httpx.Response(204))
    with pytest.raises(GatewayError) as refused:
        await client.request("GET", "/a")
    assert (refused.value.status, refused.value.detail) == (403, "not yours")
    with pytest.raises(GatewayUnavailable):
        await client.request("GET", "/b")
    with pytest.raises(GatewayError, match="plain"):
        await client.request("GET", "/c")
    with pytest.raises(GatewayError, match="bad"):
        await client.request("GET", "/d")
    assert await client.request("GET", "/e") is None
    assert await client.request("GET", "/missing", allow_404=True) is None
    gateway.down = True
    with pytest.raises(GatewayUnavailable) as down:
        await client.request("GET", "/a")
    assert TOKEN not in str(down.value)
    await client.aclose()


# ── manifest ─────────────────────────────────────────────────────────────────


def test_manifest_digest_is_canonical() -> None:
    a = RunManifest.model_validate(_manifest())
    b = RunManifest.model_validate(json.loads(json.dumps(_manifest(), sort_keys=True)))
    assert a.digest() == b.digest()
    assert a.digest().startswith("sha256:")
    assert a.digest() == canonical_digest(a.model_dump(mode="json"))
    changed = RunManifest.model_validate(_manifest(instructions="different"))
    assert changed.digest() != a.digest()
    assert RunManifest.from_json(a.model_dump_json()) == a


def test_manifest_lookups() -> None:
    manifest = RunManifest.model_validate(_manifest())
    search = manifest.tool("search_docs")
    assert search is not None
    assert search.mount == "docs"
    lookup = manifest.tool("lookup")
    assert lookup is not None
    assert lookup.mount is None
    assert manifest.tool("nope") is None


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"tools": [{"name": "x"}, {"name": "x"}]}, "listed twice"),
        ({"tools": [{"name": "x", "runs": "mcp:gone"}]}, "not mounted"),
        ({"tools": [{"name": "x", "runs": "elsewhere"}]}, "runs must be"),
        ({"surprise": 1}, "Extra inputs"),
        ({"version": 2}, "version"),
    ],
)
def test_manifest_refuses_what_it_cannot_run(over: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RunManifest.model_validate(_manifest(**over))


# ── handshake ────────────────────────────────────────────────────────────────


async def test_next_op_start_resume_stop(gateway: FakeGateway, client: GatewayClient) -> None:
    path = "/internal/v1/runner/next"
    gateway.on("GET", path, {"op": "start"})
    assert (await next_op(client)).op == "start"
    gateway.on(
        "GET", path, {"op": "resume", "resume": {"approval_id": "ap1", "decision": "approve"}}
    )
    resumed = await next_op(client)
    assert (resumed.op, resumed.approval_id, resumed.decision) == ("resume", "ap1", "approve")
    gateway.on("GET", path, {"op": "resume", "resume": {"answer": "yes"}})
    answered = await next_op(client)
    assert answered.approval_id is None
    assert answered.decision is None
    gateway.on("GET", path, {"op": "stop", "reason": "cancelled"})
    assert (await next_op(client)).reason == "cancelled"
    gateway.on("GET", path, {"op": "dance"})
    assert (await next_op(client)).op == "stop"
    gateway.on("GET", path, {"op": "resume"})
    assert (await next_op(client)).op == "stop"


async def test_fetch_manifest_checks_the_run(gateway: FakeGateway, client: GatewayClient) -> None:
    gateway.on("GET", "/internal/v1/runner/manifest", _manifest())
    assert (await fetch_manifest(client)).agent == "refunds"
    gateway.on("GET", "/internal/v1/runner/manifest", _manifest(run_id="someone-else"))
    with pytest.raises(ValueError, match="someone-else"):
        await fetch_manifest(client)


# ── the gate ─────────────────────────────────────────────────────────────────


def _event(name: str = "lookup", call_id: str = "c1", **arguments: Any) -> BeforeToolCallEvent:
    return BeforeToolCallEvent(name, call_id, dict(arguments))


async def test_gate_sends_only_the_call_and_runs_an_allowed_one(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    gateway.on(
        "POST",
        "/v1/admit",
        {
            "outcome": "allow",
            "allowed": True,
            "reason": "ok",
            "decision_token": "dt-1",
            "labels": ["read"],
            "audit_id": "a1",
            "policy_fingerprint": "fp",
        },
    )
    gate = RemoteGate(client, token_argument="decision_token")  # noqa: S106 — an argument name
    event = _event(order_id="o1")
    await gate.on_before_tool_call(event)
    assert event.cancel is False
    assert event.secret_arguments == {"decision_token": "dt-1"}
    [sent] = gateway.bodies("POST", "/v1/admit")
    assert sent == {
        "run_id": RUN_ID,
        "call_id": "c1",
        "tool": "lookup",
        "arguments": {"order_id": "o1"},
    }
    assert gate.decision_token("c1") == "dt-1"
    assert gate.decision_token("c1") is None  # given out once
    assert gate.priority == HookPriority.SECURITY_MIN
    hooks = gate.register_hooks()
    assert hooks["on_before_tool_call"]
    assert not hooks["on_after_tool_call"]


async def test_gate_cancels_a_denied_call(gateway: FakeGateway, client: GatewayClient) -> None:
    gateway.on("POST", "/v1/admit", {"outcome": "deny", "allowed": False, "reason": "never"})
    event = _event("rm")
    await RemoteGate(client).on_before_tool_call(event)
    assert event.cancel == "denied by the gateway: never"


async def test_gate_records_a_hold_it_did_not_wait_for(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    gateway.on(
        "POST",
        "/v1/admit",
        {
            "outcome": "require_human",
            "allowed": False,
            "reason": "blast radius 2",
            "approval_id": "ap1",
        },
    )
    gate = RemoteGate(client)
    event = _event("bash")
    await gate.on_before_tool_call(event)
    assert isinstance(event.cancel, str)
    assert "ap1" in event.cancel
    assert [(h.call_id, h.tool, h.approval_id) for h in gate.held] == [("c1", "bash", "ap1")]


async def test_gate_waits_for_an_approval_then_asks_again(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    answers = iter(
        [
            {"outcome": "require_human", "allowed": False, "reason": "held", "approval_id": "ap1"},
            {"outcome": "allow", "allowed": True, "reason": "approved", "decision_token": "dt-2"},
        ]
    )
    gateway.on("POST", "/v1/admit", lambda r: httpx.Response(200, json=next(answers)))
    states = iter(["pending", "approved"])
    gateway.on(
        "GET", "/v1/admit/approval/ap1", lambda r: httpx.Response(200, json={"state": next(states)})
    )
    gate = RemoteGate(client, hold_wait_s=5, poll_interval_s=0.01)
    event = _event("bash", command="ls")
    await gate.on_before_tool_call(event)
    assert event.cancel is False
    first, second = gateway.bodies("POST", "/v1/admit")
    assert "approval_id" not in first
    assert second["approval_id"] == "ap1"
    assert gate.held == []


@pytest.mark.parametrize("state", ["denied", "expired"])
async def test_gate_cancels_when_the_hold_settles_without_approval(
    gateway: FakeGateway, client: GatewayClient, state: str
) -> None:
    gateway.on(
        "POST",
        "/v1/admit",
        {"outcome": "require_human", "allowed": False, "reason": "held", "approval_id": "ap1"},
    )
    gateway.on("GET", "/v1/admit/approval/ap1", {"state": state})
    event = _event()
    await RemoteGate(client, hold_wait_s=1, poll_interval_s=0.01).on_before_tool_call(event)
    assert event.cancel == f"denied by the gateway: the hold was {state}"


async def test_gate_parks_on_a_hold_still_pending_after_its_wait(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    gateway.on(
        "POST",
        "/v1/admit",
        {"outcome": "require_human", "allowed": False, "reason": "held", "approval_id": "ap1"},
    )
    gateway.on("GET", "/v1/admit/approval/ap1", {})
    gate = RemoteGate(client, hold_wait_s=0.05, poll_interval_s=0.01)
    event = _event()
    await gate.on_before_tool_call(event)
    assert [h.approval_id for h in gate.held] == ["ap1"]


async def test_gate_fails_closed(gateway: FakeGateway, client: GatewayClient) -> None:
    gateway.down = True
    event = _event()
    await RemoteGate(client).on_before_tool_call(event)
    assert isinstance(event.cancel, str)
    assert event.cancel.startswith("not run")
    gateway.down = False
    gateway.on("POST", "/v1/admit", lambda r: httpx.Response(401, json={"detail": "bad token"}))
    refused = _event()
    await RemoteGate(client).on_before_tool_call(refused)
    assert isinstance(refused.cancel, str)
    assert "bad token" in refused.cancel


@pytest.mark.parametrize(
    "body",
    [
        "not an object",
        {"outcome": "maybe"},
        {"outcome": "allow", "allowed": False},
        {"outcome": "require_human"},  # a hold with no approval to wait on
    ],
)
async def test_gate_treats_a_strange_answer_as_a_denial(
    gateway: FakeGateway, client: GatewayClient, body: Any
) -> None:
    gateway.on("POST", "/v1/admit", body)
    event = _event()
    await RemoteGate(client).on_before_tool_call(event)
    assert isinstance(event.cancel, str)
    assert event.cancel.startswith("denied by the gateway")


def test_admit_result_hides_its_token() -> None:
    result = AdmitResult.from_body({"outcome": "allow", "decision_token": "dt-secret"})
    assert result.allowed
    assert "dt-secret" not in repr(result)


@tool
async def refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


async def test_a_real_agent_runs_only_what_the_gateway_admits(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    def decide(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        allowed = body["arguments"]["order_id"] == "o1"
        return httpx.Response(
            200,
            json={
                "outcome": "allow" if allowed else "deny",
                "allowed": allowed,
                "reason": "policy",
            },
        )

    gateway.on("POST", "/v1/admit", decide)
    model = ScriptedModel(
        [
            tool_call("refund", call_id="t1", order_id="o1"),
            tool_call("refund", call_id="t2", order_id="o2"),
            text("done"),
        ]
    )
    agent = Agent(model=model, tools=[refund], hooks=[RemoteGate(client)], reflexion=False)
    events = [e async for e in agent.run("refund both")]
    assert isinstance(events[-1], TerminateEvent)
    seen = " ".join(m.content or "" for m in model.received_messages[-1])
    assert "refunded o1" in seen
    assert "refunded o2" not in seen
    assert "denied by the gateway" in seen
    assert [b["call_id"] for b in gateway.bodies("POST", "/v1/admit")] == ["t1", "t2"]


# ── events ───────────────────────────────────────────────────────────────────


async def test_events_are_batched_and_numbered(
    gateway: FakeGateway, client: GatewayClient, tmp_path: Path
) -> None:
    path = f"/internal/v1/runs/{RUN_ID}/events"
    gateway.on("POST", path, {})
    events = GatewayEvents(client, spool=tmp_path / "s.jsonl", batch_size=2)
    for n in range(3):
        await events.emit({"type": "token", "text": str(n)})
    assert events.next_seq == 3
    assert await events.flush()
    assert await events.flush()  # nothing left: no request
    batches = gateway.bodies("POST", path)
    assert [b["seq_from"] for b in batches] == [0, 2]
    assert [len(b["events"]) for b in batches] == [2, 1]


async def test_events_refuse_what_only_the_gateway_writes(
    client: GatewayClient, tmp_path: Path
) -> None:
    events = GatewayEvents(client, spool=tmp_path / "s.jsonl")
    for kind in ("decision", "approval", "done", "run_start", "net.egress.denied", None):
        assert kind not in ALLOWED_TYPES
        with pytest.raises(ValueError, match="may not report"):
            await events.emit({"type": kind})


async def test_events_spool_through_an_outage_and_replay_in_order(
    gateway: FakeGateway, client: GatewayClient, tmp_path: Path
) -> None:
    path = f"/internal/v1/runs/{RUN_ID}/events"
    gateway.on("POST", path, {})
    spool = tmp_path / "tulip" / "events.jsonl"
    events = GatewayEvents(client, spool=spool, batch_size=10)
    gateway.down = True
    await events.emit({"type": "think", "text": "a"})
    assert not await events.flush()
    await events.emit({"type": "think", "text": "b"})
    assert not await events.flush()
    assert events.spooled == 2
    gateway.down = False
    gateway.requests.clear()
    await events.emit({"type": "think", "text": "c"})
    assert await events.flush()
    assert not spool.exists()
    assert events.spooled == 0
    assert [b["seq_from"] for b in gateway.bodies("POST", path)] == [0, 1, 2]


async def test_events_keep_what_a_partial_replay_did_not_send(
    gateway: FakeGateway, client: GatewayClient, tmp_path: Path
) -> None:
    path = f"/internal/v1/runs/{RUN_ID}/events"
    spool = tmp_path / "events.jsonl"
    events = GatewayEvents(client, spool=spool)
    gateway.down = True
    for text_ in ("a", "b"):
        await events.emit({"type": "token", "text": text_})
        await events.flush()
    gateway.down = False
    calls = {"n": 0}

    def second_fails(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200 if calls["n"] == 1 else 502)

    gateway.on("POST", path, second_fails)
    assert not await events.flush()
    remaining = [json.loads(line) for line in spool.read_text().splitlines()]
    assert [b["seq_from"] for b in remaining] == [1]


async def test_events_raise_a_refusal(
    gateway: FakeGateway, client: GatewayClient, tmp_path: Path
) -> None:
    gateway.on(
        "POST",
        f"/internal/v1/runs/{RUN_ID}/events",
        lambda r: httpx.Response(422, json={"detail": "seq"}),
    )
    events = GatewayEvents(client, spool=tmp_path / "s.jsonl", start_seq=5)
    await events.emit({"type": "token"})
    with pytest.raises(GatewayError, match="seq"):
        await events.flush()
    assert events.spooled == 0


# ── checkpoints ──────────────────────────────────────────────────────────────


async def test_checkpoints_round_trip_through_the_gateway(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    base = f"/internal/v1/runs/{RUN_ID}"
    stored: dict[str, Any] = {}

    def put(request: httpx.Request) -> httpx.Response:
        stored.update(json.loads(request.content))
        return httpx.Response(200, json={"checkpoint_id": "cp-1"})

    gateway.on("PUT", f"{base}/checkpoint", put)
    gateway.on(
        "GET",
        f"{base}/checkpoint",
        lambda r: httpx.Response(200, json={"checkpoint_id": "cp-1", "state": stored["state"]}),
    )
    gateway.on("GET", f"{base}/checkpoints", {"checkpoint_ids": ["cp-1", "cp-0"]})
    checkpointer = GatewayCheckpointer(client)
    state = AgentState()
    assert await checkpointer.save(state, "t", metadata={"why": "park"}) == "cp-1"
    assert stored["thread_id"] == "t"
    assert stored["metadata"] == {"why": "park"}
    loaded = await checkpointer.load("t", "cp-1")
    assert loaded is not None
    assert gateway.requests[-1].url.params["checkpoint_id"] == "cp-1"
    assert await checkpointer.list_checkpoints("t", limit=1) == ["cp-1"]


async def test_checkpoints_missing_and_unnamed(gateway: FakeGateway, client: GatewayClient) -> None:
    checkpointer = GatewayCheckpointer(client)
    assert await checkpointer.load("t") is None  # 404
    assert await checkpointer.list_checkpoints("t") == []
    gateway.on("PUT", f"/internal/v1/runs/{RUN_ID}/checkpoint", {})
    assert await checkpointer.save(AgentState(), "t", checkpoint_id="mine") == "mine"
    with pytest.raises(ValueError, match="did not name"):
        await checkpointer.save(AgentState(), "t")


# ── gateway-side tools ───────────────────────────────────────────────────────


async def test_remote_tool_is_performed_by_the_gateway(
    gateway: FakeGateway, client: GatewayClient
) -> None:
    path = f"/internal/v1/runs/{RUN_ID}/tools/request_secret"
    gateway.on("POST", path, {"content": "granted"})
    entry = ToolEntry(name="request_secret", runs="gateway")
    remote = RemoteTool(client, entry)
    assert remote.name == "request_secret"
    assert await remote.call("c9", {"name": "crm"}) == "granted"
    assert gateway.bodies("POST", path) == [{"call_id": "c9", "arguments": {"name": "crm"}}]
    gateway.on("POST", path, {"error": "no such secret"})
    with pytest.raises(RemoteToolError, match="no such secret"):
        await remote.call("c10", {})
    with pytest.raises(ValueError, match="not on the gateway"):
        RemoteTool(client, ToolEntry(name="lookup"))
