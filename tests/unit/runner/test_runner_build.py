# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The box runner end to end: build from a manifest, run, park, resume, report."""

# Decision tokens here are test values the fake gateway hands out.
# ruff: noqa: S106

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.unit.runner.conftest import RUN_ID, TOKEN
from tulip.core.events import ModelChunkEvent, ThinkEvent, ToolCompleteEvent, ToolStartEvent
from tulip.core.messages import ToolCall
from tulip.harness import LocalBackend
from tulip.runner import (
    ADMIT_TOKEN_VAR,
    ADMIT_URL_VAR,
    DECISION_HEADER,
    RUN_ID_VAR,
    ApiConnector,
    ApiOperation,
    ConnectorError,
    GatewayClient,
    McpClient,
    McpMount,
    RunManifest,
    RunnerConfig,
    RunnerRefused,
    RunResult,
    api_request,
    build_runtime,
    call_api,
    mint_child,
    report_result,
    run_box,
)
from tulip.runner.build import TOKEN_ARGUMENT, system_prompt
from tulip.runner.harden import command_env, make_non_dumpable, scrub_environ, withheld_names
from tulip.runner.main import EXIT_DONE, EXIT_ERROR, EXIT_REFUSED, event_for, main
from tulip.testing import ScriptedModel, text, tool_call


MODEL_KEY_ENV = "TULIP_MODEL_KEY"
API_KEY_ENV = "TULIP_CRM_KEY"
MCP_KEY_ENV = "TULIP_MCP_KEY"


def _call(name: str, call_id: str, **arguments: Any) -> Any:
    """A turn calling ``name``; unlike ``tool_call`` it can pass an argument named ``content``."""
    turn = tool_call(name, call_id=call_id)
    turn.message.tool_calls[0] = ToolCall(id=call_id, name=name, arguments=arguments)
    return turn


def _env(**extra: str) -> dict[str, str]:
    return {
        ADMIT_URL_VAR: "http://gateway.test",
        ADMIT_TOKEN_VAR: TOKEN,
        RUN_ID_VAR: RUN_ID,
        MODEL_KEY_ENV: "openshell:resolve:env:v1_model_KEY",
        API_KEY_ENV: "openshell:resolve:env:v1_crm_KEY",
        MCP_KEY_ENV: "openshell:resolve:env:v1_mcp_KEY",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        **extra,
    }


def _manifest(**over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": RUN_ID,
        "tenant": "acme",
        "agent": "box-agent",
        "definition": {"name": "box-agent", "environment": "dev"},
        "instructions": "You fix things.",
        "input": "make it so",
        "tools": [
            {"name": "write", "runs": "box"},
            {"name": "read", "runs": "box"},
            {"name": "bash", "runs": "box"},
            {"name": "ask_user", "runs": "box"},
            {
                "name": "request_network_access",
                "runs": "gateway",
                "parameters": {"type": "object", "properties": {"host": {"type": "string"}}},
            },
            {
                "name": "crm_get",
                "runs": "api:crm",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "full": {"type": "boolean"}},
                },
                "operation": {"method": "GET", "path": "/v1/customers/{id}", "query": ["full"]},
            },
            {
                "name": "lookup",
                "runs": "mcp:kb",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        ],
        "mcp": [{"id": "kb", "url": "http://kb.test/mcp", "credential_env": MCP_KEY_ENV}],
        "api": [{"id": "crm", "base_url": "http://crm.test/api", "credential_env": API_KEY_ENV}],
        "budgets": {"park_ttl_s": 0},
        "model": {
            "model": "spark-only",
            "base_url": "http://llm.test/v1",
            "api_key_env": MODEL_KEY_ENV,
        },
    }
    data.update(over)
    return data


@dataclass
class World:
    """The gateway, an API host and an MCP server, behind one mock transport."""

    manifest: dict[str, Any] = field(default_factory=_manifest)
    next_ops: list[dict[str, Any]] = field(default_factory=lambda: [{"op": "start"}])
    admit: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval_state: str = "pending"
    checkpoints: dict[str, Any] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    still_pending: bool = False
    upstream_status: int = 200

    def bodies(self, path: str) -> list[Any]:
        return [
            json.loads(r.content) if r.content else None
            for r in self.requests
            if r.url.path == path and r.url.host == "gateway.test"
        ]

    def upstream(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def _admit(self, body: dict[str, Any]) -> dict[str, Any]:
        rule = self.admit.get(body["tool"]) or {"outcome": "allow", "allowed": True}
        if (
            body.get("approval_id")
            and rule.get("outcome") == "require_human"
            and not self.still_pending
        ):
            return {"outcome": "allow", "allowed": True, "decision_token": f"dt-{body['call_id']}"}
        answer = {"reason": "policy", **rule}
        if answer.get("outcome") == "allow":
            answer.setdefault("decision_token", f"dt-{body['call_id']}")
        return answer

    def __call__(self, request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "crm.test":
            return httpx.Response(self.upstream_status, json={"id": "c1", "name": "Ada"})
        if host == "kb.test":
            message = json.loads(request.content)
            if message.get("method") == "initialize":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": message["id"], "result": {}},
                    headers={"mcp-session-id": "s-1"},
                )
            if "id" not in message:
                return httpx.Response(202)
            if self.upstream_status != 200:
                return httpx.Response(self.upstream_status, text="kb is down")
            sse = (
                "event: message\n"
                "data: "
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {"content": [{"type": "text", "text": "kb says hi"}]},
                    }
                )
                + "\n\n"
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        body = json.loads(request.content) if request.content else None
        if path == "/internal/v1/runner/next":
            return httpx.Response(200, json=self.next_ops.pop(0))
        if path == "/internal/v1/runner/manifest":
            return httpx.Response(200, json=self.manifest)
        if path == "/v1/admit":
            return httpx.Response(200, json=self._admit(body))
        if path.startswith("/v1/admit/approval/"):
            return httpx.Response(200, json={"state": self.approval_state})
        if path.endswith("/checkpoint") and request.method == "PUT":
            self.checkpoints[body["thread_id"]] = body["state"]
            return httpx.Response(200, json={"checkpoint_id": f"cp{len(self.checkpoints)}"})
        if path.endswith("/checkpoint") and request.method == "GET":
            state = self.checkpoints.get(request.url.params["thread_id"])
            if state is None:
                return httpx.Response(404, json={"detail": "none"})
            return httpx.Response(200, json={"checkpoint_id": "cp", "state": state})
        if path.endswith("/checkpoints"):
            return httpx.Response(200, json={"checkpoint_ids": list(self.checkpoints)})
        if path.endswith(("/events", "/result")):
            return httpx.Response(200, json={})
        if "/tools/" in path:
            if body["arguments"].get("host") == "evil.test":
                return httpx.Response(200, json={"error": "not a host you may ask for"})
            return httpx.Response(200, json={"content": f"granted {body['arguments']['host']}"})
        if path.endswith("/children"):
            return httpx.Response(200, json={"run_id": "run-child", "admit_token": "child-tok"})
        return httpx.Response(404, json={"detail": f"no route {path}"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def results(self, run_id: str = RUN_ID) -> list[dict[str, Any]]:
        return self.bodies(f"/internal/v1/runs/{run_id}/result")

    def events(self, run_id: str = RUN_ID) -> list[dict[str, Any]]:
        return [e for b in self.bodies(f"/internal/v1/runs/{run_id}/events") for e in b["events"]]


@pytest.fixture
def world() -> World:
    return World()


def _client(world: World, run_id: str = RUN_ID) -> GatewayClient:
    config = RunnerConfig(url="http://gateway.test", run_id=run_id, token=TOKEN)
    return GatewayClient(config, transport=world.transport)


# ── API connectors ────────────────────────────────────────────────────────────


CRM = ApiConnector(id="crm", base_url="https://crm.test/api/", credential_env=API_KEY_ENV)


def test_api_request_maps_arguments_to_one_request() -> None:
    op = ApiOperation(method="POST", path="/v1/{kind}/{id}", query=("dry",), body="json")
    request = api_request(CRM, op, {"kind": "a b", "id": 7, "dry": True, "note": "x", "n": 2})
    assert request.method == "POST"
    assert request.url == "https://crm.test/api/v1/a%20b/7"
    assert request.host == "crm.test"
    assert request.path == "/api/v1/a%20b/7"
    assert request.query == (("dry", "true"),)
    assert request.query_string == "dry=true"
    assert request.body == b'{"n":2,"note":"x"}'
    # Same arguments in another order: the same bytes.
    again = api_request(CRM, op, {"n": 2, "note": "x", "dry": True, "id": 7, "kind": "a b"})
    assert again == request


def test_api_request_refuses_what_has_nowhere_to_go() -> None:
    op = ApiOperation(path="/v1/{id}")
    with pytest.raises(ConnectorError, match="argument 'id' missing"):
        api_request(CRM, op, {})
    with pytest.raises(ConnectorError, match="takes no body"):
        api_request(CRM, op, {"id": "1", "extra": 1})
    assert api_request(CRM, ApiOperation(path="/x", query=("q",)), {"q": None}).query == ()


def test_api_operation_path_must_be_relative_to_root() -> None:
    with pytest.raises(ValueError, match="must start with"):
        ApiOperation(path="v1/x")


async def test_call_api_sends_token_and_placeholder_only(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "openshell:resolve:env:v1_crm_KEY")
    op = ApiOperation(path="/v1/customers/{id}", query=("full",))
    text_ = await call_api(
        CRM, op, {"id": "c1", "full": False}, decision_token="dt-1", transport=world.transport
    )
    assert "Ada" in text_
    [sent] = world.upstream("crm.test")
    assert sent.headers[DECISION_HEADER] == "dt-1"
    assert sent.headers["authorization"] == "Bearer openshell:resolve:env:v1_crm_KEY"
    assert sent.url.path == "/api/v1/customers/c1"
    assert sent.url.params["full"] == "false"


async def test_call_api_refuses_without_a_decision_or_credential(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = ApiOperation(path="/x")
    with pytest.raises(ConnectorError, match="no decision"):
        await call_api(CRM, op, {}, decision_token=None, transport=world.transport)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(ConnectorError, match="no credential"):
        await call_api(CRM, op, {}, decision_token="dt", transport=world.transport)
    assert world.upstream("crm.test") == []


async def test_call_api_reports_a_refusal_and_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "ph")
    op = ApiOperation(method="DELETE", path="/x")
    refused = httpx.MockTransport(lambda r: httpx.Response(403, text="no tulip writes"))
    with pytest.raises(ConnectorError, match="403: no tulip writes"):
        await call_api(CRM, op, {}, decision_token="dt", transport=refused)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(ConnectorError, match="ConnectError"):
        await call_api(CRM, op, {}, decision_token="dt", transport=httpx.MockTransport(down))


# ── MCP ─────────────────────────────────────────────────────────────────────────


async def test_mcp_initializes_once_then_calls_with_the_token(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MCP_KEY_ENV, "ph-mcp")
    client = McpClient(
        McpMount(id="kb", url="http://kb.test/mcp", credential_env=MCP_KEY_ENV),
        transport=world.transport,
    )
    assert await client.call_tool("lookup", {"q": "x"}, decision_token="dt-a") == "kb says hi"
    assert await client.call_tool("lookup", {"q": "y"}, decision_token="dt-b") == "kb says hi"
    sent = world.upstream("kb.test")
    methods = [json.loads(r.content).get("method") for r in sent]
    assert methods == ["initialize", "notifications/initialized", "tools/call", "tools/call"]
    assert DECISION_HEADER not in sent[0].headers
    assert sent[2].headers[DECISION_HEADER] == "dt-a"
    assert sent[3].headers["mcp-session-id"] == "s-1"
    assert sent[2].headers["authorization"] == "Bearer ph-mcp"


async def test_mcp_refuses_and_reports_errors() -> None:
    mount = McpMount(id="kb", url="http://kb.test/mcp")
    with pytest.raises(ConnectorError, match="no decision"):
        await McpClient(mount).call_tool("t", {}, decision_token=None)
    with pytest.raises(ConnectorError, match="streamable HTTP only"):
        await McpClient(mount.model_copy(update={"transport": "sse"})).call_tool(
            "t", {}, decision_token="dt"
        )

    def server(answer: dict[str, Any] | None, status: int = 200) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            message = json.loads(request.content)
            if message.get("method") != "tools/call":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": message.get("id")})
            if answer is None:
                return httpx.Response(status, json=[] if status == 200 else {"x": 1})
            return httpx.Response(status, json={"jsonrpc": "2.0", "id": message["id"], **answer})

        return httpx.MockTransport(handle)

    cases: list[tuple[httpx.MockTransport, str]] = [
        (server({"error": {"message": "boom"}}), "boom"),
        (server({"result": {"isError": True, "content": []}}), "the tool failed"),
        (server(None), "no answer"),
        (server({}, status=500), "answered 500"),
    ]
    for transport, message in cases:
        with pytest.raises(ConnectorError, match=message):
            await McpClient(mount, transport=transport).call_tool("t", {}, decision_token="dt")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ConnectorError, match="ReadTimeout"):
        await McpClient(mount, transport=httpx.MockTransport(down)).call_tool(
            "t", {}, decision_token="dt"
        )


# ── manifest ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        ({"name": "x", "runs": "api:crm"}, "names no operation"),
        ({"name": "x", "runs": "box", "operation": {"path": "/x"}}, "has an operation"),
        (
            {"name": "x", "runs": "api:nope", "operation": {"path": "/x"}},
            "not connected",
        ),
    ],
)
def test_manifest_checks_api_tools(tool: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RunManifest.model_validate(_manifest(tools=[tool]))


def test_manifest_lookups_for_connectors_and_mounts() -> None:
    manifest = RunManifest.model_validate(_manifest())
    assert manifest.connector("crm") is not None
    assert manifest.connector("nope") is None
    assert manifest.mount("kb") is not None
    assert manifest.mount("nope") is None
    assert manifest.tool("crm_get").connector == "crm"  # type: ignore[union-attr]


# ── build ─────────────────────────────────────────────────────────────────────────


def test_build_refuses_what_this_runner_cannot_do(world: World, tmp_path: Path) -> None:
    client = _client(world)
    unknown = RunManifest.model_validate(_manifest(tools=[{"name": "teleport", "runs": "box"}]))
    with pytest.raises(RunnerRefused, match="teleport"):
        build_runtime(unknown, client, workspace=tmp_path, environ=_env())
    anthropic = RunManifest.model_validate(
        _manifest(
            tools=[],
            model={"model": "m", "base_url": "http://x", "api_key_env": "K", "provider": "x"},
        )
    )
    with pytest.raises(RunnerRefused, match="openai only"):
        build_runtime(anthropic, client, workspace=tmp_path, environ=_env())
    no_key = RunManifest.model_validate(_manifest(tools=[]))
    env = _env()
    del env[MODEL_KEY_ENV]
    with pytest.raises(RunnerRefused, match="no model credential"):
        build_runtime(no_key, client, workspace=tmp_path, environ=env)


def test_build_model_meters_cleanly(world: World, tmp_path: Path) -> None:
    manifest = RunManifest.model_validate(_manifest(tools=[]))
    runtime = build_runtime(manifest, _client(world), workspace=tmp_path, environ=_env())
    model = runtime.agent.config.model
    assert model.config.default_headers == {"Accept-Encoding": "identity"}
    assert model.config.api_key == "openshell:resolve:env:v1_model_KEY"
    assert model.client.default_headers["Accept-Encoding"] == "identity"
    assert runtime.thread_id == RUN_ID


def test_commands_inherit_no_token_and_no_placeholder(world: World, tmp_path: Path) -> None:
    manifest = RunManifest.model_validate(_manifest())
    runtime = build_runtime(
        manifest, _client(world), workspace=tmp_path, model=ScriptedModel([]), environ=_env()
    )
    result = runtime.backend.exec("env", timeout=10)
    out = result.output if isinstance(result.output, str) else result.output.decode()
    for name in (ADMIT_TOKEN_VAR, MODEL_KEY_ENV, API_KEY_ENV, MCP_KEY_ENV):
        assert f"{name}=" not in out
    assert "PATH=" in out
    assert runtime.backend.capabilities.isolated
    assert runtime.backend.capabilities.label == "OpenShell sandbox"


def test_system_prompt_has_plan_mode_and_the_playbook() -> None:
    manifest = RunManifest.model_validate(
        _manifest(
            instructions="",
            harness={"kind": "workspace", "plan_mode": True},
            playbook={
                "name": "refunds",
                "steps": [{"id": "s1", "title": "Inspect", "instructions": "look"}, "bad"],
            },
        )
    )
    prompt = system_prompt(manifest, "## Workspace")
    assert prompt.startswith("You are a helpful agent.")
    assert "## Workspace" in prompt
    assert "## Plan mode" in prompt
    assert "- s1: Inspect -- look" in prompt
    assert "submit_decision" in prompt


def test_a_v2_playbook_is_shown_in_the_engines_own_prose() -> None:
    fixture = Path(__file__).parents[1] / "playbooks_v2" / "fixtures" / "functional-f19.v2.json"
    playbook = json.loads(fixture.read_text())
    prompt = system_prompt(RunManifest.model_validate(_manifest(playbook=playbook)))
    assert "# Playbook:" in prompt
    assert "complete_step(step_id, outputs)" in prompt


# ── the runner, end to end ───────────────────────────────────────────────────────


async def test_a_run_uses_every_kind_of_tool_then_parks_on_a_hold(
    world: World, tmp_path: Path
) -> None:
    world.admit["bash"] = {"outcome": "require_human", "approval_id": "ap-1", "reason": "exec"}
    model = ScriptedModel(
        [
            _call("write", "c1", path="notes.txt", content="hello\n"),
            tool_call("request_network_access", call_id="c2", host="pypi.org"),
            tool_call("crm_get", call_id="c3", id="c1", full=True),
            tool_call("lookup", call_id="c4", q="refunds"),
            tool_call("bash", call_id="c5", command="cat notes.txt"),
            text("never reached"),
        ]
    )
    code = await run_box(
        _env(), workspace=tmp_path, transport=world.transport, model=model, poll_interval_s=0.01
    )
    assert code == EXIT_DONE
    assert (tmp_path / "notes.txt").read_text() == "hello\n"
    [result] = world.results()
    assert result["status"] == "parked"
    assert result["waiting"] == {
        "kind": "approval",
        "call_id": "c5",
        "tool": "bash",
        "approval_id": "ap-1",
    }
    assert json.loads((tmp_path / ".tulip" / "parked.json").read_text())["call_id"] == "c5"
    assert RUN_ID in world.checkpoints
    # Only box and connector calls were asked about; the gateway tool went to its route.
    admitted = [b["tool"] for b in world.bodies("/v1/admit")]
    assert admitted == ["write", "crm_get", "lookup", "bash"]
    assert all(
        set(b) <= {"run_id", "call_id", "tool", "arguments"} for b in world.bodies("/v1/admit")
    )
    [granted] = world.bodies(f"/internal/v1/runs/{RUN_ID}/tools/request_network_access")
    assert granted == {"call_id": "c2", "arguments": {"host": "pypi.org"}}
    # The decision token reached the API host and the MCP server, nothing else.
    [crm] = world.upstream("crm.test")
    assert crm.headers[DECISION_HEADER] == "dt-c3"
    assert crm.headers["authorization"] == "Bearer openshell:resolve:env:v1_crm_KEY"
    kb_call = world.upstream("kb.test")[-1]
    assert kb_call.headers[DECISION_HEADER] == "dt-c4"
    assert TOKEN_ARGUMENT not in json.dumps(list(world.events()))
    types = [e["type"] for e in world.events()]
    assert "tool_start" in types
    assert "tool_complete" in types
    assert "think" in types


async def test_a_parked_run_resumes_after_approval_and_finishes(
    world: World, tmp_path: Path
) -> None:
    world.admit["bash"] = {"outcome": "require_human", "approval_id": "ap-1", "reason": "exec"}
    first = ScriptedModel(
        [tool_call("bash", call_id="c1", command="echo approved-run"), text("unused")]
    )
    assert (
        await run_box(_env(), workspace=tmp_path, transport=world.transport, model=first)
        == EXIT_DONE
    )
    assert world.results()[-1]["status"] == "parked"

    world.next_ops = [{"op": "resume", "resume": {"approval_id": "ap-1", "decision": "approve"}}]
    second = ScriptedModel([text("all done")])
    code = await run_box(_env(), workspace=tmp_path, transport=world.transport, model=second)
    assert code == EXIT_DONE
    result = world.results()[-1]
    assert result["status"] == "done"
    assert result["final_message"] == "all done"
    # The held call was asked about again, naming its approval, and then ran.
    resumed = world.bodies("/v1/admit")[-1]
    assert resumed["call_id"] == "c1"
    assert resumed["approval_id"] == "ap-1"
    seen = " ".join(m.content or "" for m in second.received_messages[-1])
    assert "approved-run" in seen
    execs = [e for e in world.events() if e["type"] == "harness.exec"]
    assert execs
    assert execs[-1]["attested_by"] == "runner"
    assert execs[-1]["exit_code"] == 0
    assert not (tmp_path / ".tulip" / "parked.json").exists()


async def test_a_question_parks_and_the_answer_resumes(world: World, tmp_path: Path) -> None:
    first = ScriptedModel([tool_call("ask_user", call_id="q1", question="Which env?"), text("x")])
    assert (
        await run_box(_env(), workspace=tmp_path, transport=world.transport, model=first)
        == EXIT_DONE
    )
    waiting = world.results()[-1]["waiting"]
    assert waiting == {"kind": "question", "call_id": "q1", "question": "Which env?"}

    world.next_ops = [{"op": "resume", "resume": {"answer": "staging"}}]
    second = ScriptedModel([text("using staging")])
    await run_box(_env(), workspace=tmp_path, transport=world.transport, model=second)
    assert world.results()[-1]["final_message"] == "using staging"
    seen = " ".join(m.content or "" for m in second.received_messages[-1])
    assert "staging" in seen


async def test_a_denied_call_is_reported_to_the_model(world: World, tmp_path: Path) -> None:
    world.admit["bash"] = {"outcome": "deny", "allowed": False, "reason": "exec:never"}
    model = ScriptedModel([tool_call("bash", call_id="c1", command="rm -rf x"), text("ok")])
    await run_box(_env(), workspace=tmp_path, transport=world.transport, model=model)
    assert world.results()[-1]["status"] == "done"
    seen = " ".join(m.content or "" for m in model.received_messages[-1])
    assert "denied by the gateway: exec:never" in seen


async def test_refused_stopped_and_unconfigured_runs(world: World, tmp_path: Path) -> None:
    world.manifest = _manifest(tools=[{"name": "teleport", "runs": "box"}])
    code = await run_box(_env(), workspace=tmp_path, transport=world.transport)
    assert code == EXIT_REFUSED
    assert world.results()[-1]["status"] == "refused"
    assert "teleport" in world.results()[-1]["error"]

    world.next_ops = [{"op": "stop", "reason": "cancelled"}]
    assert await run_box(_env(), workspace=tmp_path, transport=world.transport) == EXIT_DONE

    assert await run_box({}, workspace=tmp_path, transport=world.transport) == EXIT_ERROR


async def test_a_failing_run_is_reported_as_an_error(world: World, tmp_path: Path) -> None:
    class Broken(ScriptedModel):
        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model fell over")

        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model fell over")
            yield  # pragma: no cover

    code = await run_box(_env(), workspace=tmp_path, transport=world.transport, model=Broken([]))
    assert code == EXIT_ERROR
    assert world.results()[-1]["status"] == "error"
    assert "model fell over" in world.results()[-1]["error"]


async def test_an_unreachable_gateway_is_an_error(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    code = await run_box(_env(), workspace=tmp_path, transport=httpx.MockTransport(down))
    assert code == EXIT_ERROR


async def test_a_subagent_is_its_own_run_in_the_same_workspace(
    world: World, tmp_path: Path
) -> None:
    world.manifest = _manifest(
        tools=[{"name": "write", "runs": "box"}, {"name": "task", "runs": "box"}],
        harness={"kind": "workspace", "subagents": True},
    )
    model = ScriptedModel(
        [
            tool_call("task", call_id="t1", description="scout", prompt="write a note"),
            _call("write", "w1", path="child.txt", content="from child\n"),
            text("child done"),
            text("parent done"),
        ]
    )
    code = await run_box(_env(), workspace=tmp_path, transport=world.transport, model=model)
    assert code == EXIT_DONE
    assert (tmp_path / "child.txt").read_text() == "from child\n"
    [minted] = world.bodies(f"/internal/v1/runs/{RUN_ID}/children")
    assert minted == {"call_id": "t1", "agent": "general", "prompt": "write a note"}
    assert world.results("run-child")[-1] == {
        "status": "done",
        "final_message": "child done",
        "stop_reason": "complete",
    }
    child_admits = [
        r for r in world.requests if r.url.path == "/v1/admit" and b"run-child" in r.content
    ]
    assert child_admits
    assert all(r.headers["authorization"] == "Bearer child-tok" for r in child_admits)
    assert world.results()[-1]["final_message"] == "parent done"


async def test_a_failing_subagent_is_the_parents_tool_error(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tulip.runner.build as build_module

    world.manifest = _manifest(tools=[{"name": "task", "runs": "box"}])

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RunnerRefused("the child cannot be built")

    # Only the child is built through the module attribute; the parent was built already.
    monkeypatch.setattr(build_module, "build_runtime", refuse)
    model = ScriptedModel(
        [tool_call("task", call_id="t1", description="d", prompt="p"), text("parent saw it")]
    )
    code = await run_box(_env(), workspace=tmp_path, transport=world.transport, model=model)
    assert code == EXIT_DONE
    assert world.results("run-child")[-1]["status"] == "error"
    seen = " ".join(m.content or "" for m in model.received_messages[-1])
    assert "the subagent (d) failed" in seen


def test_main_runs_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ADMIT_URL_VAR, ADMIT_TOKEN_VAR, RUN_ID_VAR):
        monkeypatch.delenv(name, raising=False)
    assert main([]) == EXIT_ERROR


# ── protocol additions ───────────────────────────────────────────────────────────


async def test_children_and_results_round_trip(world: World) -> None:
    client = _client(world)
    child = await mint_child(client, call_id="c1", agent="explore", prompt="look")
    assert child.run_id == "run-child"
    assert "child-tok" not in repr(child)
    await report_result(client, RunResult(status="done", final_message="hi"))
    assert world.results() == [{"status": "done", "final_message": "hi"}]

    empty = GatewayClient(
        RunnerConfig(url="http://gateway.test", run_id=RUN_ID, token=TOKEN),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    with pytest.raises(ValueError, match="no child run"):
        await mint_child(empty, call_id="c", agent="a", prompt="p")


def test_event_mapping() -> None:
    assert event_for(ModelChunkEvent(content="hi")) == {"type": "token", "text": "hi"}
    assert event_for(ModelChunkEvent(content=None)) is None
    think = event_for(
        ThinkEvent(
            iteration=1, reasoning="r", tool_calls=[ToolCall(id="a", name="b", arguments={})]
        )
    )
    assert think == {
        "type": "think",
        "iteration": 1,
        "text": "r",
        "tool_calls": [{"id": "a", "name": "b"}],
    }
    start = ToolStartEvent(tool_name="t", tool_call_id="c", arguments={"x": 1})
    assert event_for(start) == {
        "type": "tool_start",
        "call_id": "c",
        "tool": "t",
        "arguments": {"x": 1},
    }
    done = ToolCompleteEvent(tool_name="t", tool_call_id="c", result="x" * 5000, duration_ms=1.0)
    mapped = event_for(done)
    assert mapped is not None
    assert len(mapped["result"]) == 4000
    assert event_for(ToolCompleteEvent(tool_name="t", tool_call_id="c")) is not None
    from tulip.core.events import TerminateEvent

    terminate = TerminateEvent(
        reason="complete", iterations_used=1, final_confidence=1.0, total_tool_calls=0
    )
    assert event_for(terminate) is None


# ── hardening ──────────────────────────────────────────────────────────────────────


def test_command_env_withholds_names() -> None:
    names = withheld_names(["A", None, "", "B", "A"])
    assert names == ("A", "B")
    assert command_env({"A": "1", "B": "2", "C": "3"}, names) == {"C": "3"}


def test_scrub_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_SCRUB_ME", "x")
    scrub_environ(["TULIP_SCRUB_ME", "TULIP_NOT_SET"])
    assert "TULIP_SCRUB_ME" not in os.environ


def test_non_dumpable_on_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform.startswith("linux"):
        # Run in a child so this test process stays dumpable for its debugger.
        import subprocess

        code = (
            "from tulip.runner.harden import make_non_dumpable;"
            "import sys; ok = make_non_dumpable();"
            "s = open('/proc/self/status').read();"
            "sys.exit(0 if ok else 1)"
        )
        child = subprocess.run(  # noqa: S603 — a fixed interpreter and script
            [sys.executable, "-c", code],
            check=False,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        assert child.returncode == 0
    monkeypatch.setattr(sys, "platform", "darwin")
    assert make_non_dumpable() is False


def test_local_backend_can_start_from_a_given_environment(tmp_path: Path) -> None:
    backend = LocalBackend(tmp_path, base_env={"ONLY": "this", "PATH": os.environ["PATH"]})
    result = backend.exec("env", timeout=10)
    out = result.output if isinstance(result.output, str) else result.output.decode()
    assert "ONLY=this" in out
    assert "HOME=" not in out
    assert not backend.capabilities.isolated


# ── edges ────────────────────────────────────────────────────────────────────────


async def test_tool_failures_reach_the_model_as_errors(world: World, tmp_path: Path) -> None:
    world.upstream_status = 502
    model = ScriptedModel(
        [
            tool_call("request_network_access", call_id="c1", host="evil.test"),
            tool_call("crm_get", call_id="c2", id="c1"),
            tool_call("lookup", call_id="c3", q="x"),
            text("noted"),
        ]
    )
    await run_box(_env(), workspace=tmp_path, transport=world.transport, model=model)
    seen = " ".join(m.content or "" for m in model.received_messages[-1])
    assert "Error: not a host you may ask for" in seen
    assert "answered 502" in seen
    assert "kb is down" in seen


def test_budgets_reach_the_agent(world: World, tmp_path: Path) -> None:
    manifest = RunManifest.model_validate(
        _manifest(
            tools=[],
            budgets={"max_iterations": 7, "max_tokens": 900, "max_cost_usd": 0.5, "timeout_s": 60},
        )
    )
    agent = build_runtime(
        manifest, _client(world), workspace=tmp_path, model=ScriptedModel([]), environ=_env()
    ).agent
    assert agent.config.max_iterations == 7
    assert agent.config.token_budget == 900
    assert agent.config.max_cost_usd is None  # the gateway holds the run to it, metered
    assert agent.config.time_budget_seconds == 60


async def test_a_resume_while_still_pending_parks_again(world: World, tmp_path: Path) -> None:
    world.admit["bash"] = {"outcome": "require_human", "approval_id": "ap-1", "reason": "exec"}
    first = ScriptedModel([tool_call("bash", call_id="c1", command="true"), text("x")])
    await run_box(_env(), workspace=tmp_path, transport=world.transport, model=first)
    world.still_pending = True
    world.next_ops = [{"op": "resume", "resume": {"approval_id": "ap-1", "decision": "approve"}}]
    second = ScriptedModel([text("unused")])
    code = await run_box(_env(), workspace=tmp_path, transport=world.transport, model=second)
    assert code == EXIT_DONE
    result = world.results()[-1]
    assert result["status"] == "parked"
    assert result["waiting"]["approval_id"] == "ap-1"


def test_parked_file_is_read_defensively(tmp_path: Path) -> None:
    from tulip.runner import main as runner_main

    assert runner_main._read_parked(tmp_path) == {}
    path = tmp_path / ".tulip" / "parked.json"
    path.parent.mkdir()
    path.write_text("not json")
    assert runner_main._read_parked(tmp_path) == {}
    path.write_text("[1]")
    assert runner_main._read_parked(tmp_path) == {}


async def test_a_stream_that_ends_without_an_answer_is_an_error(
    world: World, tmp_path: Path
) -> None:
    from tulip.runner import main as runner_main

    async def nothing() -> Any:
        return
        yield  # pragma: no cover

    runtime = build_runtime(
        RunManifest.model_validate(_manifest(tools=[])),
        _client(world),
        workspace=tmp_path,
        model=ScriptedModel([]),
        environ=_env(),
    )
    events = runner_main.GatewayEvents(_client(world), spool=tmp_path / "spool.jsonl")
    result = await runner_main._drive(runtime, events, nothing(), tmp_path)
    assert result.status == "error"
    assert "without an answer" in (result.error or "")


async def test_mcp_reads_sse_and_json_lists() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        if message.get("method") == "initialize":
            return httpx.Response(200, json=[{"jsonrpc": "2.0", "id": message["id"]}, "noise"])
        if "id" not in message:
            return httpx.Response(202)
        answer = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        sse = f"data: not json\ndata: [1]\nevent: x\ndata: {answer}\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = McpClient(
        McpMount(id="m", url="http://m.test/mcp"), transport=httpx.MockTransport(handle)
    )
    assert await client.call_tool("t", {}, decision_token="dt") == "ok"


def test_the_module_entry_point_exits_with_the_runners_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runpy

    for name in (ADMIT_URL_VAR, ADMIT_TOKEN_VAR, RUN_ID_VAR):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("tulip.runner", run_name="__main__")
    assert stopped.value.code == EXIT_ERROR
