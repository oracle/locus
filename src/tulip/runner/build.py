# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the agent a box runner runs, from its :class:`RunManifest` and nothing else.

:func:`build_runtime` turns a manifest into a ready :class:`~tulip.Agent`:

- **The model** — an OpenAI-compatible client on ``model.base_url``, keyed by
  the OpenShell placeholder in ``model.api_key_env`` (never a key), sending
  ``Accept-Encoding: identity`` so the box guard can read each response's
  usage and meter the run.
- **The tools, by where they run** (``ToolEntry.runs``):

  ``box``
      The harness tools (:mod:`tulip.harness`) over the box's workspace
      (``/sandbox``). ``ask_user`` pauses the run for a person's answer;
      ``task`` starts a subagent in this box as a child run.
  ``gateway``
      :class:`~tulip.runner.tools.RemoteTool`: performed server-side.
  ``mcp:<id>``
      A ``tools/call`` on that mounted MCP server, from the box.
  ``api:<id>``
      One operation of that API connector, from the box
      (:func:`~tulip.runner.connectors.api_request`).

  A box tool this runner cannot build refuses the run
  (:class:`RunnerRefused`); it never silently drops a tool.
- **The gate** — every call except the gateway's own goes through
  :class:`~tulip.runner.gate.RemoteGate` first. Its decision token reaches
  only the MCP and API tools, which send it to the box guard. A call held for
  a person is waited on up to ``budgets.park_ttl_s``; still undecided, the
  run pauses there (an interrupt, checkpointed by the gateway) and the runner
  parks.
- **Checkpoints** — :class:`~tulip.runner.checkpoint.GatewayCheckpointer`.

Commands the harness runs get the runner's environment without its gateway
token and its credential placeholders (:mod:`tulip.runner.harden`).

Plan mode and a v2 playbook shape the system prompt here; the gateway's copy
of either is the one that decides, through admission.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tulip.harness import ALL_TOOLS, ExecRecord, HarnessConfig, LocalBackend, build_harness
from tulip.runner.checkpoint import GatewayCheckpointer
from tulip.runner.client import ADMIT_TOKEN_VAR, GatewayClient, RunnerConfig
from tulip.runner.connectors import ConnectorError, McpClient, call_api
from tulip.runner.gate import RemoteGate
from tulip.runner.harden import command_env, withheld_names
from tulip.runner.outcome import RunResult, mint_child, report_result
from tulip.runner.tools import RemoteTool, RemoteToolError
from tulip.tools.decorator import Tool


if TYPE_CHECKING:
    import httpx

    from tulip.agent.agent import Agent
    from tulip.runner.manifest import RunManifest, ToolEntry
    from tulip.tools.context import ToolContext


__all__ = [
    "DEFAULT_WORKSPACE",
    "TOKEN_ARGUMENT",
    "RunnerRefused",
    "Runtime",
    "build_runtime",
    "system_prompt",
]

#: The box's persistent workspace.
DEFAULT_WORKSPACE = "/sandbox"

#: The secret argument a decision token reaches an MCP or API tool body under.
TOKEN_ARGUMENT = "__tulip_decision__"  # noqa: S105 — an argument name, not a secret

#: Box tools built here rather than by the harness.
_LOCAL_TOOLS = frozenset({"ask_user", "task"})

#: The models this runner can drive.
_PROVIDERS = frozenset({"openai"})


class RunnerRefused(Exception):  # noqa: N818 — a refusal, reported as one
    """The manifest asks for something this runner cannot do; the run is refused."""


@dataclass
class Runtime:
    """A built run: the agent and what drives it."""

    manifest: RunManifest
    agent: Agent
    gate: RemoteGate
    client: GatewayClient
    backend: LocalBackend
    #: Commands the harness ran since the last drain, for ``harness.exec`` events.
    execs: list[ExecRecord] = field(default_factory=list)

    @property
    def thread_id(self) -> str:
        """The run's conversation thread: the run id."""
        return self.manifest.run_id

    def drain_execs(self) -> list[ExecRecord]:
        """Take the exec records gathered so far."""
        taken, self.execs = self.execs, []
        return taken


def system_prompt(manifest: RunManifest, harness_fragment: str = "") -> str:
    """The system prompt: instructions, the workspace, plan mode, the playbook."""
    parts = [manifest.instructions.strip() or "You are a helpful agent."]
    if harness_fragment:
        parts.append(harness_fragment)
    harness = manifest.harness or {}
    if harness.get("plan_mode"):
        parts.append(
            "## Plan mode\n"
            "You are in plan mode: investigate and write a plan, but change nothing. "
            "When the plan is ready, present it with exit_plan_mode and wait. "
            "Calls that change the workspace are refused until the plan is accepted."
        )
    if manifest.playbook:
        parts.append(_playbook_prompt(manifest.playbook))
    return "\n\n".join(parts)


def _playbook_prompt(playbook: Mapping[str, Any]) -> str:
    """A v2 playbook in the engine's own prose; anything else as a plain outline."""
    from tulip.playbooks.v2 import (
        initial_active,
        is_playbook_v2,
        parse_playbook_v2,
        playbook_prose,
    )

    if is_playbook_v2(playbook):
        parsed = parse_playbook_v2(playbook)
        return playbook_prose(parsed, {}, initial_active(parsed))
    lines = [f"## Playbook: {playbook.get('name') or playbook.get('id') or 'unnamed'}"]
    for step in playbook.get("steps") or []:
        if isinstance(step, Mapping):
            title = step.get("title") or step.get("id") or ""
            detail = step.get("instructions") or step.get("description") or ""
            lines.append(f"- {step.get('id', '')}: {title}" + (f" -- {detail}" if detail else ""))
    lines.append(
        "Work through the steps in order. Report each with complete_step, choose branches "
        "with select_branches, and finish with submit_decision. The gateway checks every step."
    )
    return "\n".join(lines)


def _interrupt_marker(question: str, metadata: Mapping[str, Any]) -> str:
    # The runtime pauses on this marker (the shape ``ask_user`` returns) and
    # checkpoints the run before it yields the interrupt.
    return json.dumps({"__interrupt__": True, "question": question, "metadata": dict(metadata)})


def _hold_aware(tool: Tool, gate: RemoteGate) -> Tool:
    """``tool``, but a call still held for a person pauses the run instead of running."""

    async def body(ctx: ToolContext, **arguments: Any) -> Any:
        hold = gate.pending_hold(ctx.tool_call_id)
        if hold is not None:
            return _interrupt_marker(
                f"Waiting for a person to approve {hold.tool}: {hold.reason}",
                {"kind": "approval", "approval_id": hold.approval_id, "tool": hold.tool},
            )
        return await tool.execute(ctx, **arguments)

    return tool.model_copy(update={"fn": body})


def _described(entry: ToolEntry, fn: Callable[..., Any]) -> Tool:
    return Tool(
        name=entry.name,
        description=entry.description or entry.name,
        parameters=entry.parameters,
        fn=fn,
    )


def _gateway_tool(client: GatewayClient, entry: ToolEntry) -> Tool:
    remote = RemoteTool(client, entry)

    async def body(ctx: ToolContext, **arguments: Any) -> Any:
        try:
            return await remote.call(ctx.tool_call_id, arguments)
        except RemoteToolError as exc:
            return f"Error: {exc}"

    return _described(entry, body)


def _mcp_tool(
    manifest: RunManifest, entry: ToolEntry, transport: Any, environ: Mapping[str, str]
) -> Tool:
    mount = manifest.mount(entry.mount or "")
    if mount is None:  # pragma: no cover - the manifest validator refuses this
        raise RunnerRefused(f"tool {entry.name!r}: mount {entry.mount!r} missing")
    client = McpClient(mount, transport=transport, environ=environ)

    async def body(**arguments: Any) -> Any:
        token = arguments.pop(TOKEN_ARGUMENT, None)
        try:
            return await client.call_tool(entry.name, arguments, decision_token=token)
        except ConnectorError as exc:
            return f"Error: {exc}"

    return _described(entry, body)


def _api_tool(
    manifest: RunManifest, entry: ToolEntry, transport: Any, environ: Mapping[str, str]
) -> Tool:
    connector = manifest.connector(entry.connector or "")
    operation = entry.operation
    if connector is None or operation is None:  # pragma: no cover - validator refuses
        raise RunnerRefused(f"tool {entry.name!r}: connector {entry.connector!r} missing")

    async def body(**arguments: Any) -> Any:
        token = arguments.pop(TOKEN_ARGUMENT, None)
        try:
            return await call_api(
                connector,
                operation,
                arguments,
                decision_token=token,
                transport=transport,
                environ=environ,
            )
        except ConnectorError as exc:
            return f"Error: {exc}"

    return _described(entry, body)


def _ask_user_tool(entry: ToolEntry) -> Tool:
    def body(question: str, options: str | None = None) -> str:
        choices = [o.strip() for o in options.split(",")] if options else None
        return json.dumps(
            {"__interrupt__": True, "question": question, "options": choices, "metadata": {}}
        )

    return Tool(
        name="ask_user",
        description=entry.description
        or "Ask the user a question and wait for their answer before going on.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask."},
                "options": {
                    "type": "string",
                    "description": "Comma-separated choices, if the answer is one of them.",
                },
            },
            "required": ["question"],
        },
        fn=body,
    )


def _task_tool(
    entry: ToolEntry,
    parent: GatewayClient,
    make_child: Callable[[GatewayClient], Runtime],
) -> Tool:
    async def body(ctx: ToolContext, description: str, prompt: str, agent: str = "general") -> str:
        child = await mint_child(parent, call_id=ctx.tool_call_id, agent=agent, prompt=prompt)
        config = RunnerConfig(url=parent.config.url, run_id=child.run_id, token=child.admit_token)
        client = GatewayClient(config, transport=parent.transport)
        try:
            runtime = make_child(client)
            result = await runtime.agent.arun(prompt, thread_id=child.run_id)
        except Exception as exc:  # noqa: BLE001 — a child's failure is the parent's tool error
            await report_result(client, RunResult(status="error", error=str(exc)))
            return f"Error: the subagent ({description}) failed: {exc}"
        else:
            await report_result(
                client,
                RunResult(
                    status="done", final_message=result.message, stop_reason=result.stop_reason
                ),
            )
            return str(result.message or "")
        finally:
            await client.aclose()

    return Tool(
        name="task",
        description=entry.description
        or "Hand a self-contained task to a subagent that works in this same workspace.",
        parameters=entry.parameters
        if entry.parameters.get("properties")
        else {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A few words: what it is for."},
                "prompt": {"type": "string", "description": "The full task for the subagent."},
                "agent": {"type": "string", "description": "Which subagent kind to start."},
            },
            "required": ["description", "prompt"],
        },
        fn=body,
    )


def _model(manifest: RunManifest, environ: Mapping[str, str]) -> Any:
    route = manifest.model
    if route.provider not in _PROVIDERS:
        raise RunnerRefused(f"model provider {route.provider!r}: this runner speaks openai only")
    placeholder = environ.get(route.api_key_env, "")
    if not placeholder:
        raise RunnerRefused(f"{route.api_key_env} is not set: the box has no model credential")
    from tulip.models.native.openai import OpenAIModel

    return OpenAIModel(
        model=route.model,
        base_url=route.base_url,
        api_key=placeholder,
        default_headers={"Accept-Encoding": "identity"},
        # OpenShell closes a kept-alive tunnel once the box's policy generation moves on
        # (a command resolving a new host is enough); a reused connection would fail the
        # next model call with "Connection error".
        keepalive=False,
    )


def build_runtime(
    manifest: RunManifest,
    client: GatewayClient,
    *,
    workspace: str | os.PathLike[str] = DEFAULT_WORKSPACE,
    model: Any | None = None,
    environ: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    poll_interval_s: float = 2.0,
    child: bool = False,
) -> Runtime:
    """Build the run's agent from ``manifest``.

    Args:
        manifest: The run's manifest, fetched from the gateway.
        client: The runner's gateway client.
        workspace: The box's workspace directory.
        model: A model to use instead of the manifest's route (tests).
        environ: The environment to read placeholders from; the process's own
            when omitted.
        transport: An ``httpx`` transport for MCP and API calls (tests).
        poll_interval_s: Seconds between checks of a held call.
        child: Build a subagent's runtime: no ``task`` tool, and a held call is
            refused rather than parked (a child cannot park on its own).

    Raises:
        RunnerRefused: the manifest asks for a tool, a provider or a credential
            this runner does not have.
    """
    env = dict(os.environ if environ is None else environ)
    withheld = withheld_names(
        [
            ADMIT_TOKEN_VAR,
            manifest.model.api_key_env,
            *(mount.credential_env for mount in manifest.mcp),
            *(connector.credential_env for connector in manifest.api),
        ]
    )
    backend = LocalBackend(
        Path(workspace),
        base_env=command_env(env, withheld),
        label="OpenShell sandbox",
        isolated=True,
    )
    wire_tools = frozenset(
        entry.name for entry in manifest.tools if entry.mount is not None or entry.connector
    )
    server_tools = frozenset(entry.name for entry in manifest.tools if entry.runs == "gateway")
    gate = RemoteGate(
        client,
        hold_wait_s=manifest.budgets.park_ttl_s,
        poll_interval_s=poll_interval_s,
        token_argument=TOKEN_ARGUMENT,
        token_tools=wire_tools,
        server_admitted=server_tools,
        defer_unsettled=not child,
    )
    execs: list[ExecRecord] = []

    box = [entry for entry in manifest.tools if entry.runs == "box"]
    unknown = [
        entry.name
        for entry in box
        if entry.name not in ALL_TOOLS and entry.name not in _LOCAL_TOOLS
    ]
    if unknown:
        raise RunnerRefused(f"no box tool named {', '.join(map(repr, unknown))} in this runner")
    if child and any(entry.name == "task" for entry in box):
        box = [entry for entry in box if entry.name != "task"]

    tools: list[Tool] = []
    harness_names = [entry.name for entry in box if entry.name in ALL_TOOLS]
    fragment = ""
    if harness_names:
        harness = build_harness(
            backend,
            tools=harness_names,
            config=HarnessConfig(
                on_exec=execs.append,
                environment=str((manifest.definition or {}).get("environment") or "unknown"),
            ),
        )
        tools.extend(harness.tools)
        fragment = harness.prompt_fragment

    for entry in manifest.tools:
        if entry.runs == "gateway":
            tools.append(_gateway_tool(client, entry))
        elif entry.mount is not None:
            tools.append(_mcp_tool(manifest, entry, transport, env))
        elif entry.connector is not None:
            tools.append(_api_tool(manifest, entry, transport, env))
        elif entry.name == "ask_user":
            tools.append(_ask_user_tool(entry))

    from tulip.agent.agent import Agent

    budgets = manifest.budgets
    extra: dict[str, Any] = {}
    if budgets.max_tokens is not None:
        extra["token_budget"] = budgets.max_tokens
    # ``max_cost_usd`` is not passed on: the agent would need list prices for the
    # model (a house model has none), and the cost that counts is the one the box
    # guard meters on the wire, which the gateway holds the run to.
    if budgets.timeout_s is not None:
        extra["time_budget_seconds"] = budgets.timeout_s

    def make_agent(all_tools: list[Tool]) -> Agent:
        return Agent(
            model=model if model is not None else _model(manifest, env),
            tools=[_hold_aware(t, gate) for t in all_tools],
            system_prompt=system_prompt(manifest, fragment),
            max_iterations=budgets.max_iterations or 50,
            checkpointer=GatewayCheckpointer(client),
            hooks=[gate],
            tool_execution="sequential",
            name=manifest.agent,
            **extra,
        )

    task_entry = next((entry for entry in box if entry.name == "task"), None)
    if task_entry is not None:

        def make_child(child_client: GatewayClient) -> Runtime:
            return build_runtime(
                manifest,
                child_client,
                workspace=workspace,
                model=model,
                environ=env,
                transport=transport,
                poll_interval_s=poll_interval_s,
                child=True,
            )

        tools.append(_task_tool(task_entry, client, make_child))

    return Runtime(
        manifest=manifest,
        agent=make_agent(tools),
        gate=gate,
        client=client,
        backend=backend,
        execs=execs,
    )
