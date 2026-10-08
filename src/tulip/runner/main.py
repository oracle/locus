# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``python -m tulip.runner``: one run's agent loop, inside its box.

The process the gateway starts in a box. It:

1. reads its gateway, run and workload token from the environment
   (:meth:`~tulip.runner.client.RunnerConfig.from_env`), and marks itself
   non-dumpable (:func:`~tulip.runner.harden.make_non_dumpable`);
2. asks the gateway what to do (``GET /internal/v1/runner/next``): start,
   resume, or stop;
3. fetches the run's manifest and builds the agent from it
   (:func:`~tulip.runner.build.build_runtime`);
4. runs it (or resumes it from the gateway's checkpoint, re-asking about the
   held call with its approval, or answering the question it asked),
   streaming progress to the run (:class:`~tulip.runner.events.GatewayEvents`);
5. reports how it ended (``POST /internal/v1/runs/{id}/result``) and exits.

A run that waits on a person *parks*: the agent's state is checkpointed with
the gateway, what it waits for is written to ``/sandbox/.tulip/parked.json``
and reported, and the process exits 0. The box stops; when the decision
comes the gateway starts it again and step 2 says ``resume``.

Exit codes: **0** done, parked or told to stop; **2** refused (the manifest
asks for what this runner cannot do); **1** error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tulip.runner.build import DEFAULT_WORKSPACE, RunnerRefused, Runtime, build_runtime
from tulip.runner.client import GatewayClient, GatewayError, GatewayUnavailable, RunnerConfig
from tulip.runner.events import GatewayEvents
from tulip.runner.handshake import fetch_manifest, next_op
from tulip.runner.harden import make_non_dumpable
from tulip.runner.outcome import RunResult, report_result


if TYPE_CHECKING:
    import httpx

    from tulip.core.events import TulipEvent


__all__ = ["EXIT_DONE", "EXIT_ERROR", "EXIT_REFUSED", "event_for", "main", "run_box"]

logger = logging.getLogger(__name__)

EXIT_DONE = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2

#: Characters of a tool result reported on ``tool_complete`` (the gateway
#: keeps what it needs; the run's stream is for watching, not evidence).
RESULT_CHARS = 4_000


def event_for(event: TulipEvent) -> dict[str, Any] | None:
    """The gateway event for one of the agent's events, or ``None`` to send none."""
    from tulip.core.events import ModelChunkEvent, ThinkEvent, ToolCompleteEvent, ToolStartEvent

    if isinstance(event, ModelChunkEvent):
        if not event.content:
            return None
        return {"type": "token", "text": event.content}
    if isinstance(event, ThinkEvent):
        return {
            "type": "think",
            "iteration": event.iteration,
            "text": event.reasoning or "",
            "tool_calls": [{"id": call.id, "name": call.name} for call in event.tool_calls],
        }
    if isinstance(event, ToolStartEvent):
        return {
            "type": "tool_start",
            "call_id": event.tool_call_id,
            "tool": event.tool_name,
            "arguments": event.arguments,
        }
    if isinstance(event, ToolCompleteEvent):
        return {
            "type": "tool_complete",
            "call_id": event.tool_call_id,
            "tool": event.tool_name,
            "result": (event.result or "")[:RESULT_CHARS],
            "error": event.error,
            "duration_ms": event.duration_ms,
        }
    return None


def _parked_path(workspace: Path) -> Path:
    return workspace / ".tulip" / "parked.json"


def _write_parked(workspace: Path, waiting: Mapping[str, Any]) -> None:
    path = _parked_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(waiting)), encoding="utf-8")


def _read_parked(workspace: Path) -> dict[str, Any]:
    path = _parked_path(workspace)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


async def _forward(runtime: Runtime, events: GatewayEvents, event: TulipEvent) -> None:
    from tulip.core.events import ToolCompleteEvent

    mapped = event_for(event)
    if mapped is not None:
        await events.emit(mapped)
    if isinstance(event, ToolCompleteEvent):
        for record in runtime.drain_execs():
            await events.emit(
                {
                    "type": "harness.exec",
                    "call_id": event.tool_call_id,
                    "command_sha256": record.command_sha256,
                    "exit_code": record.exit_code,
                    "duration_s": record.duration_s,
                    "timed_out": record.timed_out,
                    "output_sha256": record.output_sha256,
                    "output_bytes": record.output_bytes,
                    "truncated": record.truncated,
                    "background": record.background,
                    "attested_by": "runner",
                }
            )


def _waiting(runtime: Runtime, event: Any) -> dict[str, Any]:
    hold = runtime.gate.pending_hold(event.interrupt_id)
    if hold is not None:
        return {
            "kind": "approval",
            "call_id": hold.call_id,
            "tool": hold.tool,
            "approval_id": hold.approval_id,
        }
    return {"kind": "question", "call_id": event.interrupt_id, "question": event.question}


async def _drive(
    runtime: Runtime, events: GatewayEvents, stream: Any, workspace: Path
) -> RunResult:
    try:
        return await _consume(runtime, events, stream, workspace)
    finally:
        # The loop checkpoints as its generator closes; close it while the
        # gateway client is still open, not at garbage collection.
        await stream.aclose()


async def _consume(
    runtime: Runtime, events: GatewayEvents, stream: Any, workspace: Path
) -> RunResult:
    from tulip.core.events import InterruptEvent, TerminateEvent

    async for event in stream:
        await _forward(runtime, events, event)
        if isinstance(event, InterruptEvent):
            waiting = _waiting(runtime, event)
            _write_parked(workspace, waiting)
            return RunResult(status="parked", waiting=waiting)
        if isinstance(event, TerminateEvent):
            _parked_path(workspace).unlink(missing_ok=True)
            if event.reason == "error":
                return RunResult(
                    status="error", error=event.error or "the run failed", stop_reason="error"
                )
            return RunResult(
                status="done",
                final_message=event.final_message,
                stop_reason=event.reason,
                usage_reported=event.usage,
                cost_usd_reported=event.reported_cost_usd
                if event.reported_cost_usd is not None
                else event.cost_usd,
            )
    return RunResult(status="error", error="the agent stopped without an answer")


def _stream(runtime: Runtime, op: Any, workspace: Path) -> Any:
    agent = runtime.agent
    thread = runtime.thread_id
    if op.op == "start":
        return agent.run(runtime.manifest.input, thread_id=thread, stream_tokens=True)
    parked = _read_parked(workspace)
    if op.approval_id:
        call_id = str(parked.get("call_id") or "")
        if call_id:
            runtime.gate.prime(call_id, op.approval_id)
        return agent.resume(op.decision or "approved", thread_id=thread, perform_dangling=True)
    return agent.resume(str(op.resume.get("answer", "")), thread_id=thread)


async def run_box(
    env: Mapping[str, str] | None = None,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
    transport: httpx.AsyncBaseTransport | None = None,
    model: Any | None = None,
    poll_interval_s: float = 2.0,
) -> int:
    """Run (or resume) this box's run to its end or its next park. Returns the exit code."""
    from tulip.core.errors import ApprovalPendingError

    try:
        config = RunnerConfig.from_env(env)
    except ValueError as exc:
        logger.error("%s", exc)  # noqa: TRY400 — names a variable, never a value
        return EXIT_ERROR
    make_non_dumpable()
    root = Path(workspace)
    client = GatewayClient(config, transport=transport)
    events = GatewayEvents(client, spool=root / ".tulip" / "events.jsonl")
    try:
        op = await next_op(client)
        if op.op == "stop":
            logger.info("the gateway said stop: %s", op.reason or "no reason given")
            return EXIT_DONE
        manifest = await fetch_manifest(client)
        try:
            runtime = build_runtime(
                manifest,
                client,
                workspace=root,
                model=model,
                environ=env,
                transport=transport,
                poll_interval_s=poll_interval_s,
            )
        except RunnerRefused as exc:
            await report_result(client, RunResult(status="refused", error=str(exc)))
            return EXIT_REFUSED
        try:
            result = await _drive(runtime, events, _stream(runtime, op, root), root)
        except ApprovalPendingError:
            waiting = _read_parked(root)
            result = RunResult(status="parked", waiting=waiting or None)
        except Exception as exc:  # noqa: BLE001 — every failure is reported, then exits 1
            logger.exception("the run failed")
            result = RunResult(status="error", error=f"{type(exc).__name__}: {exc}")
        await events.flush()
        await report_result(client, result)
        return EXIT_ERROR if result.status == "error" else EXIT_DONE
    except (GatewayError, GatewayUnavailable) as exc:
        logger.error("the gateway could not be used: %s", exc)  # noqa: TRY400
        return EXIT_ERROR
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    """The console entry point."""
    del argv  # the runner takes everything from its environment
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run_box())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
