# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Gated programmatic tool calling — code mode (#176).

The model writes a Python program; the program calls tools. Everyone ships
this now (it is the year's real execution-model change, with 38–64% token
reductions), and every implementation so far has the same flaw: in-sandbox
tool calls bypass the framework's hook layer, so the fast path and the
governed path fork.

Tulip's version refuses the fork. The program runs in an isolated
interpreter with **no tool implementations inside** — ``tools.call(name,
...)`` is an RPC back to the host, and the host routes every inner call
through the exact seam the loop uses: ``on_before_tool_call`` (a cancel is
honoured — the program receives a refusal string, not the effect),
hook-modified arguments, execution via the registered Tool (so a
``gate_tool`` wrapper and its ``admit()`` still stand between the call and
the side effect), then ``on_after_tool_call`` (result replacement applies).
The model gets the token savings; the gate never learns to be optional.

The interpreter is ``python -I`` with a scrubbed environment — process
isolation, the development default, same stance as
:class:`~tulip.tools.sandbox.SubprocessSandbox`. The RPC channel is the
child's stdout (JSON frames); the program's own ``print`` output is captured
and returned separately.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from typing import TYPE_CHECKING, Any

from tulip.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from tulip.agent.hook_orchestrator import HookOrchestrator
    from tulip.tools.registry import ToolRegistry

_HARNESS = textwrap.dedent(
    """
    import io, json, sys

    _rpc_out = sys.stdout
    sys.stdout = _user_out = io.StringIO()


    class _Refused(Exception):
        pass


    class _Tools:
        def call(self, name, *args, **kwargs):
            if args:
                kwargs["__positional__"] = list(args)
            _rpc_out.write(json.dumps({"rpc": "call", "name": name, "args": kwargs}) + "\\n")
            _rpc_out.flush()
            resp = json.loads(sys.stdin.readline())
            if resp.get("refused"):
                raise _Refused(resp["refused"])
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            return resp.get("result")


    tools = _Tools()
    _ns = {"tools": tools, "ToolRefused": _Refused}
    _err = None
    try:
        exec(compile(json.loads(sys.argv[1]), "<agent-code>", "exec"), _ns)
    except _Refused as exc:
        _err = "refused: " + str(exc)
    except BaseException as exc:
        _err = type(exc).__name__ + ": " + str(exc)
    _rpc_out.write(
        json.dumps(
            {
                "rpc": "done",
                "stdout": _user_out.getvalue(),
                "result": repr(_ns.get("result")) if "result" in _ns else None,
                "error": _err,
            }
        )
        + "\\n"
    )
    _rpc_out.flush()
    """
).strip()

_RESULT_CAP = 20_000


def create_code_tool(
    registry: ToolRegistry,
    orchestrator: HookOrchestrator,
    *,
    timeout: float = 60.0,
    python: str | None = None,
) -> Tool:
    """Build the ``run_code`` tool bound to a registry and hook seam.

    Auto-registered by the initializer under ``AgentConfig(code_mode=True)``.
    Inner calls may reach any registered tool except ``run_code`` itself.
    """

    async def _serve(
        proc: asyncio.subprocess.Process, call_seq: list[dict[str, Any]]
    ) -> dict[str, Any]:
        assert proc.stdout is not None
        assert proc.stdin is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return {"error": "sandbox exited without a done frame"}
            try:
                frame = json.loads(line)
            except ValueError:
                continue  # stray write on the RPC channel; user prints are captured separately
            if frame.get("rpc") == "done":
                return dict(frame)
            if frame.get("rpc") != "call":
                continue
            # Models address tools loosely: a "functions." prefix, positional
            # arguments. Meet them — a retry loop over addressing trivia
            # burns the tokens this mode exists to save.
            name = str(frame.get("name", "")).rsplit(".", 1)[-1]
            args = dict(frame.get("args") or {})
            positional = args.pop("__positional__", None)
            reply: dict[str, Any] | None = None
            resolved = None if name == "run_code" else registry.get(name)
            if resolved is None:
                reply = {"error": f"unknown tool: {name!r}"}
            elif positional:
                param_names = list((resolved.parameters.get("properties") or {}).keys())
                if len(positional) > len(param_names):
                    reply = {"error": f"{name} takes at most {len(param_names)} argument(s)"}
                else:
                    args.update(dict(zip(param_names, positional, strict=False)))
            if reply is None:
                # The same seam as a loop-issued call: before-hooks (cancel is
                # a refusal the program must handle, not a bypassed effect),
                # modified arguments, the Tool itself (any gate included),
                # after-hooks (result replacement lands in what the program
                # reads).
                before = await orchestrator.run_before_tool(
                    name, f"code:{len(call_seq)}", dict(args)
                )
                if before.cancel:
                    msg = before.cancel if isinstance(before.cancel, str) else "Cancelled by hook"
                    call_seq.append({"name": name, "refused": True})
                    reply = {"refused": msg}
                else:
                    run_args = before.arguments
                    try:
                        raw = await registry.get_or_raise(name).execute(**run_args)
                        after = await orchestrator.run_after_tool(
                            name,
                            raw,
                            None,
                            tool_call_id=f"code:{len(call_seq)}",
                            arguments=run_args,
                        )
                        replaced = after.result if after.result is not None else raw
                        call_seq.append({"name": name, "refused": False})
                        reply = {"result": replaced if isinstance(replaced, str) else str(replaced)}
                    except Exception as exc:  # noqa: BLE001 — tool bodies (and gates) raise freely; the program gets words
                        await orchestrator.run_after_tool(
                            name,
                            None,
                            str(exc),
                            tool_call_id=f"code:{len(call_seq)}",
                            arguments=run_args,
                        )
                        call_seq.append({"name": name, "refused": True})
                        # An AdmissionError is the gate holding, not a bug —
                        # surface it as the catchable refusal.
                        kind = "refused" if "Admission" in type(exc).__name__ else "error"
                        reply = {kind: str(exc)}
            proc.stdin.write((json.dumps(reply) + "\n").encode())
            await proc.stdin.drain()

    @tool(
        name="run_code",
        description=(
            "Run a Python program that can call your other tools in a loop "
            "without round-tripping each result through the conversation. "
            "Inside the program, call tools as tools.call('tool_name', "
            "arg=value); the return value is the tool's string result. A "
            "call a policy refuses raises ToolRefused — catch it if you can "
            "continue without that call. Set a variable named `result` to "
            "return a value; print() output is also returned. Use for "
            "multi-step tool workflows (loops, filtering large results, "
            "aggregation) where intermediate data does not need to be seen."
        ),
        labels={"code-exec"},
    )
    async def run_code(code: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            python or sys.executable,
            "-I",
            "-c",
            _HARNESS,
            json.dumps(code),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin"},
        )
        call_seq: list[dict[str, Any]] = []
        try:
            done = await asyncio.wait_for(_serve(proc, call_seq), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return json.dumps({"error": f"code timed out after {timeout}s", "tool_calls": call_seq})
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
        out = {
            "stdout": (done.get("stdout") or "")[:_RESULT_CAP],
            "result": (done.get("result") or "")[:_RESULT_CAP] or None,
            "error": done.get("error"),
            "tool_calls": call_seq,
        }
        return json.dumps({k: v for k, v in out.items() if v})

    return run_code
