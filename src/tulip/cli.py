# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``tulip`` command: run an agent, decide what it is waiting on, check its record.

``AGENT`` is ``module:attr`` or ``path/to/file.py:attr``, naming an ``Agent``
or a zero-argument function that returns one::

    tulip run app.py:agent "refund order 4821" --thread t1
    tulip approvals --store approvals.json list
    tulip approvals --store approvals.json approve appr-3f1c... --by alice@example.com
    tulip resume app.py:agent --thread t1 --answer approved --perform
    tulip audit verify trail.jsonl --key audit-1=audit-1.pub.pem --head 9c2e...
    tulip serve app.py:agent --port 8000

Exit codes: 0 success, 1 a check or decision failed, 2 usage error, 3 the run is
paused waiting for a person.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tulip.agent import Agent

#: The run stopped on a hold or a question and waits for a person.
EXIT_PAUSED = 3

_PREVIEW = 200


class CLIError(Exception):
    """A usage problem, reported without a traceback."""


def load_object(spec: str) -> Any:
    """The object named by ``module:attr`` or ``path/to/file.py:attr``."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise CLIError(f"expected MODULE:ATTR or FILE.py:ATTR, got {spec!r}")
    if target.endswith(".py") or os.sep in target:
        path = Path(target)
        if not path.is_file():
            raise CLIError(f"no such file: {target}")
        name = f"_tulip_cli_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise CLIError(f"cannot load {target}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
    else:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise CLIError(f"cannot import {target}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise CLIError(f"{target} has no attribute {attr!r}") from exc


def load_agent(spec: str) -> Agent:
    """An ``Agent`` from ``spec``, calling a factory when that is what it names."""
    from tulip.agent import Agent  # noqa: PLC0415

    obj = load_object(spec)
    if not isinstance(obj, Agent) and callable(obj):
        obj = obj()
    if not isinstance(obj, Agent):
        raise CLIError(f"{spec} is not an Agent or a function returning one")
    return obj


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= _PREVIEW else text[:_PREVIEW] + "..."


async def _stream(events: AsyncIterator[Any], agent_spec: str, thread_id: str | None) -> int:
    from tulip.core.events import (  # noqa: PLC0415
        InterruptEvent,
        TerminateEvent,
        ToolCompleteEvent,
        ToolStartEvent,
    )

    code = 0
    async for event in events:
        if isinstance(event, ToolStartEvent):
            print(f"-> {event.tool_name}({_preview(event.arguments)})")
        elif isinstance(event, ToolCompleteEvent):
            outcome = f"error: {event.error}" if event.error else _preview(event.result)
            print(f"<- {event.tool_name}: {outcome}")
        elif isinstance(event, InterruptEvent):
            print(f"paused: {event.question}")
            approval_id = event.metadata.get("approval_id")
            if approval_id:
                print(f"  approval: {approval_id}")
            thread = f" --thread {thread_id}" if thread_id else ""
            perform = " --perform" if approval_id else ""
            print(f"  resume with: tulip resume {agent_spec}{thread} --answer ...{perform}")
            code = EXIT_PAUSED
        elif isinstance(event, TerminateEvent):
            if event.final_message:
                print(event.final_message)
            if event.reason not in ("complete", "terminal_tool", "no_tools", "confidence_met"):
                print(f"[stopped: {event.reason}]", file=sys.stderr)
    return code


def _cmd_run(args: argparse.Namespace) -> int:
    agent = load_agent(args.agent)
    return asyncio.run(
        _stream(agent.run(args.prompt, thread_id=args.thread), args.agent, args.thread)
    )


def _cmd_resume(args: argparse.Namespace) -> int:
    agent = load_agent(args.agent)
    events = agent.resume(args.answer, thread_id=args.thread, perform_dangling=args.perform)
    return asyncio.run(_stream(events, args.agent, args.thread))


def _store(args: argparse.Namespace) -> Any:
    from tulip.control import ApprovalAuthority, FileApprovals  # noqa: PLC0415

    authority = None
    if args.authority:
        authority = load_object(args.authority)
        if callable(authority) and not isinstance(authority, ApprovalAuthority):
            authority = authority()
        if not isinstance(authority, ApprovalAuthority):
            raise CLIError(f"{args.authority} is not an ApprovalAuthority")
    return FileApprovals(args.store, authority)


def _cmd_approvals(args: argparse.Namespace) -> int:
    from tulip.control import ApprovalAuthorityError  # noqa: PLC0415

    store = _store(args)
    if args.action == "list":
        pending = store.pending()
        if args.json:
            print(json.dumps([asdict(r) for r in pending], indent=2, default=str))
        elif not pending:
            print("no pending approvals")
        for record in [] if args.json else pending:
            so_far = (
                f"  approvals so far: {', '.join(record.approvers)}" if record.approvers else ""
            )
            print(f"{record.approval_id}  {record.tool}  requested by {record.principal}{so_far}")
            print(f"    {_preview(record.arguments)}  ({record.reason})")
        return 0
    if args.action == "show":
        record = store.get(args.approval_id)
        if record is None:
            print(f"tulip: no approval {args.approval_id!r}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(record), indent=2, default=str))
        return 0
    verdict = "approved" if args.action == "approve" else "denied"
    try:
        record = store.decide(args.approval_id, verdict, by=args.by)
    except ApprovalAuthorityError as exc:
        print(f"tulip: refused: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"tulip: {exc}", file=sys.stderr)
        return 1
    print(f"{record.approval_id}: {record.status}")
    return 0


def _parse_keys(pairs: Sequence[str]) -> dict[str, bytes]:
    keys: dict[str, bytes] = {}
    for pair in pairs:
        key_id, sep, path = pair.partition("=")
        if not sep or not key_id or not path:
            raise CLIError(f"--key expects KEY_ID=PUBLIC_KEY.pem, got {pair!r}")
        try:
            keys[key_id] = Path(path).read_bytes()
        except OSError as exc:
            raise CLIError(f"cannot read public key {path}: {exc}") from exc
    return keys


def _cmd_audit_verify(args: argparse.Namespace) -> int:
    from tulip.control import verify_jsonl  # noqa: PLC0415

    keys = _parse_keys(args.key) if args.key else None
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIError(f"cannot read {args.file}: {exc}") from exc
    records = sum(1 for line in text.splitlines() if line.strip())
    if verify_jsonl(text, keys=keys, expected_head=args.head):
        signed = f", signatures checked against {len(keys)} key(s)" if keys else ", chain only"
        anchored = ", head matches" if args.head else ""
        print(f"verified: {records} record(s){signed}{anchored}")
        return 0
    print(f"FAILED: {args.file} does not verify", file=sys.stderr)
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from tulip.server import AgentServer  # noqa: PLC0415

    agent = load_agent(args.agent)
    try:
        AgentServer(agent=agent, api_key=args.api_key).run(host=args.host, port=args.port)
    except RuntimeError as exc:
        print(f"tulip: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for ``tulip``."""
    from tulip import __version__  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="tulip", description="Run an agent, decide what it waits on, check its record."
    )
    parser.add_argument("--version", action="version", version=f"tulip-agents {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an agent on a prompt, streaming what it does")
    run.add_argument("agent", help="MODULE:ATTR or FILE.py:ATTR naming an Agent or a factory")
    run.add_argument("prompt")
    run.add_argument("--thread", help="thread id; needs a checkpointer on the agent to persist")
    run.set_defaults(handler=_cmd_run)

    resume = commands.add_parser("resume", help="resume a paused run")
    resume.add_argument("agent")
    resume.add_argument("--thread", required=True)
    resume.add_argument("--answer", default="", help="the answer to the question it paused on")
    resume.add_argument(
        "--perform", action="store_true", help="re-issue a held call now that it has been decided"
    )
    resume.set_defaults(handler=_cmd_resume)

    approvals = commands.add_parser("approvals", help="list and decide held calls")
    approvals.add_argument("--store", required=True, help="the FileApprovals JSON file")
    approvals.add_argument(
        "--authority", help="MODULE:ATTR naming an ApprovalAuthority checked on every decision"
    )
    actions = approvals.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="pending approvals")
    listing.add_argument("--json", action="store_true")
    show = actions.add_parser("show", help="one approval record")
    show.add_argument("approval_id")
    for verb in ("approve", "reject"):
        decide = actions.add_parser(verb, help=f"{verb} a held call")
        decide.add_argument("approval_id")
        decide.add_argument("--by", required=True, help="who is deciding")
    approvals.set_defaults(handler=_cmd_approvals)

    audit = commands.add_parser("audit", help="check an exported audit trail")
    audit_actions = audit.add_subparsers(dest="action", required=True)
    verify = audit_actions.add_parser("verify", help="verify a JSONL export")
    verify.add_argument("file")
    verify.add_argument(
        "--key",
        action="append",
        default=[],
        metavar="KEY_ID=PEM",
        help="trusted public key; repeat for rotated keys",
    )
    verify.add_argument("--head", help="the chain head anchored out of band")
    verify.set_defaults(handler=_cmd_audit_verify)

    serve = commands.add_parser("serve", help="serve an agent over HTTP")
    serve.add_argument("agent")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", help="bearer key; or set TULIP_SERVER_API_KEY")
    serve.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``tulip`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.handler(args)
    except CLIError as exc:
        print(f"tulip: error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
