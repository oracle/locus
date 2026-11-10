# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The ``locus`` command: run an agent, decide what it is waiting on, check its record.

``AGENT`` is ``module:attr`` or ``path/to/file.py:attr``, naming an ``Agent``
or a zero-argument function that returns one::

    locus run app.py:agent "refund order 4821" --thread t1
    locus approvals --store approvals.json list
    locus approvals --store approvals.json approve appr-3f1c... --by alice@example.com
    locus resume app.py:agent --thread t1 --answer approved --perform
    locus audit verify trail.jsonl --key audit-1=audit-1.pub.pem --head 9c2e...
    locus serve app.py:agent --port 8000
    locus codex init                 # the Locus skill for Codex, in this project
    locus codex run "build an agent that triages tickets with a gated refund tool"

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

    from locus.agent import Agent

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
        name = f"_locus_cli_{path.stem}"
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
    from locus.agent import Agent  # noqa: PLC0415

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
    from locus.core.events import (  # noqa: PLC0415
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
            print(f"  resume with: locus resume {agent_spec}{thread} --answer ...{perform}")
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
    from locus.control import ApprovalAuthority, FileApprovals  # noqa: PLC0415

    authority = None
    if args.authority:
        authority = load_object(args.authority)
        if callable(authority) and not isinstance(authority, ApprovalAuthority):
            authority = authority()
        if not isinstance(authority, ApprovalAuthority):
            raise CLIError(f"{args.authority} is not an ApprovalAuthority")
    return FileApprovals(args.store, authority)


def _cmd_approvals(args: argparse.Namespace) -> int:
    from locus.control import ApprovalAuthorityError  # noqa: PLC0415

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
            print(f"locus: no approval {args.approval_id!r}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(record), indent=2, default=str))
        return 0
    verdict = "approved" if args.action == "approve" else "denied"
    try:
        record = store.decide(args.approval_id, verdict, by=args.by)
    except ApprovalAuthorityError as exc:
        print(f"locus: refused: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"locus: {exc}", file=sys.stderr)
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
    from locus.control import verify_jsonl  # noqa: PLC0415

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
    from locus.server import AgentServer  # noqa: PLC0415

    agent = load_agent(args.agent)
    try:
        AgentServer(agent=agent, api_key=args.api_key).run(host=args.host, port=args.port)
    except RuntimeError as exc:
        print(f"locus: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_codex(args: argparse.Namespace) -> int:
    from locus import codex  # noqa: PLC0415

    try:
        if args.action == "path":
            print(codex.skill_source())
            return 0
        if args.action == "init":
            project = Path(args.dir)
            dest = Path.home() / codex.SKILL_DIR if args.home else project / codex.SKILL_DIR
            written = codex.install(dest)
            print(
                f"{len(written)} file(s) written to {dest}" if written else f"{dest} is up to date"
            )
            if not args.home and codex.add_agents_section(project):
                print(f"Locus section added to {project / 'AGENTS.md'}")
            return 0
        project = Path.cwd()
        if not (Path.home() / codex.SKILL_DIR / "SKILL.md").is_file():
            codex.install(project / codex.SKILL_DIR)
            codex.add_agents_section(project)
        codex.launch(
            codex.codex_argv(
                args.prompt,
                exec_mode=args.exec_,
                codex_profile=args.codex_profile,
                extra=args.codex_args,
            )
        )
    except codex.CodexError as exc:
        raise CLIError(str(exc)) from exc
    return 0


def _load_judge(spec: str) -> Any:
    """An ``LLMJudge`` from ``spec``: a judge, a model to wrap, or a factory for either."""
    from locus.evaluation import LLMJudge  # noqa: PLC0415

    obj = load_object(spec)
    if callable(obj) and not hasattr(obj, "score") and not hasattr(obj, "complete"):
        obj = obj()
    if hasattr(obj, "score"):
        return obj
    if hasattr(obj, "complete"):
        return LLMJudge(obj)
    raise CLIError(f"{spec} is not a judge, a model, or a function returning one")


def _gate_from_args(args: argparse.Namespace) -> Any:
    from locus.evaluation import EvalGate  # noqa: PLC0415

    return EvalGate(
        min_pass_rate=args.min_pass_rate,
        min_avg_score=args.min_avg_score,
        max_not_graded=args.max_not_graded,
        max_unstable=args.max_unstable,
        no_regressions=not args.allow_regressions,
        max_avg_score_drop=args.max_avg_score_drop,
    )


def _cmd_eval(args: argparse.Namespace) -> int:
    from locus.evaluation import (  # noqa: PLC0415
        EvalReport,
        EvalRunner,
        JudgeUnavailableError,
        Rubric,
        diff_reports,
        load_cases,
    )

    def read(path: str) -> Any:
        try:
            return EvalReport.load(path)
        except (OSError, ValueError) as exc:
            raise CLIError(f"cannot read report {path}: {exc}") from exc

    if args.action == "summary":
        report = read(args.report)
        print(report.to_markdown() if args.markdown else report.summary())
        return 0
    if args.action == "diff":
        try:
            diff = diff_reports(read(args.baseline), read(args.candidate), force=args.force)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
        print(diff.to_markdown() if args.markdown else diff.summary())
        return 1 if diff.regressed else 0
    if args.action == "gate":
        report = read(args.report)
        baseline = read(args.baseline) if args.baseline else None
        result = _gate_from_args(args).check(report, baseline=baseline, force=args.force)
        print(result.summary())
        return 0 if result.passed else 1

    agent = load_agent(args.agent)
    judge = _load_judge(args.judge) if args.judge else None
    rubric = None
    if args.rubric:
        try:
            rubric = Rubric.load(args.rubric)
        except (OSError, ValueError, TypeError) as exc:
            raise CLIError(f"cannot read rubric {args.rubric}: {exc}") from exc
    try:
        cases = load_cases(args.cases, rubric=rubric)
    except (OSError, ValueError, TypeError) as exc:
        raise CLIError(f"cannot read cases {args.cases}: {exc}") from exc
    if args.tags:
        wanted = set(args.tags)
        cases = [case for case in cases if wanted & set(case.tags)]
    if not cases:
        raise CLIError("no cases to run")

    runner = EvalRunner(agent, judge=judge, concurrency=args.concurrency, repeats=args.repeats)
    try:
        report = asyncio.run(runner.arun(cases))
    except JudgeUnavailableError as exc:
        raise CLIError(f"the judge was unavailable, nothing was graded: {exc}") from exc
    print(report.summary())
    if args.report:
        print(f"report written to {report.save(args.report)}")
    baseline = read(args.baseline) if args.baseline else None
    result = _gate_from_args(args).check(report, baseline=baseline)
    print(result.summary())
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for ``locus``."""
    from locus import __version__  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="locus", description="Run an agent, decide what it waits on, check its record."
    )
    parser.add_argument("--version", action="version", version=f"locus-sdk {__version__}")
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
    serve.add_argument("--api-key", help="bearer key; or set LOCUS_SERVER_API_KEY")
    serve.set_defaults(handler=_cmd_serve)
    codex = commands.add_parser(
        "codex", help="install the Locus skill for Codex and start Codex with it"
    )
    codex_actions = codex.add_subparsers(dest="action", required=True)
    init = codex_actions.add_parser(
        "init",
        help="put the skill in DIR/.agents/skills/locus and a Locus section in DIR/AGENTS.md",
    )
    init.add_argument("dir", nargs="?", default=".", help="the project root (default: here)")
    init.add_argument(
        "--home", action="store_true", help="install for every project, in ~/.agents/skills/locus"
    )
    run_codex = codex_actions.add_parser(
        "run",
        help="install the skill here if it is missing, then start Codex with the sandbox settings it needs",
        usage="%(prog)s [-h] [--exec] [--codex-profile NAME] [PROMPT] [-- CODEX_ARGS ...]",
        epilog="Arguments after -- go to codex unchanged, before the prompt: "
        'locus codex run "build it" -- -m gpt-5.5',
    )
    run_codex.add_argument(
        "prompt", nargs="?", help="what to build; omit for an interactive session"
    )
    run_codex.add_argument(
        "--exec", dest="exec_", action="store_true", help="non-interactive (`codex exec`)"
    )
    run_codex.add_argument(
        "--codex-profile",
        help="a Codex profile, ~/.codex/<name>.config.toml, such as one that points Codex at OCI Generative AI",
    )
    run_codex.set_defaults(codex_args=[])
    codex_actions.add_parser("path", help="print where the bundled skill is")
    codex.set_defaults(handler=_cmd_codex)

    evaluate = commands.add_parser(
        "eval", help="run an evaluation suite, diff two reports, or gate a build on one"
    )
    eval_actions = evaluate.add_subparsers(dest="action", required=True)
    eval_run = eval_actions.add_parser(
        "run", help="run the cases in a file against an agent and print the report"
    )
    eval_run.add_argument("cases", help="a .json, .jsonl or .yaml file of EvalCase objects")
    eval_run.add_argument(
        "--agent", required=True, help="MODULE:ATTR or FILE.py:ATTR naming an Agent or a factory"
    )
    eval_run.add_argument(
        "--judge",
        help="MODULE:ATTR naming an LLMJudge, a model to wrap in one, or a factory for either",
    )
    eval_run.add_argument("--rubric", help="a rubric file applied to cases that name none")
    eval_run.add_argument("--repeats", type=int, default=1, help="readings per case")
    eval_run.add_argument("--concurrency", type=int, default=4, help="cases run at once")
    eval_run.add_argument("--tags", nargs="*", default=[], help="only cases carrying one of these")
    eval_run.add_argument("--report", help="write the report as JSON here")
    eval_run.add_argument("--baseline", help="a saved report to gate regressions against")
    _add_gate_arguments(eval_run)
    eval_run.set_defaults(handler=_cmd_eval)

    eval_diff = eval_actions.add_parser("diff", help="compare two saved reports case by case")
    eval_diff.add_argument("baseline")
    eval_diff.add_argument("candidate")
    eval_diff.add_argument(
        "--force", action="store_true", help="diff even when the suites' fingerprints differ"
    )
    eval_diff.add_argument("--markdown", action="store_true", help="print a Markdown table")
    eval_diff.set_defaults(handler=_cmd_eval)

    eval_gate = eval_actions.add_parser("gate", help="apply the gate to a saved report")
    eval_gate.add_argument("report")
    eval_gate.add_argument("--baseline", help="a saved report to gate regressions against")
    eval_gate.add_argument(
        "--force", action="store_true", help="gate even when the suites' fingerprints differ"
    )
    _add_gate_arguments(eval_gate)
    eval_gate.set_defaults(handler=_cmd_eval)

    eval_summary = eval_actions.add_parser("summary", help="print a saved report")
    eval_summary.add_argument("report")
    eval_summary.add_argument("--markdown", action="store_true", help="print a Markdown table")
    eval_summary.set_defaults(handler=_cmd_eval)
    return parser


def _add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-pass-rate", type=float, default=1.0, help="over graded cases (default 1.0)"
    )
    parser.add_argument("--min-avg-score", type=float, help="0-1")
    parser.add_argument("--max-not-graded", type=int, default=0, help="default 0")
    parser.add_argument("--max-unstable", type=int, help="with --repeats")
    parser.add_argument(
        "--allow-regressions",
        action="store_true",
        help="with --baseline: do not fail on a case that passed before and fails now",
    )
    parser.add_argument(
        "--max-avg-score-drop", type=float, help="with --baseline: the average may not drop by more"
    )


def _split_codex_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """``locus codex run ... -- <args>``: what follows ``--`` goes to codex unchanged, in order."""
    if argv[:2] == ["codex", "run"] and "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``locus`` command."""
    parser = build_parser()
    own, codex_args = _split_codex_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(own)
    if codex_args:
        args.codex_args = codex_args
    try:
        code: int = args.handler(args)
    except CLIError as exc:
        print(f"locus: error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
