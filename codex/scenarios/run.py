# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Run the Codex scenarios: Codex builds each one from the Locus skill, then its checks run.

Each directory beside this file is one scenario: ``PROMPT.md`` is the task Codex
gets, ``check.sh`` is run afterwards in the project Codex built and decides pass
or fail, and a ``live`` file marks a scenario that makes a live model call.

    python codex/scenarios/run.py                         # every scenario, Codex as configured
    python codex/scenarios/run.py --skip-live             # offline scenarios only
    python codex/scenarios/run.py --only 01-gated-refund-agent 05-harness-gated-workspace
    python codex/scenarios/run.py --codex-profile oci     # Codex itself on OCI Generative AI
    python codex/scenarios/run.py --locus-install "/path/to/checkout[oci]"   # unreleased code
    python codex/scenarios/run.py --parallel 3 --out out/scenarios

For each scenario the runner makes an empty project directory under ``--out``,
starts ``locus codex run --exec`` there with the preamble below followed by the
scenario's prompt, and when Codex returns runs ``check.sh`` in that directory.
Transcripts, Codex's report and the check output land under ``--out/_runs/<name>``;
a summary table is printed and written to ``--out/_runs/summary.md``. Exit status
is the number of scenarios that did not pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent

PREAMBLE = """\
Scenario task for a coding agent building with Locus (`locus-sdk`). The current directory is
empty and is the project root: write only inside it. Set up with `python3 -m venv .venv`, then
`.venv/bin/pip install "{install}" pytest`, and use `.venv/bin/python` for every command. Work
offline unless the task says live: tests use the model doubles in `locus.testing`. Never print,
log or store a credential. When the task's checks pass, write a short report (the files, the
test result line, anything you could not do) and stop.

"""


@dataclass
class Outcome:
    name: str
    live: bool
    passed: bool
    codex_exit: int | None
    check_exit: int | None
    seconds: float
    checks: list[str]
    note: str = ""


def scenarios(only: list[str] | None, *, skip_live: bool) -> list[Path]:
    found = sorted(p.parent for p in HERE.glob("*/PROMPT.md"))
    if only:
        wanted = set(only)
        found = [p for p in found if p.name in wanted]
        missing = wanted - {p.name for p in found}
        if missing:
            msg = f"no such scenario: {', '.join(sorted(missing))}"
            raise SystemExit(msg)
    if skip_live:
        found = [p for p in found if not (p / "live").exists()]
    return found


def run_one(scenario: Path, args: argparse.Namespace) -> Outcome:
    name = scenario.name
    live = (scenario / "live").exists()
    project = Path(args.out) / name
    logs = Path(args.out) / "_runs" / name
    for path in (project, logs):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
    prompt = PREAMBLE.format(install=args.locus_install) + (scenario / "PROMPT.md").read_text()
    cmd = [args.locus_bin, "codex", "run", "--exec"]
    if args.codex_profile:
        cmd += ["--codex-profile", args.codex_profile]
    cmd += ["--", "--skip-git-repo-check", "-o", str(logs / "report.md"), prompt]
    started = time.monotonic()
    codex_exit: int | None = None
    note = ""
    with (logs / "transcript.txt").open("w") as transcript:
        try:
            codex_exit = subprocess.run(  # noqa: S603 - the command is this runner's own
                cmd,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=transcript,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            note = f"codex did not finish within {args.timeout}s"
    check_exit: int | None = None
    checks: list[str] = []
    if codex_exit == 0:
        with (logs / "check.txt").open("w") as check_out:
            try:
                check_exit = subprocess.run(  # noqa: S603
                    [shutil.which("bash") or "/bin/bash", str(scenario / "check.sh")],
                    cwd=project,
                    stdin=subprocess.DEVNULL,
                    stdout=check_out,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                ).returncode
            except subprocess.TimeoutExpired:
                note = "check.sh did not finish within 600s"
        text = (logs / "check.txt").read_text()
        checks = [line[4:] for line in text.splitlines() if line.startswith("ok: ")]
        if check_exit not in (0, None) and not note:
            tail = [line for line in text.splitlines() if line.strip()][-1:]
            note = tail[0][:120] if tail else "check.sh failed"
    outcome = Outcome(
        name=name,
        live=live,
        passed=codex_exit == 0 and check_exit == 0,
        codex_exit=codex_exit,
        check_exit=check_exit,
        seconds=round(time.monotonic() - started, 1),
        checks=checks,
        note=note,
    )
    (logs / "result.json").write_text(json.dumps(asdict(outcome), indent=2) + "\n")
    return outcome


def summary(outcomes: list[Outcome]) -> str:
    lines = [
        "| Scenario | Result | Codex | Checks | Note |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        result = "PASS" if o.passed else "FAIL"
        live = " (live)" if o.live else ""
        lines.append(
            f"| `{o.name}`{live} | {result} | {o.seconds:.0f}s | {len(o.checks)} | {o.note} |"
        )
    passed = sum(o.passed for o in outcomes)
    lines.append("")
    lines.append(f"{passed} of {len(outcomes)} scenarios passed.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="out/scenarios", help="where projects and logs go")
    parser.add_argument("--only", nargs="*", help="scenario directory names to run")
    parser.add_argument("--skip-live", action="store_true", help="skip scenarios that call a model")
    parser.add_argument("--codex-profile", help="passed to `locus codex run --codex-profile`")
    parser.add_argument(
        "--locus-install",
        default="locus-sdk[oci]",
        help='what Codex installs, e.g. "locus-sdk[oci]" or "/path/to/checkout[oci]"',
    )
    parser.add_argument("--locus-bin", default=shutil.which("locus") or "locus")
    parser.add_argument("--parallel", type=int, default=1, help="scenarios run at once")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per Codex run")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = parser.parse_args(argv)

    chosen = scenarios(args.only, skip_live=args.skip_live)
    if args.list:
        for s in chosen:
            print(f"{s.name}{'  (live)' if (s / 'live').exists() else ''}")  # noqa: T201
        return 0
    if not chosen:
        print("nothing to run", file=sys.stderr)  # noqa: T201
        return 1
    (Path(args.out) / "_runs").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        outcomes = list(pool.map(lambda s: run_one(s, args), chosen))
    report = summary(outcomes)
    (Path(args.out) / "_runs" / "summary.md").write_text(report)
    print(report, end="")  # noqa: T201
    return sum(not o.passed for o in outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
