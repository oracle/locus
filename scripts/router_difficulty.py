#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Measure the difficulty of the router corpus from outcomes.

Every labelled prompt of ``benchmarks/router/cases.json`` is answered in
one reply, with no tools, by each tier (cheapest model first), and a judge
grades every answer against the answer rubric. A prompt is ``low`` when the
first tier's reply passes, ``medium`` when only a later tier's does and
``high`` when none does. The script prints the intended level against the
measured one and the cases where they differ, and writes the measurement
as JSON.

    python scripts/router_difficulty.py --tiers oci:<small-model> oci:<large-model> \
        --judge oci:<judge-model> --repeats 3

Model strings go through ``locus.models.get_model``; an ``oci:`` model reads
``OCI_PROFILE``, ``OCI_REGION`` and ``OCI_COMPARTMENT_ID`` from the
environment. Use a judge that is none of the tiers: a model grading its own
answer measures self-consistency. The tier reports go to a
``FileReportStore`` (``--reports``), one suite per tier, and the
measurement to ``--out``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from locus.evaluation import EvalReport, EvalRunner, FileReportStore, LLMJudge  # noqa: E402
from locus.models import get_model  # noqa: E402
from locus.router.benchmark import load_router_corpus  # noqa: E402
from locus.router.difficulty import (  # noqa: E402
    NOT_MEASURED,
    MeasuredDifficulty,
    ShapeAgent,
    answer_cases,
    confusion_table,
    direct_agent,
    disagreements,
    measure_difficulty,
    shape_rubric,
)


CORPUS = ROOT / "benchmarks" / "router" / "cases.json"
REPORTS = ROOT / "benchmarks" / "router" / "reports"


def _label(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec).strip("-")


async def _main(args: argparse.Namespace) -> tuple[list[MeasuredDifficulty], list[str]]:
    if len(args.tiers) < 2:
        sys.exit("--tiers needs at least two models, cheapest first")
    cases = answer_cases(load_router_corpus(args.corpus))
    if args.tags:
        wanted = set(args.tags)
        cases = [case for case in cases if wanted & set(case.tags)]
    if not cases:
        sys.exit("no cases to run")

    judge = LLMJudge(get_model(args.judge))
    store = FileReportStore(args.reports)
    reports: dict[str, EvalReport] = {}
    for spec in args.tiers:
        label = _label(spec)
        print(f"== {label}: {len(cases)} cases x {args.repeats} readings", flush=True)
        runner = EvalRunner(
            direct_agent(get_model(spec, max_tokens=args.max_tokens)),
            judge=judge,
            repeats=args.repeats,
            concurrency=args.concurrency,
        )
        report = await runner.arun(cases)
        reports[label] = report
        errors = [r.error for r in report.results if r.error]
        if len(errors) == report.total_cases:
            sys.exit(f"{label} answered nothing; every case failed with: {errors[0]}")
        stored = store.put(report, suite=f"router-difficulty-{label}", report_id=label)
        print(
            f"   {report.passed}/{report.total_cases} passed on every reading, "
            f"{report.not_graded} not graded, {len(errors)} errored; "
            f"stored at {stored.location}"
        )

    if args.shape:
        # The structured shape is the last tier, run only where no single
        # reply passed: a case a reply settles is already ``low`` or ``medium``.
        settled = {
            result.case_name
            for report in reports.values()
            for result in report.results
            if result.passed
        }
        open_cases = [case for case in cases if case.name not in settled]
        label = f"shape-{_label(args.shape)}"
        print(f"== {label}: {len(open_cases)} cases x {args.repeats} readings", flush=True)
        shape_model = get_model(args.shape, max_tokens=args.max_tokens)
        runner = EvalRunner(
            ShapeAgent(shape_model, open_cases),
            judge=judge,
            repeats=args.repeats,
            concurrency=args.concurrency,
        )
        report = await runner.arun(answer_cases(open_cases, rubric=shape_rubric()))
        reports[label] = report
        stored = store.put(report, suite=f"router-difficulty-{label}", report_id=label)
        errors = [r.error for r in report.results if r.error]
        print(
            f"   {report.passed}/{report.total_cases} passed on every reading, "
            f"{report.not_graded} not graded, {len(errors)} errored; "
            f"stored at {stored.location}"
        )

    measured = measure_difficulty(cases, reports)
    print("\n### Intended against measured difficulty\n")
    print(confusion_table(measured))
    differing = disagreements(measured)
    print(f"\n### Cases whose measured level differs ({len(differing)})\n")
    for item in differing:
        outcome = ", ".join(f"{tier}={passed}" for tier, passed in item.passed.items())
        trap = " trap" if item.trap else ""
        print(f"  {item.case:<46} {item.intended} -> {item.measured}{trap}  ({outcome})")
    unmeasured = [m.case for m in measured if m.measured == NOT_MEASURED]
    if unmeasured:
        print(f"\nnot measured ({len(unmeasured)}): {', '.join(unmeasured)}")

    return measured, list(reports)


def _write(args: argparse.Namespace, measured: list[MeasuredDifficulty], tiers: list[str]) -> None:
    payload = {
        "tiers": tiers,
        "judge": _label(args.judge),
        "repeats": args.repeats,
        "cases": [
            {
                "case": m.case,
                "intended": m.intended,
                "measured": m.measured,
                "trap": m.trap,
                "passed": m.passed,
            }
            for m in measured
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nmeasurement written to {args.out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tiers", nargs="+", required=True, help="model strings, cheapest first (two or more)"
    )
    parser.add_argument(
        "--shape",
        help="a model to run each prompt's labelled shape on, with working tools, as the "
        "last tier; only the prompts no single reply passed are run",
    )
    parser.add_argument("--judge", required=True, help="the model that grades the answers")
    parser.add_argument("--repeats", type=int, default=3, help="readings per case (default 3)")
    parser.add_argument("--concurrency", type=int, default=8, help="cases in flight at once")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="output budget of a tier's reply (default 16384); a reasoning model that "
        "runs out returns an empty reply, which reads as a failed answer",
    )
    parser.add_argument("--corpus", default=str(CORPUS), help="the cases file")
    parser.add_argument("--reports", default=str(REPORTS), help="the FileReportStore directory")
    parser.add_argument("--out", help="write the measurement to this JSON file")
    parser.add_argument("--tags", nargs="*", default=[], help="only cases carrying one of these")
    args = parser.parse_args()
    measured, tiers = asyncio.run(_main(args))
    if args.out:
        _write(args, measured, tiers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
