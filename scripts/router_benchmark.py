#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Run the router selection benchmark against one model and gate it.

Selection only: every target answers with the protocol id it would pick,
``unbounded`` for an id outside the registry or ``refused`` when nothing
was selected; no shape is built or run. The corpus is
``benchmarks/router/cases.json``; the reports go to a ``FileReportStore``
and the typed targets are gated against the free-form baseline with
``locus.router.benchmark.router_gate``. The exit code is the gate's.

    python scripts/router_benchmark.py --model oci:<model-id> --repeats 3
    python scripts/router_benchmark.py --model oci:<model-id> --targets typed freeform
    python scripts/router_benchmark.py --model oci:<model-id> --baseline reports/router-freeform/<file>.json

The model string goes through ``locus.models.get_model``; an ``oci:`` model
reads ``OCI_PROFILE``, ``OCI_REGION`` and ``OCI_COMPARTMENT_ID`` from the
environment. ``--classifier module:attr`` names a ``RequestClassifier``
factory ``(model) -> classifier`` (or an instance) for the
``typed_classifier`` target; without one that target runs the default
extractor through the classifier seam. ``--verbose`` prints every prompt
with its expected protocol and what each reading answered.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from locus.evaluation import EvalReport, EvalRunner, FileReportStore  # noqa: E402
from locus.models import get_model  # noqa: E402
from locus.router import AgentFrameExtractor, RequestClassifier  # noqa: E402
from locus.router.benchmark import (  # noqa: E402
    REFUSED,
    UNBOUNDED,
    SelectionTarget,
    frame_extractor,
    freeform_target,
    load_router_corpus,
    mean_pass_rate,
    router_gate,
    tag_table,
    typed_target,
)


TARGETS = ("typed", "typed_classifier", "freeform")
CORPUS = ROOT / "benchmarks" / "router" / "cases.json"
REPORTS = ROOT / "benchmarks" / "router" / "reports"


def _label(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec).strip("-")


def _classifier(spec: str | None, model: Any) -> RequestClassifier:
    if not spec:
        return AgentFrameExtractor(frame_extractor(model))
    module_name, sep, attr = spec.rpartition(":")
    if not sep:
        sys.exit(f"--classifier expects module:attr, got {spec!r}")
    obj = getattr(importlib.import_module(module_name), attr)
    classifier: Any = obj(model) if callable(obj) and not hasattr(obj, "classify") else obj
    if not isinstance(classifier, RequestClassifier):
        sys.exit(f"{spec} is not a RequestClassifier")
    return classifier


def _audit(target: SelectionTarget, cases: list[Any]) -> None:
    """One line per case: expected protocol and what each reading answered.

    An ``unbounded`` reading shows the id the model invented; a ``refused``
    one shows why, so a provider error is told apart from a rejected frame.
    """
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for record in target.records:
        shown = record.output
        if record.output == UNBOUNDED:
            shown = f"{UNBOUNDED}[{record.picked}]"
        elif record.output == REFUSED:
            shown = f"{REFUSED}[{record.detail[:60]}]"
        by_prompt[record.prompt].append(shown)
    for case in cases:
        expected = (case.expected_output_contains or ["(bounded)"])[0]
        readings = by_prompt.get(case.prompt, [])
        ok = all(
            r == expected if case.expected_output_contains else not r.startswith(UNBOUNDED)
            for r in readings
        )
        mark = "ok " if ok else "MISS"
        print(f"  {mark} {case.name:<44} {expected:<26} <- {', '.join(readings)}")


async def _main(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    model = get_model(args.model, **kwargs)
    cases = load_router_corpus(args.corpus)
    if args.tags:
        wanted = set(args.tags)
        cases = [case for case in cases if wanted & set(case.tags)]
    if not cases:
        sys.exit("no cases to run")

    targets: dict[str, SelectionTarget] = {}
    for name in args.targets:
        if name == "typed":
            targets[name] = typed_target(model)
        elif name == "typed_classifier":
            targets[name] = typed_target(model, classifier=_classifier(args.classifier, model))
        else:
            targets[name] = freeform_target(model)

    label = args.label or _label(args.model)
    store = FileReportStore(args.reports)
    reports: dict[str, EvalReport] = {}
    for name, target in targets.items():
        print(f"== {name}: {len(cases)} cases x {args.repeats} readings on {label}", flush=True)
        runner = EvalRunner(target, repeats=args.repeats, concurrency=args.concurrency)
        report = await runner.arun(cases)
        reports[name] = report
        stored = store.put(report, suite=f"router-{name}", report_id=label)
        mean = mean_pass_rate(report, cases)
        print(
            f"   {report.passed}/{report.total_cases} cases right on every reading, "
            f"mean per-reading accuracy {mean:.1%}; stored at {stored.location}"
            if mean is not None
            else f"   nothing graded; stored at {stored.location}"
        )
        if args.verbose:
            _audit(target, cases)

    print(f"\n### Pass rate per tag, strict (mean per reading) · {label}\n")
    print(tag_table(reports, cases))

    baseline = EvalReport.load(args.baseline) if args.baseline else reports.get("freeform")
    if baseline is None:
        print("\nno free-form report in this run and no --baseline: nothing to gate against")
        return 0
    against = args.baseline or "freeform (this run)"
    code = 0
    for name in ("typed", "typed_classifier"):
        if name not in reports:
            continue
        verdict = router_gate(reports[name], baseline)
        print(f"\n### Gate: {name} against {against}\n")
        print(verdict.to_markdown())
        if not verdict.passed:
            code = 1
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="a model string for locus.models.get_model")
    parser.add_argument("--repeats", type=int, default=3, help="readings per case (default 3)")
    parser.add_argument("--concurrency", type=int, default=8, help="cases in flight at once")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--corpus", default=str(CORPUS), help="the cases file")
    parser.add_argument("--reports", default=str(REPORTS), help="the FileReportStore directory")
    parser.add_argument("--baseline", help="a saved free-form report to gate against")
    parser.add_argument(
        "--classifier", help="module:attr of a RequestClassifier factory or instance"
    )
    parser.add_argument("--tags", nargs="*", default=[], help="only cases carrying one of these")
    parser.add_argument("--label", help="report id and column label (default: from --model)")
    parser.add_argument("--max-tokens", type=int, help="override the model's max_tokens")
    parser.add_argument("--verbose", action="store_true", help="print every prompt's readings")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
