# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The router's gate: typed selection against the free-form baseline, on one model.

Three selection-only targets run the corpus in ``benchmarks/router/cases.json``
through :class:`~locus.evaluation.EvalRunner` with three readings per case:

* ``typed``: the router with its default extractor;
* ``typed_classifier``: the router with a :class:`~locus.router.RequestClassifier`
  in its front seam. ``ROUTER_BENCHMARK_CLASSIFIER=module:attr`` names a
  factory ``(model) -> RequestClassifier`` (or an instance); without one the
  default extractor goes through the seam, so this target measures the
  same path as ``typed`` and a trained classifier replaces it later;
* ``freeform``: the model names the protocol id directly.

The rates split per tag with ``EvalReport.by_tag``, the reports go to a
``FileReportStore`` (``ROUTER_BENCHMARK_REPORTS``, else the test's temporary
directory) and the gate holds ``typed_classifier`` to the free-form report:
at least its pass rate on the whole corpus and on the ``trap`` cases, every
``injection`` case bounded, and no case that the baseline passed failing
(:class:`~locus.evaluation.EvalGate` with ``no_regressions``). The baseline is
the free-form report of this run, or the saved one
``ROUTER_BENCHMARK_BASELINE`` names. The verdict is printed as Markdown, which
is what a scheduled job posts.

Opt-in like the benchmark beside it: ``RUN_ROUTER_BENCHMARK=1``, a model
from the conftest (``LOCUS_TEST_MODEL=provider:model``), about 540 model
calls per run at the default three readings (``ROUTER_BENCHMARK_REPEATS``).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

from locus.evaluation import EvalReport, EvalRunner, FileReportStore
from locus.router import AgentFrameExtractor, RequestClassifier
from locus.router.benchmark import (
    TAG_INJECTION,
    TAG_TRAP,
    frame_extractor,
    freeform_target,
    load_router_corpus,
    mean_pass_rate,
    pass_rate,
    router_gate,
    tag_table,
    typed_target,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_ROUTER_BENCHMARK") != "1",
        reason="set RUN_ROUTER_BENCHMARK=1 to run the router gate (~540 model calls)",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks" / "router" / "cases.json"


def _classifier(model: Any) -> RequestClassifier:
    """The classifier under test: ``ROUTER_BENCHMARK_CLASSIFIER``, else the default extractor."""
    spec = os.environ.get("ROUTER_BENCHMARK_CLASSIFIER")
    if not spec:
        return AgentFrameExtractor(frame_extractor(model))
    module_name, _, attr = spec.rpartition(":")
    obj = getattr(importlib.import_module(module_name), attr)
    classifier: Any = obj(model) if callable(obj) and not hasattr(obj, "classify") else obj
    assert isinstance(classifier, RequestClassifier), f"{spec} is not a RequestClassifier"
    return classifier


@pytest.mark.asyncio
async def test_router_classifier_gate(model: object, tmp_path: Path) -> None:
    cases = load_router_corpus(CORPUS)
    repeats = int(os.environ.get("ROUTER_BENCHMARK_REPEATS", "3"))
    label = os.environ.get("BENCH_PROVIDER_LABEL") or type(model).__name__
    targets = {
        "typed": typed_target(model),
        "typed_classifier": typed_target(model, classifier=_classifier(model)),
        "freeform": freeform_target(model),
    }

    reports: dict[str, EvalReport] = {}
    for name, target in targets.items():
        reports[name] = await EvalRunner(target, repeats=repeats, concurrency=8).arun(cases)

    store = FileReportStore(os.environ.get("ROUTER_BENCHMARK_REPORTS") or tmp_path)
    for name, report in reports.items():
        stored = store.put(report, suite=f"router-{name}", report_id=f"{label}-{name}")
        print(f"{name}: report stored at {stored.location}")

    baseline_path = os.environ.get("ROUTER_BENCHMARK_BASELINE")
    baseline = EvalReport.load(baseline_path) if baseline_path else reports["freeform"]

    print(f"\n=== Router gate · {label} · n={len(cases)} · {repeats} readings per case ===\n")
    print(tag_table(reports, cases))
    for name, report in reports.items():
        mean = mean_pass_rate(report, cases)
        print(f"{name}: mean per-reading accuracy {mean:.1%}" if mean is not None else name)
    verdict = router_gate(reports["typed_classifier"], baseline)
    print()
    print(verdict.to_markdown())

    # The structural claim first, so a failure names the right thing: the
    # typed router never leaves the registered set under injection.
    for name in ("typed", "typed_classifier"):
        assert pass_rate(reports[name], TAG_INJECTION) == 1.0, (
            f"{name} emitted an unregistered protocol id under prompt injection"
        )
    trap = pass_rate(reports["typed_classifier"], TAG_TRAP)
    print(f"typed_classifier trap pass rate: {trap}")
    assert verdict.passed, verdict.summary()
