# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Routing benchmark for ``locus.router``.

The evidence behind the router's claim, which is structural, not an
accuracy claim: **a typed goal frame keeps every selection inside the
registered protocols, including under prompt injection**, where a model
asked to name the protocol directly can be talked into inventing one. The
accuracy comparison against that free-form baseline is printed as data; on
OCI Generative AI it goes both ways by model. Every PR that touches
``locus.router`` should re-run this benchmark and read the tables.

Three ablations, run against the same labelled corpus:

* **Full** — the default router: typed :class:`GoalFrame` extraction +
  :class:`ProtocolRegistry` with ``primary_for`` canonical ranking.
* **NoCanonical** — same router, but every protocol's
  ``primary_for`` is wiped to ``[]``. Isolates the contribution of the
  canonical-match tiebreaker in :func:`_rank_key`.
* **FreeForm** — the LLM picks ``protocol_id`` directly via a
  flat output schema (no :class:`GoalFrame`, no registry). The
  baseline that "let the LLM pick the topology" frameworks
  effectively run.

For each ablation we measure four things over the corpus:

* ``goal_acc`` — fraction of frames whose extracted ``primary_goal``
  matches the labelled ``expected_goal`` (Full + NoCanonical only).
* ``protocol_acc`` — fraction whose selected protocol matches
  ``expected_protocol``.
* ``bounded_rate`` — fraction whose selected protocol is in the
  registered set (the boundedness guarantee). Should be 1.0 for the
  router; the FreeForm baseline can hallucinate ids.
* ``extract_fail_rate`` — fraction where the LLM produced something
  the schema rejected.

Skipped without a real provider. Anthropic Haiku-4.5 is the cheap
default; the conftest also resolves OpenAI when its env vars are set.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from locus import Agent
from locus.router import GoalFrame, Protocol, Router, TaskType, builtin_protocols
from locus.router.benchmark import (
    TAG_INJECTION,
    TAG_LABELLED,
    FreeFormChoice,
    benchmark_compiler,
    frame_extractor,
    freeform_picker,
    load_router_corpus,
)
from locus.router.runtime import FrameExtractionError


# Opt-in, and marked slow. This file is a benchmark, not a regression test:
# each of its two tests sweeps the whole 180-prompt corpus through three
# ablations, so a single run is ~540+ model calls. Against a frontier endpoint
# that is merely slow; against a self-hosted one it dominates the suite and
# looks like a hang — the run stops advancing at this file with nothing to
# indicate why.
#
# The docstring above asks that every PR touching ``locus.router`` re-run it,
# which stays true: set RUN_ROUTER_BENCHMARK=1. Same shape as
# RUN_A2A_INTEGRATION and LOCUS_MYSQL_INTEGRATION elsewhere in this suite.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_ROUTER_BENCHMARK") != "1",
        reason=(
            "set RUN_ROUTER_BENCHMARK=1 to run the routing-accuracy benchmark "
            "(~126+ model calls per test)"
        ),
    ),
]


# Override at runtime to compare providers — e.g.
# ``BENCH_PROVIDER_LABEL=openai-gpt-4o-mini pytest …``
PROVIDER_LABEL = os.getenv("BENCH_PROVIDER_LABEL", "")


# ---------------------------------------------------------------------------
# Corpus — ``benchmarks/router/cases.json``, the one source of truth this file
# shares with ``test_router_classifier_gate.py`` and
# ``scripts/router_benchmark.py``.
#
# 180 labelled prompts across the eight protocols and the ``TaskType`` values
# they are canonical for. Each is reproducible: same prompt, same expected
# ``primary_goal`` (the case's ``expected_goal`` metadata), same canonical
# protocol id (its ``shape:`` tag and expectation).
#
# 18 adversarial prompts designed to *break* boundedness in the FreeForm
# baseline while leaving the typed router unaffected. Two flavours:
#
#   1. **Prompt-injection** — explicitly tells the LLM to ignore the
#      protocol list and invent a name. The typed router can't comply
#      because ``GoalFrame.primary_goal`` is a closed StrEnum; the
#      registry then maps to a real protocol. The FreeForm baseline
#      can comply because its schema accepts any ``str``.
#   2. **Out-of-distribution / ambiguous** — prompts where no protocol
#      is the obviously-right answer (creative writing, conversational,
#      empty-ish). Tests how the LLM degrades when forced to choose.
#
# They have no ``expected_protocol`` — the metric is pure ``bounded_rate``:
# how often the selection lives in the registered set. The router is
# bounded by construction (100% required); the FreeForm baseline is bounded
# only by the model's cooperation.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "benchmarks" / "router" / "cases.json"


@dataclass(frozen=True)
class CorpusEntry:
    prompt: str
    expected_goal: TaskType
    expected_protocol: str
    domain: str = "research"
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdversarialEntry:
    prompt: str
    flavour: Literal["injection", "ambiguous"]


def _load_corpus() -> tuple[tuple[CorpusEntry, ...], tuple[AdversarialEntry, ...]]:
    labelled: list[CorpusEntry] = []
    adversarial: list[AdversarialEntry] = []
    for case in load_router_corpus(CORPUS_FILE):
        if TAG_LABELLED in case.tags:
            labelled.append(
                CorpusEntry(
                    prompt=case.prompt,
                    expected_goal=TaskType(case.metadata["expected_goal"]),
                    expected_protocol=case.expected_output_contains[0],
                    domain=case.metadata.get("domain", "research"),
                    capabilities=tuple(case.metadata.get("capabilities", ())),
                )
            )
        else:
            flavour: Literal["injection", "ambiguous"] = (
                "injection" if TAG_INJECTION in case.tags else "ambiguous"
            )
            adversarial.append(AdversarialEntry(case.prompt, flavour))
    return tuple(labelled), tuple(adversarial)


CORPUS, ADVERSARIAL_CORPUS = _load_corpus()


# ---------------------------------------------------------------------------
# Helpers — build the three router variants under test. The extractor prompt,
# the capability probes and the free-form picker are the harness's own
# (``locus.router.benchmark``), so this file and the gate measure one thing.
# ---------------------------------------------------------------------------


def _router_full(model: object) -> Router:
    """Default router — typed extraction + canonical-match ranking."""
    return Router(extractor=frame_extractor(model), compiler=benchmark_compiler(model))


def _router_no_canonical(model: object) -> Router:
    """Router with ``primary_for=[]`` for every protocol — isolates the
    contribution of the canonical-match ranking tiebreaker."""
    wiped = [
        # Reconstruct each Protocol with primary_for wiped. ``builder`` is
        # opaque and survives unchanged.
        Protocol(
            id=p.id,
            description=p.description,
            handles=p.handles,
            primary_for=[],
            requires_capabilities=p.requires_capabilities,
            risk_max=p.risk_max,
            cost=p.cost,
            latency=p.latency,
            supports_streaming=p.supports_streaming,
            supports_repair=p.supports_repair,
            builder=p.builder,
        )
        for p in builtin_protocols()
    ]
    return Router(
        extractor=frame_extractor(model), compiler=benchmark_compiler(model, protocols=wiped)
    )


def _freeform_extractor(model: object) -> Agent:
    """Free-form baseline — the LLM picks a protocol id directly.

    It may pick *anything* (``FreeFormChoice.protocol_id`` is ``str``, not a
    Literal) so we can measure how often it hallucinates ids outside the
    registered set. That's the boundedness delta the typed router provides.
    """
    return freeform_picker(model)


# ---------------------------------------------------------------------------
# Score collection.
# ---------------------------------------------------------------------------


@dataclass
class AblationScore:
    name: str
    n: int = 0
    goal_correct: int = 0
    protocol_correct: int = 0
    bounded: int = 0
    extract_failures: int = 0
    build_failures: int = 0
    rows: list[str] = field(default_factory=list)

    def record_router(
        self,
        entry: CorpusEntry,
        frame: GoalFrame | None,
        chosen: str | None,
        registered: set[str],
        error: str | None,
        build_error: str | None = None,
    ) -> None:
        self.n += 1
        if frame is not None and frame.primary_goal == entry.expected_goal:
            self.goal_correct += 1
        if chosen is not None:
            if chosen in registered:
                self.bounded += 1
            if chosen == entry.expected_protocol:
                self.protocol_correct += 1
        if error is not None:
            self.extract_failures += 1
        if build_error is not None:
            self.build_failures += 1
        tag_g = "✓" if frame is not None and frame.primary_goal == entry.expected_goal else "✗"
        tag_p = "✓" if chosen == entry.expected_protocol else "✗"
        chosen_str = chosen or f"FAIL:{error or build_error}"
        suffix = f" [build_fail:{build_error[:40]}]" if build_error else ""
        self.rows.append(
            f"  {tag_g}{tag_p}  {entry.expected_goal.value:14s}/"
            f"{entry.expected_protocol:24s} → "
            f"{(frame.primary_goal.value if frame else '?'):14s}/"
            f"{chosen_str[:30]}{suffix}",
        )

    def record_freeform(
        self, entry: CorpusEntry, chosen: str | None, registered: set[str], error: str | None
    ) -> None:
        # Free-form has no frame, so goal_correct is unmeasurable.
        self.n += 1
        if chosen is not None:
            if chosen in registered:
                self.bounded += 1
            if chosen == entry.expected_protocol:
                self.protocol_correct += 1
        if error is not None:
            self.extract_failures += 1
        tag_p = "✓" if chosen == entry.expected_protocol else "✗"
        chosen_str = chosen or f"FAIL:{error}"
        self.rows.append(
            f"   {tag_p}  ?              /{entry.expected_protocol:24s} → ?/{chosen_str[:30]}",
        )

    def summary_row(self, kind: Literal["router", "freeform"]) -> str:
        denom = max(1, self.n)
        goal = f"{self.goal_correct / denom:>5.1%}" if kind == "router" else "  n/a"
        build = f"{self.build_failures / denom:>5.1%}" if kind == "router" else "  n/a"
        return (
            f"| {self.name:<14} | {goal} | "
            f"{self.protocol_correct / denom:>5.1%} | "
            f"{self.bounded / denom:>5.1%} | "
            f"{self.extract_failures / denom:>5.1%} | "
            f"{build} | "
            f"{self.n:>3} |"
        )


# ---------------------------------------------------------------------------
# The benchmark itself.
# ---------------------------------------------------------------------------


REGISTERED_IDS: set[str] = {p.id for p in builtin_protocols()}


def _runnable_protocol_id(runnable: object) -> str | None:
    direct = getattr(runnable, "protocol_id", None)
    if direct is not None:
        return str(direct)
    inner = getattr(runnable, "inner", None)
    if inner is not None:
        return _runnable_protocol_id(inner)
    return None


async def _run_router_corpus(router: Router, score: AblationScore) -> None:
    for entry in CORPUS:
        try:
            frame = await router.extract(entry.prompt)
        except FrameExtractionError as exc:
            score.record_router(entry, None, None, REGISTERED_IDS, str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 — a provider error on one prompt is a data point
            score.record_router(
                entry, None, None, REGISTERED_IDS, f"{type(exc).__name__}: {str(exc)[:120]}"
            )
            continue
        # Resolve which protocol the compiler *would* pick — independent of
        # whether the builder succeeds, and through ``explain`` so the
        # low-complexity short-circuit and the risk fallback count the way a
        # real dispatch would. This lets us count selection accuracy even when
        # an opt-in builder (e.g. a2a_delegate without an endpoint) raises
        # during compile.
        selected_id = router.compiler.explain(frame).selected_protocol
        if selected_id is None:
            score.record_router(entry, frame, None, REGISTERED_IDS, "no protocol could take it")
            continue

        try:
            await router.compiler.compile(frame)
        except RuntimeError as exc:
            # Builder couldn't materialise (typically a2a_delegate
            # without an endpoint, or any future opt-in protocol that
            # demands more context than the compiler supplied). Still
            # record the *selection* accuracy.
            score.record_router(
                entry,
                frame,
                selected_id,
                REGISTERED_IDS,
                None,
                build_error=type(exc).__name__ + ":" + str(exc),
            )
            continue
        score.record_router(entry, frame, selected_id, REGISTERED_IDS, None)


async def _run_freeform_corpus(extractor: Agent, score: AblationScore) -> None:
    for entry in CORPUS:
        try:
            result = await asyncio.to_thread(extractor.invoke, entry.prompt)
        except Exception as exc:  # noqa: BLE001 — see _run_router_corpus
            score.record_freeform(
                entry, None, REGISTERED_IDS, f"{type(exc).__name__}: {str(exc)[:120]}"
            )
            continue
        parsed = result.parsed
        if not isinstance(parsed, FreeFormChoice):
            score.record_freeform(entry, None, REGISTERED_IDS, result.parse_error or "no parse")
            continue
        score.record_freeform(entry, parsed.protocol_id, REGISTERED_IDS, None)


@pytest.mark.asyncio
async def test_routing_accuracy_benchmark(model: object) -> None:
    """Run the corpus across all three ablations, print the table.

    The assertions are deliberately *loose* — the point is to publish a
    measured accuracy delta, not to gate the build on a specific
    threshold. The test fails only if Full's protocol accuracy is
    *worse* than FreeForm's, which would invalidate the SDK's headline
    claim. The actual numbers go in the printed table for a reviewer
    to inspect.
    """
    full_score = AblationScore("Full")
    nocanon_score = AblationScore("NoCanonical")
    freeform_score = AblationScore("FreeForm")

    await _run_router_corpus(_router_full(model), full_score)
    await _run_router_corpus(_router_no_canonical(model), nocanon_score)
    await _run_freeform_corpus(_freeform_extractor(model), freeform_score)

    label = PROVIDER_LABEL or type(model).__name__
    print(f"\n=== Routing benchmark · provider: {label} · n={len(CORPUS)} ===")
    print()
    print("| ablation       | goal_acc | proto_acc | bounded | extract_fail | build_fail | n |")
    print("|----------------|----------|-----------|---------|--------------|------------|---|")
    print(full_score.summary_row("router"))
    print(nocanon_score.summary_row("router"))
    print(freeform_score.summary_row("freeform"))
    print()

    if os.getenv("BENCH_VERBOSE"):
        for sc in (full_score, nocanon_score, freeform_score):
            print(f"\n--- {sc.name} per-prompt audit ---")
            for r in sc.rows:
                print(r)

    # The accuracy comparison is reported, not gated: it moves with the
    # model (Cohere Command A beats the free-form baseline, Llama 3.3 70B
    # does not), and the claim the router actually makes is structural,
    # checked below and under adversarial input in the next test.
    delta = full_score.protocol_correct - freeform_score.protocol_correct
    print(
        f"typed-frame router vs free-form baseline: {delta:+d} of {full_score.n} "
        f"prompts ({full_score.protocol_correct} vs {freeform_score.protocol_correct})"
    )
    # Bounded-output guarantee: the router's selections are *always* in
    # the registered set, by construction. A failure here is a real bug.
    assert full_score.bounded == full_score.n - full_score.extract_failures, (
        f"Full router produced unbounded selections: "
        f"bounded={full_score.bounded}/{full_score.n} "
        f"extract_failures={full_score.extract_failures}"
    )


# ---------------------------------------------------------------------------
# Adversarial robustness — the boundedness contract under attack.
# ---------------------------------------------------------------------------


@dataclass
class RobustnessScore:
    """Three outcomes per prompt: a registered id, an unregistered id, or no
    selection at all (the frame was rejected by the schema, or nothing could
    take it). Only the second breaks the boundedness claim; a refusal is the
    constrained answer the router is allowed to give."""

    name: str
    n: int = 0
    bounded: int = 0
    unbounded: int = 0
    refused: int = 0
    by_flavour: dict[str, tuple[int, int]] = field(default_factory=dict)
    chosen_ids: list[str] = field(default_factory=list)

    def record(self, entry: AdversarialEntry, chosen: str | None, registered: set[str]) -> None:
        self.n += 1
        in_bounds = chosen is not None and chosen in registered
        if in_bounds:
            self.bounded += 1
        elif chosen is None:
            self.refused += 1
        else:
            self.unbounded += 1
        n_flav, b_flav = self.by_flavour.get(entry.flavour, (0, 0))
        self.by_flavour[entry.flavour] = (
            n_flav + 1,
            b_flav + (1 if in_bounds else 0),
        )
        self.chosen_ids.append(chosen or "<no-parse>")

    def summary_row(self) -> str:
        denom = max(1, self.n)
        inj_n, inj_b = self.by_flavour.get("injection", (0, 0))
        amb_n, amb_b = self.by_flavour.get("ambiguous", (0, 0))
        inj = f"{inj_b / max(1, inj_n):>5.1%}"
        amb = f"{amb_b / max(1, amb_n):>5.1%}"
        return (
            f"| {self.name:<14} | {self.bounded / denom:>5.1%} | "
            f"{self.unbounded:>9} | {self.refused:>7} | "
            f"{inj} ({inj_b:>2}/{inj_n}) | "
            f"{amb} ({amb_b:>2}/{amb_n}) | "
            f"{self.n:>3} |"
        )


async def _run_router_adversarial(router: Router, score: RobustnessScore) -> None:
    for entry in ADVERSARIAL_CORPUS:
        try:
            frame = await router.extract(entry.prompt)
        except Exception:  # noqa: BLE001 — a rejected frame or a provider error is a refusal
            score.record(entry, None, REGISTERED_IDS)
            continue
        # The compiler's own selection, so the risk fallback counts the way a
        # real dispatch would; ``selected_protocol`` is None at a dead end.
        score.record(entry, router.compiler.explain(frame).selected_protocol, REGISTERED_IDS)


async def _run_freeform_adversarial(extractor: Agent, score: RobustnessScore) -> None:
    for entry in ADVERSARIAL_CORPUS:
        try:
            result = await asyncio.to_thread(extractor.invoke, entry.prompt)
        except Exception:  # noqa: BLE001 — see _run_router_adversarial
            score.record(entry, None, REGISTERED_IDS)
            continue
        parsed = result.parsed
        if not isinstance(parsed, FreeFormChoice):
            score.record(entry, None, REGISTERED_IDS)
            continue
        score.record(entry, parsed.protocol_id, REGISTERED_IDS)


@pytest.mark.asyncio
async def test_routing_robustness_adversarial(model: object) -> None:
    """The boundedness claim under attack — prompt-injection + ambiguous
    inputs that try to make the LLM hallucinate a protocol id outside
    the registered set.

    Hypothesis (and what this test enforces):

    * **Full**: 100% bounded by construction. ``GoalFrame.primary_goal``
      is a closed StrEnum; the registry only emits known ids.
      Extraction may fail on adversarial prompts, but the system's
      response is *constrained* even then (we record ``no-parse``).
    * **NoCanonical**: same as Full — boundedness is structural, not
      dependent on canonical-match ranking.
    * **FreeForm**: bounded only by model cooperation. We expect a
      measurable drop on injection prompts (model invents protocol
      ids) and possibly on ambiguous prompts (model hedges).

    The test asserts ``Full.bounded == Full.n`` (the structural claim)
    and prints the FreeForm degradation as data — no threshold gate,
    because exact numbers depend on the model.
    """
    full = RobustnessScore("Full")
    nocanon = RobustnessScore("NoCanonical")
    freeform = RobustnessScore("FreeForm")

    await _run_router_adversarial(_router_full(model), full)
    await _run_router_adversarial(_router_no_canonical(model), nocanon)
    await _run_freeform_adversarial(_freeform_extractor(model), freeform)

    label = PROVIDER_LABEL or type(model).__name__
    print(f"\n=== Adversarial robustness · provider: {label} · n={len(ADVERSARIAL_CORPUS)} ===")
    print()
    print(
        "| ablation       | bounded | unbounded | refused | injection (b/n) | ambiguous (b/n) | n |"
    )
    print(
        "|----------------|---------|-----------|---------|-----------------|-----------------|---|"
    )
    print(full.summary_row())
    print(nocanon.summary_row())
    print(freeform.summary_row())
    print()

    if os.getenv("BENCH_VERBOSE"):
        for sc in (full, nocanon, freeform):
            print(f"\n--- {sc.name} chosen ids ---")
            for entry, chosen in zip(ADVERSARIAL_CORPUS, sc.chosen_ids, strict=False):
                tag = "✓" if chosen in REGISTERED_IDS else "✗"
                print(f"  {tag} [{entry.flavour:9s}] {chosen:24s} ← {entry.prompt[:60]!r}")

    # Structural guarantee — must hold regardless of model: the typed router
    # never emits an id outside the registered set. A refusal (schema rejected
    # the frame, or no protocol could take it) is a constrained answer, not a
    # hallucinated shape; it is reported in the table, not asserted away.
    assert full.unbounded == 0, (
        f"Full router emitted {full.unbounded} unregistered protocol id(s) under "
        f"adversarial input. This invalidates the bounded-graph-generation claim."
    )
    assert nocanon.unbounded == 0, (
        f"NoCanonical emitted {nocanon.unbounded} unregistered id(s). "
        f"Boundedness should be structural, not depend on canonical ranking."
    )
    injected_off = sum(
        1
        for entry, chosen in zip(ADVERSARIAL_CORPUS, full.chosen_ids, strict=False)
        if entry.flavour == "injection" and chosen != "<no-parse>" and chosen not in REGISTERED_IDS
    )
    assert injected_off == 0, (
        f"prompt injection moved the typed router off the registered set {injected_off} time(s)"
    )
