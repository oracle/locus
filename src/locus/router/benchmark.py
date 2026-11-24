# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Selection-only harness for the router, on the evaluation primitives.

The router's accuracy question is "which protocol would it pick?", and
answering it does not need the shape built or run. This module adapts the
selection step to :class:`~locus.evaluation.EvalRunner`'s agent contract
(``await target.arun(prompt)`` returning something with a ``message``), so
the corpus is a file of :class:`~locus.evaluation.EvalCase` objects, the
rates split per tag with ``EvalReport.by_tag``, two runs diff with
:func:`~locus.evaluation.diff_reports`, and a build gates on a
:class:`~locus.evaluation.GateResult`.

Three targets select over the same corpus:

* :class:`TypedSelectionTarget`: the typed router. A
  :class:`~locus.router.RequestClassifier` (by default the extractor agent
  filling a :class:`~locus.router.GoalFrame`) frames the request and
  :meth:`CognitiveCompiler.explain <locus.router.CognitiveCompiler.explain>`
  resolves the frame the way ``compile`` would: short-circuit, three gates,
  ranking, risk fallback. Nothing is built.
* The same target with another classifier. ``typed_target(model,
  classifier=...)`` is how a trained classifier plugs in: anything with
  ``async classify(text) -> GoalFrame`` sits in the router's front seam and
  the rest of the path is unchanged.
* :class:`FreeFormSelectionTarget`: the baseline, "ask the model directly".
  The model names a protocol id in a flat schema with no frame and no
  registry, so it can name one that does not exist.

Every target answers with one of three strings: a registered protocol id,
:data:`UNBOUNDED` for an id outside the registry, or :data:`REFUSED` when
nothing was selected (the frame was rejected, no protocol could take it, or
the provider failed). A labelled case expects its protocol id; an
adversarial case expects the answer not to be :data:`UNBOUNDED`, so a
refusal is the constrained answer it is allowed to be. The raw pick, the
frame and the reason stay on ``target.records`` for a per-prompt audit.

The corpus lives in ``benchmarks/router/cases.json``; :func:`load_router_corpus`
reads it and checks the shape the tags and expectations must have.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

from locus.evaluation.datasets import load_cases
from locus.evaluation.framework import EvalCase, EvalReport
from locus.evaluation.report import EvalGate, GateResult, diff_reports
from locus.router.capability import CapabilityIndex
from locus.router.classifier import FrameExtractionError, RequestClassifier
from locus.router.compiler import CognitiveCompiler
from locus.router.goal_frame import GoalFrame
from locus.router.policy import PolicyGate
from locus.router.protocol import Protocol, ProtocolRegistry, builtin_protocols
from locus.router.runtime import Router


if TYPE_CHECKING:
    from locus.agent.agent import Agent


__all__ = [
    "BENCHMARK_CAPABILITIES",
    "DIFFICULTIES",
    "DIFFICULTY_TAG_PREFIX",
    "GOAL_TAG_PREFIX",
    "REFUSED",
    "SHAPE_TAG_PREFIX",
    "TAG_ADVERSARIAL",
    "TAG_AMBIGUOUS",
    "TAG_INJECTION",
    "TAG_LABELLED",
    "TAG_TRAP",
    "TRANSIENT_ERROR_NAMES",
    "UNBOUNDED",
    "FreeFormChoice",
    "FreeFormSelectionTarget",
    "SelectionRecord",
    "SelectionResult",
    "SelectionTarget",
    "TypedSelectionTarget",
    "benchmark_capabilities",
    "benchmark_compiler",
    "frame_extractor",
    "freeform_picker",
    "freeform_target",
    "is_transient_error",
    "load_router_corpus",
    "mean_pass_rate",
    "pass_rate",
    "regressions",
    "router_gate",
    "tag_table",
    "typed_target",
]


#: The answer for a pick outside the registered set: the boundedness miss.
UNBOUNDED = "unbounded"
#: The answer when nothing was selected: a rejected frame, a dead end, a provider error.
REFUSED = "refused"

#: Names of the exceptions a provider client raises when a call never reached a
#: verdict: a dropped connection, a timeout, a rate limit, a failure on the
#: server. A target retries these (see :func:`is_transient_error`); a wrong
#: answer is never retried.
TRANSIENT_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
    }
)

_T = TypeVar("_T")


def is_transient_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider error worth one more attempt.

    True for the built-in connection and timeout errors, for an exception
    carrying an HTTP status of 408, 429 or 5xx (``status_code`` or
    ``status``), and for any class in :data:`TRANSIENT_ERROR_NAMES` by name,
    so the common HTTP and provider clients are covered without importing
    them. Anything else, a rejected frame or a schema the reply did not fit
    included, is an answer and is not retried.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and (status in (408, 429) or status >= 500):
        return True
    return any(cls.__name__ in TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__)


#: A case with an expected protocol; its ``shape:<id>`` tag names it.
TAG_LABELLED = "labelled"
#: A case with no right protocol, scored on boundedness alone.
TAG_ADVERSARIAL = "adversarial"
#: An adversarial case that tells the model to leave the protocol list.
TAG_INJECTION = "injection"
#: An adversarial case nothing in the catalogue fits.
TAG_AMBIGUOUS = "ambiguous"
#: A labelled case whose verb points at an expensive shape while the work is one step.
TAG_TRAP = "trap"
SHAPE_TAG_PREFIX = "shape:"
GOAL_TAG_PREFIX = "goal:"
#: The difficulty a labelled case's author intends: ``difficulty:low``,
#: ``difficulty:medium`` or ``difficulty:high``.
DIFFICULTY_TAG_PREFIX = "difficulty:"
#: The difficulty levels, in the order the table prints them.
DIFFICULTIES: tuple[str, ...] = ("low", "medium", "high")

#: The capability ids the corpus names; :func:`benchmark_capabilities` provides them.
BENCHMARK_CAPABILITIES: tuple[str, ...] = ("kb_search", "metric_probe", "alert_list")

_EXTRACTOR_PROMPT = (
    "You are a goal-frame extractor for the cognitive router. Given the "
    "user's request, fill the provided schema.\n\n"
    "Rules:\n"
    "1. Pick the single best primary_goal that matches the verb in the "
    "request.\n"
    "2. Risk: LOW for read-only/info; MEDIUM for build/modify/plan tasks; "
    "HIGH only for irreversible production operations (deletes, "
    "migrations actually being executed). Writing a plan is NOT high "
    "risk — execution of the plan is.\n"
    "3. Complexity: LOW for one-step; MEDIUM for multi-step; HIGH for "
    "fan-out across multiple specialists.\n"
    "4. required_capabilities must come from this set; do not invent "
    "ids: {capabilities}.\n"
    "If the request needs a tool that isn't in this list, leave "
    "required_capabilities empty."
)

_FREEFORM_PROMPT = (
    "You are a router. Pick the single best protocol id for the user's "
    "request. Available protocols:\n{protocols}\nReply with the chosen id."
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_router_corpus(
    path: str | Path, *, protocols: Iterable[Protocol] | None = None
) -> list[EvalCase]:
    """Read the router corpus and check every case has the shape the harness grades.

    A ``labelled`` case carries exactly one ``shape:<id>`` tag naming a
    registered protocol, and ``expected_output_contains`` is that id alone,
    so a reading passes only when the target answered with the protocol
    the label names. It also carries exactly one ``difficulty:<level>`` tag,
    the difficulty its author intends, so a report splits by difficulty. A
    ``trap`` is labelled with a low-cost shape and low difficulty: its verb
    points somewhere expensive, the work does not. An ``adversarial`` case
    carries ``injection`` or ``ambiguous`` and expects only that the answer
    is not :data:`UNBOUNDED`. Case names are unique, because reports diff by
    them.

    Args:
        path: The cases file, in any shape :func:`~locus.evaluation.load_cases` reads.
        protocols: The registry the labels must name; the eight built-in
            protocols when not given.

    Raises:
        ValueError: A case that does not fit, named.
    """
    cases = load_cases(path)
    catalogue = {p.id: p for p in (protocols if protocols is not None else builtin_protocols())}
    seen: set[str] = set()
    for case in cases:
        where = f"{Path(path).name}: case {case.name!r}"
        if case.name in seen:
            raise ValueError(f"{where} is named twice")
        seen.add(case.name)
        tags = set(case.tags)
        if TAG_LABELLED in tags:
            shapes = [
                t[len(SHAPE_TAG_PREFIX) :] for t in case.tags if t.startswith(SHAPE_TAG_PREFIX)
            ]
            if len(shapes) != 1:
                raise ValueError(f"{where} needs exactly one {SHAPE_TAG_PREFIX}<id> tag")
            shape = catalogue.get(shapes[0])
            if shape is None:
                raise ValueError(f"{where} names unregistered protocol {shapes[0]!r}")
            if case.expected_output_contains != [shape.id]:
                raise ValueError(f"{where} must expect exactly [{shape.id!r}]")
            levels = [
                t[len(DIFFICULTY_TAG_PREFIX) :]
                for t in case.tags
                if t.startswith(DIFFICULTY_TAG_PREFIX)
            ]
            if len(levels) != 1 or levels[0] not in DIFFICULTIES:
                raise ValueError(
                    f"{where} needs exactly one {DIFFICULTY_TAG_PREFIX}<level> tag, "
                    f"one of {', '.join(DIFFICULTIES)}"
                )
            if TAG_TRAP in tags and shape.cost != "low":
                raise ValueError(f"{where} is a trap, so its shape must be low-cost")
            if TAG_TRAP in tags and levels[0] != "low":
                raise ValueError(f"{where} is a trap, so its difficulty must be low")
        elif TAG_ADVERSARIAL in tags:
            flavours = tags & {TAG_INJECTION, TAG_AMBIGUOUS}
            if len(flavours) != 1:
                raise ValueError(f"{where} needs one of {TAG_INJECTION!r} or {TAG_AMBIGUOUS!r}")
            if case.expected_output_contains or case.expected_output_not_contains != [UNBOUNDED]:
                raise ValueError(f"{where} must expect only not {UNBOUNDED!r}")
        else:
            raise ValueError(f"{where} is neither {TAG_LABELLED!r} nor {TAG_ADVERSARIAL!r}")
    return cases


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRecord:
    """One selection a target made, kept for the per-prompt audit."""

    prompt: str
    output: str
    picked: str | None = None
    frame: GoalFrame | None = None
    detail: str = ""
    attempts: int = 1


class SelectionResult(BaseModel):
    """What a selection target returns; ``message`` is the string the runner grades."""

    message: str
    iterations: int = 1
    picked: str | None = None
    frame: GoalFrame | None = None
    detail: str = ""
    attempts: int = 1

    model_config = {"frozen": True}


class SelectionTarget:
    """The shared half of the targets: normalise a pick, remember it.

    Subclasses implement :meth:`arun`; :meth:`run_sync` exists for
    :meth:`EvalRunner.run <locus.evaluation.EvalRunner.run>`.
    """

    def __init__(
        self, registered: Iterable[str], *, attempts: int = 3, backoff_s: float = 1.0
    ) -> None:
        self.registered: frozenset[str] = frozenset(registered)
        self.records: list[SelectionRecord] = []
        #: Calls made for one prompt before a transient provider error counts
        #: as :data:`REFUSED`; a wrong answer is never retried.
        self.attempts = max(1, attempts)
        #: The wait before the second attempt, doubled on each one after.
        self.backoff_s = backoff_s

    async def _attempt(self, call: Callable[[], Awaitable[_T]]) -> tuple[_T, int]:
        """Await ``call()``, again after a transient provider error; the result and the attempts used."""
        for attempt in range(1, self.attempts + 1):
            try:
                return await call(), attempt
            except Exception as exc:
                if attempt == self.attempts or not is_transient_error(exc):
                    raise
                await asyncio.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _failed(self, prompt: str, exc: Exception) -> SelectionResult:
        """The refusal for a call that raised: a transient error names the attempts it used."""
        if is_transient_error(exc):
            detail = f"{type(exc).__name__} after {self.attempts} attempt(s): {exc}"
            return self._finish(prompt, None, detail=detail, attempts=self.attempts)
        return self._finish(prompt, None, detail=f"{type(exc).__name__}: {exc}")

    def _finish(
        self,
        prompt: str,
        picked: str | None,
        *,
        frame: GoalFrame | None = None,
        detail: str = "",
        attempts: int = 1,
    ) -> SelectionResult:
        if picked is None:
            output = REFUSED
        elif picked in self.registered:
            output = picked
        else:
            output = UNBOUNDED
        self.records.append(
            SelectionRecord(
                prompt=prompt,
                output=output,
                picked=picked,
                frame=frame,
                detail=detail,
                attempts=attempts,
            )
        )
        return SelectionResult(
            message=output, picked=picked, frame=frame, detail=detail, attempts=attempts
        )

    async def arun(self, prompt: str) -> SelectionResult:
        """Select for ``prompt``; the subclass decides how."""
        raise NotImplementedError

    def run_sync(self, prompt: str) -> SelectionResult:
        """The synchronous form of :meth:`arun`, for a runner without a loop."""
        return asyncio.run(self.arun(prompt))


class TypedSelectionTarget(SelectionTarget):
    """Select through the typed router; build nothing.

    ``router.extract`` runs the classifier, and ``router.compiler.explain``
    resolves the frame exactly as ``compile`` would, with no model call and
    no builder: the ``direct_response`` short-circuit, the three gates, the
    ranker and the risk fallback. An opt-in
    :class:`~locus.router.ProtocolPicker` is not consulted, as ``explain``
    documents, so the measurement is of the deterministic path. A rejected
    frame, a dead end and a provider error are all :data:`REFUSED`; a
    transient provider error (:func:`is_transient_error`) is retried up to
    ``attempts`` times first, so a dropped connection is not scored as a
    routing decision.
    """

    def __init__(self, router: Router, *, attempts: int = 3, backoff_s: float = 1.0) -> None:
        super().__init__(
            (p.id for p in router.compiler.protocols.all()), attempts=attempts, backoff_s=backoff_s
        )
        self.router = router

    async def arun(self, prompt: str) -> SelectionResult:
        try:
            frame, attempts = await self._attempt(lambda: self.router.extract(prompt))
        except FrameExtractionError as exc:
            return self._finish(prompt, None, detail=f"frame rejected: {exc}")
        except Exception as exc:  # noqa: BLE001 — a provider error on one prompt is a data point
            return self._failed(prompt, exc)
        explanation = self.router.compiler.explain(frame, goal=prompt)
        return self._finish(
            prompt,
            explanation.selected_protocol,
            frame=frame,
            detail=explanation.method or "no protocol could take the frame",
            attempts=attempts,
        )


class FreeFormChoice(BaseModel):
    """The baseline's flat output: any string, so an invented id is measurable."""

    protocol_id: str = Field(
        ...,
        description=(
            "The id of the protocol that best fits the user's request, "
            "from the list in the instructions."
        ),
    )
    reasoning: str = Field(default="", description="Why you picked it.")


class FreeFormSelectionTarget(SelectionTarget):
    """The baseline: the model names the protocol id directly.

    ``picker`` is an :class:`~locus.Agent` with ``output_schema=FreeFormChoice``
    (see :func:`freeform_picker`). Its answer is checked against the
    registered ids: a match is the id, anything else is :data:`UNBOUNDED`,
    and a reply the schema rejected or a provider error is :data:`REFUSED`.
    A transient provider error is retried the way the typed target retries it.
    """

    def __init__(
        self,
        picker: Agent,
        *,
        registered: Iterable[str] | None = None,
        attempts: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        super().__init__(
            registered if registered is not None else (p.id for p in builtin_protocols()),
            attempts=attempts,
            backoff_s=backoff_s,
        )
        self.picker = picker

    async def arun(self, prompt: str) -> SelectionResult:
        try:
            result, attempts = await self._attempt(lambda: self.picker.arun(prompt))
        except Exception as exc:  # noqa: BLE001 — see TypedSelectionTarget.arun
            return self._failed(prompt, exc)
        parsed = result.parsed
        if not isinstance(parsed, FreeFormChoice):
            return self._finish(
                prompt, None, detail=f"no parse: {result.parse_error}", attempts=attempts
            )
        return self._finish(
            prompt, parsed.protocol_id.strip(), detail=parsed.reasoning, attempts=attempts
        )


# ---------------------------------------------------------------------------
# Builders: the fixture every run of the benchmark shares
# ---------------------------------------------------------------------------


def benchmark_capabilities() -> CapabilityIndex:
    """Three inert probes under the capability ids the corpus names.

    The tools never run: selection stops at the frame. They exist so the
    capability gate sees the ids the labelled prompts ask for.
    """
    from locus.tools.decorator import tool  # noqa: PLC0415 — keep the tools package off this import
    from locus.tools.registry import create_registry  # noqa: PLC0415

    @tool
    def kb_search(query: str) -> str:
        """Search the knowledge base for a topic and return a short summary."""
        return f"kb hit for {query!r}"

    @tool
    def get_metric(name: str) -> str:
        """Return the latest value of a named metric."""
        return f"{name}=probe"

    @tool
    def list_alerts(window_minutes: int = 30) -> str:
        """List recent alerts in the given window."""
        return f"alerts in {window_minutes}min: none"

    index = CapabilityIndex(create_registry(kb_search, get_metric, list_alerts))
    index.annotate("kb_search", tool_name="kb_search", description="KB lookup.", domain="research")
    index.annotate(
        "metric_probe", tool_name="get_metric", description="Latest metric.", domain="observability"
    )
    index.annotate(
        "alert_list", tool_name="list_alerts", description="Recent alerts.", domain="observability"
    )
    return index


def frame_extractor(model: Any, *, capability_ids: Sequence[str] = BENCHMARK_CAPABILITIES) -> Agent:
    """The typed router's default classifier: an agent that fills a :class:`GoalFrame`."""
    from locus.agent.agent import Agent  # noqa: PLC0415 — as the picker module does

    return Agent(
        model=model,
        system_prompt=_EXTRACTOR_PROMPT.format(capabilities=", ".join(capability_ids)),
        output_schema=GoalFrame,
    )


def freeform_picker(model: Any, *, protocols: Iterable[Protocol] | None = None) -> Agent:
    """The baseline's agent: it is told the catalogue and asked for an id.

    Each protocol is listed with its description and the goals it is
    canonical for, which is what the typed path knows about it too.
    """
    from locus.agent.agent import Agent  # noqa: PLC0415

    lines = []
    for p in protocols if protocols is not None else builtin_protocols():
        canonical = ", ".join(t.value for t in p.primary_for)
        tail = f" Canonical for: {canonical}." if canonical else ""
        lines.append(f"  - {p.id}: {p.description}{tail}")
    return Agent(
        model=model,
        system_prompt=_FREEFORM_PROMPT.format(protocols="\n".join(lines)),
        output_schema=FreeFormChoice,
    )


def benchmark_compiler(
    model: Any,
    *,
    protocols: Iterable[Protocol] | None = None,
    capabilities: CapabilityIndex | None = None,
    policy: PolicyGate | None = None,
    **compiler_kwargs: Any,
) -> CognitiveCompiler:
    """The compiler every typed target selects with: built-ins, the probes, a default gate."""
    registry = ProtocolRegistry()
    registry.register_many(list(protocols if protocols is not None else builtin_protocols()))
    return CognitiveCompiler(
        protocols=registry,
        capabilities=capabilities if capabilities is not None else benchmark_capabilities(),
        policy=policy if policy is not None else PolicyGate(),
        model=model,
        **compiler_kwargs,
    )


def typed_target(
    model: Any,
    *,
    classifier: RequestClassifier | None = None,
    compiler: CognitiveCompiler | None = None,
) -> TypedSelectionTarget:
    """A typed target on ``model``.

    Without ``classifier`` the router frames requests with
    :func:`frame_extractor` on ``model``; with one, that classifier fills the
    frame and ``model`` is only what the compiler would hand to builders.
    This is the seam a trained classifier arrives through: implement
    ``async classify(text) -> GoalFrame`` and pass the instance.
    """
    built = compiler if compiler is not None else benchmark_compiler(model)
    if classifier is not None:
        return TypedSelectionTarget(Router(classifier=classifier, compiler=built))
    return TypedSelectionTarget(Router(extractor=frame_extractor(model), compiler=built))


def freeform_target(
    model: Any, *, protocols: Iterable[Protocol] | None = None
) -> FreeFormSelectionTarget:
    """The baseline target on ``model``."""
    catalogue = list(protocols if protocols is not None else builtin_protocols())
    return FreeFormSelectionTarget(
        freeform_picker(model, protocols=catalogue), registered=(p.id for p in catalogue)
    )


# ---------------------------------------------------------------------------
# Reading reports: rates per tag, the gate, the table
# ---------------------------------------------------------------------------


def pass_rate(report: EvalReport, tag: str | None = None) -> float | None:
    """The share of graded cases that passed, under ``tag`` or over the whole report.

    With repeats a case passes only when every reading did, so this is the
    strict rate. ``None`` when no case was graded under ``tag``.
    """
    if tag is None:
        graded = report.total_cases - report.not_graded
        return report.passed / graded if graded else None
    summary = report.by_tag.get(tag)
    if summary is None:
        return None
    graded = summary.total - summary.not_graded
    return summary.passed / graded if graded else None


def mean_pass_rate(
    report: EvalReport, cases: Sequence[EvalCase], tag: str | None = None
) -> float | None:
    """The mean per-reading pass rate under ``tag``: how often a reading was right.

    The strict rate charges a case fully for one wrong reading in three; this
    one counts each reading, which is the accuracy figure a single run reports.
    """
    wanted = {c.name for c in cases if tag is None or tag in c.tags}
    rates = [
        r.pass_rate if r.pass_rate is not None else (1.0 if r.passed else 0.0)
        for r in report.results
        if r.case_name in wanted and not r.not_graded_reason
    ]
    return sum(rates) / len(rates) if rates else None


def regressions(candidate: EvalReport, baseline: EvalReport) -> list[str]:
    """The cases ``baseline`` passed that ``candidate`` fails, by name.

    What ``per_case`` in :func:`router_gate` holds to zero; printed either
    way, since it is the list a classifier has to clear. Empty when the two
    reports are of different suites (the diff refuses them).
    """
    try:
        return list(diff_reports(baseline, candidate).regressed)
    except ValueError:
        return []


def router_gate(
    candidate: EvalReport, baseline: EvalReport, *, per_case: bool = True
) -> GateResult:
    """The router's gate: a typed report against the free-form baseline.

    Passes when the candidate's pass rate is at least the baseline's on the
    whole corpus and on the ``trap`` cases, when every ``injection`` case is
    bounded, when nothing is left not graded, and, with ``per_case``, when
    no case that passed on the baseline fails on the candidate
    (:class:`~locus.evaluation.EvalGate` with ``no_regressions``). The diff
    behind that check refuses two reports of different suites whether or
    not ``per_case`` is set. Every reason it fails is reported.

    ``per_case`` is the parity a trained classifier is held to. The default
    extractor is not one, and a one-step prompt it frames LOW goes to
    ``direct_response`` whatever the label says, so a run of the default
    router gates on the floors and reports its :func:`regressions`.
    """
    floor = pass_rate(baseline) or 0.0
    gate = EvalGate(min_pass_rate=floor, max_not_graded=0, no_regressions=per_case)
    reasons = list(gate.check(candidate, baseline=baseline).reasons)
    base_trap, cand_trap = pass_rate(baseline, TAG_TRAP), pass_rate(candidate, TAG_TRAP)
    if base_trap is not None and cand_trap is not None and cand_trap < base_trap:
        reasons.append(f"trap pass rate {cand_trap:.2f} below the baseline's {base_trap:.2f}")
    injection = pass_rate(candidate, TAG_INJECTION)
    if injection is not None and injection < 1.0:
        reasons.append(f"injection bounded {injection:.0%}, must be 100%")
    return GateResult(passed=not reasons, reasons=reasons)


def tag_table(reports: Mapping[str, EvalReport], cases: Sequence[EvalCase]) -> str:
    """A Markdown table of pass rates per tag, one column per target.

    Each cell is ``strict (mean)``: the share of cases every reading got
    right, and the share of readings that were right. ``all`` is the whole
    corpus; the other rows are the corpus's tags in a fixed order, then the
    ``difficulty:``, ``shape:`` and ``goal:`` tags.
    """
    names = list(reports)
    fixed = [TAG_LABELLED, TAG_TRAP, TAG_ADVERSARIAL, TAG_INJECTION, TAG_AMBIGUOUS]
    present = {t for c in cases for t in c.tags}
    rows = [t for t in fixed if t in present]
    rows += [t for t in (DIFFICULTY_TAG_PREFIX + level for level in DIFFICULTIES) if t in present]
    rows += sorted(t for t in present if t.startswith(SHAPE_TAG_PREFIX))
    rows += sorted(t for t in present if t.startswith(GOAL_TAG_PREFIX))

    def cell(report: EvalReport, tag: str | None) -> str:
        strict, mean = pass_rate(report, tag), mean_pass_rate(report, cases, tag)
        if strict is None or mean is None:
            return "-"
        return f"{strict:.1%} ({mean:.1%})"

    def count(tag: str | None) -> int:
        return len(cases) if tag is None else sum(1 for c in cases if tag in c.tags)

    lines = [
        "| tag | n | " + " | ".join(names) + " |",
        "|---|---:|" + "---|" * len(names),
        f"| all | {count(None)} | " + " | ".join(cell(reports[n], None) for n in names) + " |",
    ]
    lines.extend(
        f"| {tag} | {count(tag)} | " + " | ".join(cell(reports[n], tag) for n in names) + " |"
        for tag in rows
    )
    return "\n".join(lines)
