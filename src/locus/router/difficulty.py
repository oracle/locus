# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Difficulty measured from outcomes, for the router corpus.

A labelled case of the router corpus carries the difficulty its author
intends (its ``difficulty:<level>`` tag). This module measures it instead:
each prompt is answered in one reply, with no tools, by models of increasing
strength (the *tiers*), and a judge grades every answer against a rubric.
The level is read from which tier's answer passes:

- ``low``: the cheapest tier's single reply does the work;
- ``medium``: only a stronger tier's single reply does;
- ``high``: no tier's single reply does, so the request needs more than one
  reply can give (tools, several steps, a check, an approval).

The parts are the evaluation primitives: :func:`answer_cases` turns the
corpus into judged cases, :func:`direct_agent` is the single-reply agent,
an :class:`~locus.evaluation.EvalRunner` with a judge produces one report
per tier, and :func:`measure_difficulty` reads the reports. A measurement is
per model family: the same prompt can be ``low`` on one tier ladder and
``medium`` on another.

Example:
    >>> from locus.evaluation import EvalRunner, LLMJudge
    >>> from locus.router.benchmark import load_router_corpus
    >>> from locus.router.difficulty import (
    ...     answer_cases,
    ...     direct_agent,
    ...     measure_difficulty,
    ... )
    >>> cases = answer_cases(load_router_corpus("benchmarks/router/cases.json"))
    >>> judge = LLMJudge(judge_model)
    >>> reports = {
    ...     name: await EvalRunner(direct_agent(model), judge=judge, repeats=3).arun(
    ...         cases
    ...     )
    ...     for name, model in (("small", small_model), ("large", large_model))
    ... }
    >>> measured = measure_difficulty(cases, reports)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from locus.evaluation import EvalCase, EvalReport
from locus.router.benchmark import (
    DIFFICULTIES,
    DIFFICULTY_TAG_PREFIX,
    TAG_LABELLED,
    TAG_TRAP,
    benchmark_compiler,
)
from locus.router.capability import CapabilityIndex
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.policy import PolicyDeniedError
from locus.router.runtime import Router


if TYPE_CHECKING:
    from locus.agent.agent import Agent

__all__ = [
    "ANSWER_RUBRIC",
    "NOT_MEASURED",
    "WORLD_ALERTS",
    "WORLD_ARTICLES",
    "WORLD_METRICS",
    "IntendedFrameClassifier",
    "MeasuredDifficulty",
    "ShapeAgent",
    "answer_cases",
    "confusion_table",
    "direct_agent",
    "disagreements",
    "intended_difficulty",
    "intended_frame",
    "measure_difficulty",
    "shape_rubric",
    "working_capabilities",
    "world_summary",
]

#: The pass criterion a single reply is judged against. A case may carry its
#: own under ``metadata["answer_rubric"]``.
ANSWER_RUBRIC = (
    "The response, on its own, completes what the request asks for. It passes only if "
    "the work is done in the response: correct, complete, and usable as given. It fails "
    "if it invents data it could not have seen (metric values, alerts, documents, the "
    "state of a system), if it describes or outlines work the request asks to be "
    "performed, if it claims an action on a real system that a text reply cannot take, "
    "if it defers to tools, people or later steps for the substance of the request, or "
    "if it is wrong or materially incomplete."
)

#: The level of a case no decision could be read for (a tier was not graded).
NOT_MEASURED = "not_measured"

_DIRECT_PROMPT = (
    "Answer the request in one reply. You have no tools and cannot look anything up, "
    "run anything or change any system. Do the work in the reply itself when a reply "
    "can do it. Never invent data you cannot see; when the request needs something a "
    "single reply cannot provide, say plainly what is missing."
)


def intended_difficulty(case: EvalCase) -> str | None:
    """The level a case's ``difficulty:<level>`` tag names, or ``None`` without one."""
    for tag in case.tags:
        if tag.startswith(DIFFICULTY_TAG_PREFIX):
            return tag[len(DIFFICULTY_TAG_PREFIX) :]
    return None


def answer_cases(cases: Sequence[EvalCase], *, rubric: str = ANSWER_RUBRIC) -> list[EvalCase]:
    """The labelled cases of the router corpus as judged answer cases.

    Each keeps its name, prompt, tags and metadata and swaps the protocol
    expectation for a rubric: ``metadata["answer_rubric"]`` when the case
    has one, else ``rubric``. Adversarial cases have no right answer to
    grade and are left out.
    """
    out: list[EvalCase] = []
    for case in cases:
        if TAG_LABELLED not in case.tags:
            continue
        own = case.metadata.get("answer_rubric")
        out.append(
            EvalCase(
                name=case.name,
                prompt=case.prompt,
                tags=list(case.tags),
                metadata=dict(case.metadata),
                rubric=own if isinstance(own, str) and own else rubric,
            )
        )
    return out


def direct_agent(model: Any) -> Agent:
    """The single-reply agent a tier answers with: ``model``, no tools, one iteration."""
    from locus.agent.agent import Agent  # noqa: PLC0415 — keep the agent package off this import

    return Agent(model=model, system_prompt=_DIRECT_PROMPT, max_iterations=1)


# ---------------------------------------------------------------------------
# The structured shape as a tier: the labelled protocol, run with working tools
# ---------------------------------------------------------------------------

#: The metrics the working ``get_metric`` tool reads: name to latest reading.
WORLD_METRICS: dict[str, str] = {
    "cpu": "91% on checkout (up from 55% an hour ago); 38% on catalog, payments and search",
    "latency_p99": "1840 ms on checkout, 2100 ms on search (both about 400 ms yesterday); "
    "310 ms on payments",
    "errors_5xx": "4.2% on login and 3.1% on payments over the last 30 minutes; "
    "below 0.1% on the other services",
    "queue_depth": "182000 messages on the ingest queue, growing by 9000 a minute",
    "throughput": "1200 messages a second on ingest, down from 5400 an hour ago",
    "memory_rss": "6.8 GiB on recommendations, up 400 MiB an hour since the last deploy",
    "upstream_timeout": "37 timeouts a minute from the gateway to orders; zero to other upstreams",
    "gc_pause": "p99 pause 480 ms on search since last night's release; 40 ms before it",
    "cache_miss": "62% miss rate on the product cache at the top of each hour; 8% otherwise",
    "conversion": "checkout conversion 2.1% last quarter, down from 3.4% the quarter before",
    "throttle": "0.7% of requests throttled at the default limit of 100 requests a second",
    "pool_wait": "p95 wait for a connection 220 ms before the pooling change, 15 ms after",
    "v1_requests": "v1 API requests in the last week: billing 41000, reports 9300, "
    "every other service zero",
    "startup_time": "cold start p95 11.4 s on functions with the large image; 1.9 s on the slim one",
    "cert_days_left": "9 days left on the api gateway certificate; over 200 on every other",
    "cohort_retention": "30-day retention 31% for the March cohort, 44% for the February cohort",
    "replication_lag": "replication lag 95 s on the orders replica; under 1 s elsewhere",
    "connections": "database connections 480 of a limit of 500 on the orders primary",
    "disk_usage": "disk 93% and inodes 71% used on the logging cluster",
    "time_to_resolve": "median time to resolve 74 minutes this quarter, 118 a year ago",
}

#: The alerts the working ``list_alerts`` tool lists, newest first.
WORLD_ALERTS: tuple[str, ...] = (
    "CRITICAL checkout HighLatencyP99 firing for 25 minutes",
    "CRITICAL ingest QueueBacklog firing for 40 minutes",
    "WARNING login Elevated5xx firing for 30 minutes",
    "WARNING payments Elevated5xx firing for 28 minutes",
    "WARNING search GcPauseHigh firing since last night's release",
    "WARNING gateway UpstreamTimeouts(orders) firing for 15 minutes",
    "WARNING orders-db ReplicationLag firing for 20 minutes",
    "INFO api-gateway CertificateExpiresSoon(9 days) firing for 2 days",
)

#: The articles the working ``kb_search`` tool searches: title to body.
WORLD_ARTICLES: dict[str, str] = {
    "Retrieval-augmented generation": "RAG answers from retrieved passages: embed the "
    "question, fetch the nearest chunks from the vector store, and give them to the model "
    "with the question. Our default is five chunks of 800 tokens.",
    "Backpressure in the streaming platform": "Consumers signal demand; producers pause when "
    "the bounded buffer of 10000 events is full. Dropping is off by default.",
    "Cache stampede runbook": "The catalog team stopped hourly stampedes by adding jitter to "
    "expiry and a single-flight lock per key. Miss rate at the hour fell from 60% to 9%.",
    "Certificate expiry incidents": "Two outages last year came from expired gateway "
    "certificates. Renewal is automated for every service except the api gateway, which is "
    "renewed by hand by the platform team.",
    "Checkout abandonment review": "Last quarter's drop followed the new address form: mobile "
    "users abandoned at the postcode step. The form was reverted at quarter end.",
    "Rate-limit defaults": "The default of 100 requests a second per key was set from the "
    "p99 of partner traffic plus 30% headroom, reviewed yearly.",
    "Connection pooling change": "Pool size went from 20 to 60 per service with a 5 s "
    "acquire timeout, rolled out in May.",
    "v1 API deprecation": "v1 is deprecated; billing and reports are the last callers and "
    "are scheduled to move this quarter.",
    "Cold start guidance": "Cold starts are dominated by image size; use the slim base image "
    "and keep the artifact under 200 MB.",
    "Disaster recovery runbook": "Restore is tested monthly; the last test restored orders in "
    "38 minutes against a 60 minute objective. Backups run nightly.",
    "Incident response postmortems": "Postmortems this year show paging moved to service "
    "owners in March and a single incident commander since June.",
    "Service runbook: thresholds": "Page at latency_p99 above 1000 ms for 10 minutes, at "
    "errors_5xx above 1%, and at queue_depth above 100000.",
    "Capacity planning notes": "Plan for 30% headroom on cpu and connections; orders-db is "
    "budgeted for 500 connections until the next resize.",
    "Release notes": "The current release moves search to the new garbage collector "
    "settings and adds the slim base image for functions.",
}


def world_summary() -> str:
    """Everything the working tools can return, as text a judge can check an answer against."""
    metrics = "\n".join(f"- {name}: {reading}" for name, reading in WORLD_METRICS.items())
    alerts = "\n".join(f"- {alert}" for alert in WORLD_ALERTS)
    articles = "\n".join(f"- {title}: {body}" for title, body in WORLD_ARTICLES.items())
    return f"Metrics:\n{metrics}\n\nAlerts:\n{alerts}\n\nKnowledge base:\n{articles}"


def _words(value: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in value.lower()).split() if w}


def working_capabilities() -> CapabilityIndex:
    """The corpus's three capabilities as tools that answer from one fixed world.

    :func:`~locus.router.benchmark.benchmark_capabilities` provides inert
    probes, enough for selection. A shape that runs needs something to read:
    ``kb_search`` returns the articles of :data:`WORLD_ARTICLES` that share
    words with the query, ``get_metric`` the reading in :data:`WORLD_METRICS`
    and ``list_alerts`` :data:`WORLD_ALERTS`. The world is fixed, so two
    readings of a prompt see the same data and a judge can be given it.
    """
    from locus.tools.decorator import tool  # noqa: PLC0415 — keep the tools package off this import
    from locus.tools.registry import create_registry  # noqa: PLC0415

    @tool
    def kb_search(query: str) -> str:
        """Search the knowledge base for a topic and return the matching articles."""
        wanted = _words(query)
        scored = sorted(
            (
                (len(wanted & _words(f"{title} {body}")), title, body)
                for title, body in WORLD_ARTICLES.items()
            ),
            reverse=True,
        )
        hits = [f"{title}: {body}" for score, title, body in scored[:3] if score >= 2]
        return "\n\n".join(hits) if hits else f"No knowledge-base article matches {query!r}."

    @tool
    def get_metric(name: str) -> str:
        """Return the latest reading of a named metric."""
        wanted = _words(name.replace("_", " "))
        for metric, reading in WORLD_METRICS.items():
            if _words(metric.replace("_", " ")) <= wanted or wanted <= _words(
                metric.replace("_", " ")
            ):
                return f"{metric}: {reading}"
        return f"No metric named {name!r}. Known metrics: {', '.join(WORLD_METRICS)}."

    @tool
    def list_alerts(window_minutes: int = 30) -> str:
        """List the alerts that are firing."""
        return "\n".join(WORLD_ALERTS)

    index = CapabilityIndex(create_registry(kb_search, get_metric, list_alerts))
    index.annotate("kb_search", tool_name="kb_search", description="KB lookup.", domain="research")
    index.annotate(
        "metric_probe", tool_name="get_metric", description="Latest metric.", domain="observability"
    )
    index.annotate(
        "alert_list", tool_name="list_alerts", description="Recent alerts.", domain="observability"
    )
    return index


def shape_rubric(rubric: str = ANSWER_RUBRIC) -> str:
    """The pass criterion for an answer produced by a shape that could use the tools.

    The single-reply rubric fails an answer that cites data it could not have
    seen. A shape run on :func:`working_capabilities` can see the world, so
    the judge is told what that world holds: data that matches it is
    legitimate, data that contradicts or goes beyond it is invented.
    """
    return (
        f"{rubric}\n\nThe response was produced by a system that could search a knowledge "
        "base, read metrics and list alerts. Data in the response that matches the "
        "following is legitimate, not invented; data that contradicts it, or that these "
        "sources do not hold, is invented. Reading these sources is the only action the "
        f"system could take.\n\n{world_summary()}"
    )


def intended_frame(case: EvalCase) -> GoalFrame:
    """The frame a labelled case's author states in its metadata."""
    meta = case.metadata
    return GoalFrame(
        primary_goal=TaskType(meta["expected_goal"]),
        domain=meta.get("domain", "general"),
        complexity=Complexity(meta["expected_complexity"]),
        risk=Risk(meta["expected_risk"]),
        required_capabilities=list(meta.get("capabilities", [])),
    )


class IntendedFrameClassifier:
    """A ``RequestClassifier`` that answers with the frame the corpus states.

    The classifier a perfect extractor would be: through the typed target it
    selects the labelled protocol for every labelled prompt, which makes it
    the ceiling a trained classifier is measured against, and through
    :class:`ShapeAgent` it runs each prompt on the shape its label names.
    """

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        self._frames = {
            case.prompt: intended_frame(case) for case in cases if TAG_LABELLED in case.tags
        }

    async def classify(self, text: str) -> GoalFrame:
        try:
            return self._frames[text]
        except KeyError:
            raise LookupError("the corpus states no frame for this prompt") from None


class _ShapeResult(BaseModel):
    message: str
    iterations: int = 1
    protocol_id: str = ""


class ShapeAgent:
    """Answers a corpus prompt by running the protocol its intended frame selects.

    The frame is the corpus's, not an extractor's, so the outcome measures
    the shape and not the routing. The shape is built on ``model`` with
    :func:`working_capabilities` unless ``capabilities`` is given. A request
    the policy gate holds for approval answers that it was not executed,
    which a judge reads as work not done. Satisfies the evaluation runner's
    agent contract.
    """

    def __init__(
        self,
        model: Any,
        cases: Sequence[EvalCase],
        *,
        capabilities: CapabilityIndex | None = None,
    ) -> None:
        self.router = Router(
            classifier=IntendedFrameClassifier(cases),
            compiler=benchmark_compiler(
                model,
                capabilities=capabilities if capabilities is not None else working_capabilities(),
            ),
        )

    async def arun(self, prompt: str) -> _ShapeResult:
        frame = await self.router.classifier.classify(prompt)
        protocol_id = self.router.compiler.explain(frame, goal=prompt).selected_protocol or ""
        try:
            result = await self.router.dispatch(prompt, frame=frame)
        except PolicyDeniedError as exc:
            # The gate held the request for a human. That is an outcome, not a
            # failure of the run: nothing was done, and the answer says so.
            return _ShapeResult(
                message=f"Not executed: the request was held by policy ({exc}).",
                protocol_id=protocol_id,
            )
        return _ShapeResult(message=result.text or "", protocol_id=protocol_id)

    def run_sync(self, prompt: str) -> _ShapeResult:
        import asyncio  # noqa: PLC0415

        return asyncio.run(self.arun(prompt))


@dataclass(frozen=True)
class MeasuredDifficulty:
    """What the tiers' answers say about one case."""

    case: str
    #: The level the case's tag names.
    intended: str | None
    #: ``low``, ``medium``, ``high`` or :data:`NOT_MEASURED`.
    measured: str
    #: Per tier, cheapest first: ``True`` passed, ``False`` failed, ``None`` not graded.
    passed: dict[str, bool | None] = field(default_factory=dict)
    trap: bool = False

    @property
    def agrees(self) -> bool:
        """Whether the measurement is the level the author intended."""
        return self.measured == self.intended


def _outcome(report: EvalReport, name: str) -> bool | None:
    for result in report.results:
        if result.case_name == name:
            if result.not_graded_reason is not None or result.error is not None:
                return None
            return result.passed
    return None


def measure_difficulty(
    cases: Sequence[EvalCase], reports: Mapping[str, EvalReport]
) -> list[MeasuredDifficulty]:
    """Read each case's level from the tiers' reports.

    Args:
        cases: The cases the reports were run over.
        reports: One report per tier, cheapest tier first. With repeats a
            case passes a tier only when every reading passed.

    A case is ``low`` when the first tier passed it, ``medium`` when a later
    tier did, and ``high`` when every tier failed it. A tier that was not
    graded for a case (the judge abstained, the answer errored) decides
    nothing: the case is :data:`NOT_MEASURED` unless a graded tier already
    settled it.

    Raises:
        ValueError: Fewer than two tiers; one tier cannot tell ``medium`` from ``high``.
    """
    if len(reports) < 2:
        raise ValueError("measuring difficulty needs at least two tiers, cheapest first")
    tiers = list(reports)
    out: list[MeasuredDifficulty] = []
    for case in cases:
        passed = {tier: _outcome(reports[tier], case.name) for tier in tiers}
        if passed[tiers[0]]:
            level = DIFFICULTIES[0]
        elif passed[tiers[0]] is not None and any(passed[t] for t in tiers[1:]):
            level = DIFFICULTIES[1]
        elif all(passed[t] is False for t in tiers):
            level = DIFFICULTIES[2]
        else:
            level = NOT_MEASURED
        out.append(
            MeasuredDifficulty(
                case=case.name,
                intended=intended_difficulty(case),
                measured=level,
                passed=passed,
                trap=TAG_TRAP in case.tags,
            )
        )
    return out


def disagreements(measured: Sequence[MeasuredDifficulty]) -> list[MeasuredDifficulty]:
    """The measured cases whose level is not the one their author intended."""
    return [m for m in measured if m.measured != NOT_MEASURED and not m.agrees]


def confusion_table(measured: Sequence[MeasuredDifficulty]) -> str:
    """A Markdown table of intended level (rows) against measured level (columns).

    The last row counts the traps on their own: a trap is intended ``low``
    by definition, so its row is where an author's reading of "one step" is
    checked against outcomes.
    """
    columns = [*DIFFICULTIES, NOT_MEASURED]

    def row(label: str, items: Sequence[MeasuredDifficulty]) -> str:
        counts = [sum(1 for m in items if m.measured == column) for column in columns]
        return f"| {label} | {len(items)} | " + " | ".join(str(n) for n in counts) + " |"

    lines = [
        "| intended | n | " + " | ".join(f"measured {c}" for c in columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    lines.extend(row(level, [m for m in measured if m.intended == level]) for level in DIFFICULTIES)
    lines.append(row("trap", [m for m in measured if m.trap]))
    return "\n".join(lines)
