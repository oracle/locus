# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Running a v2 playbook: a step graph the model walks with three control tools.

A v1 playbook (:mod:`tulip.playbooks`) is an ordered list of steps whose
progress the runtime *infers* from the tools the model calls. A v2 playbook
(``version: "playbook.v2"``, the registry's ``definitions/playbook_v2.py``) is a
graph: steps in groups, ordered by ``after``, fanned out with ``parallel_group``
and gathered with ``joins``, routed by branches whose ``when`` is the safe
condition language in :mod:`tulip.playbooks.v2.when`, and concluded by exactly one
outcome of a decision policy. The model *declares* its progress:

* ``complete_step(step_id, outputs)`` closes an active step once every output it
  promised is present (and, under ``block``, once it made the calls its floor asks
  for). A step whose tools were allowed but ran nothing closes ``not_executed``,
  never ``done``: calling a tool with no body is not doing the step's work;
* ``select_branches(step_id, branch_ids, reason)`` routes a router step by branch id,
  the branches it does not pick waived -- but only where the data cannot: a branch whose
  ``when`` the run can tell (inputs, earlier outputs, the step's own outputs once
  ``complete_step`` gives them) is decided by it. A choice against such a verdict is
  refused, or -- when the step's own outputs settle it only at ``complete_step`` -- set
  aside; a choice that agrees is recorded as ``routed_by: "when"``. Branches without a
  condition of their own (``always``) and conditions that are ``unknown`` are still the
  model's (or a person's) to choose;
* ``submit_decision(outcome_id, rationale, evidence_refs)`` concludes the playbook
  with one of the decision policy's outcomes, and is the run's decision.

Those three are not governed (they change nothing outside the run) and are always
handed to a run that follows a v2 playbook. ``ask_user`` is how a step gets what
only the person knows (``required_from_user``): such a step closes only once every
declared answer is in its outputs, and the questions it declares are not the agent's
own -- :meth:`PlaybookRuntime.ask_is_declared` tells a per-run ``ask_user`` budget which
asks to leave out.

**Task steps** (``kind: task``, :mod:`tulip.playbooks.v2.tasks`) are done by a person.
The model has no tools in one and cannot complete it (``complete_step`` and
``select_branches`` refuse: "This step is done by a person"; while only task steps are
active every other call is refused, whatever the enforcement mode).
:meth:`PlaybookRuntime.pending_task` says which task is waiting, for the gateway to file
its hold and park the run; :meth:`PlaybookRuntime.complete_task` takes the person's
values, checks them against the form, and records them as the step's outputs. The
briefs of the steps after it show what the person gave (the ``sensitive: false`` values).

A run restored from its trace (:meth:`PlaybookRuntime.restore`) never mistakes a digest
for data: an output the trace holds only as a digest (``metadata_only`` custody) is
restored as :data:`~tulip.playbooks.v2.when.UNAVAILABLE`, a branch whose ``when`` reads it
is ``unknown``, and the step is not routed on a guess -- ``complete_step`` refuses with
``routing: "unknown"``, which the gateway turns into a hold, unless the caller restored
with the real values (``restore(events, outputs=...)``).

**Typed process data** (:mod:`tulip.playbooks.v2.fields`). A playbook declares its
``inputs``, each step's ``outputs`` (beside the old ``expected_outputs``, each a text field)
and a question's answer (``required_from_user[].field``) as typed fields. ``complete_step``
checks every typed value against its field and refuses, listing each bad one in the
registry's own words. A run's inputs (:meth:`PlaybookRuntime.set_inputs`) are read by
conditions as ``inputs.<name>``; money compares by amount and never across currencies,
dates compare as dates. A value the run cannot use -- a required input not given, an
invalid one, a digest -- is UNAVAILABLE, as above. ``sensitive: false`` marks the values the
gateway may mirror in clear (:meth:`PlaybookRuntime.public_outputs`,
:meth:`PlaybookRuntime.public_inputs`); every other value it digests. The engine emits no
event field for them: the gateway decides what to mirror.

What is enforced, before a call executes (:meth:`PlaybookRuntime.on_before_tool_call`):
the active steps' ``allowed_tools`` (``None`` = the agent's tools, ``[]`` = none, the
union across the steps active together) and each step's ``max_tool_calls``. As for
v1, **recording is the default and blocking is opt-in** (``metadata.enforcement:
block`` on the playbook, or the deployment's ``deployment`` default -- the gateway's
``TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT``): a call
outside the contract is always recorded as a ``playbook_deviation`` and is cancelled
only under ``block``. Forbidden actions are not enforced here at all: they compile
into the run's policy ``deny_for`` (:func:`forbidden_deny`), exactly as the
registry's ``effective_policy(..., playbook=)`` does, so the gate refuses them
whatever this module or the model think.

Execution is sequential. Steps of a parallel group are *active together* -- the
trace says so, and the model may work on any of them -- but one call runs at a time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tulip.hooks.provider import HookPriority, HookProvider
from tulip.playbooks.v2.approvals import (
    ResolvedApproval,
    StepApproval,
    call_context,
    describe_when,
    parse_step_approval,
    pick_rule,
)
from tulip.playbooks.v2.fields import (
    Field,
    describe,
    parse_field,
    parse_fields,
    text_field,
    typed_value,
    validate_value,
)
from tulip.playbooks.v2.notify import NotifyRule, parse_notify_rules
from tulip.playbooks.v2.results import not_executed_reason, not_executed_result, refused
from tulip.playbooks.v2.tasks import (
    TaskRequest,
    TaskSpec,
    form_problems,
    is_task,
    parse_task,
)
from tulip.playbooks.v2.when import (
    FALSE,
    TRUE,
    UNAVAILABLE,
    UNKNOWN,
    Always,
    WhenSyntaxError,
    is_digest,
    is_withheld,
    parse_when,
    when_paths,
    when_verdict,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from tulip.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent


#: The ``version`` marker of a v2 playbook.
PLAYBOOK_V2 = "playbook.v2"
#: The three tools a v2 run is driven with (the registry's ``PLAYBOOK_KERNEL_TOOLS``).
CONTROL_TOOLS: tuple[str, ...] = ("complete_step", "select_branches", "submit_decision")
#: How a step asks the person for what only they know.
ASK_USER = "ask_user"
#: Proposal tools: about the run, not a step of it, so no step allowlist governs them
#: (the policy does -- they are weighed like any governed call).
PROPOSAL_TOOLS: tuple[str, ...] = ("propose_skill", "propose_playbook")
#: The outcome every decision policy carries, checked first.
INCONCLUSIVE = "INCONCLUSIVE"

RECORD = "record"
BLOCK = "block"

STEP_EVENT = "playbook_step"
DEVIATION_EVENT = "playbook_deviation"
DECISION_EVENT = "playbook_decision"

PENDING = "pending"
ACTIVE = "active"
DONE = "done"
WAIVED = "waived"
BLOCKED = "blocked"
NOT_EXECUTED = "not_executed"
#: Statuses a step does not leave. ``not_executed`` is terminal: the record says the
#: work was not done, and the graph moves on rather than deadlocking the run.
TERMINAL = frozenset({DONE, WAIVED, NOT_EXECUTED})
#: What a gated step's ``after`` accepts as finished.
_SATISFIES = frozenset({DONE, NOT_EXECUTED})

#: Every status a step can have; a restored record must name one of these.
STATUSES = frozenset({PENDING, ACTIVE, BLOCKED, DONE, WAIVED, NOT_EXECUTED})
#: What a restored step's calls are recorded as: the count survives a move, the names do not.
RESTORED_CALL = "(before the run moved)"

#: The registry's ``PUBLISH_STAMP_KEY``, excluded from a definition's digest.
_STAMP_KEY = "tulip.publish"


class PlaybookV2Error(ValueError):
    """A v2 playbook the gateway cannot run as written. The run is refused."""


#: ``complete_step``'s refusal when a branch cannot be told true or false.
ROUTING_UNKNOWN = UNKNOWN


class RestoreError(ValueError):
    """A runtime could not be restored from the step records given; nothing was changed.

    ``reason`` says why, in the words a log line or a fail-closed notice can carry.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RestoreResult:
    """What :meth:`PlaybookRuntime.restore` put back.

    ``records`` is how many step records were read; ``statuses`` is every step's
    status, in definition order; ``active`` the steps active (or blocked) now;
    ``unavailable`` the closed steps whose outputs, or part of them, the trace holds only
    as a digest (or not at all) and nobody supplied (:meth:`PlaybookRuntime.restore`).
    """

    records: int
    statuses: Mapping[str, str]
    active: tuple[str, ...]
    unavailable: tuple[str, ...] = ()


def is_playbook_v2(definition: Any) -> bool:
    """Whether a resolved playbook definition is a v2 playbook."""
    return isinstance(definition, dict) and definition.get("version") == PLAYBOOK_V2


def definition_digest(definition: Mapping[str, Any]) -> str:
    """The registry's ``definition_digest``: canonical JSON, publish stamp excluded."""
    body = dict(definition)
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and _STAMP_KEY in metadata:
        body["metadata"] = {k: v for k, v in metadata.items() if k != _STAMP_KEY}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── the definition, as the runtime reads it ──────────────────────────────────


@dataclass(frozen=True)
class Branch:
    id: str
    label: str
    when: str
    next_step_id: str

    @property
    def conditioned(self) -> bool:
        """Whether the branch has a condition of its own: a ``when`` that is neither
        missing nor ``always``. Such a branch is decided by its condition whenever the run
        can tell it (:meth:`StepGraph.condition_verdicts`), never by a model's choice."""
        text = self.when.strip()
        if not text:
            return False
        try:
            return not isinstance(parse_when(text), Always)
        except WhenSyntaxError:
            return True  # evaluated as it always was: false

    def reads_own_outputs(self, step_id: str) -> bool:
        """Whether the condition reads the outputs of the step it belongs to."""
        try:
            paths = when_paths(parse_when(self.when))
        except WhenSyntaxError:
            return False
        own = f"outputs.{step_id}"
        return any(
            p.split(".")[0] not in ("inputs", "outputs") or p == own or p.startswith(own + ".")
            for p in paths
        )


@dataclass(frozen=True)
class Step:
    """One step and the contract it runs under (the registry's ``PlaybookStep``)."""

    id: str
    title: str
    group: str
    goal: str = ""
    required: bool = True
    after: tuple[str, ...] = ()
    parallel_group: str | None = None
    joins: str | None = None
    branches: tuple[Branch, ...] = ()
    skill_refs: tuple[str, ...] = ()
    #: ``None`` = the agent's tools; ``()`` = none.
    allowed_tools: tuple[str, ...] | None = None
    min_tool_calls: int | None = None
    max_tool_calls: int | None = None
    expected_outputs: tuple[str, ...] = ()
    #: ``(name, question)`` pairs, asked through ``ask_user``.
    required_from_user: tuple[tuple[str, str], ...] = ()
    rules: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    #: Who must approve this step's tool calls; ``None`` = no approval of its own.
    approval: StepApproval | None = None
    #: The step's outputs as typed fields (``outputs``), beside ``expected_outputs``.
    outputs: tuple[Field, ...] = ()
    #: The typed answers its questions declare (``required_from_user[].field``), by the
    #: question's name; a question without one has a sensitive text answer.
    answer_fields: tuple[Field, ...] = ()
    #: What a person is asked to do, for a task step (``kind: task``); ``None`` for a step
    #: the model does. A task step has no tools, and its form is its ``outputs``.
    task: TaskSpec | None = None

    @property
    def is_task(self) -> bool:
        """Whether a person does this step (``kind: task``), not the model."""
        return self.task is not None

    def output_fields(self) -> list[Field]:
        """The step's outputs as fields: ``expected_outputs`` merged with ``outputs``.

        The registry's ``PlaybookStep.output_fields``: in ``expected_outputs`` order, a
        name declared in ``outputs`` takes that field, a name only in ``expected_outputs``
        is a sensitive text field; the fields only ``outputs`` declares follow, in order.
        """
        typed = {f.name: f for f in self.outputs}
        merged: list[Field] = []
        seen: set[str] = set()
        for name in self.expected_outputs:
            if name in seen:
                continue
            seen.add(name)
            merged.append(typed.get(name) or text_field(name))
        merged += [f for f in self.outputs if f.name not in seen]
        return merged

    def data_fields(self) -> dict[str, Field]:
        """Every value this step produces, by name: its outputs, then its answers.

        What a condition may read of the step (``outputs.<step>.<name>``, or a bare
        ``<name>`` in the step's own branches).
        """
        fields = {f.name: f for f in self.output_fields()}
        answers = {f.name: f for f in self.answer_fields}
        for name, _ in self.required_from_user:
            fields.setdefault(name, answers.get(name) or text_field(name))
        return fields

    def typed_fields(self) -> list[Field]:
        """The fields whose values ``complete_step`` checks: the ones declared with a type.

        An ``expected_outputs`` name with no field in ``outputs`` (and a question with no
        ``field``) is untyped, as it always was: any value is accepted for it.
        """
        typed = list(self.outputs)
        names = {f.name for f in typed}
        typed += [f for f in self.answer_fields if f.name not in names]
        return typed

    def allows(self, tool: str) -> bool:
        """Whether this step's allowlist admits ``tool`` (kernel tools aside).

        Never for a task step: the model has no tools in a step a person does.
        """
        if self.task is not None:
            return False
        if self.allowed_tools is None:
            return True
        if tool == ASK_USER and self.required_from_user:
            return True
        return tool in self.allowed_tools

    @property
    def decides(self) -> bool:
        """Whether this is where the playbook concludes (it allows ``submit_decision``)."""
        return self.allowed_tools is not None and "submit_decision" in self.allowed_tools


@dataclass(frozen=True)
class Outcome:
    id: str
    priority: int
    condition: str = ""
    description: str = ""
    action: str = ""
    classification: str | None = None
    escalation_target: dict[str, Any] | None = None

    @property
    def inconclusive(self) -> bool:
        return INCONCLUSIVE in (self.id, self.classification)


@dataclass(frozen=True)
class Forbidden:
    target: str
    kind: str  # tool | label
    reason: str


@dataclass(frozen=True)
class Group:
    id: str
    title: str
    goal: str = ""


@dataclass(frozen=True)
class PlaybookV2:
    """A v2 playbook, read for running. Structural rules were checked at publish."""

    id: str
    title: str
    summary: str
    mode: str
    groups: tuple[Group, ...]
    steps: tuple[Step, ...]
    #: ``(name, description)`` of each input, as before typed data.
    inputs: tuple[tuple[str, str], ...] = ()
    decision_policy_id: str = ""
    decision_rules: tuple[str, ...] = ()
    decision_facts: tuple[str, ...] = ()
    outcomes: tuple[Outcome, ...] = ()
    forbidden: tuple[Forbidden, ...] = ()
    all_required_steps_resolved: bool = True
    recommendation_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: The inputs as typed fields (``inputs.<name>`` in a condition).
    input_fields: tuple[Field, ...] = ()
    #: Who is told when a step starts, waits, ends or fails (:mod:`~tulip.playbooks.v2.notify`).
    #: The registry sends the notices; the engine only reads the rules.
    notify_rules: tuple[NotifyRule, ...] = ()

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def outcome(self, outcome_id: str) -> Outcome | None:
        return next((o for o in self.outcomes if o.id == outcome_id), None)

    @property
    def decision_steps(self) -> list[Step]:
        return [s for s in self.steps if s.decides]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _strs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        return ()
    return tuple(_text(v) for v in value if _text(v))


def _count(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _branches(raw: Mapping[str, Any], step_id: str) -> tuple[Branch, ...]:
    branches = tuple(
        Branch(
            id=_text(b.get("id")),
            label=_text(b.get("label")),
            when=_text(b.get("when")),
            next_step_id=_text(b.get("next_step_id")),
        )
        for b in raw.get("branches") or []
        if isinstance(b, dict)
    )
    for branch in branches:
        try:
            parse_when(branch.when)
        except WhenSyntaxError as exc:
            raise PlaybookV2Error(
                f"step {step_id!r} branch {branch.id!r}: when {branch.when!r}: {exc}"
            ) from exc
    return branches


def _task_step(raw: Mapping[str, Any], step_id: str, group: str) -> Step:
    """A step a person does: no tools, no questions, no approval; its form is its outputs."""
    task = parse_task(raw)
    return Step(
        id=step_id,
        title=_text(raw.get("title")) or step_id,
        group=group,
        goal=_text(raw.get("goal")) or task.instructions,
        required=raw.get("required", True) is not False,
        after=_strs(raw.get("after")),
        parallel_group=_text(raw.get("parallel_group")) or None,
        joins=_text(raw.get("joins")) or None,
        branches=_branches(raw, step_id),
        allowed_tools=(),
        rules=_strs(raw.get("rules")),
        facts=_strs(raw.get("facts")),
        outputs=task.form,
        task=task,
    )


def _step(raw: Mapping[str, Any], group: str) -> Step:
    step_id = _text(raw.get("id"))
    if not step_id:
        raise PlaybookV2Error(f"a step in group {group!r} has no id")
    if is_task(raw):
        return _task_step(raw, step_id, group)
    allowed = raw.get("allowed_tools")
    asks = [
        r
        for r in raw.get("required_from_user") or []
        if isinstance(r, dict) and _text(r.get("name"))
    ]
    branches = _branches(raw, step_id)
    return Step(
        id=step_id,
        title=_text(raw.get("title")) or step_id,
        group=group,
        goal=_text(raw.get("goal")),
        required=raw.get("required", True) is not False,
        after=_strs(raw.get("after")),
        parallel_group=_text(raw.get("parallel_group")) or None,
        joins=_text(raw.get("joins")) or None,
        branches=branches,
        skill_refs=_strs(raw.get("skill_refs")),
        allowed_tools=None if allowed is None else _strs(allowed),
        min_tool_calls=_count(raw.get("min_tool_calls")),
        max_tool_calls=_count(raw.get("max_tool_calls")),
        expected_outputs=_strs(raw.get("expected_outputs")),
        required_from_user=tuple((_text(r.get("name")), _text(r.get("question"))) for r in asks),
        rules=_strs(raw.get("rules")),
        facts=_strs(raw.get("facts")),
        approval=parse_step_approval(raw.get("approval")),
        outputs=parse_fields(raw.get("outputs")),
        answer_fields=tuple(
            answer
            for r in asks
            if (answer := _answer_field(r.get("field"), _text(r.get("name")))) is not None
        ),
    )


def _answer_field(raw: Any, name: str) -> Field | None:
    """A question's typed answer: its ``field``, under the question's name."""
    if not isinstance(raw, Mapping):
        return None
    return parse_field({**raw, "name": name})


def parse_playbook_v2(definition: Mapping[str, Any]) -> PlaybookV2:
    """Read a resolved v2 definition, or raise :class:`PlaybookV2Error`.

    The registry validated the whole structure at publish (a 422 otherwise), so
    this re-checks only what running it depends on: every reference resolves and
    every condition parses. A playbook that fails here is refused rather than run
    with part of its contract -- its forbidden actions among them -- quietly gone.
    """
    if not is_playbook_v2(definition):
        raise PlaybookV2Error("not a playbook.v2 definition")
    groups: list[Group] = []
    steps: list[Step] = []
    for raw_group in definition.get("step_groups") or []:
        if not isinstance(raw_group, dict):
            continue
        group_id = _text(raw_group.get("id"))
        groups.append(
            Group(group_id, _text(raw_group.get("title")) or group_id, _text(raw_group.get("goal")))
        )
        steps.extend(
            _step(raw, group_id) for raw in raw_group.get("steps") or [] if isinstance(raw, dict)
        )
    if not steps:
        raise PlaybookV2Error("the playbook has no steps")
    known = {s.id for s in steps}
    if len(known) != len(steps):
        raise PlaybookV2Error("step ids repeat")
    groups_used = {s.parallel_group for s in steps if s.parallel_group}
    for s in steps:
        dangling = [d for d in s.after if d not in known]
        dangling += [b.next_step_id for b in s.branches if b.next_step_id not in known]
        if dangling:
            raise PlaybookV2Error(f"step {s.id!r} names steps that do not exist: {dangling}")
        if s.joins and s.joins not in groups_used:
            raise PlaybookV2Error(f"step {s.id!r} joins {s.joins!r}, which no step belongs to")
    policy = definition.get("decision_policy")
    policy = policy if isinstance(policy, dict) else {}
    outcomes = tuple(
        Outcome(
            id=_text(o.get("id")),
            priority=_count(o.get("priority")) or 0,
            condition=_text(o.get("condition")),
            description=_text(o.get("description")),
            action=_text(o.get("action")),
            classification=_text(o.get("classification")) or None,
            escalation_target=dict(o["escalation_target"])
            if isinstance(o.get("escalation_target"), dict)
            else None,
        )
        for o in policy.get("outcomes") or []
        if isinstance(o, dict) and _text(o.get("id"))
    )
    forbidden = tuple(
        Forbidden(
            target=_text(f.get("tool") or f.get("label")),
            kind="tool" if _text(f.get("tool")) else "label",
            reason=_text(f.get("reason")),
        )
        for f in definition.get("forbidden_actions") or []
        if isinstance(f, dict) and _text(f.get("tool") or f.get("label"))
    )
    completion = definition.get("completion")
    completion = completion if isinstance(completion, dict) else {}
    metadata = definition.get("metadata")
    input_fields = parse_fields(definition.get("inputs"))
    return PlaybookV2(
        id=_text(definition.get("id")) or "playbook",
        title=_text(definition.get("title")) or _text(definition.get("id")) or "Playbook",
        summary=_text(definition.get("summary")),
        mode=_text(definition.get("mode")) or "diagnosis",
        groups=tuple(groups),
        steps=tuple(steps),
        inputs=tuple((f.name, f.description) for f in input_fields),
        decision_policy_id=_text(policy.get("id")),
        decision_rules=_strs(policy.get("rules")),
        decision_facts=_strs(policy.get("facts")),
        outcomes=tuple(sorted(outcomes, key=lambda o: o.priority)),
        forbidden=forbidden,
        all_required_steps_resolved=completion.get("all_required_steps_resolved", True)
        is not False,
        recommendation_only=completion.get("recommendation_only") is True,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        input_fields=input_fields,
        notify_rules=parse_notify_rules(definition),
    )


# ── policy: forbidden actions are deny rules ─────────────────────────────────


def forbidden_deny(playbook: PlaybookV2) -> list[str]:
    """The ``deny_for`` the playbook's forbidden actions compile into (sorted).

    The registry's ``compile_forbidden``: a forbidden tool denies on its name, a
    forbidden label on the label. The gate puts a call's tool name in its label set
    beside the tool's declared tags, so either form denies the call.
    """
    return sorted({f.target for f in playbook.forbidden})


def with_forbidden(policy: Mapping[str, Any], deny: Iterable[str]) -> dict[str, Any]:
    """``policy`` with ``deny`` unioned into its ``deny_for``. Nothing else moves.

    The gateway half of the registry's ``effective_policy(bundle, inline,
    playbook=fragment)``: applied after the bundle/inline merge, so a playbook can
    forbid and never allow. A policy that omits ``deny_for`` reads it as the
    default (empty), which is what the gate does too.
    """
    extra = set(deny)
    if not extra:
        return dict(policy)
    current = policy.get("deny_for") or []
    return {**policy, "deny_for": sorted({str(x) for x in current} | extra)}


def enforcement_mode(playbook: PlaybookV2, deployment: str = RECORD) -> str:
    """:data:`BLOCK` or :data:`RECORD`: the playbook's own say first, then the deployment's.

    ``deployment`` is the deployment's default (the gateway passes its
    ``TULIP_GATEWAY_PLAYBOOK_ENFORCEMENT``); this module reads no environment.
    """
    declared = _text(playbook.metadata.get("enforcement")).lower()
    if declared in (BLOCK, RECORD):
        return declared
    return BLOCK if deployment.lower() == BLOCK else RECORD


# ── the graph ────────────────────────────────────────────────────────────────


class StepGraph:
    """Which steps are pending, active, done or waived, and what routes where.

    Pure state: no events, no I/O. A step becomes *ready* once every ``after`` step
    finished, every member of the group it ``joins`` is terminal, and -- when some
    branch names it -- a branch enabled it. A step named only by branches that were
    not taken is waived, and so is a step that comes after a waived one: being routed
    out propagates down. Activation follows observai's scheduler: when nothing is
    active, the first ready step activates, and with it every ready sibling of its
    parallel group.
    """

    def __init__(self, playbook: PlaybookV2) -> None:
        self.playbook = playbook
        self.status: dict[str, str] = {s.id: PENDING for s in playbook.steps}
        self.outputs: dict[str, dict[str, Any]] = {}
        #: group -> its member step ids, in order.
        self.members: dict[str, list[str]] = {}
        for s in playbook.steps:
            if s.parallel_group:
                self.members.setdefault(s.parallel_group, []).append(s.id)
        #: target step -> the steps whose branches name it.
        self.gates: dict[str, list[str]] = {}
        for s in playbook.steps:
            for b in s.branches:
                self.gates.setdefault(b.next_step_id, []).append(s.id)
        #: Branching steps whose routing is settled, and the targets they enabled.
        self.routed: dict[str, set[str]] = {}
        #: Branch ids a router step chose explicitly (``select_branches``).
        self.selected: dict[str, list[str]] = {}
        #: Why a step was waived.
        self.waived_because: dict[str, str] = {}
        #: Closed steps whose outputs this run cannot see (restored from a digest, or from
        #: a record without them): a condition that reads them is ``unknown``.
        self.withheld: set[str] = set()
        #: The playbook's inputs as conditions read them (``inputs.<name>``), set by the
        #: runtime: a value it cannot use is :data:`UNAVAILABLE`, an optional input not
        #: given is absent.
        self.inputs: dict[str, Any] = {}

    def index(self, step_id: str) -> int:
        return next(i for i, s in enumerate(self.playbook.steps) if s.id == step_id)

    def active(self) -> list[Step]:
        return [s for s in self.playbook.steps if self.status[s.id] in (ACTIVE, BLOCKED)]

    def unresolved(self) -> list[Step]:
        return [s for s in self.playbook.steps if self.status[s.id] not in TERMINAL]

    def _gate(self, step: Step) -> str:
        """``open`` / ``shut`` / ``wait`` for a step some branch names."""
        parents = self.gates.get(step.id)
        if not parents:
            return "open"
        if any(step.id in self.routed.get(p, set()) for p in parents):
            return "open"
        if all(p in self.routed for p in parents):
            return "shut"
        return "wait"

    def _readiness(self, step: Step) -> str:
        """``ready`` / ``waive`` / ``wait``."""
        gate = self._gate(step)
        if gate == "shut":
            return "waive"
        for dep in step.after:
            if self.status[dep] == WAIVED:
                return "waive"
        if gate == "wait":
            return "wait"
        if any(self.status[dep] not in _SATISFIES for dep in step.after):
            return "wait"
        if step.joins and any(
            self.status[m] not in TERMINAL for m in self.members.get(step.joins, [])
        ):
            return "wait"
        return "ready"

    def advance(self) -> tuple[list[str], list[str]]:
        """Settle what the last transition changed: ``(newly waived, newly active)``."""
        waived: list[str] = []
        changed = True
        while changed:
            changed = False
            for s in self.playbook.steps:
                if self.status[s.id] == PENDING and self._readiness(s) == "waive":
                    self.status[s.id] = WAIVED
                    self.waived_because.setdefault(
                        s.id,
                        "no branch that names it was taken"
                        if self._gate(s) == "shut"
                        else "it comes after a step that was waived",
                    )
                    waived.append(s.id)
                    changed = True
        if self.active():
            return waived, []
        ready = [
            s
            for s in self.playbook.steps
            if self.status[s.id] == PENDING and self._readiness(s) == "ready"
        ]
        if not ready:
            return waived, []
        first = ready[0]
        batch = (
            [s for s in ready if s.parallel_group == first.parallel_group]
            if first.parallel_group
            else [first]
        )
        for s in batch:
            self.status[s.id] = ACTIVE
        return waived, [s.id for s in batch]

    def branch_verdicts(
        self, step: Step, outputs: Mapping[str, Any] | None = None
    ) -> dict[str, str]:
        """Each branch of ``step``: ``"true"``, ``"false"`` or ``"unknown"``. Changes nothing.

        An explicit selection decides (chosen is true, the rest false). Otherwise each
        ``when`` is evaluated against the step's outputs -- ``outputs`` when given, the
        recorded ones otherwise -- with earlier steps' under ``outputs.<id>`` and the
        playbook's inputs under ``inputs.<name>``; a branch that reads a value this run
        cannot see (:attr:`withheld`, a withheld value inside one, an input it was not
        given) is ``"unknown"``. Typed values compare as their type says
        (:func:`~tulip.playbooks.v2.fields.typed_value`), so ordering money in two
        currencies is ``"unknown"`` too.
        """
        if step.id in self.selected:
            chosen = self.selected[step.id]
            verdicts = {b.id: TRUE if b.id in chosen else FALSE for b in step.branches}
            # A selection never goes against the data: a branch whose condition the run
            # can tell is decided by it, whatever was chosen.
            for branch_id, verdict in self.condition_verdicts(step, outputs).items():
                if verdict != UNKNOWN:
                    verdicts[branch_id] = verdict
            return verdicts
        return {b.id: when_verdict(b.when, self.context(step, outputs)) for b in step.branches}

    def condition_verdicts(
        self, step: Step, outputs: Mapping[str, Any] | None = None, *, own_pending: bool = False
    ) -> dict[str, str]:
        """Each branch of ``step`` with a condition (:attr:`Branch.conditioned`): what its
        ``when`` says, whatever ``select_branches`` chose. Changes nothing.

        ``own_pending`` is for a step whose outputs are not given yet (``select_branches``
        runs before ``complete_step``): a condition that reads them is ``"unknown"`` until
        they are, rather than read against nothing.
        """
        context = self.context(step, outputs)
        return {
            b.id: UNKNOWN
            if own_pending and b.reads_own_outputs(step.id)
            else when_verdict(b.when, context)
            for b in step.branches
            if b.conditioned
        }

    def conditions_decide(self, step: Step, outputs: Mapping[str, Any] | None = None) -> bool:
        """Whether ``step``'s conditions alone settle its routing: every branch has one,
        and the run can tell each."""
        verdicts = self.condition_verdicts(step, outputs)
        return (
            bool(step.branches)
            and len(verdicts) == len(step.branches)
            and UNKNOWN not in verdicts.values()
        )

    def context(self, step: Step, outputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What a ``when`` of ``step`` reads: its outputs bare, the earlier steps' under
        ``outputs.<id>``, the inputs under ``inputs`` -- each typed by its field."""
        own = _typed(step, self.outputs.get(step.id, {}) if outputs is None else outputs)
        earlier: dict[str, Any] = {
            step_id: _typed(self.playbook.step(step_id), values)
            for step_id, values in self.outputs.items()
        }
        earlier.update(dict.fromkeys(self.withheld, UNAVAILABLE))
        if outputs is not None:
            earlier[step.id] = own
        context: dict[str, Any] = {"outputs": earlier}
        context.update(own)
        if self.playbook.input_fields:
            # ``inputs`` always names the inputs, as the registry reads a path.
            fields = {f.name: f for f in self.playbook.input_fields}
            context["inputs"] = {
                name: typed_value(fields[name], value) for name, value in self.inputs.items()
            }
        return context

    def route(self, step: Step) -> tuple[list[str], list[str], str]:
        """Settle a branching step's routing: ``(taken, not taken, routed_by)``.

        Each branch's ``when`` is evaluated against the step's outputs (earlier steps'
        under ``outputs.<id>``). An explicit selection decides only the branches without a
        condition and the ones whose condition the run cannot tell; ``routed_by`` is
        ``"when"`` whenever the conditions alone settle it (:meth:`conditions_decide`),
        a selection that agrees with them included. When some
        branch is ``unknown`` (:meth:`branch_verdicts`) nothing is settled: ``routed_by``
        is ``"unknown"``, the unknown branches are in neither list, and every target
        keeps waiting -- a guess is never routed on. (:meth:`PlaybookRuntime.complete_step`
        refuses to close such a step before it gets here.)
        """
        if not step.branches:
            return [], [], ""
        verdicts = self.branch_verdicts(step)
        taken = [b for b, v in verdicts.items() if v == TRUE]
        not_taken = [b for b, v in verdicts.items() if v == FALSE]
        if len(taken) + len(not_taken) < len(verdicts):
            return taken, not_taken, UNKNOWN
        self.routed[step.id] = {b.next_step_id for b in step.branches if b.id in taken}
        chose = step.id in self.selected and not self.conditions_decide(step)
        return taken, not_taken, "select_branches" if chose else "when"

    def converged(self) -> bool:
        """Whether a positive decision may be submitted now.

        With a decision step (one whose allowlist names ``submit_decision``): when it
        is active. Without one: when every required step is resolved.
        """
        deciders = self.playbook.decision_steps
        if deciders:
            return any(self.status[s.id] in (ACTIVE, BLOCKED) for s in deciders)
        return all(self.status[s.id] in (DONE, WAIVED) for s in self.playbook.steps if s.required)


def _typed(step: Step | None, values: Mapping[str, Any]) -> dict[str, Any]:
    """A step's outputs with each typed one as a condition reads it."""
    fields = step.data_fields() if step is not None else {}
    return {
        name: typed_value(fields[name], value) if name in fields else value
        for name, value in values.items()
    }


# ── prose: what the model is told ────────────────────────────────────────────


def _skill_key(skill: Mapping[str, Any]) -> str:
    return _text(skill.get("name")) or _text(skill.get("id"))


def skills_by_name(skills: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Resolved skill definitions keyed by name (and by id when it differs)."""
    out: dict[str, Mapping[str, Any]] = {}
    for skill in skills:
        for key in (_text(skill.get("name")), _text(skill.get("id"))):
            if key and key not in out:
                out[key] = skill
    return out


def _tools_line(step: Step) -> str:
    if step.allowed_tools is None:
        tools = "any of your tools"
    elif not step.allowed_tools:
        tools = "none (reason over what you already have)"
    else:
        tools = ", ".join(step.allowed_tools)
    bounds = []
    if step.min_tool_calls is not None:
        bounds.append(f"at least {step.min_tool_calls}")
    if step.max_tool_calls is not None:
        bounds.append(f"at most {step.max_tool_calls}")
    return f"Tools: {tools}" + (f" ({' and '.join(bounds)} calls)" if bounds else "")


def _field_note(field: Field) -> str:
    """A field's name as the model is told it: with how to write it, when typed."""
    notes = [n for n in (describe(field), "" if field.required else "optional") if n]
    return field.name + (f" ({'; '.join(notes)})" if notes else "")


def _shown_value(value: Any) -> str:
    """A value as the model reads it in a brief: text as written, the rest as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _people_lines(people: Sequence[tuple[Step, Mapping[str, Any]]]) -> list[str]:
    """What people gave in the task steps before this one: the values that may be shown.

    A sensitive value is named, never shown; a value this run cannot see (restored from
    a digest) is said to be unavailable.
    """
    lines: list[str] = []
    for task_step, values in people:
        lines.append(f"Done by a person -- {task_step.title} ({task_step.id}):")
        fields = task_step.output_fields()
        if not fields:
            lines.append("- (the form asked for nothing)")
        for f in fields:
            if f.name not in values:
                lines.append(f"- {f.shown_as}: not given")
            elif is_withheld(values[f.name]):
                lines.append(f"- {f.shown_as}: given, not available to this run")
            elif f.sensitive:
                lines.append(f"- {f.shown_as}: given (sensitive, not shown)")
            else:
                lines.append(f"- {f.shown_as}: {_shown_value(values[f.name])}")
    return lines


def _task_brief(step: Step) -> list[str]:
    """What the model is told of a step a person does: that it is theirs, and to wait."""
    task = step.task
    assert task is not None  # only called for a task step
    lines = [
        f"This step is done by a person ({task.assignee.shown_as}), not by you. You have no "
        "tools in it and cannot complete it: the run waits until they do, and their answers "
        "become the step's outputs."
    ]
    if task.instructions:
        lines.append(f"What they are asked to do: {task.instructions}")
    if task.form:
        lines.append("What they fill in: " + ", ".join(_field_note(f) for f in task.form))
    if task.due:
        lines.append(f"Due within {task.due}.")
    return lines


def step_brief(
    step: Step,
    skills: Mapping[str, Mapping[str, Any]],
    *,
    people: Sequence[tuple[Step, Mapping[str, Any]]] = (),
) -> str:
    """Everything the model needs to do one step, its skills' instructions in full.

    ``people`` are the task steps done before it, each with its recorded outputs: their
    ``sensitive: false`` values are shown as the person gave them, the others named.
    A task step's own brief says that a person does it.
    """
    lines = [
        f"### Step {step.id}: {step.title}"
        + ("" if step.required else " (optional)")
        + (" (done by a person)" if step.is_task else "")
    ]
    if step.goal and not (step.task is not None and step.goal == step.task.instructions):
        lines.append(f"Goal: {step.goal}")
    lines.extend(_people_lines(people))
    if step.is_task:
        lines.extend(_task_brief(step))
        if step.branches:
            lines.append("Branches (routed by what the person gives):")
            lines.extend(
                f"- {b.id}{' (' + b.label + ')' if b.label else ''}: when {b.when} -> "
                f"{b.next_step_id}"
                for b in step.branches
            )
        return "\n".join(lines)
    lines.append(_tools_line(step))
    outputs = step.output_fields()
    if outputs:
        lines.append(
            "Expected outputs (pass each to complete_step): "
            + ", ".join(_field_note(f) for f in outputs)
        )
    if step.required_from_user:
        answers = {f.name: f for f in step.answer_fields}
        lines.append("Ask the person with ask_user, unless they already told you:")
        for name, question in step.required_from_user:
            shape = describe(answers[name]) if name in answers else ""
            lines.append(
                f"- {name}: {question or 'ask for ' + name}"
                + (f" (answer as {shape})" if shape else "")
            )
    if step.rules:
        lines.append("Rules:")
        lines.extend(f"- {rule}" for rule in step.rules)
    if step.facts:
        lines.append("Facts you may take as given:")
        lines.extend(f"- {fact}" for fact in step.facts)
    if step.branches:
        conditioned = [b for b in step.branches if b.conditioned]
        if len(conditioned) == len(step.branches):
            how = (
                "routed automatically by their conditions when you call complete_step; "
                "select_branches cannot choose against them"
            )
        elif conditioned:
            how = (
                "a branch with a condition is routed by it automatically when you call "
                "complete_step; choose among the others with select_branches before "
                "complete_step"
            )
        else:
            how = "decided by your outputs, or pick them with select_branches before complete_step"
        lines.append(f"Branches ({how}):")
        lines.extend(
            f"- {b.id}{' (' + b.label + ')' if b.label else ''}: when {b.when} -> {b.next_step_id}"
            for b in step.branches
        )
    for ref in step.skill_refs:
        skill = skills.get(ref)
        if skill is None:
            lines.append(f"Skill {ref}: not available to this run.")
            continue
        instructions = _text(skill.get("instructions"))
        lines.append(f"#### Skill: {ref}\n{instructions or _text(skill.get('description'))}")
    return "\n".join(lines)


def playbook_prose(
    playbook: PlaybookV2,
    skills: Mapping[str, Mapping[str, Any]],
    active: Iterable[str],
    *,
    agent_skills: Iterable[Mapping[str, Any]] = (),
    enforcement: str = RECORD,
) -> str:
    """The system-prompt section for a v2 playbook.

    Progressive disclosure: the whole procedure in outline, the decision policy and
    the forbidden actions, every skill by name and description, and only the
    *active* steps in full (their skills' instructions included). Each later step's
    brief arrives in the result of the ``complete_step`` that activates it.
    """
    active_ids = list(active)
    lines = [f"# Playbook: {playbook.title}"]
    if playbook.summary:
        lines.append(playbook.summary)
    if playbook.input_fields:
        lines.append("Inputs:")
        lines.extend(
            f"- {_field_note(f)}: {f.description}" if f.description else f"- {_field_note(f)}"
            for f in playbook.input_fields
        )
    lines.append(
        "## How this run works\n"
        "The runtime tracks this playbook step by step. Work only on the active step(s).\n"
        "- When a step's work is done, call complete_step(step_id, outputs) with every "
        "expected output in `outputs`. Its result tells you which step is active next and "
        "gives you that step's full brief.\n"
        "- A step with branches routes the run: each branch's `when` decides it "
        "automatically when complete_step closes the step, and select_branches cannot "
        "choose against a condition the run can tell. Call select_branches(step_id, "
        "branch_ids, reason) before complete_step only to choose among branches without a "
        "condition, or when the runtime says a condition cannot be decided.\n"
        + (
            "- Conclude with submit_decision(outcome_id, rationale, evidence_refs): exactly "
            "once, one of the outcomes below. If evidence is missing, the honest outcome is "
            f"{INCONCLUSIVE}.\n"
            if playbook.outcomes
            else ""
        )
        + (
            "- A step done by a person is theirs: you have no tools in it and cannot "
            "complete it. The run waits until they complete it, and what they give becomes "
            "its outputs.\n"
            if any(s.is_task for s in playbook.steps)
            else ""
        )
        + (
            "- A call outside the active step's tools is refused."
            if enforcement == BLOCK
            else "- A call outside the active step's tools is recorded as a deviation."
        )
    )
    lines.append("## Steps")
    for group in playbook.groups:
        lines.append(f"{group.title}" + (f": {group.goal}" if group.goal else ""))
        for s in playbook.steps:
            if s.group != group.id:
                continue
            notes = []
            if s.is_task:
                notes.append("done by a person")
            if not s.required:
                notes.append("optional")
            if s.after:
                notes.append("after " + ", ".join(s.after))
            if s.parallel_group:
                notes.append(f"in parallel group {s.parallel_group}")
            if s.joins:
                notes.append(f"joins {s.joins}")
            if s.branches:
                notes.append("branches to " + ", ".join(b.next_step_id for b in s.branches))
            lines.append(f"- {s.id}: {s.title}" + (f" ({'; '.join(notes)})" if notes else ""))
    if playbook.outcomes:
        lines.append("## Decision policy")
        lines.extend(f"- {rule}" for rule in playbook.decision_rules)
        lines.extend(f"- Given: {fact}" for fact in playbook.decision_facts)
        lines.append("Outcomes, checked in this order:")
        for o in playbook.outcomes:
            text = o.condition or o.description
            lines.append(
                f"- {o.id}"
                + (f" [{o.classification}]" if o.classification else "")
                + (f": {text}" if text else "")
                + (f" Then: {o.action}" if o.action else "")
            )
        if playbook.recommendation_only:
            lines.append("This playbook recommends; it does not act on its conclusion.")
    if playbook.forbidden:
        lines.append("## Never")
        lines.extend(f"- {f.target}: {f.reason}" for f in playbook.forbidden)
    catalog = skills_by_name(agent_skills)
    for ref in (r for s in playbook.steps for r in s.skill_refs):
        if ref in skills and ref not in catalog:
            catalog[ref] = skills[ref]
    if catalog:
        lines.append(
            "## Skills\nEach arrives in full with the step that uses it; until then, what it is for:"
        )
        seen: set[int] = set()
        for name, skill in catalog.items():
            if id(skill) in seen:
                continue
            seen.add(id(skill))
            desc = _text(skill.get("description"))
            lines.append(f"- {name}" + (f": {desc}" if desc else ""))
    if active_ids:
        lines.append("## Active now")
        for step_id in active_ids:
            step = playbook.step(step_id)
            if step is not None:
                lines.append(step_brief(step, skills))
    return "\n".join(lines)


def _against(verdicts: Mapping[str, str], chosen: Iterable[str]) -> list[str]:
    """The branches whose known verdict a choice of ``chosen`` contradicts."""
    picked = set(chosen)
    return [b for b, v in verdicts.items() if v != UNKNOWN and (b in picked) != (v == TRUE)]


def initial_active(playbook: PlaybookV2) -> list[str]:
    """The steps active when a run starts."""
    graph = StepGraph(playbook)
    _, active = graph.advance()
    return active


# ── the runtime ──────────────────────────────────────────────────────────────


@dataclass
class _Work:
    """What happened inside one step."""

    attempts: int = 0
    executed: list[str] = field(default_factory=list)
    unrun: dict[str, str] = field(default_factory=dict)
    asked: bool = False
    #: ``ask_user`` calls attributed to the step (asking is not one of its tool calls).
    asks: int = 0
    deviated: bool = False


def _answered(value: Any) -> bool:
    """Whether ``value`` is an answer: present, and not blank or empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        return len(value) > 0
    return True


def _public(field: Field, value: Any) -> bool:
    """Whether ``value`` of a ``sensitive: false`` field may be shown: a valid value held."""
    return value is not None and not is_withheld(value) and validate_value(field, value) is None


def validate_inputs(playbook: PlaybookV2, values: Any) -> list[str]:
    """Why ``values`` cannot start a run of ``playbook``, one sentence each (empty = fine).

    The registry's ``validate_inputs``: every required input must be given, every value
    must suit its field (:func:`~tulip.playbooks.v2.fields.validate_value`), and no value
    may name an input the playbook does not declare. A value held as a digest is not
    checked: the runtime cannot see it, and reads it as UNAVAILABLE.
    """
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        return ["The inputs must be given as names and their values."]
    fields = {f.name: f for f in playbook.input_fields}
    problems = [
        f"The playbook has no input named {name!r}." for name in values if name not in fields
    ]
    for name, f in fields.items():
        value = values.get(name)
        if is_withheld(value):
            continue
        problem = validate_value(f, value)
        if problem is not None:
            problems.append(problem)
    return problems


def _by_person(steps: Sequence[Step]) -> str:
    """Why the model cannot act in these task steps, in one sentence."""
    named = ", ".join(
        f"{s.id} ({s.task.assignee.shown_as})" if s.task is not None else s.id for s in steps
    )
    return (
        f"This step is done by a person: {named}. The run waits until they complete it."
        if len(steps) == 1
        else f"These steps are done by people: {named}. The run waits until they complete them."
    )


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PlaybookRuntime(HookProvider):
    """Runs one v2 playbook alongside the agent loop, and records what happened.

    Emits ``playbook_step`` (every status change), ``playbook_deviation`` (every
    departure, and whether it was blocked) and ``playbook_decision`` through
    ``emit``; hands each step's terminal status and the decision to ``record``, for the
    run's chain, so the run's evidence bundle carries the outcome and the trail.

    ``record(event, fields)`` gets the audit record's kind (:data:`STEP_EVENT` or
    :data:`DECISION_EVENT`) and its fields, ``correlation_id`` and ``principal``
    included -- exactly the fields of the gateway's ``PlaybookStepAudit`` and
    ``PlaybookDecisionAudit``. Whoever owns the chain turns them into its record; this
    module writes nothing itself.
    """

    def __init__(
        self,
        playbook: PlaybookV2,
        *,
        emit: Callable[[dict[str, Any]], None],
        skills: Mapping[str, Mapping[str, Any]] | None = None,
        enforce: bool = False,
        ref: Mapping[str, Any] | None = None,
        record: Callable[[str, dict[str, Any]], None] | None = None,
        correlation_id: str = "run",
        principal: str = "agent",
    ) -> None:
        self.playbook = playbook
        self.graph = StepGraph(playbook)
        self._emit = emit
        self._skills = dict(skills or {})
        self.enforce = enforce
        ref = dict(ref or {})
        self.version = _text(ref.get("version"))
        self.digest = _text(ref.get("digest"))
        self._record = record
        self._correlation_id = correlation_id
        self._principal = principal
        self._work: dict[str, _Work] = {s.id: _Work() for s in playbook.steps}
        #: tool -> the steps its in-flight calls were attributed to, oldest first.
        self._owners: dict[str, list[str]] = {}
        self.decision: dict[str, Any] | None = None
        self._finished = False
        self._started = False
        self._restored = False
        self.violations: list[str] = []
        #: The inputs the run was started with (:meth:`set_inputs`); ``None`` = none given.
        self._given_inputs: dict[str, Any] | None = None
        self.graph.inputs = self._input_context(None, withheld=False)

    @property
    def name(self) -> str:
        return "TulipPlaybookRuntime"

    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    def bind(self, enforcer: Any) -> None:
        """The v1 tracker's seam; a v2 run has no SDK enforcer to adopt."""

    # ── events ───────────────────────────────────────────────────────────────

    def _step_event(self, step_id: str, *, reason: str = "", **extra: Any) -> None:
        step = self.playbook.step(step_id)
        assert step is not None  # every caller names a step of this playbook
        status = self.graph.status[step_id]
        work = self._work[step_id]
        event: dict[str, Any] = {
            "type": STEP_EVENT,
            "playbook": self.playbook.id,
            "step": step.id,
            "index": self.graph.index(step.id),
            "total": len(self.playbook.steps),
            "status": status,
            "required": step.required,
            "expected_tools": list(step.allowed_tools or ()),
            "description": step.goal or step.title,
            "verified": status != NOT_EXECUTED and not work.deviated,
            "version": PLAYBOOK_V2,
            "step_id": step.id,
            "title": step.title,
            "group": step.group,
            "parallel_group": step.parallel_group,
            "allowed_tools": None if step.allowed_tools is None else list(step.allowed_tools),
            "tool_calls": len(work.executed),
            **({"reason": reason} if reason else {}),
            **extra,
        }
        self._emit(event)
        if status in TERMINAL:
            self._audit_step(step, status, reason)

    def _deviation(
        self,
        step: Step | None,
        tool: str,
        violation: str,
        *,
        blocked: bool = False,
        **detail: Any,
    ) -> None:
        """Record a departure from the contract; ``detail`` says what it was about."""
        if step is not None:
            self._work[step.id].deviated = True
        self.violations.append(violation)
        self._emit(
            {
                **detail,
                "type": DEVIATION_EVENT,
                "playbook": self.playbook.id,
                "step": step.id if step is not None else "",
                "index": self.graph.index(step.id) if step is not None else -1,
                "tool": tool,
                "violation": violation,
                "expected_tools": list(step.allowed_tools or ()) if step is not None else [],
                "required": step.required if step is not None else False,
                "enforcement": BLOCK if self.enforce else RECORD,
                "blocked": blocked,
                "version": PLAYBOOK_V2,
            }
        )

    def _audit_step(self, step: Step, status: str, reason: str) -> None:
        if self._record is None:
            return
        self._record(
            STEP_EVENT,
            {
                "correlation_id": self._correlation_id,
                "principal": self._principal,
                "playbook": self.playbook.id,
                "playbook_version": self.version,
                "playbook_digest": self.digest,
                "step": step.id,
                "status": status,
                "required": step.required,
                "tool_calls": len(self._work[step.id].executed),
                "reason": reason,
            },
        )

    def _settle(self) -> list[str]:
        """Advance the graph and record what it changed; the newly active step ids."""
        waived, active = self.graph.advance()
        for step_id in waived:
            self._step_event(step_id, reason=self.graph.waived_because.get(step_id, ""))
        for step_id in active:
            self._step_event(step_id)
        return active

    def start(self, inputs: Mapping[str, Any] | None = None) -> None:
        """Record every step as pending, then activate the first. Idempotent.

        ``inputs`` are the run's input values, as :meth:`set_inputs` takes them (whose
        result says what was wrong with them); they are set even when the run has started.
        """
        if inputs is not None:
            self.set_inputs(inputs)
        if self._started:
            return
        self._started = True
        for s in self.playbook.steps:
            self._step_event(s.id)
        self._settle()

    # ── process data: the inputs, and what may be shown in clear ─────────────

    def set_inputs(self, values: Mapping[str, Any]) -> list[str]:
        """Give the run its input values; what was wrong with them, one sentence each.

        Each declared input's value is checked against its field (:func:`validate_inputs`,
        the registry's run-start check; the gateway refuses a run whose inputs fail it).
        The engine stays tolerant: a value that fails -- or a required input not given,
        or a value held as a digest -- reads as UNAVAILABLE, so a branch that reads it is
        ``unknown`` and the step is not routed on a guess. A value for an input the
        playbook does not declare is reported and left out of the conditions. Values can
        be set at any time; a restored run needs them set again (:meth:`restore`).
        """
        given = dict(values)
        self._given_inputs = given
        self.graph.inputs = self._input_context(given, withheld=False)
        return validate_inputs(self.playbook, given)

    def _input_context(self, values: Mapping[str, Any] | None, *, withheld: bool) -> dict[str, Any]:
        """The inputs as conditions read them: valid values, UNAVAILABLE for the rest."""
        context: dict[str, Any] = {}
        for f in self.playbook.input_fields:
            value = (values or {}).get(f.name)
            if withheld or is_withheld(value) or validate_value(f, value) is not None:
                context[f.name] = UNAVAILABLE
            elif value is not None:
                context[f.name] = value
        return context

    def unavailable_inputs(self) -> list[str]:
        """The declared inputs this run cannot read: not given, not valid, or withheld."""
        return [name for name, value in self.graph.inputs.items() if is_withheld(value)]

    def public_inputs(self) -> dict[str, Any]:
        """The run's input values that may be shown in clear: the ``sensitive: false`` ones.

        For the gateway's mirror: these may be kept in clear, every other value is
        digested. Only valid values the run holds are included.
        """
        given = self._given_inputs or {}
        return {
            f.name: given[f.name]
            for f in self.playbook.input_fields
            if not f.sensitive and _public(f, given.get(f.name))
        }

    def public_outputs(self, step_id: str) -> dict[str, Any]:
        """A closed step's output values that may be shown in clear (``sensitive: false``).

        Its outputs and its answers, as :meth:`Step.data_fields` declares them; an
        untyped or undeclared value is sensitive and never included, nor is a value the
        run cannot see or one that is not valid for its field. Empty for a step that has
        recorded no outputs, or one this playbook does not have.
        """
        step = self.playbook.step(_text(step_id))
        if step is None:
            return {}
        values = self.graph.outputs.get(step.id, {})
        return {
            name: values[name]
            for name, f in step.data_fields().items()
            if not f.sensitive and name in values and _public(f, values[name])
        }

    # ── restoring a run from its own records ─────────────────────────────────

    @property
    def restored(self) -> bool:
        """Whether this runtime's state came from :meth:`restore` rather than :meth:`start`."""
        return self._restored

    def restore(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        outputs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RestoreResult:
        """Put the runtime where a run's ``playbook_step`` records left it; emit nothing.

        For a run rebuilt somewhere else (another pod) from its trace: without this the
        rebuilt runtime would start over, ``owner_of`` would attribute nothing, and a call
        a step's approval holds would run unasked. ``events`` are the payloads this engine
        emitted as ``playbook_step`` (``{"type": "playbook_step", "playbook", "step_id",
        "status", ...}``), in the order they were emitted (the trace's ``seq``). A payload
        whose ``type`` names another event (a deviation, a decision) is skipped; one
        without a ``type`` is read as a step record.

        Each step's last record gives its status, why it was waived, how many calls it
        made (and whether it deviated), and -- for a step that closed ``done`` -- its
        outputs, which a later branch's ``when`` may read. A branching step that ended
        routed the run: the targets its record says it enabled (``enabled_steps``), or,
        for a record without them, the targets that were not waived. The runtime then
        counts as started: :meth:`start` records no fresh step tree over the real one.

        **A digest is never data.** Under the gateway's ``metadata_only`` custody a
        record's ``outputs`` is a digest (``{"redacted": true, "sha256", "bytes"}``), not
        the values. Such a step -- and a closed step whose record carries no outputs at
        all, as a ``not_executed`` one never does -- is restored with its outputs
        UNAVAILABLE (:meth:`unavailable_outputs`); so is any single output value the
        record holds as a digest, and a step whose ``verified`` is a digest is restored
        unverified. A branch whose ``when`` reads one is ``unknown``, and
        :meth:`complete_step` will not route on it. ``outputs`` lets a caller that holds
        the real values (the gateway's data-plane checkpoint) supply them, by step id: a
        supplied mapping is used for a step that closed, after it is checked against the
        record -- its digest must be the record's ``sha256`` (or equal the record's own
        plain outputs). Supplied outputs for a step that has not closed are ignored.

        All or nothing. :class:`RestoreError` is raised, and nothing changes, when a
        record is not a mapping, names another playbook or a step this playbook does not
        have, carries a status the engine does not know, when there are no step records,
        or when a step of the playbook has none (:meth:`start` records them all); and
        when supplied ``outputs`` name a step this playbook does not have, are not a
        mapping, or are not the outputs the record holds.

        What the records do not carry is not restored: a selection made with
        ``select_branches`` on a step still active, whether (and how often) a step already
        asked the person, which of its tools did not execute, and the decision. Nor are
        the run's inputs: the ones :meth:`set_inputs` gave this runtime carry over, and
        without them every input is UNAVAILABLE until they are set.
        """
        playbook = self.playbook
        known = {s.id for s in playbook.steps}
        supplied: dict[str, dict[str, Any]] = {}
        for step_id, values in (outputs or {}).items():
            if step_id not in known:
                raise RestoreError(
                    f"outputs supplied for a step this playbook does not have ({step_id!r})"
                )
            if not isinstance(values, Mapping):
                raise RestoreError(f"the outputs supplied for step {step_id} are not a mapping")
            supplied[step_id] = dict(values)
        last: dict[str, Mapping[str, Any]] = {}
        records = 0
        for event in events:
            if not isinstance(event, Mapping):
                raise RestoreError("a step record without its fields")
            kind = event.get("type")
            if kind is not None and kind != STEP_EVENT:
                continue
            if str(event.get("playbook") or "") != playbook.id:
                raise RestoreError(f"a step record of another playbook ({event.get('playbook')!r})")
            step_id = str(event.get("step_id") or event.get("step") or "")
            if step_id not in known:
                raise RestoreError(
                    f"a step record for a step this playbook does not have ({step_id!r})"
                )
            if str(event.get("status") or "") not in STATUSES:
                raise RestoreError(
                    f"a step record with an unknown status ({event.get('status')!r})"
                )
            last[step_id] = event
            records += 1
        if not last:
            raise RestoreError("the trace has no step records")
        missing = sorted(known - set(last))
        if missing:
            raise RestoreError(f"the trace has no record of step(s) {', '.join(missing)}")

        # Everything read: build the new state aside, then swap it in.
        graph = StepGraph(playbook)
        work: dict[str, _Work] = {s.id: _Work() for s in playbook.steps}
        for step_id, data in last.items():
            status = str(data["status"])
            graph.status[step_id] = status
            if status == WAIVED and data.get("reason") and not is_withheld(data["reason"]):
                graph.waived_because[step_id] = str(data["reason"])
            if status in (DONE, NOT_EXECUTED):
                self._restore_outputs(graph, step_id, data.get("outputs"), supplied.get(step_id))
            calls = data.get("tool_calls")
            if isinstance(calls, int) and not isinstance(calls, bool) and calls > 0:
                work[step_id].executed = [RESTORED_CALL] * calls
                work[step_id].attempts = calls
            # A digest in place of ``verified`` is not a yes: a step the record cannot
            # vouch for stays unverified.
            verified = data.get("verified")
            if (verified is False or is_withheld(verified)) and status != NOT_EXECUTED:
                work[step_id].deviated = True
        for step in playbook.steps:
            if not step.branches or graph.status[step.id] not in TERMINAL:
                continue
            enabled = last[step.id].get("enabled_steps")
            targets = {b.next_step_id for b in step.branches}
            if isinstance(enabled, list | tuple):
                graph.routed[step.id] = {str(t) for t in enabled} & targets
            else:
                graph.routed[step.id] = {t for t in targets if graph.status[t] != WAIVED}
        # The records never carry the inputs: the ones set on this runtime carry over, and
        # without them every input is UNAVAILABLE until :meth:`set_inputs` gives them.
        graph.inputs = self._input_context(self._given_inputs, withheld=self._given_inputs is None)
        self.graph = graph
        self._work = work
        self._owners = {}
        self._started = True
        self._restored = True
        return RestoreResult(
            records=records,
            statuses={s.id: graph.status[s.id] for s in playbook.steps},
            active=tuple(s.id for s in graph.active()),
            unavailable=tuple(s.id for s in playbook.steps if s.id in self.unavailable_outputs()),
        )

    @staticmethod
    def _restore_outputs(
        graph: StepGraph, step_id: str, recorded: Any, supplied: dict[str, Any] | None
    ) -> None:
        """A closed step's outputs, from what the caller supplied or what its record holds."""
        if supplied is not None:
            if is_digest(recorded):
                matches = recorded.get("sha256") == _sha256(supplied)
            elif isinstance(recorded, Mapping):
                matches = _sha256(dict(recorded)) == _sha256(supplied)
            else:
                matches = True  # the record says nothing to check them against
            if not matches:
                raise RestoreError(
                    f"the outputs supplied for step {step_id} are not the ones its record holds"
                )
            graph.outputs[step_id] = supplied
            return
        if isinstance(recorded, Mapping) and not is_withheld(recorded):
            graph.outputs[step_id] = {
                str(k): UNAVAILABLE if is_withheld(v) else v for k, v in recorded.items()
            }
            return
        graph.withheld.add(step_id)

    def unavailable_outputs(self) -> set[str]:
        """The closed steps whose outputs, or part of them, this run cannot see.

        Only a restored run has any (:meth:`restore`): the trace held them as digests, or
        not at all, and nobody supplied them. A branch that reads one is ``unknown``.
        """
        partial = {
            step_id
            for step_id, values in self.graph.outputs.items()
            if any(is_withheld(v) for v in values.values())
        }
        return set(self.graph.withheld) | partial

    def hold_unstarted(self) -> None:
        """Count the runtime as started without recording a step tree or activating a step.

        For a run whose steps could not be restored (:class:`RestoreError`) and that fails
        closed instead: :meth:`start` would otherwise record a fresh tree that was never
        true over the real one. Nothing is active afterwards, so ``owner_of`` attributes
        nothing.
        """
        self._started = True

    def brief(self, step_ids: Iterable[str]) -> list[str]:
        """Each step's brief; a step after a task step done by a person shows what they gave.

        The task steps shown are the ones earlier in the playbook that closed ``done``,
        their ``sensitive: false`` values as given and the rest named
        (:func:`step_brief`'s ``people``).
        """
        return [
            step_brief(step, self._skills, people=self._people_before(step))
            for step_id in step_ids
            if (step := self.playbook.step(step_id)) is not None
        ]

    def _people_before(self, step: Step) -> list[tuple[Step, Mapping[str, Any]]]:
        """The task steps done before ``step``, each with its recorded outputs."""
        index = self.graph.index(step.id)
        return [
            (
                s,
                dict.fromkeys((f.name for f in s.output_fields()), UNAVAILABLE)
                if s.id in self.graph.withheld
                else self.graph.outputs.get(s.id, {}),
            )
            for i, s in enumerate(self.playbook.steps)
            if i < index and s.is_task and self.graph.status[s.id] == DONE
        ]

    # ── enforcement at the hook seam ─────────────────────────────────────────

    def owner_of(self, tool: str) -> Step | None:
        """The active step a call to ``tool`` would be attributed to, or ``None``.

        The first active step (in definition order) whose allowlist admits ``tool``.
        """
        return next((s for s in self.graph.active() if s.allows(tool)), None)

    _owner = owner_of

    def approval_for(
        self, tool: str, args: Mapping[str, Any] | None = None
    ) -> ResolvedApproval | None:
        """Who must approve a call to ``tool`` with ``args``, or ``None`` if nobody need.

        The step that owns the call (:meth:`owner_of`) decides: ``None`` when no active
        step owns it or the step has no ``approval``. Otherwise its rules are read, first
        match wins (:func:`~tulip.playbooks.v2.approvals.pick_rule`), each ``when``
        against what the step's branches read -- the run's inputs (``inputs.<name>``),
        the earlier steps' outputs (``outputs.<id>.<name>``) -- plus the call's arguments
        (``args.<name>``). A rule that cannot be told (a withheld value, two currencies,
        an argument that is not the number it is ordered against) means the strictest of
        it and the rules after it: a call never needs fewer approvers for what the run
        cannot see. Changes nothing.
        """
        step = self.owner_of(tool)
        if step is None or step.approval is None:
            return None
        context = self.graph.context(step)
        context["args"] = call_context(args)
        return pick_rule(step.approval, context, step=step.id)

    def active_steps(self) -> list[Step]:
        """The steps active now (``active`` or ``blocked``), in definition order."""
        return self.graph.active()

    # ── questions: the playbook's, or the agent's own ────────────────────────

    def declared_question(self, step_id: str, name: str) -> bool:
        """Whether ``name`` is a question step ``step_id`` declares (``required_from_user``)."""
        step = self.playbook.step(_text(step_id))
        return step is not None and _text(name) in {n for n, _ in step.required_from_user}

    def _declaring(self) -> Step | None:
        """The first active step that still has declared questions it has not asked."""
        return next(
            (
                s
                for s in self.graph.active()
                if s.required_from_user and self._work[s.id].asks < len(s.required_from_user)
            ),
            None,
        )

    def _ask_owner(self) -> Step | None:
        return self._declaring() or self._owner(ASK_USER)

    def ask_is_declared(self) -> bool:
        """Whether the next ``ask_user`` call asks one of the playbook's declared questions.

        For a per-run budget of the agent's own questions: a declared question
        (``required_from_user``) is the playbook asking, not the agent, and should not
        count. Ask it *before* the call is admitted: an active step that declares ``n``
        questions makes its first ``n`` asks declared, and every ask after that -- or one
        no such step owns -- is the agent's own. The allowance is per step, so a step
        cannot buy unlimited questions by declaring one. A restored run counts its
        steps' asks afresh (the records do not carry them).
        """
        return self._declaring() is not None

    def _admit(self, tool: str) -> str:
        """Attribute a call about to run; ``""`` to let it run, else why it is refused."""
        self.start()
        if tool in CONTROL_TOOLS or tool in PROPOSAL_TOOLS:
            return ""
        active = self.graph.active()
        if self.decision is not None or not self.graph.unresolved():
            self._deviation(None, tool, "playbook_complete", blocked=self.enforce)
            return (
                f"the playbook {self.playbook.id} is concluded; {tool} is not part of it"
                if self.enforce
                else ""
            )
        owner = self._ask_owner() if tool == ASK_USER else self._owner(tool)
        if owner is None and active and all(s.is_task for s in active):
            # Only people's steps are active: the model has nothing to do but wait, and
            # this is refused whatever the enforcement mode.
            self._deviation(active[0], tool, "unexpected_tool", blocked=True)
            return _by_person(active) + f" {tool} cannot run until they complete it."
        if owner is None:
            allowed = sorted({t for s in active for t in (s.allowed_tools or ())})
            anchor = active[0] if active else None
            self._deviation(anchor, tool, "unexpected_tool", blocked=self.enforce)
            if not self.enforce:
                return ""
            return (
                f"{tool} is not allowed in the active step"
                f"{'s' if len(active) > 1 else ''} "
                f"({', '.join(s.id for s in active) or 'none'}). Allowed: "
                f"{', '.join(allowed) or 'no tools'}. Finish the step with complete_step "
                "to move on."
            )
        work = self._work[owner.id]
        if owner.max_tool_calls is not None and work.attempts >= owner.max_tool_calls:
            self._deviation(
                owner, tool, "too_many_calls", blocked=self.enforce, limit=owner.max_tool_calls
            )
            if self.enforce:
                return (
                    f"step {owner.id} allows at most {owner.max_tool_calls} tool calls and has "
                    "made them; call complete_step with what you have."
                )
        if tool == ASK_USER:
            work.asked = True
            work.asks += 1
        else:
            work.attempts += 1
        self._owners.setdefault(tool, []).append(owner.id)
        return ""

    def _credit(self, tool: str, result: Any, *, error: Any = None) -> None:
        """Attribute a finished call to the step it was admitted under."""
        owners = self._owners.get(tool) or []
        step_id = owners.pop(0) if owners else ""
        if not step_id or tool == ASK_USER or error:
            return
        if refused(result):
            return
        why = not_executed_reason(result)
        work = self._work[step_id]
        if why is not None:
            work.unrun[tool] = why
            self._deviation(
                self.playbook.step(step_id),
                tool,
                "tool_not_executed",
                reason=f"tool {tool} did not execute: {why}",
            )
            return
        work.executed.append(tool)

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        refusal = self._admit(str(event.tool_name))
        if refusal:
            event.cancel = f"Playbook {self.playbook.id}: {refusal}"

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self._credit(
            str(event.tool_name),
            getattr(event, "result", None),
            error=getattr(event, "error", None),
        )

    def observe(self, tool: str, *, arguments: Any = None, not_executed: str = "") -> None:
        """A call performed on resume, which no hook saw (see the v1 tracker's ``observe``)."""
        if self._admit(tool) and self.enforce:
            return
        if not_executed:
            self._credit(tool, not_executed_result(tool, not_executed, {}))
            return
        self._credit(tool, None)

    # ── pauses ───────────────────────────────────────────────────────────────

    def pause(self, reason: str) -> None:
        """The run is waiting on a person: its active steps are blocked until it resumes.

        A call admitted but not yet finished when the run pauses is the call held for that
        person: it has not run. Its admission is released, so the same call redelivered on
        resume is counted once -- otherwise a step with ``max_tool_calls: 1`` refuses its own
        approved call (``too_many_calls``) and then cannot close (``insufficient_effort``).
        """
        self._release_pending()
        for s in self.graph.active():
            if self.graph.status[s.id] == ACTIVE:
                self.graph.status[s.id] = BLOCKED
                self._step_event(s.id, reason=reason)

    def _release_pending(self) -> None:
        """Forget the admissions of calls that never finished (see :meth:`pause`)."""
        for tool, step_ids in self._owners.items():
            if tool == ASK_USER:
                continue
            for step_id in step_ids:
                work = self._work.get(step_id)
                if work is not None and work.attempts > 0:
                    work.attempts -= 1
        self._owners = {}

    def unpause(self) -> None:
        for s in self.graph.active():
            if self.graph.status[s.id] == BLOCKED:
                self.graph.status[s.id] = ACTIVE
                self._step_event(s.id)

    # ── the control tools ────────────────────────────────────────────────────

    def _active_ids(self) -> list[str]:
        return [s.id for s in self.graph.active()]

    def complete_step(self, step_id: str, outputs: Mapping[str, Any] | None) -> dict[str, Any]:
        """``complete_step``: close an active step once its obligations are met."""
        self.start()
        step = self.playbook.step(_text(step_id))
        if step is None:
            return {
                "ok": False,
                "error": f"no step {step_id!r} in this playbook",
                "active": self._active_ids(),
            }
        if step.is_task:
            return {
                "ok": False,
                "error": _by_person([step]) + " You cannot complete it.",
                "done_by_person": True,
                "active": self._active_ids(),
            }
        if self.graph.status[step.id] not in (ACTIVE, BLOCKED):
            return {
                "ok": False,
                "error": f"step {step.id} is {self.graph.status[step.id]}, not active",
                "active": self._active_ids(),
            }
        given = dict(outputs or {})
        missing = [f.name for f in step.output_fields() if f.required and f.name not in given]
        if missing:
            return {
                "ok": False,
                "error": f"step {step.id} still owes outputs: {', '.join(missing)}",
                "missing": missing,
            }
        work = self._work[step.id]
        unanswered = [name for name, _ in step.required_from_user if not _answered(given.get(name))]
        if unanswered:
            return {
                "ok": False,
                "error": f"step {step.id} still needs the person's answer to "
                f"{', '.join(unanswered)}: ask them with ask_user (unless they already told "
                "you), then pass each answer in outputs under its name",
                "missing": unanswered,
            }
        invalid = {
            f.name: problem
            for f in step.typed_fields()
            if f.name in given and (problem := validate_value(f, given[f.name])) is not None
        }
        if invalid:
            return {
                "ok": False,
                "error": f"step {step.id} cannot close: "
                + " ".join(invalid.values())
                + " Fix each of these outputs and call complete_step again.",
                "invalid": invalid,
            }
        if step.branches and step.id not in self.graph.selected:
            verdicts = self.graph.branch_verdicts(step, given)
            unknown = [b for b, v in verdicts.items() if v == UNKNOWN]
            if unknown:
                return self._unroutable(step, unknown)
        set_aside = ""
        if step.id in self.graph.selected:
            # Chosen before the outputs were given: now they are, the conditions they let
            # the run tell decide, and the choice yields to them.
            verdicts = self.graph.condition_verdicts(step, given)
            against = _against(verdicts, self.graph.selected[step.id])
            if against:
                set_aside = (
                    self._conditions_say(step, verdicts, against, given)
                    + " Your select_branches choice was set aside where it went against them."
                )
        self.graph.outputs[step.id] = given
        if not work.executed and work.unrun:
            reason = "; ".join(f"tool {t} did not execute: {why}" for t, why in work.unrun.items())
            self.graph.status[step.id] = NOT_EXECUTED
            self._step_event(step.id, reason=reason)
            if step.required:
                self._deviation(
                    step, next(iter(work.unrun)), "required_step_not_executed", reason=reason
                )
            return self._after_close(step, status=NOT_EXECUTED, reason=reason, set_aside=set_aside)
        floor = step.min_tool_calls or 0
        if len(work.executed) < floor:
            short = floor - len(work.executed)
            self._deviation(
                step,
                "complete_step",
                "insufficient_effort",
                calls_short=short,
                blocked=self.enforce,
            )
            if self.enforce:
                return {
                    "ok": False,
                    "error": f"step {step.id} needs {short} more tool call(s) before it can close",
                }
        self.graph.status[step.id] = DONE
        return self._after_close(step, status=DONE, outputs=given, set_aside=set_aside)

    def _unroutable(self, step: Step, unknown: list[str]) -> dict[str, Any]:
        """``complete_step``'s refusal for a step whose branches cannot be told."""
        unseen = sorted(self.unavailable_outputs())
        read = {
            path.split(".")[1]
            for b in step.branches
            if b.id in unknown
            for path in when_paths(parse_when(b.when))
            if path.startswith("inputs.")
        }
        inputs = [name for name in self.unavailable_inputs() if name in read]
        why: list[str] = []
        if unseen:
            why.append(
                f"reads outputs this run cannot see (of {', '.join(unseen)}) -- the trace "
                "does not hold their values"
            )
        if inputs:
            why.append(
                f"reads inputs this run does not have ({', '.join(inputs)}): not given, "
                "not valid, or not restored"
            )
        if not why:
            why.append(
                "cannot be decided from the data this run holds (it may order amounts of "
                "money in different currencies)"
            )
        return {
            "ok": False,
            "error": f"step {step.id} cannot be routed: branch {', '.join(unknown)} "
            + "; it ".join(why)
            + ". A person must choose the branches, or the run must be given that data.",
            "routing": ROUTING_UNKNOWN,
            "branches_unknown": unknown,
            "unavailable": unseen,
            "unavailable_inputs": inputs,
        }

    def _after_close(
        self,
        step: Step,
        *,
        status: str,
        reason: str = "",
        outputs: Mapping[str, Any] | None = None,
        set_aside: str = "",
        completed_by: str = "",
    ) -> dict[str, Any]:
        taken, not_taken, routed_by = self.graph.route(step)
        routing: dict[str, Any] = {}
        if step.branches:
            routing = {
                "routed_by": routed_by,
                "branches_taken": taken,
                "branches_not_taken": not_taken,
                "enabled_steps": sorted(self.graph.routed.get(step.id, set())),
            }
        if status == DONE:
            done_by = {"completed_by": completed_by} if completed_by else {}
            self._step_event(step.id, outputs=dict(outputs or {}), **routing, **done_by)
        active = self._settle()
        result: dict[str, Any] = {
            "ok": status == DONE,
            "step": step.id,
            "status": status,
            **({"reason": reason} if reason else {}),
            **routing,
            **({"selection_set_aside": set_aside} if set_aside else {}),
            "active": self._active_ids(),
            "waived": sorted(s for s, st in self.graph.status.items() if st == WAIVED),
        }
        if active:
            result["next"] = self.brief(active)
        elif self.playbook.outcomes and self.decision is None:
            result["next"] = (
                "No step is waiting for work. Conclude with submit_decision "
                f"(one of: {', '.join(o.id for o in self.playbook.outcomes)})."
            )
        return result

    # ── task steps: the steps people do ─────────────────────────────────────

    def pending_task(self) -> TaskRequest | None:
        """The task step active now and waiting on its person, or ``None``.

        The first active (or blocked) task step in definition order. The gateway files the
        task hold from it (:meth:`TaskRequest.hold_fields
        <tulip.playbooks.v2.tasks.TaskRequest.hold_fields>`) and parks the run; the person's
        values come back through :meth:`complete_task`. ``None`` once the playbook is
        decided, or when no task step is active. Changes nothing.
        """
        if self.decision is not None:
            return None
        step = next((s for s in self.graph.active() if s.task is not None), None)
        if step is None or step.task is None:
            return None
        task = step.task
        return TaskRequest(
            step_id=step.id,
            title=step.title,
            instructions=task.instructions or step.goal,
            form=task.form,
            assignee=task.assignee,
            due_seconds=task.due_seconds,
            escalate_to=task.escalate_to,
        )

    def complete_task(self, step_id: str, values: Any, by: str) -> dict[str, Any]:
        """A person completed task step ``step_id`` with ``values``: close it, or say why not.

        ``values`` are checked against the step's form, in the registry's words
        (:func:`~tulip.playbooks.v2.tasks.form_problems`): each names a field, each suits its
        field, every required field is given. Then they are the step's outputs -- typed, as
        conditions read them -- the step closes ``done`` and the run routes on as it does
        after ``complete_step`` (a branch whose ``when`` reads the form is decided by it).
        The ``playbook_step`` ``done`` event carries ``completed_by`` (``by``), the one
        field a task step adds. ``by`` is who completed it; whether they may is the
        registry's to check (it holds the grants), not this method's.

        The result is ``complete_step``'s: the step, its routing, the active steps and the
        briefs of the next (which show what the person gave, the ``sensitive: false``
        values in clear). A refusal (``ok: false``) changes nothing: ``invalid`` names each
        bad value by field, ``routing: "unknown"`` says a branch cannot be told.
        """
        self.start()
        step = self.playbook.step(_text(step_id))
        if step is None:
            return {
                "ok": False,
                "error": f"no step {step_id!r} in this playbook",
                "active": self._active_ids(),
            }
        if step.task is None:
            return {
                "ok": False,
                "error": f"step {step.id} is not a task step: the agent completes it",
                "active": self._active_ids(),
            }
        if self.graph.status[step.id] not in (ACTIVE, BLOCKED):
            return {
                "ok": False,
                "error": f"step {step.id} is {self.graph.status[step.id]}, not active",
                "active": self._active_ids(),
            }
        who = _text(by)
        if not who:
            return {"ok": False, "error": "say who completed the task: `by` is required"}
        invalid = form_problems(step.task.form, values)
        if invalid:
            return {
                "ok": False,
                "error": f"The task {step.title} cannot be completed: "
                + " ".join(invalid.values()),
                "invalid": invalid,
            }
        given = dict(values)
        if step.branches:
            verdicts = self.graph.branch_verdicts(step, given)
            unknown = [b for b, v in verdicts.items() if v == UNKNOWN]
            if unknown:
                return self._unroutable(step, unknown)
        self.graph.outputs[step.id] = given
        self.graph.status[step.id] = DONE
        result = self._after_close(step, status=DONE, outputs=given, completed_by=who)
        result["completed_by"] = who
        return result

    def select_branches(
        self, step_id: str, branch_ids: Iterable[str], reason: str
    ) -> dict[str, Any]:
        """``select_branches``: route an active router step by branch id."""
        self.start()
        step = self.playbook.step(_text(step_id))
        if step is None or not step.branches:
            return {"ok": False, "error": f"{step_id!r} is not a step with branches"}
        if step.is_task:
            return {
                "ok": False,
                "error": _by_person([step]) + " What they give routes it.",
                "done_by_person": True,
            }
        if self.graph.status[step.id] not in (ACTIVE, BLOCKED):
            return {
                "ok": False,
                "error": f"step {step.id} is {self.graph.status[step.id]}; route it while active",
            }
        chosen = [_text(b) for b in branch_ids if _text(b)]
        known = [b.id for b in step.branches]
        unknown = [b for b in chosen if b not in known]
        if unknown:
            return {"ok": False, "error": f"unknown branches {unknown}", "branches": known}
        # The data decides whenever it can: a choice against a condition the run can tell
        # is refused. A condition on the step's own outputs waits for them (complete_step).
        verdicts = self.graph.condition_verdicts(step, own_pending=True)
        against = _against(verdicts, chosen)
        if against:
            return {
                "ok": False,
                "error": self._conditions_say(step, verdicts, against)
                + " select_branches cannot choose against the step's conditions: call "
                f"complete_step({step.id!r}, outputs) and they route it.",
                "routed_by": "when",
                "branches_taken": [b for b, v in verdicts.items() if v == TRUE],
                "branches_not_taken": [b for b, v in verdicts.items() if v == FALSE],
            }
        self.graph.selected[step.id] = chosen
        return {
            "ok": True,
            "step": step.id,
            "selected": chosen,
            "waives": [b for b in known if b not in chosen],
            "reason": _text(reason),
            "next": f"Call complete_step({step.id!r}, outputs) to close the step and take "
            "these branches.",
        }

    def _conditions_say(
        self,
        step: Step,
        verdicts: Mapping[str, str],
        against: list[str],
        outputs: Mapping[str, Any] | None = None,
    ) -> str:
        """Why a choice goes against the conditions, in plain words: "This step routes by
        its conditions: the amount is more than $10,000, so it goes to Large payment."
        """
        context = self.graph.context(step, outputs)
        named = {b.id: b for b in step.branches}

        def words(branch_id: str) -> str:
            try:
                return describe_when(parse_when(named[branch_id].when), context)
            except WhenSyntaxError:
                return f"`{named[branch_id].when}`"

        to = [b for b in against if verdicts[b] == TRUE]
        if to:
            said = [f"{words(b)}, so it goes to {named[b].label or b}" for b in to]
        else:
            said = [
                f"{words(b)} does not hold, so it does not go to {named[b].label or b}"
                for b in against
            ]
        return "This step routes by its conditions: " + "; and ".join(said) + "."

    def submit_decision(
        self, outcome_id: str, rationale: str, evidence_refs: Iterable[str]
    ) -> dict[str, Any]:
        """``submit_decision``: conclude the playbook with one outcome."""
        self.start()
        if not self.playbook.outcomes:
            return {"ok": False, "error": "this playbook has no decision policy"}
        if self.decision is not None:
            return {
                "ok": False,
                "error": f"the playbook is already decided ({self.decision['outcome_id']})",
            }
        outcome = self.playbook.outcome(_text(outcome_id))
        if outcome is None:
            return {
                "ok": False,
                "error": f"{outcome_id!r} is not an outcome of this playbook",
                "outcomes": [o.id for o in self.playbook.outcomes],
            }
        if not _text(rationale):
            return {"ok": False, "error": "say why: the rationale is required"}
        converged = self.graph.converged()
        if not converged and not outcome.inconclusive:
            pending = [s.id for s in self.graph.unresolved()]
            self._deviation(
                None,
                "submit_decision",
                "decision_before_convergence",
                blocked=self.enforce,
                reason="steps still open: " + ", ".join(pending),
            )
            if self.enforce:
                return {
                    "ok": False,
                    "error": "the playbook has not converged: finish "
                    + ", ".join(pending)
                    + f" first (or conclude {INCONCLUSIVE} if the evidence cannot be had)",
                }
        refs = [_text(r) for r in evidence_refs if _text(r)]
        # The decision closes the step that asks for it, and only that one: a step that
        # was merely active when the model concluded is waived, not credited.
        deciders = [s for s in self.graph.active() if s.decides]
        for s in deciders:
            self.graph.outputs[s.id] = {"outcome_id": outcome.id}
            self.graph.status[s.id] = DONE
            self._step_event(s.id, outputs={"outcome_id": outcome.id})
        for s in self.graph.unresolved():
            if s.required and not outcome.inconclusive:
                self._deviation(s, "submit_decision", "required_step_skipped")
            self.graph.status[s.id] = WAIVED
            self.graph.waived_because[s.id] = f"the playbook concluded {outcome.id}"
            self._step_event(s.id, reason=self.graph.waived_because[s.id])
        self.decision = {
            "outcome_id": outcome.id,
            "classification": outcome.classification,
            "priority": outcome.priority,
            "action": outcome.action,
            "escalation_target": outcome.escalation_target,
        }
        self._emit(
            {
                "type": DECISION_EVENT,
                "playbook": self.playbook.id,
                "version": PLAYBOOK_V2,
                "step": deciders[0].id if deciders else "",
                "outcome_id": outcome.id,
                "classification": outcome.classification,
                "priority": outcome.priority,
                "rationale": _text(rationale),
                "evidence_refs": refs,
                "action": outcome.action,
                "escalation_target": outcome.escalation_target,
                "recommendation_only": self.playbook.recommendation_only,
                "converged": converged,
                "enforcement": BLOCK if self.enforce else RECORD,
            }
        )
        if self._record is not None:
            self._record(
                DECISION_EVENT,
                {
                    "correlation_id": self._correlation_id,
                    "principal": self._principal,
                    "playbook": self.playbook.id,
                    "playbook_version": self.version,
                    "playbook_digest": self.digest,
                    "outcome_id": outcome.id,
                    "classification": outcome.classification or "",
                    "converged": converged,
                    "rationale_sha256": _sha256(_text(rationale)),
                    "evidence_refs_sha256": _sha256(refs),
                    "steps": dict(self.graph.status),
                    "deviations": list(self.violations),
                },
            )
        return {
            "ok": True,
            "outcome_id": outcome.id,
            "classification": outcome.classification,
            "action": outcome.action,
            "escalation_target": outcome.escalation_target,
            "recommendation_only": self.playbook.recommendation_only,
            "next": "The playbook is concluded. Write your final answer: the outcome, why, and "
            "the evidence it rests on"
            + (
                " -- as a recommendation; do not act on it."
                if self.playbook.recommendation_only
                else "."
            ),
        }

    def finish(self) -> None:
        """Close the record at the end of a run: what was left open, and no decision."""
        if self._finished:
            return
        self._finished = True
        self.start()
        if self.decision is None:
            for s in self.graph.unresolved():
                if s.required:
                    self._deviation(
                        s, "", "required_step_skipped", reason="the run ended before it"
                    )
            if self.playbook.outcomes:
                self._deviation(None, "", "no_decision", reason="the run ended undecided")

    @property
    def outcome(self) -> dict[str, Any] | None:
        """``{outcome_id, classification}`` once decided, for the run's ``done``."""
        if self.decision is None:
            return None
        return {
            "outcome_id": self.decision["outcome_id"],
            "classification": self.decision["classification"],
        }


# ── the control tools, as tools ──────────────────────────────────────────────


def control_tool(name: str, runtime: PlaybookRuntime) -> Any:
    """The SDK tool for one control tool name, bound to ``runtime``."""
    from tulip.tools.decorator import tool

    if name == "complete_step":

        @tool(name="complete_step")
        def complete_step(step_id: str, outputs: dict[str, Any]) -> str:
            """Close an active playbook step. ``outputs`` maps each expected output to its value.

            Each answer the person gave a step's question goes in ``outputs`` too, under the
            question's name. A typed value is written as the step's brief says: money as
            ``{"amount": <number>, "currency": "<ISO 4217 code>"}``, a date as YYYY-MM-DD.

            Call it once the step's work is done. The result says which step is
            active next and gives its full brief, or why the step cannot close yet.
            """
            return json.dumps(runtime.complete_step(step_id, outputs), default=str)

        return complete_step
    if name == "select_branches":

        @tool(name="select_branches")
        def select_branches(step_id: str, branch_ids: list[str], reason: str) -> str:
            """Choose which branches of the active router step to take, by branch id.

            The branches not chosen are waived. Then close the step with complete_step.
            """
            return json.dumps(runtime.select_branches(step_id, branch_ids, reason), default=str)

        return select_branches

    @tool(name="submit_decision")
    def submit_decision(outcome_id: str, rationale: str, evidence_refs: list[str]) -> str:
        """Conclude the playbook with exactly one outcome of its decision policy.

        ``rationale`` says why; ``evidence_refs`` names what it rests on (tool calls,
        records, documents). Submitted once; it is the run's decision.
        """
        return json.dumps(
            runtime.submit_decision(outcome_id, rationale, evidence_refs), default=str
        )

    return submit_decision


__all__ = [
    "ACTIVE",
    "ASK_USER",
    "BLOCK",
    "BLOCKED",
    "CONTROL_TOOLS",
    "DECISION_EVENT",
    "DEVIATION_EVENT",
    "DONE",
    "INCONCLUSIVE",
    "NOT_EXECUTED",
    "PENDING",
    "PLAYBOOK_V2",
    "PROPOSAL_TOOLS",
    "RECORD",
    "RESTORED_CALL",
    "ROUTING_UNKNOWN",
    "STATUSES",
    "STEP_EVENT",
    "TERMINAL",
    "WAIVED",
    "PlaybookRuntime",
    "PlaybookV2",
    "PlaybookV2Error",
    "RestoreError",
    "RestoreResult",
    "StepGraph",
    "TaskRequest",
    "TaskSpec",
    "control_tool",
    "definition_digest",
    "enforcement_mode",
    "forbidden_deny",
    "initial_active",
    "is_playbook_v2",
    "parse_playbook_v2",
    "playbook_prose",
    "skills_by_name",
    "step_brief",
    "validate_inputs",
    "with_forbidden",
]
