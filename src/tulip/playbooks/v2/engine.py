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
* ``select_branches(step_id, branch_ids, reason)`` routes a router step by branch id
  instead of by its outputs; the branches it does not pick are waived;
* ``submit_decision(outcome_id, rationale, evidence_refs)`` concludes the playbook
  with one of the decision policy's outcomes, and is the run's decision.

Those three are not governed (they change nothing outside the run) and are always
handed to a run that follows a v2 playbook. ``ask_user`` is how a step gets what
only the person knows (``required_from_user``).

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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tulip.hooks.provider import HookPriority, HookProvider
from tulip.playbooks.v2.results import not_executed_reason, not_executed_result, refused
from tulip.playbooks.v2.when import WhenSyntaxError, evaluate_when, parse_when


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

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

#: The registry's ``PUBLISH_STAMP_KEY``, excluded from a definition's digest.
_STAMP_KEY = "tulip.publish"


class PlaybookV2Error(ValueError):
    """A v2 playbook the gateway cannot run as written. The run is refused."""


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

    def allows(self, tool: str) -> bool:
        """Whether this step's allowlist admits ``tool`` (kernel tools aside)."""
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
    inputs: tuple[tuple[str, str], ...] = ()
    decision_policy_id: str = ""
    decision_rules: tuple[str, ...] = ()
    decision_facts: tuple[str, ...] = ()
    outcomes: tuple[Outcome, ...] = ()
    forbidden: tuple[Forbidden, ...] = ()
    all_required_steps_resolved: bool = True
    recommendation_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

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


def _step(raw: Mapping[str, Any], group: str) -> Step:
    step_id = _text(raw.get("id"))
    if not step_id:
        raise PlaybookV2Error(f"a step in group {group!r} has no id")
    allowed = raw.get("allowed_tools")
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
        required_from_user=tuple(
            (_text(r.get("name")), _text(r.get("question")))
            for r in raw.get("required_from_user") or []
            if isinstance(r, dict) and _text(r.get("name"))
        ),
        rules=_strs(raw.get("rules")),
        facts=_strs(raw.get("facts")),
    )


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
    return PlaybookV2(
        id=_text(definition.get("id")) or "playbook",
        title=_text(definition.get("title")) or _text(definition.get("id")) or "Playbook",
        summary=_text(definition.get("summary")),
        mode=_text(definition.get("mode")) or "diagnosis",
        groups=tuple(groups),
        steps=tuple(steps),
        inputs=tuple(
            (_text(i.get("name")), _text(i.get("description")))
            for i in definition.get("inputs") or []
            if isinstance(i, dict) and _text(i.get("name"))
        ),
        decision_policy_id=_text(policy.get("id")),
        decision_rules=_strs(policy.get("rules")),
        decision_facts=_strs(policy.get("facts")),
        outcomes=tuple(sorted(outcomes, key=lambda o: o.priority)),
        forbidden=forbidden,
        all_required_steps_resolved=completion.get("all_required_steps_resolved", True)
        is not False,
        recommendation_only=completion.get("recommendation_only") is True,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
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

    def route(self, step: Step) -> tuple[list[str], list[str], str]:
        """Settle a branching step's routing: ``(taken, not taken, routed_by)``.

        An explicit selection wins; otherwise each branch's ``when`` is evaluated
        against the step's outputs (earlier steps' under ``outputs.<id>``).
        """
        if not step.branches:
            return [], [], ""
        if step.id in self.selected:
            taken = [b.id for b in step.branches if b.id in self.selected[step.id]]
            routed_by = "select_branches"
        else:
            context: dict[str, Any] = {"outputs": dict(self.outputs)}
            context.update(self.outputs.get(step.id, {}))
            taken = [b.id for b in step.branches if evaluate_when(b.when, context)]
            routed_by = "when"
        self.routed[step.id] = {b.next_step_id for b in step.branches if b.id in taken}
        not_taken = [b.id for b in step.branches if b.id not in taken]
        return taken, not_taken, routed_by

    def converged(self) -> bool:
        """Whether a positive decision may be submitted now.

        With a decision step (one whose allowlist names ``submit_decision``): when it
        is active. Without one: when every required step is resolved.
        """
        deciders = self.playbook.decision_steps
        if deciders:
            return any(self.status[s.id] in (ACTIVE, BLOCKED) for s in deciders)
        return all(self.status[s.id] in (DONE, WAIVED) for s in self.playbook.steps if s.required)


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


def step_brief(step: Step, skills: Mapping[str, Mapping[str, Any]]) -> str:
    """Everything the model needs to do one step, its skills' instructions in full."""
    lines = [f"### Step {step.id}: {step.title}" + ("" if step.required else " (optional)")]
    if step.goal:
        lines.append(f"Goal: {step.goal}")
    lines.append(_tools_line(step))
    if step.expected_outputs:
        lines.append(
            "Expected outputs (pass each to complete_step): " + ", ".join(step.expected_outputs)
        )
    if step.required_from_user:
        lines.append("Ask the person with ask_user, unless they already told you:")
        lines.extend(
            f"- {name}: {question or 'ask for ' + name}"
            for name, question in step.required_from_user
        )
    if step.rules:
        lines.append("Rules:")
        lines.extend(f"- {rule}" for rule in step.rules)
    if step.facts:
        lines.append("Facts you may take as given:")
        lines.extend(f"- {fact}" for fact in step.facts)
    if step.branches:
        lines.append(
            "Branches (decided by your outputs, or pick them with select_branches "
            "before complete_step):"
        )
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
    if playbook.inputs:
        lines.append("Inputs:")
        lines.extend(f"- {name}: {desc}" if desc else f"- {name}" for name, desc in playbook.inputs)
    lines.append(
        "## How this run works\n"
        "The runtime tracks this playbook step by step. Work only on the active step(s).\n"
        "- When a step's work is done, call complete_step(step_id, outputs) with every "
        "expected output in `outputs`. Its result tells you which step is active next and "
        "gives you that step's full brief.\n"
        "- A step with branches routes the run: its outputs decide (each branch's `when`), "
        "or call select_branches(step_id, branch_ids, reason) before complete_step.\n"
        + (
            "- Conclude with submit_decision(outcome_id, rationale, evidence_refs): exactly "
            "once, one of the outcomes below. If evidence is missing, the honest outcome is "
            f"{INCONCLUSIVE}.\n"
            if playbook.outcomes
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
    deviated: bool = False


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
        self.violations: list[str] = []

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

    def start(self) -> None:
        """Record every step as pending, then activate the first. Idempotent."""
        if self._started:
            return
        self._started = True
        for s in self.playbook.steps:
            self._step_event(s.id)
        self._settle()

    def brief(self, step_ids: Iterable[str]) -> list[str]:
        return [
            step_brief(step, self._skills)
            for step_id in step_ids
            if (step := self.playbook.step(step_id)) is not None
        ]

    # ── enforcement at the hook seam ─────────────────────────────────────────

    def _owner(self, tool: str) -> Step | None:
        return next((s for s in self.graph.active() if s.allows(tool)), None)

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
        owner = self._owner(tool)
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
        """The run is waiting on a person: its active steps are blocked until it resumes."""
        for s in self.graph.active():
            if self.graph.status[s.id] == ACTIVE:
                self.graph.status[s.id] = BLOCKED
                self._step_event(s.id, reason=reason)

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
        if self.graph.status[step.id] not in (ACTIVE, BLOCKED):
            return {
                "ok": False,
                "error": f"step {step.id} is {self.graph.status[step.id]}, not active",
                "active": self._active_ids(),
            }
        given = dict(outputs or {})
        missing = [name for name in step.expected_outputs if name not in given]
        if missing:
            return {
                "ok": False,
                "error": f"step {step.id} still owes outputs: {', '.join(missing)}",
                "missing": missing,
            }
        work = self._work[step.id]
        unasked = [name for name, _ in step.required_from_user if name not in given]
        if unasked and not work.asked:
            return {
                "ok": False,
                "error": "this step needs answers only the person has: ask them with ask_user "
                "(or pass what they already told you as outputs)",
                "missing": unasked,
            }
        self.graph.outputs[step.id] = given
        if not work.executed and work.unrun:
            reason = "; ".join(f"tool {t} did not execute: {why}" for t, why in work.unrun.items())
            self.graph.status[step.id] = NOT_EXECUTED
            self._step_event(step.id, reason=reason)
            if step.required:
                self._deviation(
                    step, next(iter(work.unrun)), "required_step_not_executed", reason=reason
                )
            return self._after_close(step, status=NOT_EXECUTED, reason=reason)
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
        return self._after_close(step, status=DONE, outputs=given)

    def _after_close(
        self,
        step: Step,
        *,
        status: str,
        reason: str = "",
        outputs: Mapping[str, Any] | None = None,
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
            self._step_event(step.id, outputs=dict(outputs or {}), **routing)
        active = self._settle()
        result: dict[str, Any] = {
            "ok": status == DONE,
            "step": step.id,
            "status": status,
            **({"reason": reason} if reason else {}),
            **routing,
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

    def select_branches(
        self, step_id: str, branch_ids: Iterable[str], reason: str
    ) -> dict[str, Any]:
        """``select_branches``: route an active router step by branch id."""
        self.start()
        step = self.playbook.step(_text(step_id))
        if step is None or not step.branches:
            return {"ok": False, "error": f"{step_id!r} is not a step with branches"}
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
    "STEP_EVENT",
    "TERMINAL",
    "WAIVED",
    "PlaybookRuntime",
    "PlaybookV2",
    "PlaybookV2Error",
    "StepGraph",
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
    "with_forbidden",
]
