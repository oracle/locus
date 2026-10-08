# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Scripted playbook runs whose full trace must not change when the engine moves.

Each scenario drives a :class:`PlaybookRuntime` through a fixed script -- start, admitted
and refused calls (through the hook seam and directly), outputs, branch picks, pauses,
a decision, finish -- and returns everything observable: the events emitted, the audit
records handed over, every call's return value, the final graph and the prose the model
is shown. ``fixtures/golden.json`` holds the trace the gateway's engine produced before
the move (``make_golden.py``); ``test_golden.py`` replays the same scripts on this
package's engine and wants the identical trace.

Engine-agnostic: :class:`Engine` adapts the one seam that differs (the gateway's runtime
takes an ``audit`` sink of pydantic records and reads its enforcement default from the
environment; this package's takes a ``record`` callable and a ``deployment`` argument).
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


FIXTURES = Path(__file__).parent / "fixtures"


def refund_example() -> dict[str, Any]:
    body: dict[str, Any] = yaml.safe_load((FIXTURES / "refund-dispute.v2.yaml").read_text("utf-8"))
    return body


def f19_playbook() -> dict[str, Any]:
    body: dict[str, Any] = json.loads((FIXTURES / "functional-f19.v2.json").read_text("utf-8"))
    return body


F19_LOOK = "functional_p0_pb_look"
F19_WARRANTY = "functional_p0_pb_warranty"
SKILLS = {"refund-checklist": {"name": "refund-checklist", "instructions": "CHECK"}}


@dataclass
class Engine:
    """The engine under test: its module, and how to build a runtime on it."""

    engine: ModuleType
    results: ModuleType
    when: ModuleType
    make: Callable[..., Any]
    mode: Callable[[Any, str], str]


@dataclass
class _Event:
    """A stand-in for the SDK's before/after tool-call events: the fields read."""

    tool_name: str
    result: Any = None
    error: Any = None
    cancel: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, sort_keys=True))


class _Script:
    def __init__(self, eng: Engine, body: dict[str, Any], *, enforce: bool) -> None:
        self.eng = eng
        self.events: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.rt = eng.make(
            eng.engine.parse_playbook_v2(body),
            emit=self.events.append,
            audit=self.audit,
            skills=SKILLS,
            enforce=enforce,
            ref={"version": "7", "digest": "sha256:feed"},
            correlation_id="run-golden",
            principal="gw-golden",
        )

    def op(self, name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self.rt, name)
        out = fn(*args, **kwargs)
        self.ops.append({"op": name, "args": _jsonable([args, kwargs]), "out": _jsonable(out)})
        return out

    def call(self, tool: str, result: Any = '{"ok": true}', *, error: Any = None) -> None:
        """One tool call through the hook seam, as the agent loop makes it."""
        before = _Event(tool_name=tool)
        asyncio.run(self.rt.on_before_tool_call(before))
        self.ops.append({"op": "before", "tool": tool, "cancel": before.cancel})
        if before.cancel:
            return
        after = _Event(tool_name=tool, result=result, error=error)
        asyncio.run(self.rt.on_after_tool_call(after))
        self.ops.append({"op": "after", "tool": tool})

    def trace(self) -> dict[str, Any]:
        graph = self.rt.graph
        return _jsonable(
            {
                "events": self.events,
                "audit": self.audit,
                "ops": self.ops,
                "status": graph.status,
                "violations": self.rt.violations,
                "decision": self.rt.decision,
                "outcome": self.rt.outcome,
                "converged": graph.converged(),
            }
        )


def _refund_triage(s: _Script, branches: list[str]) -> None:
    s.call("lookup_order")
    s.op(
        "complete_step",
        "triage",
        {"dispute_kind": "duplicate", "selected_branch_ids": branches, "customer_statement": "x"},
    )


def scenario_refund_record(eng: Engine) -> dict[str, Any]:
    """The happy path, recording: a duplicate charge, refunded."""
    s = _Script(eng, refund_example(), enforce=False)
    s.op("start")
    s.op("start")  # idempotent
    s.call("get_balance")  # outside triage's allowlist: recorded, not refused
    _refund_triage(s, ["duplicate_charge"])
    s.call("lookup_order")
    s.op("complete_step", "check_duplicates", {"duplicate_charges": ["ch_2"]})
    s.op("submit_decision", "REFUND_DUPLICATE", "charged twice", ["ch_1", "ch_2"])
    s.call("lookup_order")  # after the decision
    s.op("finish")
    return s.trace()


def scenario_refund_block(eng: Engine) -> dict[str, Any]:
    """Blocking: refusals, the effort floor, a premature and an unknown decision."""
    s = _Script(eng, refund_example(), enforce=True)
    s.op("start")
    s.call("get_balance")  # refused
    s.op("submit_decision", "REFUND_DUPLICATE", "too early", [])
    s.op("complete_step", "triage", {"dispute_kind": "x"})  # outputs owed
    s.op("complete_step", "check_amount", {"amount_delta": 1})  # not active
    s.op("select_branches", "triage", ["nope"], "bad id")
    s.op("select_branches", "triage", ["duplicate", "amount"], "both")
    _refund_triage(s, [])
    s.call("lookup_order", "⛔ denied: by policy")  # the gate's refusal is not the step's work
    s.call("lookup_order", error="boom")  # a failed call neither
    s.call("get_balance")
    s.op("complete_step", "check_amount", {"amount_delta": 3})
    s.call("lookup_order")
    s.op("complete_step", "check_duplicates", {"duplicate_charges": []})
    s.op("submit_decision", "NOPE", "unknown", [])
    s.op("submit_decision", "REFUND_DIFFERENCE", "short by 3", ["ch_9"])
    s.op("finish")
    return s.trace()


def scenario_refund_pause_and_observe(eng: Engine) -> dict[str, Any]:
    """A run parked on a person, resumed, with calls performed on resume."""
    s = _Script(eng, refund_example(), enforce=False)
    s.op("start")
    s.op("pause", "awaiting approval")
    s.op("unpause")
    s.op("observe", "lookup_order", arguments={"order_id": "o1"})
    s.op("observe", "lookup_order", not_executed="label-only tool")
    s.call("ask_user")
    s.op(
        "complete_step",
        "triage",
        {
            "dispute_kind": "amount",
            "selected_branch_ids": ["amount_mismatch"],
            "customer_statement": "y",
        },
    )
    s.call("lookup_order", eng.results.not_executed_result("lookup_order", "no body", {}))
    s.op("complete_step", "check_amount", {"amount_delta": 0})
    s.op("finish")  # no decision: what was left open
    return s.trace()


def scenario_f19_damaged(eng: Engine) -> dict[str, Any]:
    """F19's playbook as the live check runs it: damaged, so the warranty is checked."""
    s = _Script(eng, f19_playbook(), enforce=False)
    s.op("start")
    s.call(F19_LOOK, json.dumps({"order_id": "o1", "condition": "damaged"}))
    s.op("complete_step", "inspect", {"condition": "damaged"})
    s.call(F19_WARRANTY, json.dumps({"covered": True}))
    s.op("complete_step", "check_warranty", {"covered": True})
    s.call("functional_p0_pb_refund")  # outside decide's allowlist
    s.op("submit_decision", "REPLACE", "damaged and covered", [F19_LOOK, F19_WARRANTY])
    s.op("finish")
    return s.trace()


def scenario_f19_intact_block(eng: Engine) -> dict[str, Any]:
    """F19's playbook, blocking: intact, so the charge is explained and nothing called."""
    s = _Script(eng, f19_playbook(), enforce=True)
    s.op("start")
    s.call(F19_LOOK, json.dumps({"order_id": "o1", "condition": "intact"}))
    s.call(F19_LOOK)
    s.call(F19_LOOK)  # over max_tool_calls
    s.op("complete_step", "inspect", {"condition": "arrived intact"})
    s.call(F19_WARRANTY)  # the waived branch's tool
    s.op("complete_step", "explain_charge", {"note": "fine"})
    s.op("submit_decision", "INCONCLUSIVE", "could not tell", [])
    s.op("finish")
    return s.trace()


def scenario_static(eng: Engine) -> dict[str, Any]:
    """What is computed without running: digests, deny rules, modes, the model's prose."""
    out: dict[str, Any] = {}
    for name, body in (("refund", refund_example()), ("f19", f19_playbook())):
        e = eng.engine
        pb = e.parse_playbook_v2(body)
        stamped = copy.deepcopy(body)
        stamped.setdefault("metadata", {})["tulip.publish"] = {"at": "now"}
        out[name] = {
            "is_v2": e.is_playbook_v2(body),
            "digest": e.definition_digest(body),
            "digest_stamped": e.definition_digest(stamped),
            "deny": e.forbidden_deny(pb),
            "with_forbidden": e.with_forbidden({"max_blast_radius": 2}, e.forbidden_deny(pb)),
            "mode_record": eng.mode(pb, "record"),
            "mode_block": eng.mode(pb, "block"),
            "initial": e.initial_active(pb),
            "prose": e.playbook_prose(pb, SKILLS, e.initial_active(pb)),
            "prose_block": e.playbook_prose(
                pb,
                SKILLS,
                [s.id for s in pb.steps],
                agent_skills=[{"name": "x"}],
                enforcement="block",
            ),
            "briefs": [e.step_brief(step, SKILLS) for step in pb.steps],
            "skills": sorted(e.skills_by_name([{"name": "a"}, {"id": "b"}, {}])),
        }
    return _jsonable(out)


WHEN_CASES: list[tuple[str, dict[str, Any]]] = [
    ("always", {}),
    (
        "selected_branch_ids contains duplicate_charge",
        {"selected_branch_ids": ["duplicate_charge"]},
    ),
    ("selected_branch_ids contains duplicate_charge", {"selected_branch_ids": []}),
    ("condition contains damaged", {"condition": "arrived damaged"}),
    ("NOT (condition contains damaged)", {"condition": "intact"}),
    (
        "outputs.router.scope == 'CLUSTER-WIDE' AND outputs.router.error_count > 10",
        {"outputs": {"router": {"scope": "CLUSTER-WIDE", "error_count": 11}}},
    ),
    ("outputs.router.error_count > 10", {"outputs": {"router": {}}}),
    ("NOT (findings is empty)", {"findings": []}),
    ("findings is empty", {}),
    ("amount >= 3 OR kind equals refund", {"amount": 2, "kind": "refund"}),
    ("amount < 'x'", {"amount": 2}),
    ("a.b.c != null", {"a": {"b": {"c": 1}}}),
]


def scenario_when(eng: Engine) -> dict[str, Any]:
    """The ``when`` language: parse and evaluate a fixed corpus, and its syntax errors."""
    w = eng.when
    out: list[Any] = []
    for text, ctx in WHEN_CASES:
        node = w.parse_when(text)
        out.append(
            {"when": text, "holds": w.evaluate_when(node, ctx), "paths": sorted(w.when_paths(node))}
        )
    for bad in ("", "a ==", "(a", "a contains", "x" * 5000, "a === b"):
        try:
            w.parse_when(bad)
            out.append({"when": bad[:40], "error": None})
        except w.WhenSyntaxError as exc:
            out.append({"when": bad[:40], "error": str(exc)})
    return _jsonable(out)


SCENARIOS: dict[str, Callable[[Engine], dict[str, Any] | list[Any]]] = {
    "refund_record": scenario_refund_record,
    "refund_block": scenario_refund_block,
    "refund_pause_and_observe": scenario_refund_pause_and_observe,
    "f19_damaged": scenario_f19_damaged,
    "f19_intact_block": scenario_f19_intact_block,
    "static": scenario_static,
    "when": scenario_when,
}


def run_all(eng: Engine) -> dict[str, Any]:
    return {name: fn(eng) for name, fn in SCENARIOS.items()}
