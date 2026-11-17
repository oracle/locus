"""Ask it many different things and check what it actually chose.

The router's claim is that the MODEL never invents topology: an extractor
authors a typed `GoalFrame` and the compiler picks a shape from it by rule. A
claim like that is only worth what a spread of goals says about it, so this
asks fourteen different kinds of question and asserts on the choice each time.

What the router selects, and what it does not, is worth stating because they
are easy to conflate:

* the PROTOCOL SHAPE — yes, deterministically from the frame;
* the TOOLS — yes, by capability, filtered to what a shape requires;
* the SKILLS — by domain, from whatever `SkillIndex` it is handed;
* the AGENT — no. That is the meta-agent's delegation, a different mechanism;
* the PLAYBOOK — no. Nothing selects one; it is static on the agent definition.

"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from locus.router.capability import CapabilityIndex
from locus.router.compiler import CognitiveCompiler
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.policy import PolicyDeniedError, PolicyGate
from locus.router.protocol import ProtocolRegistry, _rank_key, builtin_protocols
from locus.tools.decorator import tool
from locus.tools.registry import create_registry


class _Model:
    """A model that is never called: compiling a shape must not need one."""

    name = "stub"

    async def complete(self, *_a: object, **_k: object) -> object:  # pragma: no cover
        raise AssertionError("compiling a protocol must not call the model")

    async def stream(self, *_a: object, **_k: object) -> object:  # pragma: no cover
        raise AssertionError("compiling a protocol must not call the model")
        yield


@tool
def look_up(query: str) -> str:
    """Look something up."""
    return query


@tool
def run_code(code: str) -> str:
    """Execute code in a sandbox."""
    return code


@tool
def restart_service(name: str) -> str:
    """Restart a service."""
    return name


def _capabilities() -> CapabilityIndex:
    tools = create_registry()
    for fn in (look_up, run_code, restart_service):
        tools.register(fn)
    return CapabilityIndex(tools)


def _compiler(policy: PolicyGate | None = None, **kw: object) -> CognitiveCompiler:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    return CognitiveCompiler(
        protocols=registry,
        capabilities=_capabilities(),
        policy=policy or PolicyGate(),
        model=_Model(),
        **kw,  # type: ignore[arg-type]
    )


def _registry() -> ProtocolRegistry:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    return registry


def _frame(**kw: object) -> GoalFrame:
    base: dict[str, object] = {
        "primary_goal": TaskType.ANSWER,
        "domain": "general",
        "complexity": Complexity.LOW,
        "risk": Risk.LOW,
    }
    base.update(kw)
    return GoalFrame(**base)  # type: ignore[arg-type]


def _shape(runnable: Any) -> str | None:
    """The protocol id a compiled runnable carries, through an approval wrapper."""
    direct = getattr(runnable, "protocol_id", None)
    if direct is not None:
        return str(direct)
    inner = getattr(runnable, "inner", None)
    return None if inner is None else str(inner.protocol_id)


#: (what a person asked, the frame an extractor would author for it).
#: The point is the SPREAD — a router that answers everything with one shape
#: passes any single case.
ASKS: list[tuple[str, GoalFrame]] = [
    ("what is our refund policy?", _frame(primary_goal=TaskType.ANSWER)),
    ("explain why this charge failed", _frame(primary_goal=TaskType.EXPLAIN)),
    (
        "plan the migration off the old billing system",
        _frame(primary_goal=TaskType.PLAN, complexity=Complexity.HIGH),
    ),
    (
        "write a script that reconciles payouts",
        _frame(primary_goal=TaskType.GENERATE_CODE, requires_code_generation=True),
    ),
    (
        "build me a tool that scrapes our status page",
        _frame(primary_goal=TaskType.BUILD, requires_code_generation=True),
    ),
    (
        "why is the database slow?",
        _frame(primary_goal=TaskType.DIAGNOSE, complexity=Complexity.HIGH),
    ),
    (
        "restart the payments service",
        _frame(primary_goal=TaskType.REMEDIATE, risk=Risk.HIGH, approval_required=True),
    ),
    (
        "refund this customer £1,200",
        _frame(primary_goal=TaskType.REMEDIATE, risk=Risk.HIGH, approval_required=True),
    ),
    (
        "research what our competitors charge",
        _frame(primary_goal=TaskType.RESEARCH, complexity=Complexity.MEDIUM),
    ),
    (
        "compare these two vendors for us",
        _frame(primary_goal=TaskType.COMPARE, complexity=Complexity.MEDIUM),
    ),
    (
        "should we ship on Friday? argue both sides",
        _frame(primary_goal=TaskType.COMPARE, complexity=Complexity.HIGH),
    ),
    (
        "coordinate with the billing agent to close this",
        _frame(primary_goal=TaskType.COORDINATE, requires_multi_agent=True),
    ),
    (
        "escalate this to whoever owns it",
        _frame(primary_goal=TaskType.ESCALATE),
    ),
    (
        "watch the error rate and tell me if it moves",
        _frame(primary_goal=TaskType.MONITOR),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("ask", "frame"), ASKS, ids=[a for a, _ in ASKS])
async def test_every_kind_of_ask_compiles_to_something_runnable(ask: str, frame: GoalFrame) -> None:
    """No question type falls through the registry with nothing to run."""
    runnable = await _compiler().compile(frame)
    assert runnable is not None, f"nothing compiled for: {ask}"


@pytest.mark.asyncio
async def test_the_shape_actually_varies_with_the_ask() -> None:
    """A router that answers everything the same way is not routing.

    This is the assertion the whole feature rests on, and the one a single
    happy-path case cannot make.
    """
    shapes = set()
    for _ask, frame in ASKS:
        runnable = await _compiler().compile(frame)
        shapes.add(getattr(runnable, "protocol_id", None) or type(runnable).__name__)
    assert len(shapes) > 1, f"every ask compiled to the same thing: {shapes}"


@pytest.mark.asyncio
async def test_a_simple_question_does_not_get_a_committee() -> None:
    """Cheap asks must stay cheap — the cost of a shape is a real cost."""
    simple = await _compiler().compile(_frame(primary_goal=TaskType.ANSWER))
    shape = getattr(simple, "protocol_id", "")
    assert shape not in ("debate", "specialist_fanout"), (
        f"'what is our refund policy?' routed to {shape}"
    )


@pytest.mark.asyncio
async def test_a_risky_ask_is_gated_before_anything_runs() -> None:
    """The shape itself is policy-gated, not just the tool calls inside it."""
    compiler = _compiler(policy=PolicyGate(max_risk=Risk.LOW))

    with pytest.raises(Exception) as caught:  # noqa: PT011 — the type is the point
        await compiler.compile(
            _frame(primary_goal=TaskType.REMEDIATE, risk=Risk.HIGH, approval_required=True)
        )
    assert "risk" in str(caught.value).lower() or "polic" in str(caught.value).lower()


@pytest.mark.asyncio
async def test_the_choices_that_are_clearly_right_stay_right() -> None:
    """The mapping a person would defend in a review, pinned.

    Not every ask has an obviously correct shape, so this covers only the ones
    where a wrong answer would be plainly wrong — a question answered by a
    committee, or a multi-step code request that never reaches the code loop.
    """
    expected = [
        (TaskType.ANSWER, Complexity.LOW, "direct_response"),
        (TaskType.EXPLAIN, Complexity.LOW, "direct_response"),
        (TaskType.PLAN, Complexity.HIGH, "plan_execute_validate"),
        # Flipped deliberately (#342): a LOW-complexity frame compiles to
        # `direct_response` before the goal gate, whatever its verb, so a
        # one-step code request ("write the one-liner that…") is one reply.
        # The row below keeps the code loop pinned for the request it exists
        # for, the multi-step one.
        (TaskType.GENERATE_CODE, Complexity.LOW, "direct_response"),
        (TaskType.GENERATE_CODE, Complexity.MEDIUM, "codegen_test_validate"),
        (TaskType.DIAGNOSE, Complexity.HIGH, "specialist_fanout"),
    ]
    for task, complexity, shape in expected:
        frame = _frame(
            primary_goal=task,
            complexity=complexity,
            requires_code_generation=task is TaskType.GENERATE_CODE,
        )
        runnable = await _compiler().compile(frame)
        assert _shape(runnable) == shape, (
            f"{task.value}/{complexity.value} routed to {_shape(runnable)}, expected {shape}"
        )


@pytest.mark.asyncio
async def test_a_risky_ask_becomes_an_approval_before_it_becomes_a_shape() -> None:
    """Risk is resolved first: the person is asked before any topology runs."""
    runnable = await _compiler().compile(
        _frame(primary_goal=TaskType.REMEDIATE, risk=Risk.HIGH, approval_required=True)
    )
    assert type(runnable).__name__ == "_ApprovalRunnable"


@pytest.mark.asyncio
async def test_asking_to_BUILD_something_does_not_reach_the_code_loop() -> None:
    """Documented, not endorsed — this looks wrong and I have not confirmed intent.

    "build me a tool that scrapes our status page" carries
    `requires_code_generation=True` and still compiles to
    `plan_execute_validate` rather than `codegen_test_validate`. Either BUILD
    means "produce a plan for building" — in which case the flag is misleading —
    or the code loop is being missed for exactly the request it exists to serve.

    Asserted as-is so the behaviour is visible and a deliberate change to it
    breaks this test rather than passing silently. At MEDIUM complexity: a
    LOW-complexity BUILD is one reply and takes the short-circuit to
    `direct_response`, which says nothing about BUILD versus the code loop.
    """
    runnable = await _compiler().compile(
        _frame(
            primary_goal=TaskType.BUILD,
            complexity=Complexity.MEDIUM,
            requires_code_generation=True,
        )
    )
    assert getattr(runnable, "protocol_id", None) == "plan_execute_validate"


# ---------------------------------------------------------------------------
# `direct_response` handles DIAGNOSE (#342) — ranked by rule alone, with no
# short-circuit in play, so the `handles` row is pinned on its own.
# ---------------------------------------------------------------------------


def test_a_one_step_diagnosis_ranks_the_direct_answer_first() -> None:
    """ "Why did this alert fire?" at LOW complexity is one reply, not a committee."""
    registry = _registry()
    frame = _frame(primary_goal=TaskType.DIAGNOSE)
    ranked = sorted(registry.filter_candidates(frame), key=lambda p: _rank_key(p, frame))
    assert [p.id for p in ranked] == ["direct_response", "specialist_fanout"]
    assert registry.select(frame).id == "direct_response"


@pytest.mark.parametrize("complexity", [Complexity.MEDIUM, Complexity.HIGH])
def test_a_harder_diagnosis_still_fans_out(complexity: Complexity) -> None:
    """The canonical shape keeps every diagnosis that is more than one step."""
    chosen = _registry().select(_frame(primary_goal=TaskType.DIAGNOSE, complexity=complexity))
    assert chosen.id == "specialist_fanout"


# ---------------------------------------------------------------------------
# The LOW-complexity short-circuit (#342): one step of work is one reply,
# whatever verb the extractor framed it with.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("task", list(TaskType), ids=[t.value for t in TaskType])
async def test_a_one_step_ask_is_answered_directly_whatever_its_verb(task: TaskType) -> None:
    compiler = _compiler()
    frame = _frame(primary_goal=task)
    runnable = await compiler.compile(frame)
    assert _shape(runnable) == "direct_response", f"{task.value} routed to {_shape(runnable)}"

    explanation = compiler.explain(frame)
    assert explanation.selected_protocol == "direct_response"
    assert explanation.method == "low_complexity_direct"
    assert [c.protocol_id for c in explanation.candidates] == ["direct_response"]


@pytest.mark.asyncio
async def test_the_short_circuit_names_itself_on_the_selection_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = AsyncMock()
    monkeypatch.setattr("locus.router.compiler.emit_protocol_selected", selected)
    await _compiler().compile(_frame(primary_goal=TaskType.GENERATE_CODE), run_id="r1")
    assert selected.await_count == 1
    assert selected.await_args.kwargs == {"method": "low_complexity_direct", "rationale": None}


@pytest.mark.asyncio
async def test_the_short_circuit_needs_low_complexity() -> None:
    """MEDIUM is more than one step: the ordinary filter-and-rank path runs."""
    compiler = _compiler()
    frame = _frame(primary_goal=TaskType.GENERATE_CODE, complexity=Complexity.MEDIUM)
    assert _shape(await compiler.compile(frame)) == "codegen_test_validate"
    assert compiler.explain(frame).method == "rule_based"


@pytest.mark.asyncio
async def test_the_short_circuit_respects_the_direct_protocols_risk_cap() -> None:
    """`direct_response` caps at LOW risk; a MEDIUM-risk one-step plan is not its to answer."""
    compiler = _compiler()
    frame = _frame(primary_goal=TaskType.PLAN, risk=Risk.MEDIUM)
    assert _shape(await compiler.compile(frame)) == "plan_execute_validate"
    assert compiler.explain(frame).method == "rule_based"


@pytest.mark.asyncio
async def test_a_denied_direct_protocol_does_not_short_circuit() -> None:
    """A deployment that bans the shape gets the ordinary path, not a refusal."""
    compiler = _compiler(policy=PolicyGate(denied_protocols={"direct_response"}))
    frame = _frame(primary_goal=TaskType.GENERATE_CODE)
    assert _shape(await compiler.compile(frame)) == "codegen_test_validate"
    assert compiler.explain(frame).method == "rule_based"
    # A goal only the denied shape handles is refused by the gate, as before.
    with pytest.raises(PolicyDeniedError, match="denied by policy"):
        await compiler.compile(_frame(primary_goal=TaskType.ANSWER))


@pytest.mark.asyncio
async def test_a_direct_protocol_missing_a_capability_does_not_short_circuit() -> None:
    direct = next(p for p in builtin_protocols() if p.id == "direct_response")
    registry = ProtocolRegistry()
    registry.register(direct.model_copy(update={"requires_capabilities": ["crystal_ball"]}))
    registry.register_many([p for p in builtin_protocols() if p.id != "direct_response"])
    compiler = CognitiveCompiler(
        protocols=registry, capabilities=_capabilities(), policy=PolicyGate(), model=_Model()
    )
    assert _shape(await compiler.compile(_frame(primary_goal=TaskType.GENERATE_CODE))) == (
        "codegen_test_validate"
    )


@pytest.mark.asyncio
async def test_the_short_circuit_still_runs_the_policy_gate() -> None:
    """Resolving the shape early does not skip the gate: a held ask stays held."""
    runnable = await _compiler().compile(
        _frame(primary_goal=TaskType.ANSWER, approval_required=True)
    )
    assert type(runnable).__name__ == "_ApprovalRunnable"
    assert _shape(runnable) == "direct_response"


@pytest.mark.asyncio
async def test_the_short_circuit_can_be_switched_off() -> None:
    compiler = _compiler(low_complexity_direct=None)
    frame = _frame(primary_goal=TaskType.GENERATE_CODE)
    assert _shape(await compiler.compile(frame)) == "codegen_test_validate"
    assert compiler.explain(frame).method == "rule_based"


def test_explain_accounts_for_every_protocol_under_the_short_circuit() -> None:
    """No protocol vanishes: the ones the gates would have stopped say which gate,
    the ones that would have survived say they were never considered."""
    compiler = _compiler()
    explanation = compiler.explain(_frame(primary_goal=TaskType.GENERATE_CODE))

    accounted = {c.protocol_id for c in explanation.candidates} | {
        r.protocol_id for r in explanation.rejected
    }
    assert accounted == {p.id for p in compiler.protocols.all()}
    by_id = {r.protocol_id: r for r in explanation.rejected}
    assert by_id["codegen_test_validate"].gate == "complexity"
    assert "before the goal gate" in by_id["codegen_test_validate"].reason
    assert by_id["debate"].gate == "goal"
    assert explanation.allowed
    assert not explanation.requires_approval
