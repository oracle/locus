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

See REVIEW-what-optic-has-that-tulip-lost.md for the last of those.
"""

from __future__ import annotations

import pytest

from tulip.router.capability import CapabilityIndex
from tulip.router.compiler import CognitiveCompiler
from tulip.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from tulip.router.policy import PolicyGate
from tulip.router.protocol import ProtocolRegistry, builtin_protocols
from tulip.tools.decorator import tool
from tulip.tools.registry import create_registry


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


def _frame(**kw: object) -> GoalFrame:
    base: dict[str, object] = {
        "primary_goal": TaskType.ANSWER,
        "domain": "general",
        "complexity": Complexity.LOW,
        "risk": Risk.LOW,
    }
    base.update(kw)
    return GoalFrame(**base)  # type: ignore[arg-type]


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
    committee, or a code request that never reaches the code loop.
    """
    expected = {
        TaskType.ANSWER: "direct_response",
        TaskType.EXPLAIN: "direct_response",
        TaskType.PLAN: "plan_execute_validate",
        TaskType.GENERATE_CODE: "codegen_test_validate",
        TaskType.DIAGNOSE: "specialist_fanout",
    }
    for task, shape in expected.items():
        frame = _frame(
            primary_goal=task,
            complexity=Complexity.HIGH
            if task in (TaskType.PLAN, TaskType.DIAGNOSE)
            else Complexity.LOW,
            requires_code_generation=task is TaskType.GENERATE_CODE,
        )
        runnable = await _compiler().compile(frame)
        assert getattr(runnable, "protocol_id", None) == shape, (
            f"{task.value} routed to {getattr(runnable, 'protocol_id', None)}, expected {shape}"
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
    breaks this test rather than passing silently.
    """
    runnable = await _compiler().compile(
        _frame(primary_goal=TaskType.BUILD, requires_code_generation=True)
    )
    assert getattr(runnable, "protocol_id", None) == "plan_execute_validate"
