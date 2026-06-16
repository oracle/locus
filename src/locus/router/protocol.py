# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Protocol model + registry + the v1 built-in builders.

A :class:`Protocol` is a declarative orchestration shape. The registry
filters protocols against a :class:`~locus.router.goal_frame.GoalFrame`
deterministically — the LLM never picks a protocol, it just produces
the frame.

Three builders ship in v1, covering the cardinal shapes:

- ``direct_response``  → :class:`~locus.Agent` (single call)
- ``plan_execute_validate`` → :class:`~locus.SequentialPipeline`
- ``specialist_fanout`` → :class:`~locus.Orchestrator`

The other five builders sketched in the design plan ship in follow-up
branches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from locus.router.capability import Capability
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.runnable import (
    Runnable,
    wrap_a2a,
    wrap_agent,
    wrap_debate,
    wrap_pipeline,
)


if TYPE_CHECKING:
    from locus.router.capability import CapabilityIndex


class NoMatchingProtocolError(LookupError):
    """Raised by :meth:`ProtocolRegistry.select` when no protocol fits the frame."""


class BuilderContext(BaseModel):
    """Bundle of inputs passed to every protocol builder.

    Builders need the model, the tool-resolution view, and the skill
    catalogue; passing a typed context keeps the builder signature
    stable as we add follow-up protocols.
    """

    model: Any = Field(..., description="A locus model instance or model string.")
    capabilities: Any = Field(
        ...,
        description="The CapabilityIndex used to resolve Capability -> Tool.",
    )
    skills: Any = Field(
        default=None,
        description=(
            "Optional SkillIndex. Builders attach a SkillsPlugin to every "
            "emitted Agent, scoped to the frame's domain."
        ),
    )
    a2a_endpoint: str | None = Field(
        default=None,
        description=(
            "Optional remote A2A URL. Required by the ``a2a_delegate`` "
            "builder; absent endpoints simply make that protocol "
            "unselectable when the policy gate runs."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}


Builder = Callable[[GoalFrame, list[Capability], BuilderContext], Runnable]


class Protocol(BaseModel):
    """Declarative description of an orchestration shape."""

    id: str
    description: str
    handles: list[TaskType]
    primary_for: list[TaskType] = Field(
        default_factory=list,
        description=(
            "Task types this protocol is the *canonical* choice for "
            "(strict subset of ``handles``). When the registry has to "
            "pick between two protocols that both ``handle`` the same "
            "primary_goal, the one that lists it in ``primary_for`` "
            "wins. Empty list = never canonical (an opt-in protocol)."
        ),
    )
    requires_capabilities: list[str] = Field(default_factory=list)
    risk_max: Risk = Field(default=Risk.MEDIUM)
    cost: Literal["low", "medium", "high"] = "medium"
    latency: Literal["low", "medium", "high"] = "medium"
    supports_streaming: bool = True
    supports_repair: bool = True
    builder: Builder

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


class ProtocolRegistry:
    """Deterministic mapping from :class:`GoalFrame` to :class:`Protocol`."""

    def __init__(self) -> None:
        self._protocols: dict[str, Protocol] = {}

    def register(self, protocol: Protocol) -> None:
        if protocol.id in self._protocols:
            raise ValueError(f"Protocol already registered: {protocol.id!r}")
        self._protocols[protocol.id] = protocol

    def register_many(self, protocols: list[Protocol]) -> None:
        for p in protocols:
            self.register(p)

    def get(self, protocol_id: str) -> Protocol:
        if protocol_id not in self._protocols:
            available = sorted(self._protocols.keys())
            raise KeyError(
                f"Unknown protocol id {protocol_id!r}. Available: {available}",
            )
        return self._protocols[protocol_id]

    def all(self) -> list[Protocol]:
        return list(self._protocols.values())

    def filter_candidates(
        self,
        frame: GoalFrame,
        *,
        available_capabilities: set[str] | None = None,
    ) -> list[Protocol]:
        """Return every protocol that passes the three gates.

        The shared filter for both ``select()`` (rule-based ranker) and
        any opt-in selector that wants to choose among already-qualified
        candidates. Gates:

        - ``frame.primary_goal in p.handles``
        - ``p.risk_max >= frame.risk``
        - ``set(p.requires_capabilities).issubset(available_capabilities)``

        Returns the survivors in registration order. Empty list is a
        valid result — callers raise :class:`NoMatchingProtocolError`
        when appropriate (this method is filter-only by design).
        """
        caps_available = available_capabilities or set()
        return [
            p
            for p in self._protocols.values()
            if frame.primary_goal in p.handles
            and p.risk_max >= frame.risk
            and set(p.requires_capabilities).issubset(caps_available)
        ]

    def select(
        self,
        frame: GoalFrame,
        *,
        available_capabilities: set[str] | None = None,
    ) -> Protocol:
        """Pick the best protocol for ``frame``.

        Filters by ``handles ∋ primary_goal`` ∧ ``risk_max ≥ frame.risk``
        ∧ ``required_capabilities ⊆ available_capabilities``, then ranks
        candidates by complexity-fit + cost.
        """
        if not self._protocols:
            raise NoMatchingProtocolError("No protocols registered.")

        caps_available = available_capabilities or set()
        candidates = self.filter_candidates(frame, available_capabilities=caps_available)
        if not candidates:
            raise NoMatchingProtocolError(
                f"No protocol handles primary_goal={frame.primary_goal!r} "
                f"at risk={frame.risk!r} with capabilities={sorted(caps_available)}.",
            )

        return min(candidates, key=lambda p: _rank_key(p, frame))


def _rank_key(protocol: Protocol, frame: GoalFrame) -> tuple[int, int, int, int]:
    """Lower is better. Ranks are layered:

    1. ``distance``: how close the protocol's cost matches the frame's
       complexity (0 = perfect match). The frame's complexity is the
       *first* signal — a LOW-complexity task should never get a
       HIGH-cost protocol just because that protocol claims to be
       "canonical" for the goal type.
    2. ``canonical``: 0 if the protocol declares this primary_goal in
       ``primary_for``, else 1. Breaks distance ties — when two
       protocols both fit the complexity, the one designed for this
       specific goal wins.
    3. ``cost_rank``: lower-cost protocols win at the next tier.
    4. ``handles``: fewer ``handles`` wins as a final tiebreaker (more
       specific protocol). Rarely activates.
    """
    cost_rank = {"low": 0, "medium": 1, "high": 2}
    complexity_rank = {Complexity.LOW: 0, Complexity.MEDIUM: 1, Complexity.HIGH: 2}
    distance = abs(cost_rank[protocol.cost] - complexity_rank[frame.complexity])
    canonical = 0 if frame.primary_goal in protocol.primary_for else 1
    return (distance, canonical, cost_rank[protocol.cost], len(protocol.handles))


# ---------------------------------------------------------------------------
# v1 builders
# ---------------------------------------------------------------------------


def _resolve_tools(capabilities: list[Capability], capability_index: CapabilityIndex) -> list[Any]:
    """Return real Tool objects for every non-human capability."""
    tools: list[Any] = []
    for cap in capabilities:
        if cap.is_human:
            continue
        tools.append(capability_index.resolve_tool(cap))
    return tools


def _domain_skill_plugins(frame: GoalFrame, ctx: BuilderContext) -> list[Any]:
    """Build the plugin list each Agent should receive for ``frame.domain``.

    Returns ``[]`` when no SkillIndex is wired or the index has no skills
    matching the frame's domain — keeping the v1 path that needs no
    skills exactly as it was. When skills *are* registered for the
    domain, the agent loop's progressive-disclosure flow takes over: L1
    catalogue in the system prompt, L2 activation via the ``skills``
    tool, L3 resource files on demand.
    """
    if ctx.skills is None:
        return []
    matching = ctx.skills.for_domain(frame.domain)
    if not matching:
        return []
    from locus.skills.plugin import SkillsPlugin

    return [SkillsPlugin(skills=matching)]


def _build_direct_response(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Single-agent builder. Used for ANSWER / EXPLAIN / RESEARCH."""
    from locus.agent.agent import Agent

    tools = _resolve_tools(capabilities, ctx.capabilities)
    plugins = _domain_skill_plugins(frame, ctx)
    system_prompt = (
        f"You are answering a {frame.primary_goal.value} request in the "
        f"{frame.domain} domain. Be concise and direct. "
        f"Success criteria: {frame.success_criteria or ['the user gets a useful answer']}."
    )
    agent = Agent(model=ctx.model, tools=tools, system_prompt=system_prompt, plugins=plugins)
    return wrap_agent(agent, "direct_response", frame)


def _build_plan_execute_validate(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Three-stage SequentialPipeline: planner → executor → validator."""
    from locus.agent.agent import Agent
    from locus.agent.composition import SequentialPipeline

    tools = _resolve_tools(capabilities, ctx.capabilities)
    plugins = _domain_skill_plugins(frame, ctx)
    success_str = "; ".join(frame.success_criteria) or "(none specified)"

    planner = Agent(
        model=ctx.model,
        tools=[],
        system_prompt=(
            f"You are the planner stage for a {frame.primary_goal.value} task in "
            f"the {frame.domain} domain. Produce a concrete numbered plan. "
            f"Success criteria: {success_str}."
        ),
        plugins=plugins,
    )
    executor = Agent(
        model=ctx.model,
        tools=tools,
        system_prompt=(
            f"You are the executor stage. Carry out the plan above using available "
            f"tools. Domain: {frame.domain}. Be decisive."
        ),
        plugins=plugins,
    )
    validator = Agent(
        model=ctx.model,
        tools=[],
        system_prompt=(
            f"You are the validator stage. Compare the executor's output against the "
            f"success criteria: {success_str}. State 'PASS' or 'FAIL: <reason>' on the "
            f"first line, then summarize."
        ),
        plugins=plugins,
    )
    pipeline = SequentialPipeline(agents=[planner, executor, validator])
    return wrap_pipeline(pipeline, "plan_execute_validate", frame)


def _build_specialist_fanout(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Fan out to one tool-bound :class:`Agent` per capability.

    We use :class:`ParallelPipeline` rather than
    :class:`Orchestrator` + :class:`Specialist` because ``Specialist``
    does a single-turn ``model.complete()`` (it sends tool schemas but
    doesn't loop on tool calls), so models that respond with "I'll do
    X" never actually invoke their bound tool. ``Agent`` runs the full
    locus agent loop and processes tool calls properly. The outputs
    from each capability-bound agent are concatenated; correlation
    across specialists is a follow-up.
    """
    from locus.agent.agent import Agent
    from locus.agent.composition import ParallelPipeline

    plugins = _domain_skill_plugins(frame, ctx)
    agents: list[Agent] = []
    for cap in capabilities:
        if cap.is_human:
            continue
        tool_obj = ctx.capabilities.resolve_tool(cap)
        agents.append(
            Agent(
                model=ctx.model,
                tools=[tool_obj],
                system_prompt=(
                    f"You are the {cap.id} specialist for the {cap.domain} "
                    f"domain. You have exactly one tool: {tool_obj.name}.\n\n"
                    f"PROTOCOL — strict, no exceptions:\n"
                    f"1. Your FIRST action is to call {tool_obj.name}. Do "
                    f"NOT write any text before the tool call. Do NOT say "
                    f"'I will' or 'let me' — just call the tool.\n"
                    f"2. After the tool returns, your SECOND action is to "
                    f"summarise the tool's output in 1-2 sentences. Then "
                    f"stop.\n"
                    f"3. If you find yourself wanting to answer from memory, "
                    f"call {tool_obj.name} instead. The tool is the only "
                    f"trustworthy source for {cap.domain} data."
                ),
                plugins=plugins,
            ),
        )
    pipeline = ParallelPipeline(agents=agents, merge_strategy="concatenate")
    return wrap_pipeline(pipeline, "specialist_fanout", frame)


def _build_debate(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Two debaters argue, a judge picks the winner.

    The pattern is a fan-out (:class:`ParallelPipeline`) of two
    differently-instructed debater agents on the same prompt, followed
    by a judge :class:`Agent` that reads the joined transcript and
    rules. Both stages get any domain-scoped skills attached.
    """
    from locus.agent.agent import Agent
    from locus.agent.composition import ParallelPipeline

    tools = _resolve_tools(capabilities, ctx.capabilities)
    plugins = _domain_skill_plugins(frame, ctx)

    pro = Agent(
        model=ctx.model,
        tools=tools,
        system_prompt=(
            "You are Debater A. Argue strongly *for* the proposition implied by "
            "the user's question. Cite at least two concrete reasons. Label your "
            "answer 'A:' on the first line."
        ),
        plugins=plugins,
    )
    con = Agent(
        model=ctx.model,
        tools=tools,
        system_prompt=(
            "You are Debater B. Argue strongly *against* the proposition implied "
            "by the user's question. Cite at least two concrete reasons. Label "
            "your answer 'B:' on the first line."
        ),
        plugins=plugins,
    )
    judge = Agent(
        model=ctx.model,
        tools=[],
        system_prompt=(
            "You are an impartial judge. Read both debater transcripts and pick "
            "the stronger argument on the merits. Be terse and end with: "
            "'WINNER: A | B | inconclusive'."
        ),
        plugins=plugins,
    )
    debaters = ParallelPipeline(agents=[pro, con], merge_strategy="concatenate")
    return wrap_debate(debaters, judge, "debate", frame)


def _build_codegen_test_validate(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Loop a tool-using :class:`Agent` until output declares ``PASS``.

    Wraps :class:`LoopAgent`. The agent has the resolved capability
    tools (typically a code-runner / test-runner), and the loop
    condition checks for ``PASS`` on the first output line.
    """
    from locus.agent.agent import Agent
    from locus.agent.composition import LoopAgent

    tools = _resolve_tools(capabilities, ctx.capabilities)
    plugins = _domain_skill_plugins(frame, ctx)

    coder = Agent(
        model=ctx.model,
        tools=tools,
        system_prompt=(
            "You are a code-generate-and-test loop. Each iteration:\n"
            "1. Produce or revise the code to satisfy the request.\n"
            "2. Use available tools to run it / its tests.\n"
            "3. On the very first line, write 'PASS' if all tests passed and the "
            "spec is met, else 'FAIL: <one-line reason>'.\n"
            "Keep iterating until you can write PASS."
        ),
        plugins=plugins,
    )
    loop = LoopAgent(
        agent=coder,
        condition=lambda output: output.strip().upper().startswith("PASS"),
        max_loops=4,
        loop_prompt=(
            "Previous attempt's output:\n{previous_output}\n\n"
            "Original task: {task}\n\n"
            "Fix the failures and re-run. First line must be PASS or FAIL."
        ),
    )
    return wrap_pipeline(loop, "codegen_test_validate", frame)


def _build_approval_gated_execution(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Single-agent execution with a forced approval interrupt.

    Same shape as :func:`_build_direct_response` (so the inspectable
    graph is one :class:`Agent`), but the protocol's ``handles`` /
    ``risk_max`` / surrounding policy gate force an approval step
    before the agent runs. The forcing happens in the compiler when
    ``approval_required=True`` on the frame, so the policy gate sees
    ``require_approval=True`` and wraps the runnable.
    """
    from locus.agent.agent import Agent

    tools = _resolve_tools(capabilities, ctx.capabilities)
    plugins = _domain_skill_plugins(frame, ctx)
    agent = Agent(
        model=ctx.model,
        tools=tools,
        system_prompt=(
            f"You are executing a {frame.primary_goal.value} request in the "
            f"{frame.domain} domain. This action requires explicit human "
            "approval (already obtained before you ran). Be precise and "
            "report exactly what you changed."
        ),
        plugins=plugins,
    )
    return wrap_agent(agent, "approval_gated_execution", frame)


def _build_a2a_delegate(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Forward the task to a remote agent over the A2A protocol.

    The remote endpoint comes from ``BuilderContext.a2a_endpoint``; if
    none is configured, this builder raises :class:`RuntimeError` so
    the protocol effectively isn't selectable. Use the
    :class:`ProtocolRegistry`-level ``requires_capabilities`` mechanism
    to gate it on caller-known tags if you want fancier control.
    """
    from locus.a2a.protocol import A2AClient

    if not ctx.a2a_endpoint:
        raise RuntimeError(
            "a2a_delegate protocol requires a BuilderContext.a2a_endpoint",
        )
    # capabilities are intentionally ignored — the remote agent owns its
    # own tool surface. Listed for forward compatibility (e.g. capability
    # negotiation against the remote agent card).
    _ = capabilities
    client = A2AClient(url=ctx.a2a_endpoint)
    return wrap_a2a(client, "a2a_delegate", frame)


def _build_handoff_chain(
    frame: GoalFrame, capabilities: list[Capability], ctx: BuilderContext
) -> Runnable:
    """Sequential chain — each agent gets one capability and forwards.

    Uses :class:`SequentialPipeline` of real :class:`Agent` instances
    rather than locus's native :class:`HandoffAgent` /
    :meth:`Handoff.chain_handoff` because those run a single-turn
    ``model.complete()`` and therefore don't process tool calls (same
    limitation surfaced by :class:`Specialist`).
    """
    from locus.agent.agent import Agent
    from locus.agent.composition import SequentialPipeline

    plugins = _domain_skill_plugins(frame, ctx)
    agents: list[Agent] = []
    for cap in capabilities:
        if cap.is_human:
            continue
        tool_obj = ctx.capabilities.resolve_tool(cap)
        agents.append(
            Agent(
                model=ctx.model,
                tools=[tool_obj],
                system_prompt=(
                    f"You are link {cap.id} in a handoff chain for the "
                    f"{cap.domain} domain. Your one tool is {tool_obj.name}.\n\n"
                    f"1. Read what the previous link produced.\n"
                    f"2. Use {tool_obj.name} to add the next concrete fact.\n"
                    f"3. Hand off cleanly: end your output with one line stating "
                    f"what the next link should focus on."
                ),
                plugins=plugins,
            ),
        )
    if not agents:
        # Without capabilities the chain has no real work — fall back to a
        # single tool-less agent so the runnable is still well-formed.
        agents.append(
            Agent(
                model=ctx.model,
                tools=[],
                system_prompt=(
                    f"You are a single-link handoff chain in the {frame.domain} "
                    "domain. Answer directly."
                ),
                plugins=plugins,
            ),
        )
    pipeline = SequentialPipeline(agents=agents)
    return wrap_pipeline(pipeline, "handoff_chain", frame)


def builtin_protocols() -> list[Protocol]:
    """The full eight-protocol catalogue. Compose your own
    :class:`ProtocolRegistry` from these (or a subset)."""
    return [
        Protocol(
            id="direct_response",
            description="Single-agent answer for direct questions and explanations.",
            handles=[TaskType.ANSWER, TaskType.EXPLAIN, TaskType.RESEARCH],
            primary_for=[TaskType.ANSWER, TaskType.EXPLAIN],
            risk_max=Risk.LOW,
            cost="low",
            latency="low",
            builder=_build_direct_response,
        ),
        Protocol(
            id="plan_execute_validate",
            description="Three-stage pipeline for tasks needing structure and validation.",
            handles=[
                TaskType.PLAN,
                TaskType.BUILD,
                TaskType.MODIFY,
                TaskType.GENERATE_CODE,
                TaskType.REMEDIATE,
            ],
            primary_for=[TaskType.PLAN, TaskType.BUILD, TaskType.MODIFY],
            risk_max=Risk.MEDIUM,
            cost="medium",
            latency="medium",
            builder=_build_plan_execute_validate,
        ),
        Protocol(
            id="specialist_fanout",
            description="Fan out to multiple specialists and correlate findings.",
            handles=[
                TaskType.DIAGNOSE,
                TaskType.COMPARE,
                TaskType.MONITOR,
                TaskType.COORDINATE,
                TaskType.RESEARCH,
            ],
            primary_for=[TaskType.DIAGNOSE, TaskType.MONITOR, TaskType.RESEARCH],
            risk_max=Risk.MEDIUM,
            cost="high",
            latency="high",
            builder=_build_specialist_fanout,
        ),
        Protocol(
            id="debate",
            description="Two debaters argue opposing positions; a judge picks the winner.",
            handles=[TaskType.COMPARE, TaskType.RESEARCH],
            primary_for=[TaskType.COMPARE],
            risk_max=Risk.LOW,
            cost="high",
            latency="high",
            builder=_build_debate,
        ),
        Protocol(
            id="codegen_test_validate",
            description=(
                "Loop a tool-using agent that must declare PASS/FAIL on every "
                "iteration; stops when output starts with PASS."
            ),
            handles=[TaskType.GENERATE_CODE, TaskType.BUILD],
            primary_for=[TaskType.GENERATE_CODE],
            risk_max=Risk.MEDIUM,
            cost="medium",
            latency="high",
            builder=_build_codegen_test_validate,
        ),
        Protocol(
            id="approval_gated_execution",
            description=(
                "Single-agent execution behind a forced approval interrupt — "
                "for risky REMEDIATE / MODIFY / ESCALATE actions."
            ),
            handles=[
                TaskType.REMEDIATE,
                TaskType.MODIFY,
                TaskType.ESCALATE,
            ],
            primary_for=[TaskType.ESCALATE, TaskType.REMEDIATE],
            risk_max=Risk.HIGH,
            cost="medium",
            latency="medium",
            builder=_build_approval_gated_execution,
        ),
        Protocol(
            id="a2a_delegate",
            description=(
                "Forward the task to a remote agent via the A2A protocol. "
                "Requires BuilderContext.a2a_endpoint to be set at compile time."
            ),
            handles=[TaskType.COORDINATE, TaskType.ESCALATE],
            primary_for=[],  # opt-in only
            risk_max=Risk.MEDIUM,
            cost="medium",
            latency="high",
            builder=_build_a2a_delegate,
        ),
        Protocol(
            id="handoff_chain",
            description=(
                "Sequential chain of one-tool agents — each link adds one fact "
                "and hands off to the next."
            ),
            handles=[TaskType.PLAN, TaskType.RESEARCH, TaskType.COORDINATE],
            primary_for=[TaskType.COORDINATE],
            risk_max=Risk.MEDIUM,
            cost="medium",
            latency="high",
            builder=_build_handoff_chain,
        ),
    ]
