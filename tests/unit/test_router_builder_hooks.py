# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Hooks reach every agent a protocol builder emits — leaves included.

A caller governs a run by passing a hook whose before-tool-call handler can
cancel the call. That works when the caller constructs the agent. It did not work
for anything the router compiled: each builder constructed its own agents, so the
hook reached the one hand-built shape and none of the leaves inside a pipeline, a
fan-out, a debate or a handoff chain. Seven of the eight shapes ran ungoverned,
and the ones with leaves are exactly the ones worth governing.

Nothing about that was visible from the outside. The hook was attached, the run
produced output, and the tools it called along the way were never offered to the
gate — so a test asserting only on the top-level agent passes while the defect is
live. These tests walk the built graph to the leaves instead, and they are
table-driven over ``builtin_protocols()`` so a protocol added later is covered by
construction rather than by whoever remembers to extend the list.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.router.capability import Capability, CapabilityIndex
from tulip.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from tulip.router.protocol import BuilderContext, builtin_protocols
from tulip.tools.decorator import tool
from tulip.tools.registry import create_registry


class _Sentinel:
    """Stands in for a governance hook — identity is all these tests check."""


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


@pytest.fixture
def ctx() -> BuilderContext:
    tools = create_registry(lookup)
    caps = CapabilityIndex(tools)
    caps.annotate("kb_lookup", tool_name="lookup", description="Lookup.", domain="research")
    return BuilderContext(
        model="anthropic:claude-sonnet-4-6",
        capabilities=caps,
        a2a_endpoint="https://remote.example/a2a",
        hooks=[_Sentinel()],
    )


@pytest.fixture
def capabilities(ctx: BuilderContext) -> list[Capability]:
    return [ctx.capabilities.lookup(["kb_lookup"])[0]]


def _frame(**over: Any) -> GoalFrame:
    body: dict[str, Any] = {
        "primary_goal": TaskType.ANSWER,
        "domain": "research",
        "complexity": Complexity.MEDIUM,
        "risk": Risk.LOW,
        "success_criteria": ["the caller gets an answer"],
        "required_capabilities": ["kb_lookup"],
    }
    body.update(over)
    return GoalFrame(**body)


def _agents(node: Any, seen: set[int] | None = None) -> list[Any]:
    """Every agent reachable from a built runnable, however deeply nested.

    Written structurally rather than per-protocol: it follows ``agent`` /
    ``agents`` / ``pipeline`` / ``debaters`` / ``judge`` / ``sub_agents`` wherever
    they appear. A shape that nests differently still gets walked, which is the
    point — the failure this guards against is a leaf nobody thought to look at.
    """
    seen = seen if seen is not None else set()
    if node is None or id(node) in seen:
        return []
    seen.add(id(node))
    found: list[Any] = []
    if hasattr(node, "_hooks") or hasattr(node, "invoke"):
        # An Agent. Recurse anyway — an agent may itself hold sub-agents.
        if hasattr(node, "_hooks"):
            found.append(node)
    for attr in ("agent", "pipeline", "judge", "inner"):
        found.extend(_agents(getattr(node, attr, None), seen))
    for attr in ("agents", "debaters", "sub_agents"):
        value = getattr(node, attr, None)
        if isinstance(value, list):
            for item in value:
                found.extend(_agents(item, seen))
        else:
            found.extend(_agents(value, seen))
    return found


def _hooks_of(agent: Any) -> list[Any]:
    # `_hooks` is a PrivateAttr; the config is where the constructor put them.
    from_private = list(getattr(agent, "_hooks", []) or [])
    config = getattr(agent, "config", None)
    from_config = list(getattr(config, "hooks", []) or []) if config is not None else []
    return [*from_private, *from_config]


# ── the shapes that build agents locally ─────────────────────────────────────
#: ``a2a_delegate`` is excluded deliberately, not overlooked: it forwards the task
#: to a remote agent that owns its own tool surface, so there is no local agent to
#: attach a hook to. That shape is governed by the remote's own gate, which is a
#: real limitation worth stating rather than papering over.
_LOCAL_SHAPES = [p for p in builtin_protocols() if p.id != "a2a_delegate"]


@pytest.mark.parametrize("protocol", _LOCAL_SHAPES, ids=lambda p: p.id)
def test_every_agent_a_shape_emits_carries_the_hooks(
    protocol: Any, ctx: BuilderContext, capabilities: list[Capability]
) -> None:
    frame = _frame(primary_goal=protocol.handles[0])
    built = protocol.builder(frame, capabilities, ctx)

    agents = _agents(built)
    assert agents, f"{protocol.id} built no agents to check — the walk missed the graph"
    ungoverned = [a for a in agents if not any(isinstance(h, _Sentinel) for h in _hooks_of(a))]
    assert not ungoverned, (
        f"{protocol.id}: {len(ungoverned)} of {len(agents)} agents have no hooks — "
        "every tool call they make bypasses the gate"
    )


@pytest.mark.parametrize("protocol", _LOCAL_SHAPES, ids=lambda p: p.id)
def test_a_shape_with_no_hooks_configured_still_builds(
    protocol: Any, ctx: BuilderContext, capabilities: list[Capability]
) -> None:
    """The ungoverned path stays valid — the SDK is usable without a gateway."""
    plain = ctx.model_copy(update={"hooks": []})
    built = protocol.builder(_frame(primary_goal=protocol.handles[0]), capabilities, plain)
    assert _agents(built)


def test_the_multi_agent_shapes_really_do_have_leaves() -> None:
    """Guards the guard.

    If ``_agents`` silently found only the top-level agent, every assertion above
    would pass on a graph whose leaves are ungoverned — the exact defect. So pin
    that at least one shape yields more than one agent.
    """
    tools = create_registry(lookup)
    caps = CapabilityIndex(tools)
    caps.annotate("kb_lookup", tool_name="lookup", description="Lookup.", domain="research")
    ctx = BuilderContext(model="anthropic:claude-sonnet-4-6", capabilities=caps, hooks=[])
    counts = {
        p.id: len(
            _agents(p.builder(_frame(primary_goal=p.handles[0]), caps.lookup(["kb_lookup"]), ctx))
        )
        for p in _LOCAL_SHAPES
    }
    assert max(counts.values()) > 1, f"the walk never reached a leaf: {counts}"


def test_a_builder_may_add_its_own_hooks_on_top() -> None:
    """Merged, not overwritten — a shape-specific hook must not drop the caller's."""
    from tulip.router.protocol import _agent

    tools = create_registry(lookup)
    caps = CapabilityIndex(tools)
    extra = _Sentinel()
    ctx = BuilderContext(model="anthropic:claude-sonnet-4-6", capabilities=caps, hooks=[extra])
    own = _Sentinel()
    agent = _agent(ctx, tools=[], system_prompt="x", hooks=[own])
    hooks = _hooks_of(agent)
    assert extra in hooks
    assert own in hooks
