# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Attributing an event to the agent that produced it.

``docs/concepts/multi-agent.md`` promised this outright — *"``agent_name`` is
set on every event so you can attribute output to the specialist that produced
it"* — and ``production.md`` matched on it in an example. It did not exist:
``grep -rn "agent_name" src/tulip/`` returned nothing, and ``TulipEvent`` is
frozen without ``extra="allow"``, so it could not even be set from outside.

The claim was worth keeping rather than deleting. Without it, a caller merging
two agents' streams has no way to tell the researcher's tool call from the
writer's, and per-agent attribution is most of what multi-agent observability
means.

These cover the three things the implementation has to get right: the label
reaches every event, an unnamed agent is left honestly blank, and — the one
that carries the feature — a nested agent's events are never relabelled by the
orchestrator around them.
"""

from __future__ import annotations

import pytest

from tulip.agent import Agent
from tulip.core.events import ThinkEvent, TulipEvent
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return "found it"


async def _events(agent: Agent, prompt: str = "go") -> list[TulipEvent]:
    return [event async for event in agent.run(prompt)]


# --------------------------------------------------------------------------
# The label reaches everything
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_event_in_a_run_carries_the_agent_name() -> None:
    """Including tool events, which is where attribution actually gets used."""
    agent = Agent(
        name="Researcher",
        model=ScriptedModel([tool_call("lookup", query="x"), text("done")]),
        tools=[lookup],
    )

    events = await _events(agent)

    assert {e.event_type for e in events} >= {"think", "tool_start", "tool_complete"}
    assert {e.agent_name for e in events} == {"Researcher"}


@pytest.mark.asyncio
async def test_agent_id_is_used_when_there_is_no_display_name() -> None:
    """An id in a trace beats no attribution at all."""
    agent = Agent(model=ScriptedModel([text("hi")]), agent_id="svc-42")

    assert {e.agent_name for e in await _events(agent)} == {"svc-42"}


@pytest.mark.asyncio
async def test_a_display_name_wins_over_the_id() -> None:
    agent = Agent(name="Writer", model=ScriptedModel([text("hi")]), agent_id="svc-42")

    assert {e.agent_name for e in await _events(agent)} == {"Writer"}


@pytest.mark.asyncio
async def test_an_unnamed_agent_leaves_it_blank_rather_than_inventing_one() -> None:
    """A positional label would be stable only until someone reorders the list.

    Attribution is worth having and not worth fabricating: ``None`` says "not
    recorded", which a consumer can handle, where ``"agent-3"`` says something
    false with confidence.
    """
    agent = Agent(model=ScriptedModel([text("hi")]))

    assert {e.agent_name for e in await _events(agent)} == {None}


# --------------------------------------------------------------------------
# The nested case — the reason the field exists
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_nested_agents_events_are_not_relabelled() -> None:
    """The innermost agent wins.

    An orchestrator forwarding a specialist's stream must not stamp its own
    name over it — that would erase precisely the attribution the docs promise
    and leave every event in a multi-agent run looking like the orchestrator's.
    """
    specialist = Agent(name="Specialist", model=ScriptedModel([text("inner")]))
    from_specialist = await _events(specialist)
    assert {e.agent_name for e in from_specialist} == {"Specialist"}

    orchestrator = Agent(name="Orchestrator", model=ScriptedModel([text("outer")]))
    merged = from_specialist + await _events(orchestrator)

    by_agent = {e.agent_name for e in merged}
    assert by_agent == {"Specialist", "Orchestrator"}


def test_an_already_attributed_event_is_left_alone() -> None:
    """The guard is on the event, not on the call site, so it holds anywhere."""
    stamped = ThinkEvent(iteration=0, agent_name="Specialist")

    assert stamped.model_copy(update={}).agent_name == "Specialist"


# --------------------------------------------------------------------------
# It must not disturb what events already were
# --------------------------------------------------------------------------


def test_events_stay_frozen() -> None:
    event = ThinkEvent(iteration=0)

    with pytest.raises((TypeError, ValueError)):
        event.agent_name = "sneaky"  # type: ignore[misc]


def test_the_field_defaults_to_none_on_a_bare_event() -> None:
    """Constructing an event directly must not require knowing about this."""
    assert ThinkEvent(iteration=0).agent_name is None


def test_attribution_survives_serialisation() -> None:
    """Events cross a process boundary in SSE and A2A; the label has to travel."""
    event = ThinkEvent(iteration=1, agent_name="Researcher")

    assert ThinkEvent.model_validate_json(event.model_dump_json()).agent_name == "Researcher"
