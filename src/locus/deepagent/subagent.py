# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent-spawned subagent dispatch via a single ``task()`` tool.

Mirrors deepagents' ``task`` middleware: the parent agent — mid-run —
calls ``task(subagent_type="reviewer", description="…")`` to spawn a
*stateless one-shot* subagent. The subagent runs to completion with
its own tools and prompt, returns its final message as the tool
result, and is discarded.

Different from :class:`locus.Orchestrator` (router-style top-down
delegation): this is *the parent agent decides mid-research to
delegate*, not "the router picks a specialist to handle the user
request".

Subagents are stateless — no multi-turn within one call, no shared
memory between calls. If you want a long-running specialist
delegating back and forth, use ``Orchestrator + Specialist`` or
``Handoff`` instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from locus.tools.decorator import tool


class SubAgentDef(BaseModel):
    """Declarative definition of a subagent the parent can spawn.

    Mirrors deepagents' ``SubAgent`` TypedDict, mapped to a Pydantic
    model so locus's typed-config conventions hold.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    name: str = Field(..., description="Unique id the parent passes as `subagent_type`.")
    description: str = Field(
        ...,
        description="Free text shown to the parent so it can decide when to call this subagent.",
    )
    system_prompt: str
    tools: list[Any] = Field(default_factory=list)
    model: Any = Field(
        default=None,
        description="Optional model override; falls back to the parent's model if None.",
    )
    max_iterations: int = 10


def task_tool(
    subagents: list[SubAgentDef],
    *,
    parent_model: Any,
) -> Any:
    """Build a single ``task()`` tool the parent agent can call.

    The tool's parameters are flat: ``subagent_type`` (str) and
    ``description`` (str). On call:

    1. Look up ``subagent_type`` in the registered subagents.
    2. Spawn a fresh ``locus.Agent`` with the subagent's tools, prompt,
       and (optional) model override.
    3. Run the subagent on ``description`` using ``agent.run()`` and
       capture the final ``TerminateEvent.final_message``.
    4. Return that string as the tool's output to the parent.

    Args:
        subagents: List of :class:`SubAgentDef` declaring the
            available subagent types.
        parent_model: Model used when a subagent doesn't override.

    Returns:
        A locus ``Tool`` instance ready to splice into an Agent's
        tool list.
    """
    by_name: dict[str, SubAgentDef] = {sa.name: sa for sa in subagents}
    if not by_name:
        msg = "task_tool requires at least one SubAgentDef"
        raise ValueError(msg)

    # Build the docstring at runtime so the parent's prompt sees the
    # actual subagent catalogue.
    catalog = "\n".join(f"- {sa.name}: {sa.description}" for sa in subagents)
    available = ", ".join(sorted(by_name))

    @tool
    async def task(subagent_type: str, description: str) -> str:
        """Spawn a one-shot subagent to handle a focused subproblem.

        Args:
            subagent_type: Which subagent to spawn (one of: see catalogue
                in the system prompt).
            description: Detailed instructions for the subagent.

        Returns:
            The subagent's final message.
        """
        if subagent_type not in by_name:
            return f"unknown subagent_type {subagent_type!r}; available: {available}"
        defn = by_name[subagent_type]

        import time as _time

        from locus.agent.agent import Agent
        from locus.core.events import TerminateEvent
        from locus.observability.emit import (
            EV_DEEPAGENT_SUBAGENT_COMPLETED,
            EV_DEEPAGENT_SUBAGENT_SPAWNED,
            emit,
        )

        await emit(
            EV_DEEPAGENT_SUBAGENT_SPAWNED,
            subagent_type=subagent_type,
            description_preview=description[:200],
            max_iterations=defn.max_iterations,
        )
        _started = _time.perf_counter()

        sub_model = defn.model if defn.model is not None else parent_model
        sub_agent = Agent(
            model=sub_model,
            tools=defn.tools,
            system_prompt=defn.system_prompt,
            max_iterations=defn.max_iterations,
            reflexion=False,  # Subagents are short-lived; reflexion is overkill.
            grounding=False,
        )
        final_message = ""
        async for event in sub_agent.run(description):
            if isinstance(event, TerminateEvent):
                final_message = event.final_message or ""
        await emit(
            EV_DEEPAGENT_SUBAGENT_COMPLETED,
            subagent_type=subagent_type,
            output_length=len(final_message),
            duration_ms=(_time.perf_counter() - _started) * 1000,
            success=bool(final_message),
        )
        return final_message

    # Stash the catalogue on the tool so callers can inject it into
    # the parent's prompt if they want the parent to see what's
    # available without overloading the docstring.
    task._subagent_catalog = catalog  # type: ignore[attr-defined]  # noqa: SLF001 — extension point

    return task
