# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""A ``task`` tool: delegate to a typed subagent, in parallel, resumably.

The tool coding agents converge on for delegation — Claude Code's ``Agent``,
Codex's ``spawn_agent`` / ``send_input``, opencode's ``task`` — built from
the SDK's own pieces rather than beside them:

- each call runs a :class:`~tulip.agent.subagent.Subagent` with its own
  conversation, so the parent's context holds the subagent's *answer*, not
  the dozens of reads that produced it;
- several calls in one model turn run concurrently (the parent's
  ``tool_execution="concurrent"``, bounded by ``max_concurrency``);
- every call returns a ``task_id``; passing it back continues that
  subagent's conversation instead of starting cold;
- the subagent's spend, time and cancellation belong to the parent run
  (see :mod:`tulip.agent.subagent`), and its events stream live on the
  parent's stream as :class:`~tulip.core.events.SubagentEvent`;
- what a subagent may use is chosen from the parent's own tools by its
  :class:`~tulip.agent.specs.AgentSpec`, and the parent's hooks run inside
  it — a subagent type can narrow the parent's powers and never widen them.

The harness supplies the agent types and the tool pool::

    from tulip import Agent
    from tulip.agent.specs import AgentSpec
    from tulip.agent.tasks import task_tool

    explore = AgentSpec(
        name="explore",
        description="Read-only search of the codebase. Use for 'where is X'.",
        prompt="You search code and report file:line findings.",
        tools=("read", "grep", "glob"),
        mode="subagent",
    )
    tools = [read, grep, glob, edit, bash]
    task = task_tool([explore], tools=tools, model="openai:gpt-5.5")
    agent = Agent(model="openai:gpt-5.5", tools=[*tools, task])
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

from tulip.agent.specs import AgentSpec
from tulip.agent.subagent import Subagent, SubagentResult, parent_hooks
from tulip.tools.decorator import Tool, tool


__all__ = ["DEFAULT_SUBAGENT_PROMPT", "TaskRegistry", "task_depth", "task_tool"]

#: The system prompt of a subagent type that brings none.
DEFAULT_SUBAGENT_PROMPT = (
    "You are a subagent. Another agent delegated the task below to you and will "
    "read only your final message, not your tool calls — so finish with a "
    "concise, self-contained answer: what you found or did, with file paths "
    "and line numbers where they matter, and anything you could not settle."
)

#: How deep the current code is in a chain of task calls: 0 in a top-level
#: agent, 1 inside a subagent, and so on. A contextvar, so it follows the
#: child loop into the tasks it creates for its own tool calls.
_TASK_DEPTH: ContextVar[int] = ContextVar("tulip_task_depth", default=0)


def task_depth() -> int:
    """How many task calls deep the current code is running."""
    return _TASK_DEPTH.get()


class TaskRegistry:
    """The resumable subagents of one session, by ``task_id``.

    Bounded: past ``max_tasks`` the least recently used conversation is
    dropped, and a later attempt to resume it says so rather than failing
    in some stranger way.
    """

    def __init__(self, max_tasks: int = 64) -> None:
        self.max_tasks = max_tasks
        self._tasks: OrderedDict[str, tuple[str, Subagent]] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, agent_type: str, subagent: Subagent) -> None:
        with self._lock:
            self._tasks[subagent.task_id] = (agent_type, subagent)
            self._tasks.move_to_end(subagent.task_id)
            while len(self._tasks) > self.max_tasks:
                self._tasks.popitem(last=False)

    def get(self, task_id: str) -> tuple[str, Subagent] | None:
        """``(agent_type, subagent)`` for ``task_id``, marking it recently used."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is not None:
                self._tasks.move_to_end(task_id)
            return entry

    def __len__(self) -> int:
        return len(self._tasks)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._tasks)


def _describe(specs: Iterable[AgentSpec]) -> str:
    lines = [f"- {s.name}: {s.description or '(no description)'}" for s in specs]
    return "\n".join(lines)


def _footer(result: SubagentResult, agent_type: str) -> str:
    parts = [f"task_id={result.task_id}", f"subagent={agent_type}", f"turns={result.iterations}"]
    if not result.success:
        parts.append(f"stopped={result.stop_reason}")
    return f"\n\n[{', '.join(parts)} — pass task_id to continue this subagent's conversation]"


def task_tool(  # noqa: PLR0913, C901 — the whole delegation policy, passed in one place
    agents: Sequence[AgentSpec] | Mapping[str, AgentSpec],
    *,
    tools: Sequence[Any] | Callable[[], Sequence[Any]],
    model: Any,
    resolve_model: Callable[[str], Any] | None = None,
    hooks: Sequence[Any] = (),
    inherit_hooks: bool = True,
    max_depth: int = 2,
    max_iterations: int = 20,
    registry: TaskRegistry | None = None,
    name: str = "task",
    default_type: str | None = None,
    agent_kwargs: Mapping[str, Any] | None = None,
) -> Tool:
    """Build the ``task`` tool for an agent.

    Args:
        agents: The subagent types it may spawn. Specs whose ``mode`` is
            ``primary`` are left out.
        tools: The pool each subagent's tools are chosen from — normally the
            parent's own toolset (this tool included, for nesting), or a
            callable returning it, read on every call. A spec's ``tools`` and
            ``disallowed_tools`` select from it; nothing outside it is ever
            given to a subagent.
        model: The model a spec without ``model`` runs on (the parent's).
        resolve_model: Turns a spec's ``model`` string into what ``Agent``
            accepts — a harness's aliases, a provider prefix. Identity when
            omitted.
        hooks: Extra lifecycle hooks for every subagent.
        inherit_hooks: Also run the calling agent's own hooks inside the
            subagent, so a policy attached to the parent gates the child's
            calls too. On by default: delegation must not be a way around a
            gate.
        max_depth: How deep task calls may nest. A subagent at the last level
            is not given this tool.
        max_iterations: The iteration cap for a spec without ``max_turns``.
        registry: Where resumable subagents are kept. One per session; a new
            one is made when omitted.
        name: The tool's name.
        default_type: The type used when the model names none (default: the
            first type).
        agent_kwargs: Further ``AgentConfig`` fields for every subagent
            (``temperature``, ``token_budget``, ...).

    Returns:
        An async :class:`~tulip.tools.decorator.Tool` that streams its
        subagents' events (``emits_progress=True``).
    """
    specs = list(agents.values()) if isinstance(agents, Mapping) else list(agents)
    by_name = {s.name: s for s in specs if s.is_subagent}
    if not by_name:
        msg = "task_tool needs at least one agent spec with mode 'subagent' or 'all'"
        raise ValueError(msg)
    if max_depth < 1:
        msg = "max_depth must be at least 1"
        raise ValueError(msg)
    fallback = default_type or next(iter(by_name))
    if fallback not in by_name:
        msg = f"default_type {fallback!r} is not one of the subagent types: {', '.join(by_name)}"
        raise ValueError(msg)
    tasks = registry if registry is not None else TaskRegistry()
    extra = dict(agent_kwargs or {})
    available = ", ".join(by_name)

    def _pool() -> list[Any]:
        return list(tools() if callable(tools) else tools)

    def _child_tools(spec: AgentSpec, depth: int) -> list[Any]:
        chosen = spec.select_tools(_pool())
        if depth >= max_depth:
            # The last level does not delegate: unbounded nesting is a
            # fork bomb with a model in the loop.
            chosen = [t for t in chosen if getattr(t, "name", None) != name]
        return chosen

    def _model_for(spec: AgentSpec) -> Any:
        if spec.model is None:
            return model
        return resolve_model(spec.model) if resolve_model is not None else spec.model

    description = (
        "Delegate a task to a subagent that works in its own context window and "
        "returns only its final answer. Use it for searches and investigations "
        "that would fill your context with material you do not need afterwards, "
        "and for independent pieces of work. Several calls in one turn run in "
        "parallel. Give the subagent everything it needs in `prompt` — it cannot "
        "see this conversation. The result ends with a task_id: pass it back as "
        "`task_id` to continue that subagent with a follow-up instead of starting "
        "a new one.\n\nSubagent types:\n" + _describe(by_name.values())
    )

    async def task(
        description: str,
        prompt: str,
        subagent_type: str = fallback,
        task_id: str | None = None,
    ) -> str:
        """Run a subagent and return its final answer.

        Args:
            description: A short (3-5 word) label for the task, for whoever is
                watching.
            prompt: The full task: what to do, what to return, and any
                context the subagent needs. It sees nothing else.
            subagent_type: Which kind of subagent to use (see the list).
            task_id: Continue an earlier subagent's conversation instead of
                starting a new one. Its type is kept.
        """
        # ``description`` is for whoever watches: it reaches them on the
        # ToolStartEvent's arguments, and the subagent never needs it.
        depth = _TASK_DEPTH.get() + 1
        if depth > max_depth:
            return (
                f"refused: task calls may nest {max_depth} deep and this would be "
                f"level {depth}. Do the work directly."
            )
        if task_id:
            entry = tasks.get(task_id)
            if entry is None:
                known = ", ".join(tasks.ids()[-10:]) or "none"
                return (
                    f"no subagent with task_id {task_id!r} (it may have been dropped "
                    f"to make room). Known: {known}. Start a new task instead."
                )
            agent_type, subagent = entry
        else:
            spec = by_name.get(subagent_type)
            if spec is None:
                return f"unknown subagent_type {subagent_type!r}; available: {available}"
            agent_type = spec.name
            child_kwargs: dict[str, Any] = dict(extra)
            if spec.temperature is not None:
                child_kwargs["temperature"] = spec.temperature
            subagent = Subagent(
                model=_model_for(spec),
                tools=_child_tools(spec, depth),
                system_prompt=spec.prompt or DEFAULT_SUBAGENT_PROMPT,
                name=spec.name,
                max_iterations=spec.max_turns or max_iterations,
                hooks=[*(parent_hooks() if inherit_hooks else ()), *hooks],
                **child_kwargs,
            )
            tasks.add(agent_type, subagent)

        token = _TASK_DEPTH.set(depth)
        try:
            result = await subagent.send(prompt)
        finally:
            _TASK_DEPTH.reset(token)
        text = result.text.strip() or "(the subagent finished without a final message)"
        return text + _footer(result, agent_type)

    return tool(name=name, description=description, emits_progress=True)(task)
