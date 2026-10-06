# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Skills plugin — progressive disclosure of skill instructions.

Implements the AgentSkills.io three-level content model:
- L1: XML catalog injected into system prompt (names + descriptions)
- L2: Full instructions returned when agent activates a skill
- L3: Resource file listing for agent to read on demand

A host that decides in code which skills a run needs passes them as
``active``: their instructions are in the prompt from the first model call,
with no catalog and no ``skills`` tool unless ``catalog=True``. A host whose
choice depends on the message passes a ``router`` instead, and one plugin (so
one Agent) serves every route: the router is asked once per run, before the
run's first model call. With ``enforce_allowed_tools=True`` the run's active
skills' ``allowed-tools`` stop being advice: a call to any other tool is
cancelled before it runs.

What a run activates (routed, or through the ``skills`` tool) belongs to that
run alone, so one plugin on a shared Agent never lets one conversation's skills
widen another's tool allowlist.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

from tulip.hooks.plugin import Plugin, hook
from tulip.skills.models import Skill
from tulip.tools.decorator import tool as tool_decorator


if TYPE_CHECKING:
    from tulip.core.events import RunInfo


logger = logging.getLogger(__name__)

#: ``(latest user message, run) -> skill names`` — sync or async.
SkillRouter = Callable[
    [str, "RunInfo | None"], "Iterable[str] | None | Awaitable[Iterable[str] | None]"
]

#: The default text before the active skills' instructions.
DEFAULT_ACTIVE_PREAMBLE = "Follow these skills for this conversation turn:\n\n"
#: The default text before the catalog.
DEFAULT_CATALOG_PREAMBLE = (
    "The following skills are available. To activate a skill, "
    "call the `skills` tool with the skill name.\n\n"
)

#: Runs whose skills are remembered at once. A run's entry is dropped when the
#: run ends; the bound only matters for runs that never reach their end.
_MAX_TRACKED_RUNS = 1024


@dataclass
class _RunSkills:
    """What one run has active: routed (or ``active``) plus model-activated."""

    active: tuple[str, ...]
    activated: list[str] = field(default_factory=list)


class SkillsPlugin(Plugin):
    """Plugin that provides AgentSkills.io skill discovery and activation.

    Injects a compact XML catalog of available skills into the system prompt.
    Registers a `skills` tool that the agent calls to load full instructions.

    Example:
        >>> from tulip.skills import Skill, SkillsPlugin
        >>>
        >>> plugin = SkillsPlugin(
        ...     skills=[
        ...         Skill.from_file("./skills/code-review"),
        ...         Skill(
        ...             name="summarize", description="Summarize text", instructions="..."
        ...         ),
        ...     ]
        ... )
        >>>
        >>> agent = Agent(
        ...     config=AgentConfig(
        ...         model=model,
        ...         plugins=[plugin],
        ...     )
        ... )
    """

    name = "skills"

    #: Name of the activation tool the model calls (``catalog=True``).
    ACTIVATION_TOOL = "skills"

    def __init__(
        self,
        skills: list[Skill | str | Path],
        max_resource_files: int = 20,
        *,
        active: Iterable[str] = (),
        catalog: bool | None = None,
        enforce_allowed_tools: bool = False,
        show_paths: bool = True,
        router: SkillRouter | None = None,
        active_preamble: str = DEFAULT_ACTIVE_PREAMBLE,
        catalog_preamble: str = DEFAULT_CATALOG_PREAMBLE,
        render_skill: Callable[[Skill], str] | None = None,
        skill_footer: bool = True,
    ) -> None:
        """Initialize with skill sources.

        Args:
            skills: List of Skill instances, paths to skill directories,
                   or paths to parent directories containing skills.
            max_resource_files: Max resource files to list per skill.
            active: Names of skills the host activates for the run up front.
                Their full instructions are sent with every model call (as a
                system message after the system prompt, never written to the
                conversation state). Unknown names raise ``ValueError``.
            catalog: Whether the model sees the L1 catalog and gets the
                ``skills`` tool to activate more skills itself. Defaults to
                True without ``active`` and False with it: a host that routes
                skills in code usually does not want the model to route too.
            enforce_allowed_tools: Make the active skills' ``allowed-tools``
                a hard limit. A tool call outside the union of the lists the
                active skills declare is cancelled before it runs, with a
                reason the model reads. Skills that declare no list add no
                tools; when no active skill declares one there is no limit.
                The ``skills`` tool itself is always allowed.
            show_paths: Include each skill's filesystem location in the
                catalog and in activation responses. Set False for a server,
                where paths reveal the deployment layout and help nobody.
            router: ``(text, run) -> skill names`` choosing the active skills
                PER RUN, in code: ``text`` is the run's latest user message and
                ``run`` its :class:`~tulip.core.events.RunInfo` (``run.metadata``
                is what ``agent.run(..., metadata=)`` passed). Sync or async.
                Asked once per run, before its first model call; what it
                returns replaces ``active`` for that run (``None`` keeps
                ``active``; an empty list means no skill). An unknown name is
                logged and ignored, and a router that raises leaves the run on
                ``active`` — a live turn is never lost to a routing mistake.
                The router cannot change the tools the model is offered; use
                ``enforce_allowed_tools`` to make the routed skills' lists bind.
            active_preamble: The text before the active skills' instructions.
            catalog_preamble: The text before the catalog.
            render_skill: ``(skill) -> text`` rendering one active skill. The
                default is ``<skill name="...">`` around its instructions and,
                with ``skill_footer``, its metadata footer.
            skill_footer: Whether the default rendering of an active skill ends
                with its ``Allowed tools`` / ``Compatibility`` / location
                footer and resource listing. The model never needs them when
                the host routes skills and enforces the tool list.
        """
        self._skills: dict[str, Skill] = {}
        self._max_resource_files = max_resource_files
        self._show_paths = show_paths
        self._enforce = enforce_allowed_tools
        self._router = router
        self._active_preamble = active_preamble
        self._catalog_preamble = catalog_preamble
        self._render_skill = render_skill
        self._skill_footer = skill_footer
        # Per-run activation, keyed by run id; ``None`` is the bucket for calls
        # made outside any run (a tool called directly, an event built by hand).
        self._runs: dict[str | None, _RunSkills] = {}
        self._last_activated: list[str] = []

        for source in skills:
            if isinstance(source, Skill):
                self._skills[source.name] = source
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if (path / "SKILL.md").exists():
                    skill = Skill.from_file(path)
                    self._skills[skill.name] = skill
                elif path.is_dir():
                    for skill in Skill.from_directory(path):
                        self._skills[skill.name] = skill

        active_names = list(dict.fromkeys(active))
        unknown = [n for n in active_names if n not in self._skills]
        if unknown:
            available = ", ".join(sorted(self._skills)) or "none"
            msg = f"Unknown active skill(s): {', '.join(unknown)}. Available: {available}"
            raise ValueError(msg)
        self._active: tuple[str, ...] = tuple(active_names)
        self._last_activated = list(self._active)
        self._catalog = (not self._active and router is None) if catalog is None else catalog

    # ------------------------------------------------------------------
    # Per-run state
    # ------------------------------------------------------------------

    def _run(self, run_id: str | None) -> _RunSkills:
        """The run's skills, created on ``active`` when the run is new."""
        found = self._runs.get(run_id)
        if found is None:
            found = _RunSkills(active=self._active, activated=list(self._active))
            self._remember(run_id, found)
        return found

    def _remember(self, run_id: str | None, entry: _RunSkills) -> None:
        if run_id not in self._runs and len(self._runs) >= _MAX_TRACKED_RUNS:
            self._runs.pop(next(iter(self._runs)))
        self._runs[run_id] = entry
        self._last_activated = entry.activated

    async def _route(self, run: RunInfo | None, messages: list[Any]) -> _RunSkills:
        """Ask the router once for this run; later calls reuse the answer."""
        run_id = run.run_id if run is not None else None
        found = self._runs.get(run_id)
        if found is not None or self._router is None:
            return found if found is not None else self._run(run_id)
        names: Iterable[str] | None
        try:
            chosen = self._router(_latest_user_text(messages), run)
            names = await chosen if inspect.isawaitable(chosen) else chosen
        except Exception:  # noqa: BLE001 — a routing bug must not cost the turn
            logger.warning(
                "skills router failed; run %s keeps the default skills", run_id, exc_info=True
            )
            names = None
        if names is None:
            picked = self._active
        else:
            ordered = list(dict.fromkeys(str(n) for n in names))
            unknown = [n for n in ordered if n not in self._skills]
            if unknown:
                logger.warning(
                    "skills router chose unknown skill(s) %s; ignored", ", ".join(unknown)
                )
            picked = tuple(n for n in ordered if n in self._skills)
        entry = _RunSkills(active=picked, activated=list(picked))
        self._remember(run_id, entry)
        return entry

    def _generate_catalog_xml(self) -> str:
        """Generate XML catalog of available skills.

        Returns compact XML with skill names and descriptions only.
        Full instructions are NOT included (progressive disclosure L1).
        """
        if not self._skills:
            return ""

        lines = ["<available_skills>"]
        for skill in self._skills.values():
            lines.append("<skill>")
            lines.append(f"<name>{escape(skill.name)}</name>")
            lines.append(f"<description>{escape(skill.description)}</description>")
            if skill.path and self._show_paths:
                lines.append(f"<location>{escape(str(skill.path / 'SKILL.md'))}</location>")
            lines.append("</skill>")
        lines.append("</available_skills>")

        return "\n".join(lines)

    def _format_skill_response(self, skill: Skill, *, footer: bool = True) -> str:
        """Format full skill response for activation (L2 + L3).

        Returns instructions plus metadata and resource listing (without
        ``footer``, the instructions alone).
        """
        parts = [skill.instructions]
        if not footer:
            return "\n".join(parts)

        # Metadata footer
        meta: list[str] = []
        if skill.allowed_tools:
            meta.append(f"Allowed tools: {', '.join(skill.allowed_tools)}")
        if skill.compatibility:
            meta.append(f"Compatibility: {skill.compatibility}")
        if skill.path and self._show_paths:
            meta.append(f"Location: {skill.path}")

        if meta:
            parts.append("\n---\n" + "\n".join(meta))

        # Resource listing (L3) — relative paths, but only useful to a model
        # that can read the files, i.e. when paths are shown at all.
        resources = (
            skill.list_resources(max_files=self._max_resource_files) if self._show_paths else []
        )
        if resources:
            parts.append("\n---\nResource files:\n" + "\n".join(f"- {r}" for r in resources))

        return "\n".join(parts)

    def _active_instructions(self, names: Iterable[str] | None = None) -> str:
        """The active skills' instructions, one section per skill."""
        sections = []
        for name in self._active if names is None else names:
            skill = self._skills[name]
            if self._render_skill is not None:
                sections.append(self._render_skill(skill))
                continue
            body = self._format_skill_response(skill, footer=self._skill_footer).strip()
            sections.append(f'<skill name="{escape(name)}">\n{body}\n</skill>')
        return "\n\n".join(sections)

    @hook
    async def on_before_model_call(self, event: Any) -> None:
        """Inject the active skills and/or the catalog before each model call.

        Only the messages sent to the model change; the run's state (and so
        every checkpoint) never holds them.
        """
        from tulip.core.messages import Message

        current = await self._route(getattr(event, "run", None), list(event.messages))
        injected: list[Message] = []
        if current.active:
            injected.append(
                Message.system(self._active_preamble + self._active_instructions(current.active))
            )
        if self._catalog:
            catalog = self._generate_catalog_xml()
            if catalog:
                injected.append(Message.system(self._catalog_preamble + catalog))
        if not injected:
            return

        # Insert after the first system message (if any)
        messages = list(event.messages)
        insert_idx = 1 if messages and messages[0].role.value == "system" else 0
        messages[insert_idx:insert_idx] = injected
        event.messages = messages

    def allowed_tools(self, run_id: str | None = None) -> frozenset[str] | None:
        """The tools a run's active skills allow, or None when nothing limits them.

        The union of the ``allowed-tools`` lists the run's active skills
        declare (routed, or ``active``); None when no active skill declares
        one (or none is active). Also whatever the model activated itself
        through the ``skills`` tool in that run. ``run_id=None`` is the state
        outside any run, which starts from ``active``.
        """
        current = self._runs.get(run_id)
        names = current.activated if current is not None else list(self._active)
        declared = [
            self._skills[n].allowed_tools
            for n in names
            if n in self._skills and self._skills[n].allowed_tools is not None
        ]
        if not declared:
            return None
        return frozenset(t for tools in declared for t in (tools or ()))

    @hook
    async def on_before_tool_call(self, event: Any) -> None:
        """With ``enforce_allowed_tools``, cancel calls the skills do not allow."""
        if not self._enforce or event.tool_name == self.ACTIVATION_TOOL:
            return
        run = getattr(event, "run", None)
        allowed = self.allowed_tools(run.run_id if run is not None else None)
        if allowed is None or event.tool_name in allowed:
            return
        event.cancel = (
            f"The tool {event.tool_name} is not available for this request. "
            f"Use one of: {', '.join(sorted(allowed)) or 'no tools'}."
        )

    def get_tools(self) -> list[Any]:
        """The ``skills`` activation tool when the catalog is on, else none.

        (``AgentConfig.skills`` registers it itself; this is what lets a
        configured plugin in ``AgentConfig.plugins`` work the same way.)
        """
        return [self.get_activation_tool()] if self._catalog else []

    def get_activation_tool(self) -> Any:
        """Create the skills activation tool.

        Returns a Tool that the agent calls to load skill instructions.
        """
        skills_dict = self._skills
        plugin = self

        @tool_decorator(
            name=self.ACTIVATION_TOOL,
            description="Activate a skill to load its instructions. "
            "Call with the skill name from the available_skills catalog.",
        )
        def skills(skill_name: str) -> str:  # noqa: ARG001
            """Load a skill's full instructions.

            Args:
                skill_name: Name of the skill to activate.
            """
            if not skill_name:
                return "Error: skill_name is required."

            skill = skills_dict.get(skill_name)
            if skill is None:
                available = ", ".join(sorted(skills_dict.keys()))
                return f"Unknown skill: '{skill_name}'. Available: {available}"

            # Track activation, in the calling run's own state.
            plugin._note_activation(skill_name)

            # Telemetry — opt-in. ``emit_sync`` no-ops outside an
            # active run_context, so SDK users who never enter one
            # pay nothing for this line.
            try:
                from tulip.observability.emit import EV_SKILL_ACTIVATED, emit_sync  # noqa: PLC0415

                emit_sync(
                    EV_SKILL_ACTIVATED,
                    skill_name=skill_name,
                    has_resources=bool(skill.list_resources(max_files=1)),
                    instructions_length=len(skill.instructions or ""),
                )
            except Exception:  # noqa: BLE001 — telemetry must never break the SDK
                pass

            return plugin._format_skill_response(skill)

        return skills

    def _note_activation(self, skill_name: str) -> None:
        """Record a model activation on the run whose tool call this is."""
        from tulip.tools.context import current_tool_context  # noqa: PLC0415

        ctx = current_tool_context()
        current = self._run(ctx.run_id if ctx is not None else None)
        if skill_name in current.activated:
            current.activated.remove(skill_name)
        current.activated.append(skill_name)
        self._last_activated = current.activated

    @hook
    async def on_after_invocation(self, state: Any, success: bool) -> None:  # noqa: ARG002
        """Forget the run's skills once it ends (``activated_skills`` keeps them)."""
        run_id = getattr(state, "run_id", None)
        entry = self._runs.pop(run_id, None)
        if entry is not None:
            self._last_activated = entry.activated

    @property
    def activated_skills(self) -> list[str]:
        """Skills active in the most recent run (most recent activation last).

        On a shared Agent serving runs at once this is whichever run touched
        its skills last; use :meth:`allowed_tools` with a run id, or a hook
        reading ``event.run``, for one run's own view.
        """
        return list(self._last_activated)

    @property
    def available_skills(self) -> list[str]:
        """Get list of available skill names."""
        return sorted(self._skills.keys())


def _latest_user_text(messages: list[Any]) -> str:
    """The content of the last user message, or ``""``."""
    for message in reversed(messages):
        role = getattr(getattr(message, "role", None), "value", getattr(message, "role", None))
        if role == "user":
            content = getattr(message, "content", None)
            return content if isinstance(content, str) else str(content or "")
    return ""
