# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Skills plugin — progressive disclosure of skill instructions.

Implements the AgentSkills.io three-level content model:
- L1: XML catalog injected into system prompt (names + descriptions)
- L2: Full instructions returned when agent activates a skill
- L3: Resource file listing for agent to read on demand

A host that decides in code which skills a run needs (a router) passes them as
``active``: their instructions are in the prompt from the first model call,
with no catalog and no ``skills`` tool unless ``catalog=True``. With
``enforce_allowed_tools=True`` the active skills' ``allowed-tools`` stop being
advice: a call to any other tool is cancelled before it runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from tulip.hooks.plugin import Plugin, hook
from tulip.skills.models import Skill
from tulip.tools.decorator import tool as tool_decorator


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
        """
        self._skills: dict[str, Skill] = {}
        self._max_resource_files = max_resource_files
        self._activated: list[str] = []
        self._show_paths = show_paths
        self._enforce = enforce_allowed_tools

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
        self._activated.extend(self._active)
        self._catalog = (not self._active) if catalog is None else catalog

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

    def _format_skill_response(self, skill: Skill) -> str:
        """Format full skill response for activation (L2 + L3).

        Returns instructions plus metadata and resource listing.
        """
        parts = [skill.instructions]

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

    def _active_instructions(self) -> str:
        """The active skills' instructions, one section per skill."""
        sections = []
        for name in self._active:
            skill = self._skills[name]
            body = self._format_skill_response(skill).strip()
            sections.append(f'<skill name="{escape(name)}">\n{body}\n</skill>')
        return "\n\n".join(sections)

    @hook
    async def on_before_model_call(self, event: Any) -> None:
        """Inject the active skills and/or the catalog before each model call.

        Only the messages sent to the model change; the run's state (and so
        every checkpoint) never holds them.
        """
        from tulip.core.messages import Message

        injected: list[Message] = []
        if self._active:
            injected.append(
                Message.system(
                    "Follow these skills for this conversation turn:\n\n"
                    + self._active_instructions()
                )
            )
        if self._catalog:
            catalog = self._generate_catalog_xml()
            if catalog:
                injected.append(
                    Message.system(
                        "The following skills are available. To activate a skill, "
                        "call the `skills` tool with the skill name.\n\n" + catalog
                    )
                )
        if not injected:
            return

        # Insert after the first system message (if any)
        messages = list(event.messages)
        insert_idx = 1 if messages and messages[0].role.value == "system" else 0
        messages[insert_idx:insert_idx] = injected
        event.messages = messages

    def allowed_tools(self) -> frozenset[str] | None:
        """The tools the active skills allow, or None when nothing limits them.

        The union of the ``allowed-tools`` lists the active skills declare;
        None when no active skill declares one (or none is active). Also
        whatever the model activated itself through the ``skills`` tool.
        """
        declared = [
            self._skills[n].allowed_tools
            for n in self._activated
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
        allowed = self.allowed_tools()
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

            # Track activation
            if skill_name in plugin._activated:
                plugin._activated.remove(skill_name)
            plugin._activated.append(skill_name)

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

    @property
    def activated_skills(self) -> list[str]:
        """Get list of activated skill names (most recent last)."""
        return list(self._activated)

    @property
    def available_skills(self) -> list[str]:
        """Get list of available skill names."""
        return sorted(self._skills.keys())
