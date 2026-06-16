# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Skills plugin — progressive disclosure of skill instructions.

Implements the AgentSkills.io three-level content model:
- L1: XML catalog injected into system prompt (names + descriptions)
- L2: Full instructions returned when agent activates a skill
- L3: Resource file listing for agent to read on demand
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from locus.hooks.plugin import Plugin, hook
from locus.skills.models import Skill
from locus.tools.decorator import tool as tool_decorator


class SkillsPlugin(Plugin):
    """Plugin that provides AgentSkills.io skill discovery and activation.

    Injects a compact XML catalog of available skills into the system prompt.
    Registers a `skills` tool that the agent calls to load full instructions.

    Example:
        >>> from locus.skills import Skill, SkillsPlugin
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

    def __init__(
        self,
        skills: list[Skill | str | Path],
        max_resource_files: int = 20,
    ) -> None:
        """Initialize with skill sources.

        Args:
            skills: List of Skill instances, paths to skill directories,
                   or paths to parent directories containing skills.
            max_resource_files: Max resource files to list per skill.
        """
        self._skills: dict[str, Skill] = {}
        self._max_resource_files = max_resource_files
        self._activated: list[str] = []

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
            if skill.path:
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
        if skill.path:
            meta.append(f"Location: {skill.path}")

        if meta:
            parts.append("\n---\n" + "\n".join(meta))

        # Resource listing (L3)
        resources = skill.list_resources(max_files=self._max_resource_files)
        if resources:
            parts.append("\n---\nResource files:\n" + "\n".join(f"- {r}" for r in resources))

        return "\n".join(parts)

    @hook
    async def on_before_model_call(self, event: Any) -> None:
        """Inject skills catalog XML into messages before model call."""
        catalog = self._generate_catalog_xml()
        if not catalog:
            return

        from locus.core.messages import Message

        # Inject catalog as a system message at the beginning
        catalog_msg = Message.system(
            "The following skills are available. To activate a skill, "
            "call the `skills` tool with the skill name.\n\n" + catalog
        )

        # Insert after the first system message (if any)
        messages = list(event.messages)
        insert_idx = 1 if messages and messages[0].role.value == "system" else 0
        messages.insert(insert_idx, catalog_msg)
        event.messages = messages

    def get_activation_tool(self) -> Any:
        """Create the skills activation tool.

        Returns a Tool that the agent calls to load skill instructions.
        """
        skills_dict = self._skills
        plugin = self

        @tool_decorator(
            name="skills",
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
                from locus.observability.emit import EV_SKILL_ACTIVATED, emit_sync  # noqa: PLC0415

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
