# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""AgentSkills.io compliant skills system for Tulip.

Skills are packaged instruction bundles (SKILL.md files) that agents
load on demand via progressive disclosure:
- L1: Agent sees skill catalog (names + descriptions) in system prompt
- L2: Agent activates a skill → full instructions loaded
- L3: Agent reads resource files (scripts/, references/, assets/)

Example:
    from tulip.skills import Skill, SkillsPlugin

    # Load from filesystem
    skills = Skill.from_directory("./skills")

    # Or create programmatically
    skill = Skill(
        name="code-review",
        description="Use when reviewing code for quality and security issues.",
        instructions="# Code Review Checklist\\n1. Check error handling...",
    )

    # Attach to agent
    agent = Agent(config=AgentConfig(
        model=model,
        skills=[skill],  # or paths to skill directories
    ))
"""

from tulip.skills.models import Skill
from tulip.skills.plugin import SkillsPlugin


__all__ = ["Skill", "SkillsPlugin"]
