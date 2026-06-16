# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""AgentSkills.io compliant skills system for Locus.

Skills are packaged instruction bundles (SKILL.md files) that agents
load on demand via progressive disclosure:
- L1: Agent sees skill catalog (names + descriptions) in system prompt
- L2: Agent activates a skill → full instructions loaded
- L3: Agent reads resource files (scripts/, references/, assets/)

Example:
    from locus.skills import Skill, SkillsPlugin

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

from locus.skills.models import Skill
from locus.skills.plugin import SkillsPlugin


__all__ = ["Skill", "SkillsPlugin"]
