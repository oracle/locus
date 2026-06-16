# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Skill data model — AgentSkills.io compliant.

A Skill is a packaged instruction bundle with YAML frontmatter metadata
and a Markdown body. Skills teach agents HOW to do something — they are
not executable functions (those are tools).

SKILL.md format:
    ---
    name: my-skill
    description: What it does and when to use it.
    allowed-tools: tool1 tool2
    license: Apache-2.0
    metadata:
      author: acme
    ---

    # Instructions
    Step-by-step guidance for the agent...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # PyYAML ships no inline types


# AgentSkills.io name validation: kebab-case, 1-64 chars, no consecutive hyphens
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_CONSECUTIVE_HYPHENS = re.compile(r"--")

# Resource directories per AgentSkills.io spec
_RESOURCE_DIRS = ("scripts", "references", "assets")


def validate_skill_name(name: str, strict: bool = False) -> bool:
    """Validate a skill name per AgentSkills.io specification.

    Args:
        name: Skill name to validate.
        strict: If True, raise ValueError on invalid. If False, return bool.

    Returns:
        True if valid, False if invalid (when strict=False).
    """
    errors: list[str] = []

    if not name:
        errors.append("Skill name cannot be empty")
    elif len(name) > 64:
        errors.append(f"Skill name exceeds 64 chars: {len(name)}")
    elif not _SKILL_NAME_PATTERN.match(name):
        errors.append(
            f"Invalid skill name '{name}': must be kebab-case (lowercase alphanumeric + hyphens)"
        )
    elif _CONSECUTIVE_HYPHENS.search(name):
        errors.append(f"Skill name '{name}' contains consecutive hyphens")

    if errors and strict:
        msg = "; ".join(errors)
        raise ValueError(msg)

    return len(errors) == 0


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
        content: Raw file content starting with ---

    Returns:
        Tuple of (frontmatter_dict, markdown_body)
    """
    if not content.startswith("---"):
        return {}, content

    # Find closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if end_match is None:
        return {}, content

    yaml_text = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :].strip()

    try:
        frontmatter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        return {}, content

    return frontmatter, body


@dataclass
class Skill:
    """A skill — packaged instructions for an agent.

    Skills follow the AgentSkills.io specification.

    Example:
        >>> skill = Skill(
        ...     name="code-review",
        ...     description="Review code for quality and security issues.",
        ...     instructions="# Code Review\\n1. Check error handling...",
        ... )

        >>> # Load from filesystem
        >>> skill = Skill.from_file(Path("./skills/code-review"))
    """

    name: str
    description: str
    instructions: str = ""
    path: Path | None = None
    allowed_tools: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    compatibility: str | None = None

    @classmethod
    def from_file(cls, path: Path | str) -> Skill:
        """Load a skill from a directory containing SKILL.md.

        Args:
            path: Path to skill directory or SKILL.md file.

        Returns:
            Loaded Skill instance.

        Raises:
            FileNotFoundError: If SKILL.md not found.
            ValueError: If required fields missing.
        """
        path = Path(path)

        if path.name == "SKILL.md":
            skill_file = path
            skill_dir = path.parent
        elif path.is_dir():
            skill_file = path / "SKILL.md"
            skill_dir = path
        else:
            msg = f"Expected directory or SKILL.md file, got: {path}"
            raise FileNotFoundError(msg)

        if not skill_file.exists():
            msg = f"SKILL.md not found in {skill_dir}"
            raise FileNotFoundError(msg)

        content = skill_file.read_text(encoding="utf-8")
        return cls.from_content(content, path=skill_dir)

    @classmethod
    def from_content(cls, content: str, path: Path | None = None) -> Skill:
        """Parse a skill from raw SKILL.md content.

        Args:
            content: Raw SKILL.md file content.
            path: Optional filesystem path for resource resolution.

        Returns:
            Parsed Skill instance.

        Raises:
            ValueError: If required fields (name, description) missing.
        """
        frontmatter, body = _parse_frontmatter(content)

        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

        if not name:
            msg = "SKILL.md missing required field: name"
            raise ValueError(msg)
        if not description:
            msg = "SKILL.md missing required field: description"
            raise ValueError(msg)

        # Parse allowed-tools (space-delimited string or list)
        allowed_tools_raw = frontmatter.get("allowed-tools")
        allowed_tools: list[str] | None = None
        if isinstance(allowed_tools_raw, str):
            allowed_tools = allowed_tools_raw.split()
        elif isinstance(allowed_tools_raw, list):
            allowed_tools = [str(t) for t in allowed_tools_raw]

        return cls(
            name=name,
            description=description,
            instructions=body,
            path=path,
            allowed_tools=allowed_tools,
            metadata=frontmatter.get("metadata", {}),
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
        )

    @classmethod
    def from_directory(cls, path: Path | str) -> list[Skill]:
        """Load all skills from a parent directory.

        Scans subdirectories for SKILL.md files.

        Args:
            path: Parent directory containing skill subdirectories.

        Returns:
            List of loaded skills.
        """
        path = Path(path)
        skills: list[Skill] = []

        if not path.is_dir():
            msg = f"Not a directory: {path}"
            raise FileNotFoundError(msg)

        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "SKILL.md").exists():
                try:
                    skills.append(cls.from_file(child))
                except (ValueError, FileNotFoundError):
                    continue  # Skip invalid skills

        return skills

    def list_resources(self, max_files: int = 20) -> list[str]:
        """List resource files from scripts/, references/, assets/ directories.

        Args:
            max_files: Maximum number of files to list.

        Returns:
            List of relative file paths.
        """
        if self.path is None:
            return []

        resources: list[str] = []
        for dir_name in _RESOURCE_DIRS:
            resource_dir = self.path / dir_name
            if resource_dir.is_dir():
                for f in sorted(resource_dir.iterdir()):
                    if f.is_file() and not f.name.startswith("."):
                        resources.append(f"{dir_name}/{f.name}")
                        if len(resources) >= max_files:
                            return resources

        return resources
