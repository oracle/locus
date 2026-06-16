# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Skill metadata layered on top of :class:`locus.skills.Skill`.

The router uses skills the same way the agent loop already does — via
:class:`locus.skills.SkillsPlugin` and its progressive-disclosure
catalog. The index here only adds the *domain* metadata the router
needs to scope skills per :class:`GoalFrame.domain`, and resolves the
skills the chosen builder should attach to every emitted ``Agent``.

There is no parallel skill storage. The :class:`Skill` instances live
where they were loaded (``Skill.from_directory(...)``) — the index just
holds them with a domain tag so the compiler can hand the right subset
to each builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from locus.skills.models import Skill


class SkillIndex:
    """A domain-tagged view over a list of :class:`Skill` instances."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._domains: dict[str, str] = {}  # skill_name -> domain (empty = global)

    def register(self, skill: Skill, *, domain: str = "") -> None:
        """Add a skill to the index with an optional domain tag.

        ``domain=""`` (the default) means "applies to every domain";
        :meth:`for_domain` always returns these alongside domain-specific
        matches.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name!r}")
        self._skills[skill.name] = skill
        self._domains[skill.name] = domain

    def register_many(self, skills: list[Skill], *, domain: str = "") -> None:
        """Register multiple skills under the same domain."""
        for skill in skills:
            self.register(skill, domain=domain)

    def get(self, name: str) -> Skill:
        """Return a skill by name. Raises :class:`KeyError` if missing."""
        if name not in self._skills:
            available = sorted(self._skills.keys())
            raise KeyError(
                f"Unknown skill {name!r}. Available: {available}",
            )
        return self._skills[name]

    def for_domain(self, domain: str) -> list[Skill]:
        """Skills tagged with ``domain``, plus every globally-tagged skill.

        The router calls this with :attr:`GoalFrame.domain` to pick the
        catalogue handed to each compiled :class:`Agent`. Globally-tagged
        skills (registered with ``domain=""``) are always included so a
        common catalogue (e.g. "communication tone", "safety checks") is
        available everywhere.
        """
        return [self._skills[name] for name, d in self._domains.items() if d in (domain, "")]

    def all(self) -> list[Skill]:
        """Every registered skill, in registration order."""
        return list(self._skills.values())

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)
