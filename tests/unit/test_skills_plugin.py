# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.skills.plugin`` (SkillsPlugin)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from locus.core.messages import Message
from locus.skills.models import Skill
from locus.skills.plugin import SkillsPlugin


@pytest.fixture
def alpha() -> Skill:
    return Skill(
        name="alpha",
        description="First test skill.",
        instructions="Step 1.\nStep 2.",
    )


@pytest.fixture
def beta() -> Skill:
    return Skill(
        name="beta",
        description="Second test skill.",
        instructions="Do beta things.",
        allowed_tools=["foo", "bar"],
        compatibility="locus>=0.1",
    )


# ---------------------------------------------------------------------------
# Construction — accepts Skills, paths, or parent dirs
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_accepts_skill_instances(self, alpha: Skill, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha, beta])
        assert plugin.available_skills == ["alpha", "beta"]

    def test_accepts_skill_directory_path_str(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: Review code.\n---\nBody"
        )
        plugin = SkillsPlugin(skills=[str(skill_dir)])
        assert plugin.available_skills == ["code-review"]

    def test_accepts_skill_directory_path_obj(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: review\ndescription: x.\n---\n")
        plugin = SkillsPlugin(skills=[skill_dir])
        assert plugin.available_skills == ["review"]

    def test_accepts_parent_directory_with_multiple_skills(self, tmp_path: Path) -> None:
        for n in ("a-one", "b-two"):
            d = tmp_path / n
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {n}\ndescription: x.\n---\n")
        plugin = SkillsPlugin(skills=[tmp_path])
        assert plugin.available_skills == ["a-one", "b-two"]

    def test_silently_skips_nonexistent_path(self, tmp_path: Path) -> None:
        plugin = SkillsPlugin(skills=[tmp_path / "nonexistent"])
        assert plugin.available_skills == []

    def test_silently_skips_unsupported_source_type(self) -> None:
        # Source is neither a Skill nor a path-like — both branches of
        # the isinstance checks fall through; the loop just continues.
        plugin = SkillsPlugin(skills=[42])  # type: ignore[list-item]
        assert plugin.available_skills == []


# ---------------------------------------------------------------------------
# Catalog generation (L1)
# ---------------------------------------------------------------------------


class TestCatalogXml:
    def test_includes_name_and_description(self, alpha: Skill, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha, beta])
        xml = plugin._generate_catalog_xml()
        assert "<name>alpha</name>" in xml
        assert "<description>First test skill.</description>" in xml
        assert "<name>beta</name>" in xml

    def test_escapes_xml_special_chars(self) -> None:
        skill = Skill(name="esc", description="A & B < C")
        plugin = SkillsPlugin(skills=[skill])
        xml = plugin._generate_catalog_xml()
        assert "A &amp; B &lt; C" in xml

    def test_includes_location_when_path_set(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "loc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: loc\ndescription: x.\n---\n")
        plugin = SkillsPlugin(skills=[skill_dir])
        xml = plugin._generate_catalog_xml()
        assert "<location>" in xml
        assert "SKILL.md" in xml

    def test_no_location_when_path_unset(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        xml = plugin._generate_catalog_xml()
        assert "<location>" not in xml

    def test_empty_when_no_skills(self) -> None:
        plugin = SkillsPlugin(skills=[])
        assert plugin._generate_catalog_xml() == ""


# ---------------------------------------------------------------------------
# Skill response formatting (L2 + L3)
# ---------------------------------------------------------------------------


class TestFormatSkillResponse:
    def test_returns_instructions_for_minimal_skill(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        out = plugin._format_skill_response(alpha)
        assert "Step 1." in out
        assert "---" not in out  # no metadata footer

    def test_includes_allowed_tools(self, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[beta])
        out = plugin._format_skill_response(beta)
        assert "Allowed tools: foo, bar" in out

    def test_includes_compatibility(self, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[beta])
        out = plugin._format_skill_response(beta)
        assert "Compatibility: locus>=0.1" in out

    def test_includes_location_when_path_set(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "loc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: loc\ndescription: x.\n---\nBody")
        plugin = SkillsPlugin(skills=[skill_dir])
        skill = next(iter(plugin._skills.values()))
        out = plugin._format_skill_response(skill)
        assert f"Location: {skill_dir}" in out

    def test_includes_resource_files(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "with-resources"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: with-resources\ndescription: x.\n---\n")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("#!/bin/sh")
        plugin = SkillsPlugin(skills=[skill_dir])
        skill = next(iter(plugin._skills.values()))
        out = plugin._format_skill_response(skill)
        assert "Resource files:" in out
        assert "scripts/run.sh" in out


# ---------------------------------------------------------------------------
# on_before_model_call hook (catalog injection)
# ---------------------------------------------------------------------------


@dataclass
class _ModelEvent:
    """Minimal stand-in for ``BeforeModelCallEvent``."""

    messages: list[Message]


class TestOnBeforeModelCall:
    @pytest.mark.asyncio
    async def test_injects_catalog_when_skills_present(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        event = _ModelEvent(
            messages=[
                Message.system("Original system."),
                Message.user("hello"),
            ]
        )
        await plugin.on_before_model_call(event)
        # Catalog inserted as second message (after the existing system).
        assert len(event.messages) == 3
        assert event.messages[1].role.value == "system"
        assert "available_skills" in event.messages[1].content

    @pytest.mark.asyncio
    async def test_inserts_at_index_zero_when_no_system_message(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        event = _ModelEvent(messages=[Message.user("hello")])
        await plugin.on_before_model_call(event)
        assert event.messages[0].role.value == "system"
        assert "available_skills" in event.messages[0].content

    @pytest.mark.asyncio
    async def test_noop_when_no_skills(self) -> None:
        plugin = SkillsPlugin(skills=[])
        event = _ModelEvent(messages=[Message.user("hello")])
        await plugin.on_before_model_call(event)
        # Untouched.
        assert len(event.messages) == 1


# ---------------------------------------------------------------------------
# Activation tool
# ---------------------------------------------------------------------------


class TestActivationTool:
    def test_returns_tool(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        tool = plugin.get_activation_tool()
        assert tool.name == "skills"

    def test_activate_loads_skill_response(self, alpha: Skill, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha, beta])
        tool = plugin.get_activation_tool()
        out = tool.fn("alpha")
        assert "Step 1." in out

    def test_activate_unknown_skill_lists_available(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        tool = plugin.get_activation_tool()
        out = tool.fn("nonexistent")
        assert "Unknown skill" in out
        assert "alpha" in out

    def test_activate_empty_name_returns_error(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        tool = plugin.get_activation_tool()
        assert "skill_name is required" in tool.fn("")

    def test_activation_tracked_in_order(self, alpha: Skill, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha, beta])
        tool = plugin.get_activation_tool()
        tool.fn("alpha")
        tool.fn("beta")
        assert plugin.activated_skills == ["alpha", "beta"]

    def test_re_activation_moves_to_end(self, alpha: Skill, beta: Skill) -> None:
        # Re-activating an existing skill bumps it to the most-recent slot.
        plugin = SkillsPlugin(skills=[alpha, beta])
        tool = plugin.get_activation_tool()
        tool.fn("alpha")
        tool.fn("beta")
        tool.fn("alpha")
        assert plugin.activated_skills == ["beta", "alpha"]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_available_skills_sorted(self, alpha: Skill, beta: Skill) -> None:
        plugin = SkillsPlugin(skills=[beta, alpha])
        assert plugin.available_skills == ["alpha", "beta"]

    def test_activated_skills_initially_empty(self, alpha: Skill) -> None:
        plugin = SkillsPlugin(skills=[alpha])
        assert plugin.activated_skills == []

    def test_name_attribute(self) -> None:
        plugin = SkillsPlugin(skills=[])
        assert plugin.name == "skills"
