# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.skills.models`` (Skill + frontmatter parsing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from locus.skills.models import Skill, validate_skill_name


# ---------------------------------------------------------------------------
# validate_skill_name
# ---------------------------------------------------------------------------


class TestValidateSkillName:
    """The kebab-case rules from the AgentSkills.io spec."""

    @pytest.mark.parametrize(
        "name",
        ["a", "code-review", "my-skill-1", "abc123", "0", "ab"],
    )
    def test_accepts_valid_names(self, name: str) -> None:
        assert validate_skill_name(name) is True

    @pytest.mark.parametrize(
        "name",
        ["", "Code-Review", "skill_name", "-leading", "trailing-", "a--b"],
    )
    def test_rejects_invalid_names(self, name: str) -> None:
        assert validate_skill_name(name) is False

    def test_rejects_overlong_name(self) -> None:
        assert validate_skill_name("a" * 65) is False

    def test_strict_mode_raises_on_invalid(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_skill_name("", strict=True)

    def test_strict_mode_raises_on_overlong(self) -> None:
        with pytest.raises(ValueError, match="exceeds 64 chars"):
            validate_skill_name("a" * 65, strict=True)

    def test_strict_mode_raises_on_pattern_violation(self) -> None:
        with pytest.raises(ValueError, match="kebab-case"):
            validate_skill_name("CamelCase", strict=True)

    def test_strict_mode_passes_on_valid(self) -> None:
        # Smoke test — strict path on a valid name should not raise.
        assert validate_skill_name("ok-name", strict=True) is True


# ---------------------------------------------------------------------------
# Skill.from_content (frontmatter parsing)
# ---------------------------------------------------------------------------


class TestSkillFromContent:
    """Frontmatter + body extraction from SKILL.md content."""

    def test_parses_minimum_skill(self) -> None:
        content = "---\nname: my-skill\ndescription: Does a thing.\n---\n# Body"
        skill = Skill.from_content(content)
        assert skill.name == "my-skill"
        assert skill.description == "Does a thing."
        assert skill.instructions == "# Body"

    def test_parses_allowed_tools_string(self) -> None:
        content = "---\nname: s\ndescription: d.\nallowed-tools: a b c\n---\nBody"
        skill = Skill.from_content(content)
        assert skill.allowed_tools == ["a", "b", "c"]

    def test_parses_allowed_tools_list(self) -> None:
        content = "---\nname: s\ndescription: d.\nallowed-tools:\n  - tool_a\n  - tool_b\n---\nBody"
        skill = Skill.from_content(content)
        assert skill.allowed_tools == ["tool_a", "tool_b"]

    def test_no_allowed_tools_yields_none(self) -> None:
        skill = Skill.from_content("---\nname: s\ndescription: d.\n---\n")
        assert skill.allowed_tools is None

    def test_parses_metadata_license_compatibility(self) -> None:
        content = (
            "---\n"
            "name: s\n"
            "description: d.\n"
            "license: Apache-2.0\n"
            "compatibility: locus>=1\n"
            "metadata:\n  author: acme\n"
            "---\nBody"
        )
        skill = Skill.from_content(content)
        assert skill.license == "Apache-2.0"
        assert skill.compatibility == "locus>=1"
        assert skill.metadata == {"author": "acme"}

    def test_missing_name_raises(self) -> None:
        content = "---\ndescription: d.\n---\nBody"
        with pytest.raises(ValueError, match="missing required field: name"):
            Skill.from_content(content)

    def test_missing_description_raises(self) -> None:
        content = "---\nname: s\n---\nBody"
        with pytest.raises(ValueError, match="missing required field: description"):
            Skill.from_content(content)

    def test_no_frontmatter_returns_empty_meta_with_full_body(self) -> None:
        # Without frontmatter, both ``name`` and ``description`` are
        # missing from the parsed dict — we hit the same validation gate.
        with pytest.raises(ValueError):
            Skill.from_content("Just a body, no frontmatter.")

    def test_unterminated_frontmatter_treated_as_no_frontmatter(self) -> None:
        # The parser bails out (no closing ``---``), so the whole thing
        # is treated as the body and validation fails for missing name.
        with pytest.raises(ValueError):
            Skill.from_content("---\nname: s\nNo closing fence.")

    def test_invalid_yaml_treated_as_no_frontmatter(self) -> None:
        # Malformed YAML inside fences → empty frontmatter, validation fails.
        bad = "---\nname: s\n\tbroken: : :\n---\nBody"
        with pytest.raises(ValueError):
            Skill.from_content(bad)


# ---------------------------------------------------------------------------
# Skill.from_file
# ---------------------------------------------------------------------------


class TestSkillFromFile:
    """Filesystem loading."""

    def test_loads_from_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: Review code.\n---\nBody"
        )
        skill = Skill.from_file(skill_dir)
        assert skill.name == "code-review"
        assert skill.path == skill_dir

    def test_loads_from_skill_md_path(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: code-review\ndescription: x.\n---\nBody")
        skill = Skill.from_file(skill_md)
        assert skill.name == "code-review"
        assert skill.path == skill_dir

    def test_missing_skill_md_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match=r"SKILL\.md not found"):
            Skill.from_file(empty)

    def test_neither_dir_nor_skill_md_raises(self, tmp_path: Path) -> None:
        random_file = tmp_path / "random.txt"
        random_file.write_text("hi")
        with pytest.raises(FileNotFoundError, match=r"Expected directory or SKILL\.md"):
            Skill.from_file(random_file)


# ---------------------------------------------------------------------------
# Skill.from_directory
# ---------------------------------------------------------------------------


class TestSkillFromDirectory:
    def test_loads_all_valid_skills(self, tmp_path: Path) -> None:
        for name in ("alpha", "beta"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x.\n---\nBody")
        skills = Skill.from_directory(tmp_path)
        names = sorted(s.name for s in skills)
        assert names == ["alpha", "beta"]

    def test_skips_invalid_skills(self, tmp_path: Path) -> None:
        valid = tmp_path / "good"
        valid.mkdir()
        (valid / "SKILL.md").write_text("---\nname: good\ndescription: x.\n---\n")

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("---\nname: bad\n---\n")  # missing desc

        skills = Skill.from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "good"

    def test_skips_dirs_without_skill_md(self, tmp_path: Path) -> None:
        (tmp_path / "no_skill").mkdir()
        skills = Skill.from_directory(tmp_path)
        assert skills == []

    def test_non_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.txt"
        f.write_text("x")
        with pytest.raises(FileNotFoundError, match="Not a directory"):
            Skill.from_directory(f)


# ---------------------------------------------------------------------------
# Skill.list_resources
# ---------------------------------------------------------------------------


class TestSkillListResources:
    def test_empty_when_no_path(self) -> None:
        skill = Skill(name="s", description="d.")
        assert skill.list_resources() == []

    def test_lists_resource_dirs(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d.\n---\n")
        for sub in ("scripts", "references", "assets"):
            (skill_dir / sub).mkdir()
        (skill_dir / "scripts" / "run.sh").write_text("#!/bin/sh")
        (skill_dir / "references" / "spec.md").write_text("spec")
        (skill_dir / "assets" / "logo.png").write_bytes(b"\x89PNG")

        skill = Skill.from_file(skill_dir)
        resources = skill.list_resources()
        assert "scripts/run.sh" in resources
        assert "references/spec.md" in resources
        assert "assets/logo.png" in resources

    def test_skips_dotfiles(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d.\n---\n")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / ".hidden").write_text("x")
        (skill_dir / "scripts" / "visible.sh").write_text("x")

        skill = Skill.from_file(skill_dir)
        resources = skill.list_resources()
        assert "scripts/visible.sh" in resources
        assert "scripts/.hidden" not in resources

    def test_max_files_enforced(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d.\n---\n")
        (skill_dir / "scripts").mkdir()
        for i in range(10):
            (skill_dir / "scripts" / f"f{i}.sh").write_text("x")

        skill = Skill.from_file(skill_dir)
        resources = skill.list_resources(max_files=3)
        assert len(resources) == 3

    def test_no_resource_dirs_yields_empty(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d.\n---\n")
        skill = Skill.from_file(skill_dir)
        assert skill.list_resources() == []
