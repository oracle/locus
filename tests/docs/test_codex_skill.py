# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The Codex skill describes the SDK that is installed; this keeps the two in step.

``.agents/skills/locus/`` is what Codex reads to build with Locus. Its
front matter has to be what Codex expects, every reference the skill names
has to exist and be named, and every Python snippet in it has to pass the
same check the documentation pages get, so a renamed parameter or a removed
class fails here before it misleads an agent.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.check_snippets import check


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".agents" / "skills" / "locus"
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def test_front_matter_names_the_skill() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    assert match is not None, "SKILL.md starts with YAML front matter"
    fields = dict(
        line.split(":", 1) for line in match.group(1).splitlines() if re.match(r"^[a-z]", line)
    )
    assert fields["name"].strip() == "locus"
    description = fields["description"].strip()
    assert "Locus" in description and "locus-sdk" in description
    assert len(description) <= 1024, "Codex shortens long descriptions in its skills list"
    assert text[match.end() :].lstrip().startswith("# Locus\n")


def test_every_reference_is_named_and_present() -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    on_disk = sorted(p.name for p in (SKILL / "references").glob("*.md"))
    named = sorted(set(re.findall(r"`references/([a-z-]+\.md)`", skill)))
    assert on_disk, "the skill ships reference files"
    assert named == on_disk, f"SKILL.md names {named}, the directory holds {on_disk}"
    for name in on_disk:
        text = (SKILL / "references" / name).read_text(encoding="utf-8")
        assert text.startswith("# "), f"{name} starts with a title"


def test_every_snippet_in_the_skill_checks_against_the_sdk() -> None:
    checked, problems = check(SKILL)
    assert checked > 0, "the skill has Python snippets to check"
    assert problems == [], "\n".join(str(problem) for problem in problems)


def test_the_skill_carries_no_credential_shaped_text() -> None:
    for path in SKILL.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for pattern in (
            r"sk-[A-Za-z0-9_-]{20,}",
            r"ocid1\.(tenancy|compartment)\.oc1\.\.[a-z0-9]{20,}",
        ):
            assert re.search(pattern, text) is None, (
                f"{path.name} holds a value that looks like a credential"
            )
