"""The grouped playbook shape is validated against the corpus it came from.

`required_probes`, `uses` and the `step_groups` adapter were ported from
observai/optic. A port is a claim about a format, and the only honest way to
check it is to load that format's real files rather than the paraphrase I wrote
while reading them.

The corpus lives outside this repository, so this SKIPS when it is absent —
loudly, naming what it would have checked. A skip that reads as a pass is how
the studio's browser tier sat green for months without ever running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulip.playbooks.loader import load_playbook
from tulip.skills.models import Skill


CORPUS = Path.home() / "Projects/observai/observai-7ee831391/optic/data/fusion"

pytestmark = pytest.mark.skipif(
    not CORPUS.is_dir(),
    reason=(
        f"optic corpus not present at {CORPUS} — these cases load its real "
        "playbooks and skills to check the ported format against its source"
    ),
)


def _playbooks() -> list[Path]:
    return sorted((CORPUS / "playbooks").glob("*.json"))


def _skills() -> list[Path]:
    return sorted(d for d in (CORPUS / "skills").iterdir() if (d / "SKILL.md").exists())


def test_every_optic_playbook_loads() -> None:
    """Grouped `step_groups` in, flat ordered steps out — for all of them."""
    failures: list[str] = []
    for path in _playbooks():
        try:
            playbook = load_playbook(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001 — the failure IS the finding
            failures.append(f"{path.name}: {exc}")
            continue
        if not playbook.steps:
            failures.append(f"{path.name}: loaded with no steps")
    assert not failures, "optic playbooks that do not load:\n" + "\n".join(failures)


def test_every_optic_skill_loads() -> None:
    failures: list[str] = []
    for path in _skills():
        try:
            Skill.from_file(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {exc}")
    assert not failures, "optic skills that do not load:\n" + "\n".join(failures)


def test_the_skill_references_actually_arrive() -> None:
    """Loading is not enough — the LINK is the thing that was missing here.

    A format adapter that quietly dropped `guidance.skill_refs` would still
    report every file as loading.
    """
    carried = sum(
        len(step.uses)
        for path in _playbooks()
        for step in load_playbook(json.loads(path.read_text())).steps
    )
    assert carried > 0, "no step in the corpus carried a skill reference into `uses`"


def test_the_declared_evidence_actually_arrives() -> None:
    carried = sum(len(Skill.from_file(path).required_probes) for path in _skills())
    assert carried > 0, "no skill in the corpus carried a required_probe"


def test_a_step_and_its_skill_meet() -> None:
    """End to end: a step's `uses` resolves to a skill, and its probes apply.

    This is the whole chain — playbook → skill → evidence — checked on real
    artefacts rather than a fixture written to pass.
    """
    skills = {path.name: Skill.from_file(path) for path in _skills()}
    by_id = {
        str(skill.metadata.get("optic-skill-id") or name): skill for name, skill in skills.items()
    }
    by_id.update(skills)

    resolved = 0
    for path in _playbooks():
        for step in load_playbook(json.loads(path.read_text())).steps:
            for ref in step.uses:
                if ref in by_id:
                    resolved += 1
    assert resolved > 0, (
        "no step reference resolved to a skill — the two halves of the corpus "
        "load but do not meet, which would make `uses` decorative"
    )
