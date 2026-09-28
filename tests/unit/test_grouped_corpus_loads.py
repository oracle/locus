"""The grouped playbook shape, loaded from files on disk.

`test_grouped_playbook_shape.py` pins each rule with inline fixtures. This
loads a small corpus from tests/fixtures/grouped_corpus the way a deployment
would, and checks the whole chain: playbook -> step `uses` -> skill -> probes.
The two files deliberately use both spellings of the probe key.
"""

from __future__ import annotations

import json
from pathlib import Path

from tulip.playbooks.loader import load_playbook
from tulip.skills.models import Skill


CORPUS = Path(__file__).parent.parent / "fixtures" / "grouped_corpus"


def _playbooks() -> list[Path]:
    return sorted((CORPUS / "playbooks").glob("*.json"))


def _skills() -> list[Path]:
    return sorted(d for d in (CORPUS / "skills").iterdir() if (d / "SKILL.md").exists())


def test_the_corpus_is_present() -> None:
    """An empty glob would make every other case here pass vacuously."""
    assert len(_playbooks()) >= 2
    assert len(_skills()) >= 2


def test_every_playbook_loads() -> None:
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
    assert not failures, "playbooks that do not load:\n" + "\n".join(failures)


def test_every_skill_loads_with_its_probes() -> None:
    """Both spellings of the probe key arrive; neither is silently dropped."""
    for path in _skills():
        assert Skill.from_file(path).required_probes, f"{path.name} lost its required_probes"


def test_a_step_and_its_skill_meet() -> None:
    """End to end: every step's `uses` resolves to a skill in the corpus."""
    skills = {path.name: Skill.from_file(path) for path in _skills()}
    by_id = {str(skill.metadata.get("skill-id") or name): skill for name, skill in skills.items()}
    by_id.update(skills)

    refs = [
        ref
        for path in _playbooks()
        for step in load_playbook(json.loads(path.read_text())).steps
        for ref in step.uses
    ]
    assert refs, "no step carried a skill reference into `uses`"
    unresolved = [ref for ref in refs if ref not in by_id]
    assert not unresolved, f"step references with no skill: {unresolved}"
