# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus codex``: the skill lands where Codex looks, once, and Codex starts with the right settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from locus import codex
from locus.cli import main


if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_SKILL = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "locus"


def test_the_bundled_skill_is_the_repository_copy() -> None:
    source = codex.skill_source()
    assert (source / "SKILL.md").is_file()
    text = (source / "SKILL.md").read_text()
    assert text.startswith("---\nname: locus\n")
    assert (
        source.resolve() == REPO_SKILL.resolve()
        or (source / "SKILL.md").read_bytes() == (REPO_SKILL / "SKILL.md").read_bytes()
    )


def test_init_installs_the_skill_and_the_agents_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["codex", "init", str(tmp_path)]) == 0
    skill = tmp_path / ".agents" / "skills" / "locus"
    assert (skill / "SKILL.md").read_bytes() == (REPO_SKILL / "SKILL.md").read_bytes()
    for reference in (REPO_SKILL / "references").glob("*.md"):
        assert (skill / "references" / reference.name).read_bytes() == reference.read_bytes()
    agents = (tmp_path / "AGENTS.md").read_text()
    assert agents.startswith("# Agent instructions\n")
    assert "## Locus" in agents
    assert "`.agents/skills/locus/SKILL.md`" in agents
    out = capsys.readouterr().out
    assert "written to" in out
    assert "AGENTS.md" in out


def test_init_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "AGENTS.md").write_text("# My project\n\nHouse rules.\n")
    assert main(["codex", "init", str(tmp_path)]) == 0
    first = (tmp_path / "AGENTS.md").read_text()
    assert first.startswith("# My project\n\nHouse rules.\n")
    assert first.count("## Locus") == 1
    capsys.readouterr()

    assert main(["codex", "init", str(tmp_path)]) == 0
    assert (tmp_path / "AGENTS.md").read_text() == first
    out = capsys.readouterr().out
    assert "up to date" in out
    assert "AGENTS.md" not in out


def test_install_rewrites_only_what_differs(tmp_path: Path) -> None:
    written = codex.install(tmp_path)
    assert written
    assert codex.install(tmp_path) == []
    (tmp_path / "SKILL.md").write_text("stale")
    (tmp_path / "notes.md").write_text("mine")
    assert codex.install(tmp_path) == [tmp_path / "SKILL.md"]
    assert (tmp_path / "notes.md").read_text() == "mine"


def test_init_home_installs_for_every_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert main(["codex", "init", "--home"]) == 0
    assert (tmp_path / ".agents" / "skills" / "locus" / "SKILL.md").is_file()
    assert not (tmp_path / "AGENTS.md").exists()
    assert str(tmp_path / ".agents" / "skills" / "locus") in capsys.readouterr().out


def test_path_prints_the_bundled_skill(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["codex", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(codex.skill_source())


@pytest.mark.parametrize(
    ("prompt", "kwargs", "expected_tail"),
    [
        ("build it", {}, ["build it"]),
        (None, {"exec_mode": True}, []),
        ("build it", {"codex_profile": "oci"}, ["-p", "oci", "build it"]),
        ("build it", {"extra": ["--", "-m", "gpt-5.5"]}, ["-m", "gpt-5.5", "build it"]),
        ("build it", {"extra": ["-m", "gpt-5.5"]}, ["-m", "gpt-5.5", "build it"]),
    ],
)
def test_codex_argv(prompt: str | None, kwargs: dict[str, Any], expected_tail: list[str]) -> None:
    argv = codex.codex_argv(prompt, **kwargs)
    head = ["codex", "exec"] if kwargs.get("exec_mode") else ["codex"]
    assert argv == [*head, *codex.CODEX_FLAGS, *expected_tail]


def test_run_installs_here_and_hands_over_to_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    calls: list[tuple[str, Sequence[str]]] = []
    monkeypatch.setattr(codex.shutil, "which", lambda name: f"/opt/bin/{name}")
    monkeypatch.setattr(codex.os, "execv", lambda exe, argv: calls.append((exe, argv)))

    code = main(["codex", "run", "--codex-profile", "oci", "triage tickets", "--", "-m", "x"])

    assert code == 0
    assert (tmp_path / ".agents" / "skills" / "locus" / "SKILL.md").is_file()
    assert "## Locus" in (tmp_path / "AGENTS.md").read_text()
    assert calls == [
        (
            "/opt/bin/codex",
            ["codex", *codex.CODEX_FLAGS, "-p", "oci", "-m", "x", "triage tickets"],
        )
    ]


def test_run_exec_mode_without_a_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    calls: list[Sequence[str]] = []
    monkeypatch.setattr(codex.shutil, "which", lambda name: f"/opt/bin/{name}")
    monkeypatch.setattr(codex.os, "execv", lambda exe, argv: calls.append(argv))

    assert main(["codex", "run", "--exec"]) == 0
    assert calls == [["codex", "exec", *codex.CODEX_FLAGS]]


def test_run_with_the_skill_in_home_leaves_the_project_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    codex.install(home / codex.SKILL_DIR)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(codex.shutil, "which", lambda name: f"/opt/bin/{name}")
    monkeypatch.setattr(codex.os, "execv", lambda exe, argv: None)

    assert main(["codex", "run", "hello"]) == 0
    assert not (project / ".agents").exists()
    assert not (project / "AGENTS.md").exists()


def test_run_without_codex_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    monkeypatch.setattr(codex.shutil, "which", lambda name: None)

    assert main(["codex", "run", "hello"]) == 2
    err = capsys.readouterr().err
    assert "npm install -g @openai/codex" in err


def test_a_missing_skill_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(codex, "SKILL_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(codex.resources, "files", lambda name: tmp_path / "nothing")
    with pytest.raises(codex.CodexError, match="missing"):
        codex.skill_source()
