# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The Codex scenario series is complete and consistent.

Every scenario directory has the task Codex gets and the check that judges
what it built; the runner discovers exactly those directories; and the README
table names each one, so a scenario cannot be added without being described.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "codex" / "scenarios"
NAME = re.compile(r"^\d{2}-[a-z][a-z0-9-]+$")


def _dirs() -> list[Path]:
    return sorted(p for p in SCENARIOS.iterdir() if p.is_dir() and not p.name.startswith("_"))


def test_every_scenario_has_a_prompt_and_an_executable_check() -> None:
    dirs = _dirs()
    assert len(dirs) >= 10, "the series has at least ten scenarios"
    for d in dirs:
        assert NAME.match(d.name), f"{d.name}: directories are NN-slug"
        prompt = (d / "PROMPT.md").read_text(encoding="utf-8")
        assert "Locus" in prompt and "test" in prompt.lower(), (
            f"{d.name}: the task names Locus and tests"
        )
        check = d / "check.sh"
        text = check.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/bash\n"), f"{d.name}: check.sh has a bash shebang"
        assert "set -euo pipefail" in text, f"{d.name}: check.sh fails on the first error"
        assert "ok:" in text, f"{d.name}: check.sh reports what it confirmed"
        assert os.access(check, os.X_OK), f"{d.name}: check.sh is executable"


def _listed(*flags: str) -> list[str]:
    out = subprocess.run(  # noqa: S603 - the repository's own script
        [sys.executable, str(SCENARIOS / "run.py"), "--list", *flags],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def test_the_runner_discovers_the_same_scenarios() -> None:
    assert _listed() == [d.name for d in _dirs()]
    offline = _listed("--skip-live")
    assert all(not (SCENARIOS / name / "live").exists() for name in offline)
    assert len(offline) < len(_dirs()), "at least one scenario is live"


def test_the_readme_table_names_every_scenario() -> None:
    readme = (SCENARIOS / "README.md").read_text(encoding="utf-8")
    for d in _dirs():
        assert f"| `{d.name}` |" in readme, f"{d.name} is missing from the README table"
        row = next(line for line in readme.splitlines() if line.startswith(f"| `{d.name}` |"))
        is_live = (d / "live").exists()
        assert row.rstrip().endswith("| yes |") == is_live, f"{d.name}: the Live column is wrong"
