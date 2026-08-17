# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for the release-metadata consistency check.

The check exists because a human checklist failed: SECURITY.md claimed 2.1.x
support while the package shipped 2.10.0. So the tests that matter are the
ones proving it *fails* on each kind of drift — a consistency checker that
only ever passes is indistinguishable from no checker at all.

Each test builds a complete, self-consistent repo in a tmp_path and then
breaks exactly one thing, so a failure names the drift rather than a fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release_consistency.py"


def _load(root: Path):
    """Import the script with its ROOT pointed at a throwaway repo."""
    spec = importlib.util.spec_from_file_location(f"_rc_{root.name}", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


def _repo(
    tmp_path: Path,
    *,
    pyproject: str = "2.11.0",
    dunder: str = "2.11.0",
    changelog: str = "2.11.0",
    security_major: str = "2",
) -> Path:
    """A complete repo, consistent unless a caller asks for drift."""
    (tmp_path / "src" / "tulip").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "tulip-agents"\nversion = "{pyproject}"\n'
    )
    (tmp_path / "src" / "tulip" / "__init__.py").write_text(f'__version__ = "{dunder}"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{changelog}] - 2026-08-16\n\n- a change\n"
    )
    (tmp_path / "SECURITY.md").write_text(
        f"# Security Policy\n\n## Supported Versions\n\n"
        f"| Latest `{security_major}.x` minor | yes |\n"
    )
    return tmp_path


# ------------------------------------------------------------------ passes --


def test_a_consistent_repo_passes(tmp_path: Path) -> None:
    module = _load(_repo(tmp_path))
    version = module.check_versions_agree(verbose=False)
    module.check_changelog(version, verbose=False)
    module.check_security_major(version, verbose=False)
    assert version == "2.11.0"


def test_the_real_repo_is_consistent() -> None:
    """The check must pass against the repo it ships in."""
    module = _load(SCRIPT.parent.parent)
    version = module.check_versions_agree(verbose=False)
    module.check_changelog(version, verbose=False)
    module.check_security_major(version, verbose=False)


# ----------------------------------------------------------------- catches --


def test_desynced_versions_name_both_values(tmp_path: Path) -> None:
    module = _load(_repo(tmp_path, pyproject="2.12.0", dunder="2.11.0"))
    with pytest.raises(module.ConsistencyError) as exc:
        module.check_versions_agree(verbose=False)
    assert "2.12.0" in str(exc.value)
    assert "2.11.0" in str(exc.value)


def test_missing_changelog_entry_lists_what_was_found(tmp_path: Path) -> None:
    module = _load(_repo(tmp_path, changelog="2.99.0"))
    with pytest.raises(module.ConsistencyError) as exc:
        module.check_changelog("2.11.0", verbose=False)
    assert "no entry for 2.11.0" in str(exc.value)
    assert "2.99.0" in str(exc.value)


def test_unreleased_heading_alone_is_not_enough(tmp_path: Path) -> None:
    """`## [Unreleased]` is the working area, not a record of what shipped."""
    root = _repo(tmp_path)
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n- pending\n")
    module = _load(root)
    with pytest.raises(module.ConsistencyError, match=r"no entry for 2\.11\.0"):
        module.check_changelog("2.11.0", verbose=False)


def test_security_covering_the_wrong_major_is_caught(tmp_path: Path) -> None:
    module = _load(_repo(tmp_path, security_major="9"))
    with pytest.raises(module.ConsistencyError) as exc:
        module.check_security_major("2.11.0", verbose=False)
    assert "`2.x`" in str(exc.value)
    assert "['9']" in str(exc.value)


def test_a_missing_document_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "SECURITY.md").unlink()
    module = _load(root)
    with pytest.raises(module.ConsistencyError, match=r"SECURITY\.md is missing"):
        module.check_security_major("2.11.0", verbose=False)


def test_missing_dunder_version_is_reported(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "src" / "tulip" / "__init__.py").write_text("# no version here\n")
    module = _load(root)
    with pytest.raises(module.ConsistencyError, match=r"declares no __version__"):
        module.dunder_version()


# -------------------------------------------------------------------- main --


def test_main_returns_zero_on_a_consistent_repo(tmp_path: Path, monkeypatch) -> None:
    module = _load(_repo(tmp_path))
    monkeypatch.setattr(sys, "argv", ["check_release_consistency.py"])
    assert module.main() == 0


def test_main_returns_one_and_explains_on_drift(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load(_repo(tmp_path, pyproject="2.12.0", dunder="2.11.0"))
    monkeypatch.setattr(sys, "argv", ["check_release_consistency.py"])
    assert module.main() == 1
    assert "cannot agree with itself" in capsys.readouterr().err


def test_verbose_reports_each_passing_check(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load(_repo(tmp_path))
    monkeypatch.setattr(sys, "argv", ["check_release_consistency.py", "--verbose"])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "CHANGELOG.md" in out
    assert "SECURITY.md" in out
