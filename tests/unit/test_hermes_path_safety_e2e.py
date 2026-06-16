# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration test for path safety (A.4) wired through a real file tool.

Builds a tiny ``@tool`` that opens a model-supplied path under a
fixed base directory and demonstrates that
:func:`~locus.tools.path_safety.safe_resolve` blocks every escape
vector against real filesystem state (symlinks, traversal, absolute
paths, non-existent siblings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locus.core.errors import ValidationError
from locus.tools.path_safety import safe_resolve


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a base workspace plus a sibling 'secret' dir outside it."""
    base = tmp_path / "workspace"
    base.mkdir()
    (base / "data.txt").write_text("safe content")
    (base / "subdir").mkdir()
    (base / "subdir" / "nested.txt").write_text("nested content")

    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "creds").write_text("super secret")

    return base


def _read_file_tool(base: Path, user_path: str) -> str:
    """Stand-in for a user @tool that reads a file under ``base``."""
    target = safe_resolve(base, user_path)
    return target.read_text()


# ---------------------------------------------------------------------------
# Happy path: legitimate reads work end-to-end.
# ---------------------------------------------------------------------------


class TestLegitimateAccess:
    def test_read_root_file(self, workspace: Path) -> None:
        assert _read_file_tool(workspace, "data.txt") == "safe content"

    def test_read_nested_file(self, workspace: Path) -> None:
        assert _read_file_tool(workspace, "subdir/nested.txt") == "nested content"


# ---------------------------------------------------------------------------
# Traversal vectors — all blocked end-to-end.
# ---------------------------------------------------------------------------


class TestTraversalBlocked:
    @pytest.mark.parametrize(
        "evil",
        [
            "../secret/creds",
            "../../secret/creds",
            "subdir/../../secret/creds",
            "./subdir/../../secret/creds",
        ],
    )
    def test_dotdot_rejected(self, workspace: Path, evil: str) -> None:
        with pytest.raises(ValidationError, match="outside the allowed base"):
            _read_file_tool(workspace, evil)

    def test_absolute_outside_base_rejected(self, workspace: Path) -> None:
        outside = workspace.parent / "secret" / "creds"
        with pytest.raises(ValidationError):
            _read_file_tool(workspace, str(outside))


# ---------------------------------------------------------------------------
# Symlinks: the resolver follows them and contains accordingly.
# ---------------------------------------------------------------------------


class TestSymlinks:
    def test_symlink_inside_base_resolves_through(self, workspace: Path) -> None:
        # link -> existing real file inside base
        link = workspace / "link.txt"
        link.symlink_to(workspace / "data.txt")
        assert _read_file_tool(workspace, "link.txt") == "safe content"

    def test_symlink_pointing_outside_base_rejected(self, workspace: Path) -> None:
        # Create a link inside base that points to the outside secret.
        outside = workspace.parent / "secret" / "creds"
        link = workspace / "escape"
        link.symlink_to(outside)
        with pytest.raises(ValidationError, match="outside the allowed base"):
            _read_file_tool(workspace, "escape")


# ---------------------------------------------------------------------------
# Non-existent target is permitted (caller decides what to do).
# ---------------------------------------------------------------------------


class TestNonexistent:
    def test_missing_file_resolves_then_open_fails(self, workspace: Path) -> None:
        # ``safe_resolve`` allows the resolution; the read raises FileNotFound.
        with pytest.raises(FileNotFoundError):
            _read_file_tool(workspace, "missing.txt")
