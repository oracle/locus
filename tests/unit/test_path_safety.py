# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.tools.path_safety.safe_resolve``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from locus.core.errors import ValidationError
from locus.tools.path_safety import safe_resolve


class TestHappyPath:
    def test_simple_relative_file(self, tmp_path: Path) -> None:
        result = safe_resolve(tmp_path, "data.txt")
        assert result == (tmp_path / "data.txt").resolve()

    def test_nested_relative_directory(self, tmp_path: Path) -> None:
        result = safe_resolve(tmp_path, "sub/dir/file.txt")
        assert result == (tmp_path / "sub/dir/file.txt").resolve()

    def test_empty_string_returns_base(self, tmp_path: Path) -> None:
        # ``Path(base) / ""`` is idiomatic for "the base itself".
        assert safe_resolve(tmp_path, "") == tmp_path.resolve()

    def test_dot_returns_base(self, tmp_path: Path) -> None:
        assert safe_resolve(tmp_path, ".") == tmp_path.resolve()

    def test_base_accepts_string(self, tmp_path: Path) -> None:
        result = safe_resolve(str(tmp_path), "data.txt")
        assert result == (tmp_path / "data.txt").resolve()


class TestTraversalBlocked:
    @pytest.mark.parametrize(
        "attack",
        [
            "../etc/passwd",
            "../../../../etc/passwd",
            "./../../secret",
            "sub/../../escape",
            "a/b/c/../../../../out",
        ],
    )
    def test_dotdot_rejected(self, tmp_path: Path, attack: str) -> None:
        with pytest.raises(ValidationError, match="outside the allowed base"):
            safe_resolve(tmp_path, attack)

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="outside the allowed base"):
            safe_resolve(tmp_path, "/etc/passwd")

    def test_sibling_of_base_rejected(self, tmp_path: Path) -> None:
        # tmp_path is /private/var/.../T/pytest-.../test-0; a sibling
        # directory shares the parent but is not contained.
        sibling_name = tmp_path.name + "-sibling"
        with pytest.raises(ValidationError, match="outside the allowed base"):
            safe_resolve(tmp_path, f"../{sibling_name}/file")


class TestSymlinkHandling:
    def test_symlink_inside_base_resolved(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("hi")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        result = safe_resolve(tmp_path, "link.txt")
        # Resolved to the real file (which lives inside base), accepted.
        assert result == target.resolve()

    def test_symlink_pointing_outside_base_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside-target"
        outside.write_text("secret")
        try:
            link = tmp_path / "escape"
            link.symlink_to(outside)

            with pytest.raises(ValidationError, match="outside the allowed base"):
                safe_resolve(tmp_path, "escape")
        finally:
            if outside.exists():
                outside.unlink()


class TestInputValidation:
    @pytest.mark.parametrize("bogus", [None, 42, b"bytes", ["list"]])
    def test_non_string_rejected(self, tmp_path: Path, bogus: object) -> None:
        with pytest.raises(ValidationError, match="must be a string"):
            safe_resolve(tmp_path, bogus)  # type: ignore[arg-type]


class TestMissingTargetsTolerated:
    def test_nonexistent_child_is_ok(self, tmp_path: Path) -> None:
        # The tool may be checking whether to create a file; missing
        # targets must not raise (open() will, later, if needed).
        result = safe_resolve(tmp_path, "new/deeply/nested/file.txt")
        assert result.is_relative_to(tmp_path.resolve())
        assert result.name == "file.txt"
        assert not result.exists()


class TestRealWorldShapes:
    def test_url_encoded_traversal_not_decoded(self, tmp_path: Path) -> None:
        # ``safe_resolve`` is a filesystem guard, not a URL decoder.
        # A literal ``%2e%2e`` in the path is a valid (weird) directory
        # name, not a traversal. Callers that expect URL-encoded input
        # must decode first before handing off to this helper.
        result = safe_resolve(tmp_path, "%2e%2e/child")
        assert result.is_relative_to(tmp_path.resolve())

    def test_windows_backslash_treated_as_filename_on_posix(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("POSIX semantics only")
        # On POSIX, '\\' is a valid filename char — no interpretation.
        result = safe_resolve(tmp_path, "a\\..\\b")
        # On POSIX this is a single filename that lives under base.
        assert result.is_relative_to(tmp_path.resolve())
