# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for deepagent filesystem and state backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from locus.deepagent.backends.filesystem import FilesystemBackend
from locus.deepagent.backends.protocol import BackendError
from locus.deepagent.backends.state import StateBackend


# ---------------------------------------------------------------------------
# StateBackend — in-memory, synchronous, paths must start with /
# ---------------------------------------------------------------------------


class TestStateBackend:
    def test_write_and_read(self) -> None:
        b = StateBackend()
        b.write("/file.txt", "hello world")
        raw = b.read("/file.txt")
        assert "hello world" in raw

    def test_read_missing_raises(self) -> None:
        b = StateBackend()
        with pytest.raises(BackendError):
            b.read("/nonexistent.txt")

    def test_path_must_start_with_slash(self) -> None:
        b = StateBackend()
        with pytest.raises(BackendError):
            b.write("no-slash.txt", "content")

    def test_overwrite(self) -> None:
        b = StateBackend()
        b.write("/f.txt", "v1")
        b.write("/f.txt", "v2")
        assert "v2" in b.read("/f.txt")

    def test_exists_true(self) -> None:
        b = StateBackend()
        b.write("/x.txt", "x")
        assert b.exists("/x.txt") is True

    def test_exists_false(self) -> None:
        b = StateBackend()
        assert b.exists("/missing.txt") is False

    def test_edit_replaces_content(self) -> None:
        b = StateBackend()
        b.write("/e.txt", "alpha beta gamma")
        b.edit("/e.txt", "beta", "delta")
        assert "delta" in b.read("/e.txt")
        assert "beta" not in b.read("/e.txt")

    def test_edit_missing_raises(self) -> None:
        b = StateBackend()
        with pytest.raises(BackendError):
            b.edit("/nope.txt", "old", "new")

    def test_ls_returns_file_infos(self) -> None:
        b = StateBackend()
        b.write("/notes.txt", "content")
        items = b.ls("/")
        paths = [i.path for i in items]
        assert any("notes.txt" in p for p in paths)

    def test_independent_instances(self) -> None:
        a = StateBackend()
        c = StateBackend()
        a.write("/shared.txt", "from a")
        assert not c.exists("/shared.txt")


# ---------------------------------------------------------------------------
# FilesystemBackend — disk-backed, synchronous, same path conventions
# ---------------------------------------------------------------------------


class TestFilesystemBackend:
    def test_write_and_read(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/notes.txt", "research notes")
        assert "research notes" in b.read("/notes.txt")

    def test_creates_subdirectories(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/subdir/file.txt", "nested content")
        assert "nested content" in b.read("/subdir/file.txt")

    def test_ls_lists_files(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/one.txt", "1")
        b.write("/two.txt", "2")
        items = b.ls("/")
        paths = [i.path for i in items]
        assert any("one.txt" in p for p in paths)
        assert any("two.txt" in p for p in paths)

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises(BackendError):
            b.read("/missing.txt")

    def test_exists(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/here.txt", "x")
        assert b.exists("/here.txt") is True
        assert b.exists("/nothere.txt") is False

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        FilesystemBackend(root=tmp_path).write("/persist.txt", "data")
        assert "data" in FilesystemBackend(root=tmp_path).read("/persist.txt")

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises((BackendError, OSError, ValueError)):
            b.write("/../../../etc/passwd", "bad")

    def test_glob_matches_pattern(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/a.txt", "a")
        b.write("/b.py", "b")
        matches = b.glob("*.txt")
        paths = [m.path for m in matches]
        assert any("a.txt" in p for p in paths)
        assert not any("b.py" in p for p in paths)

    def test_grep_finds_pattern(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/log.txt", "error: something went wrong\ninfo: all ok")
        hits = b.grep("error", path="/log.txt")
        assert len(hits) >= 1
        assert any("error" in h.text for h in hits)

    def test_grep_no_match(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/clean.txt", "everything is fine")
        hits = b.grep("error", path="/clean.txt")
        assert hits == []

    def test_grep_missing_path_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises(BackendError):
            b.grep("anything", path="/nope.txt")

    def test_grep_recursive_across_files(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/a.txt", "alpha match")
        b.write("/sub/b.txt", "beta match")
        hits = b.grep("match")
        assert len(hits) >= 2

    def test_grep_non_recursive(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/top.txt", "match here")
        b.write("/sub/deep.txt", "match here too")
        hits = b.grep("match", recursive=False)
        paths = [h.path for h in hits]
        assert any("top.txt" in p for p in paths)

    def test_read_offset_beyond_end_returns_empty(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/short.txt", "only one line")
        assert b.read("/short.txt", offset=100) == ""

    def test_read_with_limit_zero_returns_all(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/multi.txt", "line1\nline2\nline3\nline4\nline5")
        result = b.read("/multi.txt", limit=0)
        assert "line1" in result
        assert "line5" in result

    def test_read_with_offset_and_limit(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/paged.txt", "a\nb\nc\nd\ne")
        result = b.read("/paged.txt", offset=1, limit=2)
        assert "b" in result
        assert "c" in result
        assert "a" not in result

    def test_read_directory_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/mydir/file.txt", "x")
        with pytest.raises(BackendError):
            b.read("/mydir")

    def test_ls_on_single_file(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/solo.txt", "content")
        items = b.ls("/solo.txt")
        assert len(items) == 1
        assert "solo.txt" in items[0].path

    def test_ls_missing_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises(BackendError):
            b.ls("/nonexistent/")

    def test_ls_recursive(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/a/b/c.txt", "deep")
        items = b.ls("/", recursive=True)
        paths = [i.path for i in items]
        assert any("c.txt" in p for p in paths)

    def test_glob_missing_base_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises(BackendError):
            b.glob("*.txt", path="/nope/")

    def test_edit_ambiguous_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/dup.txt", "abc abc")
        with pytest.raises(ValueError, match="2 times"):
            b.edit("/dup.txt", "abc", "xyz")

    def test_edit_missing_old_str_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/f.txt", "hello world")
        with pytest.raises(ValueError, match="not found"):
            b.edit("/f.txt", "nothere", "x")

    def test_edit_directory_raises(self, tmp_path: Path) -> None:
        b = FilesystemBackend(root=tmp_path)
        b.write("/d/file.txt", "x")
        with pytest.raises(BackendError):
            b.edit("/d", "x", "y")
