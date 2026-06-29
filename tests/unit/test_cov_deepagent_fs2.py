# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage-gap tests for filesystem/state backends and filesystem tools.

Targets uncovered branches in:
  - tulip.deepagent.backends.filesystem
  - tulip.deepagent.backends.state
  - tulip.deepagent.tools.filesystem
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from tulip.deepagent.backends.filesystem import FilesystemBackend
from tulip.deepagent.backends.protocol import BackendError
from tulip.deepagent.backends.state import StateBackend, _normalize
from tulip.deepagent.tools.filesystem import make_filesystem_tools


# ---------------------------------------------------------------------------
# _normalize — edge-cases in state.py
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_empty_path_raises(self) -> None:
        with pytest.raises(BackendError) as exc:
            _normalize("")
        assert exc.value.code == "invalid_path"

    def test_non_slash_path_raises(self) -> None:
        with pytest.raises(BackendError) as exc:
            _normalize("no-slash")
        assert exc.value.code == "invalid_path"

    def test_root_returns_slash(self) -> None:
        assert _normalize("/") == "/"

    def test_trailing_slash_stripped(self) -> None:
        assert _normalize("/notes/") == "/notes"


# ---------------------------------------------------------------------------
# StateBackend — uncovered branches
# ---------------------------------------------------------------------------


class TestStateBackendCoverage:
    def test_write_non_string_contents_coerced(self) -> None:
        """write() with a non-str converts it rather than crashing (line 80)."""
        b = StateBackend()
        b.write("/num.txt", 42)  # type: ignore[arg-type]
        assert "42" in b.read("/num.txt")

    def test_write_to_implicit_dir_raises(self) -> None:
        """Writing to a path that is an implicit dir raises is_directory (line 83)."""
        b = StateBackend()
        b.write("/dir/child.txt", "x")
        with pytest.raises(BackendError) as exc:
            b.write("/dir", "data")
        assert exc.value.code == "is_directory"

    def test_exists_for_implicit_dir(self) -> None:
        """exists() returns True for implicit directories (line 112)."""
        b = StateBackend()
        b.write("/a/b.txt", "content")
        assert b.exists("/a") is True

    def test_exists_bad_path_returns_false(self) -> None:
        """exists() on a bad path returns False not raises (lines 109-110)."""
        b = StateBackend()
        assert b.exists("no-slash") is False

    def test_ls_missing_path_raises(self) -> None:
        """ls() on a nonexistent path raises file_not_found (line 129)."""
        b = StateBackend()
        with pytest.raises(BackendError) as exc:
            b.ls("/nonexistent")
        assert exc.value.code == "file_not_found"

    def test_ls_non_recursive_shows_dir_once(self) -> None:
        """Non-recursive ls surfaces a nested dir as a single is_dir entry (lines 135-138)."""
        b = StateBackend()
        b.write("/top/a/x.txt", "1")
        b.write("/top/a/y.txt", "2")
        entries = b.ls("/top")
        paths = [(e.path, e.is_dir) for e in entries]
        # /top/a should appear exactly once as a directory
        dir_entries = [p for p, is_d in paths if is_d]
        assert len([d for d in dir_entries if "a" in d]) == 1
        # nested files should NOT appear
        assert "/top/a/x.txt" not in [p for p, _ in paths]

    def test_glob_root_prefix(self) -> None:
        """glob with path='/' uses a bare '/' prefix — hits line 170."""
        b = StateBackend()
        b.write("/notes.md", "hello")
        b.write("/other.txt", "world")
        hits = b.glob("*.md", path="/")
        assert any("notes.md" in fi.path for fi in hits)
        assert not any("other.txt" in fi.path for fi in hits)

    def test_glob_with_subpath_prefix(self) -> None:
        """glob restricted to a sub-path filters correctly (line 174 path)."""
        b = StateBackend()
        b.write("/notes/a.md", "a")
        b.write("/notes/b.md", "b")
        b.write("/other/c.md", "c")
        hits = b.glob("*.md", path="/notes")
        hit_paths = {fi.path for fi in hits}
        assert hit_paths == {"/notes/a.md", "/notes/b.md"}

    def test_grep_non_recursive_skips_subdirs(self) -> None:
        """grep with recursive=False skips files in nested dirs (lines 206-210)."""
        b = StateBackend()
        b.write("/top/match.txt", "needle here")
        b.write("/top/sub/deep.txt", "needle here too")
        hits = b.grep("needle", path="/top", recursive=False)
        hit_paths = {m.path for m in hits}
        assert "/top/match.txt" in hit_paths
        assert "/top/sub/deep.txt" not in hit_paths

    def test_grep_non_recursive_at_root(self) -> None:
        """grep non-recursive at root doesn't skip any files (prefix == '/')."""
        b = StateBackend()
        b.write("/a.txt", "needle")
        b.write("/sub/b.txt", "needle")
        # At root, prefix == "/" so the "not recursive and prefix != '/'" check
        # is False, meaning all files match regardless.
        hits = b.grep("needle", path="/", recursive=False)
        hit_paths = {m.path for m in hits}
        # Both files are in scope because prefix == "/"
        assert "/a.txt" in hit_paths

    def test_ls_skips_out_of_prefix_files(self) -> None:
        """ls() skips files outside the requested prefix — hits line 135 continue."""
        b = StateBackend()
        b.write("/a/file.txt", "x")
        b.write("/b/other.txt", "y")
        entries = b.ls("/a")
        paths = [e.path for e in entries]
        assert any("file.txt" in p for p in paths)
        # /b/other.txt is outside the /a/ prefix — hits the `continue` at line 135
        assert not any("other.txt" in p for p in paths)

    def test_ls_file_path_rest_is_empty(self) -> None:
        """ls() on a file path: the fp == path entry has empty rest, hitting line 138 continue."""
        b = StateBackend()
        b.write("/notes.txt", "content")
        # Listing the file path itself — StateBackend will iterate the dict
        # and see fp="/notes.txt" which matches (fp == path), but then
        # rest = fp[len("/notes.txt/"):] = "" → continue at line 138.
        # Result is an empty list since no CHILD of "/notes.txt" exists.
        entries = b.ls("/notes.txt")
        assert entries == []

    def test_grep_out_of_scope_file_skipped(self) -> None:
        """grep() skips files outside the search path — hits line 206 continue."""
        b = StateBackend()
        b.write("/target/match.txt", "needle")
        b.write("/other/unrelated.txt", "needle")
        hits = b.grep("needle", path="/target")
        hit_paths = {m.path for m in hits}
        # /other/unrelated.txt is outside /target/ prefix → hits line 206 continue
        assert "/target/match.txt" in hit_paths
        assert "/other/unrelated.txt" not in hit_paths


# ---------------------------------------------------------------------------
# FilesystemBackend — permission errors + _file_info stat failure
# ---------------------------------------------------------------------------


class TestFilesystemBackendCoverage:
    def test_write_to_existing_dir_raises(self, tmp_path: Path) -> None:
        """write() when path is an existing directory raises is_directory (line 100)."""
        b = FilesystemBackend(root=tmp_path)
        (tmp_path / "mydir").mkdir()
        with pytest.raises(BackendError) as exc:
            b.write("/mydir", "content")
        assert exc.value.code == "is_directory"

    def test_edit_missing_file_raises(self, tmp_path: Path) -> None:
        """edit() on a non-existent file raises file_not_found (line 112)."""
        b = FilesystemBackend(root=tmp_path)
        with pytest.raises(BackendError) as exc:
            b.edit("/nope.txt", "old", "new")
        assert exc.value.code == "file_not_found"

    def test_read_permission_error(self, tmp_path: Path) -> None:
        """read() surfaces PermissionError as permission_denied (lines 87-88)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/secret.txt", "data")
        with (
            mock.patch("pathlib.Path.read_text", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.read("/secret.txt")
        assert exc.value.code == "permission_denied"

    def test_write_permission_error(self, tmp_path: Path) -> None:
        """write() surfaces PermissionError as permission_denied (lines 106-107)."""
        b = FilesystemBackend(root=tmp_path)
        with (
            mock.patch("pathlib.Path.write_text", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.write("/out.txt", "data")
        assert exc.value.code == "permission_denied"

    def test_edit_read_permission_error(self, tmp_path: Path) -> None:
        """edit() surfaces PermissionError during read as permission_denied (lines 117-118)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/f.txt", "original content here")
        with (
            mock.patch("pathlib.Path.read_text", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.edit("/f.txt", "original", "replacement")
        assert exc.value.code == "permission_denied"

    def test_edit_write_permission_error(self, tmp_path: Path) -> None:
        """edit() surfaces PermissionError during write as permission_denied (lines 129-132)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/f.txt", "hello world")
        original_read = Path.read_text

        def mock_read_text(self, *args, **kwargs):
            return original_read(self, *args, **kwargs)

        with (
            mock.patch("pathlib.Path.read_text", mock_read_text),
            mock.patch("pathlib.Path.write_text", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.edit("/f.txt", "hello", "goodbye")
        assert exc.value.code == "permission_denied"

    def test_file_info_stat_error(self, tmp_path: Path) -> None:
        """_file_info gracefully handles stat() raising FileNotFoundError (lines 151-153).

        We delete the file between writing it and calling _file_info directly so
        that is_dir() returns False (file gone) but stat() then raises
        FileNotFoundError — which is caught by the except block.
        """
        b = FilesystemBackend(root=tmp_path)
        b.write("/x.txt", "content")
        target = tmp_path / "x.txt"
        # Remove the file so stat() raises FileNotFoundError on the explicit call
        # while is_dir() returns False (the path no longer exists)
        target.unlink()
        fi = b._file_info(target, "/x.txt")
        # size and modified_at should be None when stat fails
        assert fi.size is None
        assert fi.modified_at is None

    def test_ls_permission_error(self, tmp_path: Path) -> None:
        """ls() raises permission_denied when iterdir() raises PermissionError (lines 170-171)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/sub/f.txt", "x")
        with (
            mock.patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.ls("/sub")
        assert exc.value.code == "permission_denied"

    def test_glob_permission_error(self, tmp_path: Path) -> None:
        """glob() raises permission_denied when rglob raises PermissionError (lines 193-194)."""
        b = FilesystemBackend(root=tmp_path)
        with (
            mock.patch("pathlib.Path.rglob", side_effect=PermissionError("denied")),
            pytest.raises(BackendError) as exc,
        ):
            b.glob("*.txt")
        assert exc.value.code == "permission_denied"

    def test_glob_full_agent_path_match(self, tmp_path: Path) -> None:
        """glob() matches entries by full agent_path as fallback (lines 201-202)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/notes/readme.md", "hello")
        # Pattern with leading / matches against full agent path
        matches = b.glob("/notes/readme.md")
        assert len(matches) == 1
        assert matches[0].path == "/notes/readme.md"

    def test_grep_permission_error(self, tmp_path: Path) -> None:
        """grep() raises permission_denied when iterating the rglob result raises PermissionError (lines 226-227).

        rglob() returns a lazy iterator; the PermissionError surfaces when
        sorted() tries to consume it, not when rglob() is called.
        """

        def _raising_iter(*_args, **_kwargs):
            """Generator that raises PermissionError on first iteration."""
            raise PermissionError("denied")
            yield  # makes this a generator function

        b = FilesystemBackend(root=tmp_path)
        with (
            mock.patch.object(type(tmp_path), "rglob", _raising_iter),
            pytest.raises(BackendError) as exc,
        ):
            b.grep("pattern")
        assert exc.value.code == "permission_denied"

    def test_grep_one_read_error_skips_file(self, tmp_path: Path) -> None:
        """_grep_one silently skips files where read_text raises OSError (lines 237-238)."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/good.txt", "needle here")
        b.write("/bad.txt", "needle here too")

        good_path = tmp_path / "good.txt"
        bad_path = tmp_path / "bad.txt"
        original_read = Path.read_text

        def conditional_read(self, *args, **kwargs):
            if self == bad_path:
                raise OSError("unreadable")
            return original_read(self, *args, **kwargs)

        with mock.patch("pathlib.Path.read_text", conditional_read):
            hits = b.grep("needle")

        # Only good.txt should appear since bad.txt raises OSError
        assert all(h.path == "/good.txt" for h in hits)

    def test_grep_non_recursive_dir(self, tmp_path: Path) -> None:
        """grep non-recursive on a directory only scans top-level files."""
        b = FilesystemBackend(root=tmp_path)
        b.write("/top.txt", "match")
        b.write("/subdir/deep.txt", "match")
        hits = b.grep("match", recursive=False)
        paths = [h.path for h in hits]
        assert "/top.txt" in paths

    def test_exists_with_invalid_path_returns_false(self, tmp_path: Path) -> None:
        """exists() catches BackendError from _resolve and returns False (lines 137-138)."""
        b = FilesystemBackend(root=tmp_path)
        # A relative path (no leading /) makes _resolve raise BackendError("invalid_path")
        # which exists() catches and converts to False.
        assert b.exists("not-absolute") is False

    def test_symlink_outside_root(self, tmp_path: Path) -> None:
        """Symlinks escaping the root are rejected (lines 68-72)."""
        outside = tmp_path.parent / "outside_for_symlink_test.txt"
        outside.write_text("secret")
        link = tmp_path / "escape_link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        b = FilesystemBackend(tmp_path)
        with pytest.raises(BackendError) as exc:
            b.read("/escape_link.txt")
        assert exc.value.code == "invalid_path"
        outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# tools/filesystem.py — uncovered error + success paths
# ---------------------------------------------------------------------------


class TestFilesystemToolsCoverage:
    @pytest.mark.asyncio
    async def test_write_file_backend_error_returns_code(self) -> None:
        """write_file returns error code when backend raises (lines 55-56)."""
        backend = StateBackend()
        # Create an implicit dir at /dir so writing to /dir raises is_directory
        backend.write("/dir/inner.txt", "x")
        tools = make_filesystem_tools(backend)
        write_file = next(t for t in tools if t.name == "write_file")
        result = await write_file.execute(path="/dir", contents="data")
        assert result == "is_directory"

    @pytest.mark.asyncio
    async def test_ls_tool_backend_error_returns_code(self) -> None:
        """ls returns error code when backend raises (lines 89-90)."""
        backend = StateBackend()
        tools = make_filesystem_tools(backend)
        ls_tool = next(t for t in tools if t.name == "ls")
        result = await ls_tool.execute(path="/nonexistent")
        assert result == "file_not_found"

    @pytest.mark.asyncio
    async def test_edit_file_success_returns_path(self) -> None:
        """edit_file returns the path on success (line 120)."""
        backend = StateBackend()
        backend.write("/doc.txt", "alpha beta gamma")
        tools = make_filesystem_tools(backend)
        edit_tool = next(t for t in tools if t.name == "edit_file")
        result = await edit_tool.execute(path="/doc.txt", old_str="beta", new_str="BETA")
        assert result == "/doc.txt"

    @pytest.mark.asyncio
    async def test_edit_file_backend_error_returns_code(self) -> None:
        """edit_file returns error code on BackendError (line 117)."""
        backend = StateBackend()
        tools = make_filesystem_tools(backend)
        edit_tool = next(t for t in tools if t.name == "edit_file")
        result = await edit_tool.execute(path="/nope.txt", old_str="x", new_str="y")
        assert result == "file_not_found"

    @pytest.mark.asyncio
    async def test_glob_tool_success_returns_json(self) -> None:
        """glob tool returns JSON on success (lines 130-134)."""
        backend = StateBackend()
        backend.write("/docs/a.md", "x")
        backend.write("/docs/b.txt", "y")
        tools = make_filesystem_tools(backend)
        glob_tool = next(t for t in tools if t.name == "glob")
        result = await glob_tool.execute(pattern="*.md", path="/docs")
        decoded = json.loads(result)
        assert any("a.md" in d["path"] for d in decoded)
        assert not any("b.txt" in d["path"] for d in decoded)

    @pytest.mark.asyncio
    async def test_glob_tool_backend_error_returns_code(self, tmp_path: Path) -> None:
        """glob tool returns error code on BackendError (lines 132-133)."""
        backend = FilesystemBackend(tmp_path)
        tools = make_filesystem_tools(backend)
        glob_tool = next(t for t in tools if t.name == "glob")
        result = await glob_tool.execute(pattern="*.md", path="/nonexistent")
        assert result == "file_not_found"

    @pytest.mark.asyncio
    async def test_grep_tool_backend_error_returns_code(self, tmp_path: Path) -> None:
        """grep tool returns error code on BackendError (lines 155-156)."""
        backend = FilesystemBackend(tmp_path)
        tools = make_filesystem_tools(backend)
        grep_tool = next(t for t in tools if t.name == "grep")
        result = await grep_tool.execute(pattern="needle", path="/nonexistent")
        assert result == "file_not_found"
