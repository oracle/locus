# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the deepagent filesystem-as-memory toolset.

Two backends (StateBackend in-memory + FilesystemBackend rooted on
disk), six tools, four standardized error codes — all asserted here.
The tests don't touch a model provider; they exercise the backend +
tool surface directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from locus.deepagent import (
    BackendError,
    BackendProtocol,
    FilesystemBackend,
    StateBackend,
    create_deepagent,
    make_filesystem_tools,
)


# ---------------------------------------------------------------------------
# StateBackend roundtrips
# ---------------------------------------------------------------------------


class TestStateBackend:
    def test_implements_protocol(self) -> None:
        assert isinstance(StateBackend(), BackendProtocol)

    def test_write_and_read_with_line_numbers(self) -> None:
        sb = StateBackend()
        sb.write("/notes/draft.md", "alpha\nbeta\ngamma")
        out = sb.read("/notes/draft.md")
        assert "     1\talpha" in out
        assert "     2\tbeta" in out
        assert "     3\tgamma" in out

    def test_read_with_offset_and_limit_paginates(self) -> None:
        sb = StateBackend()
        sb.write("/big.txt", "\n".join(f"line {i}" for i in range(1, 51)))
        out = sb.read("/big.txt", offset=10, limit=3)
        # offset is 0-based — start at line 11.
        assert "    11\tline 11" in out
        assert "    12\tline 12" in out
        assert "    13\tline 13" in out
        assert "line 14" not in out

    def test_read_offset_past_eof_returns_empty(self) -> None:
        sb = StateBackend()
        sb.write("/short.txt", "one\ntwo")
        assert sb.read("/short.txt", offset=99, limit=10) == ""

    def test_read_missing_file_raises_file_not_found(self) -> None:
        sb = StateBackend()
        with pytest.raises(BackendError) as exc:
            sb.read("/nope.txt")
        assert exc.value.code == "file_not_found"

    def test_read_directory_raises_is_directory(self) -> None:
        sb = StateBackend()
        sb.write("/dir/inner.txt", "x")
        with pytest.raises(BackendError) as exc:
            sb.read("/dir")
        assert exc.value.code == "is_directory"

    def test_invalid_path_rejected(self) -> None:
        sb = StateBackend()
        with pytest.raises(BackendError) as exc:
            sb.read("relative.txt")
        assert exc.value.code == "invalid_path"

    def test_ls_root_lists_implicit_dirs_then_files(self) -> None:
        sb = StateBackend()
        sb.write("/notes/a.md", "a")
        sb.write("/notes/b.md", "b")
        sb.write("/top.md", "t")
        entries = sb.ls("/")
        paths = [(e.path, e.is_dir) for e in entries]
        assert ("/notes", True) in paths
        assert ("/top.md", False) in paths
        assert ("/notes/a.md", False) not in paths  # not recursive

    def test_ls_recursive_walks_subtree(self) -> None:
        sb = StateBackend()
        sb.write("/x/y/z.txt", "deep")
        sb.write("/x/top.txt", "shallow")
        entries = sb.ls("/", recursive=True)
        paths = {e.path for e in entries}
        assert "/x/y/z.txt" in paths
        assert "/x/top.txt" in paths

    def test_edit_unique_match_succeeds(self) -> None:
        sb = StateBackend()
        sb.write("/draft.md", "Header\nbody one\nfooter")
        sb.edit("/draft.md", "body one", "BODY ONE")
        assert "BODY ONE" in sb.read("/draft.md")

    def test_edit_zero_match_raises_value_error(self) -> None:
        sb = StateBackend()
        sb.write("/draft.md", "Header")
        with pytest.raises(ValueError, match="not found"):
            sb.edit("/draft.md", "missing", "X")

    def test_edit_multi_match_raises_value_error(self) -> None:
        sb = StateBackend()
        sb.write("/draft.md", "ping pong ping pong")
        with pytest.raises(ValueError, match=r"multiple|matches.*times"):
            sb.edit("/draft.md", "ping", "PING")

    def test_glob_matches_extension(self) -> None:
        sb = StateBackend()
        sb.write("/notes/a.md", "")
        sb.write("/notes/b.md", "")
        sb.write("/notes/c.txt", "")
        hits = sb.glob("*.md", path="/notes")
        paths = {fi.path for fi in hits}
        assert paths == {"/notes/a.md", "/notes/b.md"}

    def test_grep_returns_line_and_text(self) -> None:
        sb = StateBackend()
        sb.write(
            "/log.txt",
            "INFO start\nERROR something failed\nINFO continue",
        )
        hits = sb.grep("ERROR", path="/")
        assert len(hits) == 1
        assert hits[0].path == "/log.txt"
        assert hits[0].line == 2
        assert hits[0].text == "ERROR something failed"


# ---------------------------------------------------------------------------
# FilesystemBackend — path safety + traversal guards
# ---------------------------------------------------------------------------


class TestFilesystemBackend:
    def test_implements_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FilesystemBackend(tmp_path), BackendProtocol)

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        fb = FilesystemBackend(tmp_path)
        fb.write("/scratch/note.md", "hello\nworld")
        out = fb.read("/scratch/note.md")
        assert "     1\thello" in out
        assert "     2\tworld" in out
        # Real disk effect — file actually exists under the root.
        assert (tmp_path / "scratch" / "note.md").exists()

    def test_traversal_rejected_with_invalid_path(self, tmp_path: Path) -> None:
        fb = FilesystemBackend(tmp_path)
        with pytest.raises(BackendError) as exc:
            fb.read("/../../../etc/passwd")
        assert exc.value.code == "invalid_path"

    def test_relative_path_rejected(self, tmp_path: Path) -> None:
        fb = FilesystemBackend(tmp_path)
        with pytest.raises(BackendError) as exc:
            fb.write("relative.md", "x")
        assert exc.value.code == "invalid_path"

    def test_symlink_outside_root_rejected(self, tmp_path: Path) -> None:
        # Lay a symlink pointing outside the root, then try to read it
        # through the agent-visible path.
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        try:
            inside_link = tmp_path / "link.txt"
            inside_link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this OS")
        fb = FilesystemBackend(tmp_path)
        with pytest.raises(BackendError) as exc:
            fb.read("/link.txt")
        assert exc.value.code == "invalid_path"
        # Cleanup
        outside.unlink()

    def test_missing_file_returns_file_not_found(self, tmp_path: Path) -> None:
        fb = FilesystemBackend(tmp_path)
        with pytest.raises(BackendError) as exc:
            fb.read("/nope.md")
        assert exc.value.code == "file_not_found"

    def test_is_directory_signal(self, tmp_path: Path) -> None:
        fb = FilesystemBackend(tmp_path)
        fb.write("/sub/x.md", "x")
        with pytest.raises(BackendError) as exc:
            fb.read("/sub")
        assert exc.value.code == "is_directory"


# ---------------------------------------------------------------------------
# Tool surface — schema + delegation
# ---------------------------------------------------------------------------


class TestFilesystemTools:
    def test_six_tools_with_expected_names(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        names = {t.name for t in tools}
        assert names == {
            "write_file",
            "read_file",
            "ls",
            "edit_file",
            "glob",
            "grep",
        }

    def test_write_file_schema_has_flat_fields(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        write_file = next(t for t in tools if t.name == "write_file")
        props = write_file.parameters.get("properties", {})
        assert "path" in props
        assert "contents" in props

    def test_read_file_schema_has_pagination_fields(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        read_file = next(t for t in tools if t.name == "read_file")
        props = read_file.parameters.get("properties", {})
        assert {"path", "offset", "limit"} <= set(props.keys())

    def test_ls_schema_has_recursive(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        ls = next(t for t in tools if t.name == "ls")
        props = ls.parameters.get("properties", {})
        assert {"path", "recursive"} <= set(props.keys())

    @pytest.mark.asyncio
    async def test_write_then_read_via_tools(self) -> None:
        backend = StateBackend()
        tools = make_filesystem_tools(backend)
        tool_map = {t.name: t for t in tools}
        result = await tool_map["write_file"].execute(path="/scratch/x.md", contents="hello")
        assert result == "/scratch/x.md"
        out = await tool_map["read_file"].execute(path="/scratch/x.md")
        assert "hello" in out

    @pytest.mark.asyncio
    async def test_read_missing_returns_error_code(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        read_file = next(t for t in tools if t.name == "read_file")
        result = await read_file.execute(path="/missing.md")
        assert result == "file_not_found"

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error_code(self) -> None:
        tools = make_filesystem_tools(StateBackend())
        read_file = next(t for t in tools if t.name == "read_file")
        result = await read_file.execute(path="relative.md")
        assert result == "invalid_path"

    @pytest.mark.asyncio
    async def test_edit_zero_match_surfaces_message(self) -> None:
        backend = StateBackend()
        backend.write("/draft.md", "header")
        tools = make_filesystem_tools(backend)
        edit = next(t for t in tools if t.name == "edit_file")
        result = await edit.execute(path="/draft.md", old_str="missing", new_str="X")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_ls_returns_json_payload(self) -> None:
        backend = StateBackend()
        backend.write("/a.md", "x")
        backend.write("/b.md", "y")
        tools = make_filesystem_tools(backend)
        ls = next(t for t in tools if t.name == "ls")
        result = await ls.execute(path="/")
        decoded = json.loads(result)
        paths = {entry["path"] for entry in decoded}
        assert paths == {"/a.md", "/b.md"}

    @pytest.mark.asyncio
    async def test_grep_returns_json_with_line_numbers(self) -> None:
        backend = StateBackend()
        backend.write("/log.txt", "INFO\nERROR boom\nINFO done")
        tools = make_filesystem_tools(backend)
        grep = next(t for t in tools if t.name == "grep")
        result = await grep.execute(pattern="ERROR", path="/")
        decoded = json.loads(result)
        assert len(decoded) == 1
        assert decoded[0]["line"] == 2
        assert decoded[0]["text"] == "ERROR boom"


# ---------------------------------------------------------------------------
# Factory wiring — enable_filesystem flag
# ---------------------------------------------------------------------------


class TestCreateDeepagentFilesystem:
    def _stub_oci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCI_PROFILE", "DEFAULT")

    def test_filesystem_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            reflexion=False,
            grounding=False,
        )
        names = {t.name for t in agent.config.tools}
        # No FS tools attached.
        assert "write_file" not in names
        assert "read_file" not in names

    def test_enable_filesystem_attaches_six_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            enable_filesystem=True,
            reflexion=False,
            grounding=False,
        )
        names = {t.name for t in agent.config.tools}
        assert {
            "write_file",
            "read_file",
            "ls",
            "edit_file",
            "glob",
            "grep",
        } <= names

    def test_enable_filesystem_default_backend_is_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an explicit backend, the FS tools must work in-memory
        with StateBackend semantics — write then read should round-trip
        within a single agent's tool list, with no disk side effect."""
        self._stub_oci(monkeypatch)
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            enable_filesystem=True,
            reflexion=False,
            grounding=False,
        )
        tool_map = {t.name: t for t in agent.config.tools}

        import asyncio

        write_result = asyncio.run(
            tool_map["write_file"].execute(path="/scratch/x.md", contents="hello")
        )
        assert write_result == "/scratch/x.md"
        read_result = asyncio.run(tool_map["read_file"].execute(path="/scratch/x.md"))
        assert "hello" in read_result
        # No disk side effect — the path doesn't exist anywhere on
        # the real filesystem (the default backend is in-memory).
        assert not os.path.exists("/scratch/x.md")

    def test_explicit_backend_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_oci(monkeypatch)
        custom = StateBackend()
        custom.write("/seed.md", "preloaded")
        agent = create_deepagent(
            model="oci:openai.gpt-4o-mini",
            tools=[],
            system_prompt="be helpful",
            enable_filesystem=True,
            backend=custom,
            reflexion=False,
            grounding=False,
        )
        # Calling read_file through the agent's tools should hit `custom`.
        read_file = next(t for t in agent.config.tools if t.name == "read_file")
        import asyncio

        out = asyncio.run(read_file.execute(path="/seed.md"))
        assert "preloaded" in out
