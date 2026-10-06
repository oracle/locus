# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""read, write, edit, multi_edit and ls, on every backend.

Ported from tulip-code's tool tests: the edges rather than the happy path —
what happens outside the workspace, on a file that is not there, on a match
that is not unique, on a file the agent has not read.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.harness.conftest import get, put
from tulip.core.media import has_images
from tulip.harness import HarnessConfig, WorkspaceBackend, build_harness
from tulip.harness.backend import MAX_READ_BYTES
from tulip.harness.toolset import Harness
from tulip.tools.output import ToolOutput


def harness(backend: WorkspaceBackend, **config: object) -> Harness:
    return build_harness(backend, config=HarnessConfig(**config))  # type: ignore[arg-type]


async def call(h: Harness, name: str, **kwargs: object) -> str:
    result: str = await h.tool(name).execute(**kwargs)
    return result


# ------------------------------------------------------------------- read --


async def test_read_numbers_the_lines(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "one\ntwo\n")
    out = await call(harness(workspace), "read", path="a.py")
    assert "1\tone" in out
    assert "2\ttwo" in out


async def test_read_says_so_rather_than_raising_on_a_missing_file(
    workspace: WorkspaceBackend,
) -> None:
    # The model can act on a sentence; it cannot act on a traceback.
    assert "no such file" in await call(harness(workspace), "read", path="ghost.py")


async def test_read_refuses_a_directory_and_says_what_to_use(workspace: WorkspaceBackend) -> None:
    put(workspace, "src/a.py", "x")
    assert "use ls or glob" in await call(harness(workspace), "read", path="src")


async def test_read_windows_a_long_file(workspace: WorkspaceBackend) -> None:
    put(workspace, "big.txt", "\n".join(str(i) for i in range(500)))
    out = await call(harness(workspace), "read", path="big.txt", offset=10, limit=5)
    assert "11\t10" in out
    assert "more lines (of 500) — read again with offset=15" in out


async def test_a_large_file_is_read_a_window_at_a_time(workspace: WorkspaceBackend) -> None:
    lines = [f"line {i} " + "x" * 100 for i in range(5000)]
    put(workspace, "huge.log", "\n".join(lines))
    out = await call(harness(workspace), "read", path="huge.log", offset=4990, limit=5)
    assert "4991\tline 4990" in out
    assert "line 4995" not in out
    assert "read again with offset=4995" in out


async def test_one_window_never_returns_more_than_the_cap(workspace: WorkspaceBackend) -> None:
    put(workspace, "huge.log", "\n".join("y" * 1000 for _ in range(1000)))
    out = await call(harness(workspace), "read", path="huge.log")
    assert len(out) <= MAX_READ_BYTES + 200
    assert "more lines (of 1,000)" in out


async def test_a_very_long_line_is_cut_and_says_so(workspace: WorkspaceBackend) -> None:
    put(workspace, "bundle.js", "z" * (MAX_READ_BYTES + 10))
    out = await call(harness(workspace), "read", path="bundle.js")
    assert f"line is {MAX_READ_BYTES + 10:,} characters" in out
    assert len(out) < 5000


async def test_an_offset_past_the_end_says_how_long_the_file_is(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.txt", "one\ntwo\n")
    assert "has 2 lines" in await call(harness(workspace), "read", path="a.txt", offset=10)


async def test_an_empty_file_says_so(workspace: WorkspaceBackend) -> None:
    put(workspace, "empty.txt", "")
    assert "is empty" in await call(harness(workspace), "read", path="empty.txt")


async def test_read_survives_bytes_that_are_not_text(workspace: WorkspaceBackend) -> None:
    put(workspace, "blob.bin", b"\xff\xfe\x00hello")
    assert "hello" in await call(harness(workspace), "read", path="blob.bin")


@pytest.mark.parametrize("path", ["../escape.py", "../../etc/passwd", "src/../../out.py"])
async def test_nothing_reads_outside_the_workspace(workspace: WorkspaceBackend, path: str) -> None:
    """The whole attack surface of a file-writing agent is ``..``."""
    if workspace.capabilities.root == "/":
        pytest.skip("nothing is outside a namespace rooted at /")
    h = harness(workspace)
    for name, kwargs in [
        ("read", {"path": path}),
        ("write", {"path": path, "content": "x"}),
        ("edit", {"path": path, "old": "a", "new": "b"}),
        ("multi_edit", {"path": path, "edits": [{"old": "a", "new": "b"}]}),
        ("ls", {"path": path}),
        ("glob", {"pattern": "*", "path": path}),
        ("grep", {"pattern": "x", "path": path}),
    ]:
        out = await call(h, name, **kwargs)
        assert isinstance(out, ToolOutput), name
        assert out.is_error, name
        assert "escapes the workspace" in out


async def test_an_image_is_shown_to_a_model_that_can_see(workspace: WorkspaceBackend) -> None:
    put(workspace, "shot.png", b"\x89PNG\r\n\x1a\nfakeimage")
    out = await call(harness(workspace, vision=True), "read", path="shot.png")
    assert has_images(out)
    assert isinstance(out, ToolOutput)
    assert out.content_blocks
    assert out.content_blocks[0]["mimeType"] == "image/png"


async def test_an_image_is_declined_for_a_model_that_cannot(workspace: WorkspaceBackend) -> None:
    put(workspace, "shot.png", b"\x89PNG")
    out = await call(harness(workspace), "read", path="shot.png")
    assert "cannot see images" in out
    assert not has_images(out)


async def test_an_oversized_image_is_declined(
    workspace: WorkspaceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tulip.harness.tools import files

    monkeypatch.setattr(files, "MAX_IMAGE_BYTES", 3)
    put(workspace, "shot.jpg", b"\xff\xd8\xff\xe0")
    out = await call(harness(workspace, vision=True), "read", path="shot.jpg")
    assert "over the 3-byte limit" in out


async def test_a_notebook_is_read_cell_by_cell(workspace: WorkspaceBackend) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "m1", "source": ["# Title\n"]},
            {
                "cell_type": "code",
                "id": "c1",
                "source": "print('hi')",
                "outputs": [{"text": ["hi\n"]}, {"data": {"text/plain": "42"}}, {"other": 1}],
            },
            "not a cell",
        ],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    put(workspace, "nb.ipynb", json.dumps(notebook))
    out = await call(harness(workspace), "read", path="nb.ipynb")
    assert "--- cell 0 id=m1 [markdown]" in out
    assert "--- cell 1 id=c1 [code]" in out
    assert "print('hi')" in out
    assert "> hi" in out
    assert "> 42" in out


@pytest.mark.parametrize("body", ["{not json", '{"cells": 3}'])
async def test_a_broken_notebook_is_read_as_text(workspace: WorkspaceBackend, body: str) -> None:
    put(workspace, "nb.ipynb", body)
    assert body in await call(harness(workspace), "read", path="nb.ipynb")


# ------------------------------------------------------------------ write --


async def test_write_creates_and_reports_the_size(workspace: WorkspaceBackend) -> None:
    out = await call(harness(workspace), "write", path="new.py", content="x = 1\n")
    assert get(workspace, "new.py") == "x = 1\n"
    assert out == "created new.py (6 bytes)"


async def test_write_makes_the_parent_directory(workspace: WorkspaceBackend) -> None:
    await call(harness(workspace), "write", path="deep/down/here.py", content="ok")
    assert get(workspace, "deep/down/here.py") == "ok"


async def test_write_says_overwrote_and_shows_what_it_displaced(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "before\n")
    out = await call(
        harness(workspace, require_read=False), "write", path="a.py", content="after\n"
    )
    assert out.startswith("overwrote a.py")
    assert "-before" in out
    assert "+after" in out


async def test_write_onto_a_directory_is_refused(workspace: WorkspaceBackend) -> None:
    put(workspace, "src/a.py", "x")
    assert "is a directory" in await call(harness(workspace), "write", path="src", content="x")


async def test_overwriting_a_crlf_file_keeps_its_line_endings(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "win.txt", b"one\r\ntwo\r\n")
    h = harness(workspace)
    await call(h, "read", path="win.txt")
    await call(h, "write", path="win.txt", content="uno\ndos\n")
    assert workspace.read_bytes("win.txt") == b"uno\r\ndos\r\n"


# ------------------------------------------------------- read before edit --


async def test_an_edit_to_a_file_never_read_is_refused(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    out = await call(harness(workspace), "edit", path="a.py", old="x = 1", new="x = 2")
    assert "has not been read" in out
    assert get(workspace, "a.py") == "x = 1\n"


async def test_an_edit_after_a_read_goes_through(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    h = harness(workspace)
    await call(h, "read", path="a.py")
    out = await call(h, "edit", path="a.py", old="x = 1", new="x = 2")
    assert out.startswith("edited a.py")
    assert get(workspace, "a.py") == "x = 2\n"


async def test_a_file_changed_since_the_read_is_refused(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    h = harness(workspace)
    await call(h, "read", path="a.py")
    put(workspace, "a.py", "x = 1\ny = 2\n")  # a formatter, a checkout, a person
    out = await call(h, "edit", path="a.py", old="x = 1", new="x = 3")
    assert "has changed since you last read it" in out
    assert get(workspace, "a.py") == "x = 1\ny = 2\n"


async def test_rewriting_identical_bytes_does_not_make_a_file_stale(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "x = 1\n")
    h = harness(workspace)
    await call(h, "read", path="a.py")
    put(workspace, "a.py", "x = 1\n")
    assert (await call(h, "edit", path="a.py", old="x = 1", new="x = 2")).startswith("edited")


async def test_the_agents_own_edit_does_not_make_the_file_stale(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "a = 1\nb = 2\n")
    h = harness(workspace)
    await call(h, "read", path="a.py")
    await call(h, "edit", path="a.py", old="a = 1", new="a = 10")
    out = await call(h, "edit", path="a.py", old="b = 2", new="b = 20")
    assert out.startswith("edited")


async def test_overwriting_a_file_never_read_is_refused(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "keep")
    out = await call(harness(workspace), "write", path="a.py", content="clobber")
    assert "has not been read" in out
    assert get(workspace, "a.py") == "keep"


async def test_creating_a_file_needs_no_read_and_counts_as_one(
    workspace: WorkspaceBackend,
) -> None:
    h = harness(workspace)
    await call(h, "write", path="new.py", content="a = 1\n")
    assert (await call(h, "edit", path="new.py", old="a = 1", new="a = 2")).startswith("edited")


async def test_the_read_rule_can_be_lifted(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    h = harness(workspace, require_read=False)
    assert (await call(h, "edit", path="a.py", old="x = 1", new="x = 2")).startswith("edited")


async def test_multi_edit_needs_a_read_too(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    out = await call(
        harness(workspace), "multi_edit", path="a.py", edits=[{"old": "x", "new": "y"}]
    )
    assert "has not been read" in out


# ------------------------------------------------------------------- edit --


async def test_edit_replaces_an_exact_string(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "a = 1\nb = 2\n")
    await call(
        harness(workspace, require_read=False), "edit", path="a.py", old="b = 2", new="b = 3"
    )
    assert get(workspace, "a.py") == "a = 1\nb = 3\n"


async def test_edit_refuses_an_ambiguous_match_and_names_the_lines(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "x = 1\nx = 1\n")
    out = await call(
        harness(workspace, require_read=False), "edit", path="a.py", old="x = 1", new="x = 2"
    )
    assert "matches 2 times" in out
    assert "1, 2" in out
    assert get(workspace, "a.py") == "x = 1\nx = 1\n"


async def test_replace_all_changes_every_one(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\nx = 1\n")
    out = await call(
        harness(workspace, require_read=False),
        "edit",
        path="a.py",
        old="x = 1",
        new="x = 2",
        replace_all=True,
    )
    assert "2 places" in out
    assert get(workspace, "a.py") == "x = 2\nx = 2\n"


async def test_a_miss_says_what_is_wrong(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    out = await call(
        harness(workspace, require_read=False), "edit", path="a.py", old="y = 2", new="z"
    )
    assert "not found" in out


async def test_a_near_miss_is_applied_and_reported_with_its_diff(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "def f():\n    return 1\n")
    out = await call(
        harness(workspace, require_read=False),
        "edit",
        path="a.py",
        old="def f():\n  return 1",
        new="def f():\n  return 2",
    )
    assert "did not match exactly" in out
    assert "check the diff" in out
    assert get(workspace, "a.py") == "def f():\n    return 2\n"


async def test_an_edit_keeps_a_crlf_file_crlf(workspace: WorkspaceBackend) -> None:
    put(workspace, "win.txt", b"one\r\ntwo\r\n")
    out = await call(
        harness(workspace, require_read=False), "edit", path="win.txt", old="one\ntwo", new="1\n2"
    )
    assert "CRLF" in out
    assert workspace.read_bytes("win.txt") == b"1\r\n2\r\n"


async def test_identical_old_and_new_is_refused(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    out = await call(harness(workspace, require_read=False), "edit", path="a.py", old="x", new="x")
    assert "identical" in out


async def test_edit_on_a_missing_file_says_so(workspace: WorkspaceBackend) -> None:
    out = await call(harness(workspace), "edit", path="ghost.py", old="a", new="b")
    assert "no such file" in out


async def test_edit_on_a_directory_says_so(workspace: WorkspaceBackend) -> None:
    put(workspace, "src/a.py", "x")
    out = await call(harness(workspace), "edit", path="src", old="a", new="b")
    assert "is a directory" in out


# ------------------------------------------------------------- multi_edit --


async def test_multi_edit_applies_every_change(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "a = 1\nb = 2\n")
    out = await call(
        harness(workspace, require_read=False),
        "multi_edit",
        path="a.py",
        edits=[{"old": "a = 1", "new": "a = 10"}, {"old": "b = 2", "new": "b = 20"}],
    )
    assert "applied 2 edit(s)" in out
    assert get(workspace, "a.py") == "a = 10\nb = 20\n"


async def test_multi_edit_is_all_or_nothing(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "a = 1\nb = 2\n")
    out = await call(
        harness(workspace, require_read=False),
        "multi_edit",
        path="a.py",
        edits=[{"old": "a = 1", "new": "a = 10"}, {"old": "zzz", "new": "y"}],
    )
    assert "edit 2" in out
    assert "unchanged" in out
    assert get(workspace, "a.py") == "a = 1\nb = 2\n"


async def test_multi_edit_rejects_an_edit_with_no_target(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "a = 1\n")
    out = await call(
        harness(workspace, require_read=False), "multi_edit", path="a.py", edits=[{"new": "x"}]
    )
    assert "has no 'old' text" in out


async def test_multi_edit_reports_near_misses_and_can_replace_all(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "a.py", "def f():\n    return 1\nx = 0\nx = 0\n")
    out = await call(
        harness(workspace, require_read=False),
        "multi_edit",
        path="a.py",
        edits=[
            {"old": "def f():\n  return 1", "new": "def f():\n  return 2"},
            {"old": "x = 0", "new": "x = 9", "replace_all": True},
        ],
    )
    assert "edit 1 (old text did not match exactly" in out
    assert "edit 2 (2 places)" in out
    assert get(workspace, "a.py") == "def f():\n    return 2\nx = 9\nx = 9\n"


async def test_a_missing_file_in_multi_edit_says_so(workspace: WorkspaceBackend) -> None:
    out = await call(harness(workspace), "multi_edit", path="ghost.py", edits=[])
    assert "no such file" in out


# --------------------------------------------------------------------- ls --


async def test_ls_shows_the_tree_and_skips_the_noise(workspace: WorkspaceBackend) -> None:
    put(workspace, "README.md", "x")
    put(workspace, "src/app.py", "x")
    put(workspace, "node_modules/junk.js", "x")
    put(workspace, ".hidden/secret", "x")
    out = await call(harness(workspace), "ls")
    assert "README.md" in out
    assert "src/" in out
    assert "app.py" in out
    assert "node_modules" not in out
    assert ".hidden" not in out


async def test_ls_stops_at_the_depth_it_was_given(workspace: WorkspaceBackend) -> None:
    put(workspace, "a/b/c/deep.txt", "x")
    out = await call(harness(workspace), "ls", depth=2)
    assert "a/" in out
    assert "b/" not in out
    assert "deep.txt" not in out
    assert "deep.txt" in await call(harness(workspace), "ls", depth=4)


async def test_ls_truncates_a_directory_with_too_many_files(workspace: WorkspaceBackend) -> None:
    for i in range(45):
        put(workspace, f"many/f{i:02}.txt", "x")
    out = await call(harness(workspace), "ls", path="many")
    assert "... 5 more files" in out


async def test_ls_stops_a_listing_that_runs_too_long(workspace: WorkspaceBackend) -> None:
    for d in range(12):
        for i in range(40):
            put(workspace, f"d{d:02}/f{i:02}.txt", "x")
    out = await call(harness(workspace), "ls")
    assert out.endswith("use a smaller depth or a subdirectory")


async def test_ls_on_a_file_says_it_is_one(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "12345")
    assert "a.py is a file (5 bytes)" in await call(harness(workspace), "ls", path="a.py")


async def test_ls_on_a_missing_directory_says_so(workspace: WorkspaceBackend) -> None:
    assert "no such directory" in await call(harness(workspace), "ls", path="nowhere")


async def test_ls_of_an_empty_directory(workspace: WorkspaceBackend) -> None:
    if workspace.capabilities.root == "/":
        pytest.skip("the memory backend has no empty directories")
    workspace.exec("mkdir empty", timeout=10)
    assert await call(harness(workspace), "ls", path="empty") == "(empty)"


# ------------------------------------------------------------------ todos --


async def test_todos_round_trip(workspace: WorkspaceBackend) -> None:
    h = harness(workspace)
    out = await call(
        h,
        "todo_write",
        items=[
            {"task": "read the code", "status": "done"},
            {"task": "fix it", "status": "in_progress"},
            {"content": "test it", "status": "nonsense"},
            {"task": "   "},
            "not an item",
        ],
    )
    assert out.startswith("1/3 done")
    assert "[x] read the code" in out
    assert "[~] fix it" in out
    assert "[ ] test it" in out
    assert await call(h, "todo_read") == out.split("\n", 1)[1]
    assert [t.status for t in h.context.todos.snapshot()] == ["completed", "in_progress", "pending"]


async def test_an_empty_plan_says_so(workspace: WorkspaceBackend) -> None:
    assert await call(harness(workspace), "todo_read") == "(no todos)"
