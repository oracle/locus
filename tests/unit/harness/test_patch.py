# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``apply_patch``: GPT's edit format, with ``edit``'s containment.

Ported from tulip-code. The format is the easy part; what these pin is the
contract: paths stay inside the workspace, the whole patch is worked out
before anything is written, and a patch that does not fully apply changes
nothing.
"""

from __future__ import annotations

import pytest

from tests.unit.harness.conftest import get, put
from tulip.harness import BackendError, FileChange, HarnessConfig, WorkspaceBackend, build_harness
from tulip.harness.tools.patch import PatchError, apply_hunks, parse


def _patch(*body: str) -> str:
    return "\n".join(["*** Begin Patch", *body, "*** End Patch"])


async def apply(workspace: WorkspaceBackend, text: str, **config: object) -> str:
    h = build_harness(workspace, tools=["apply_patch"], config=HarnessConfig(**config))  # type: ignore[arg-type]
    out: str = await h.tool("apply_patch").execute(input=text)
    return out


async def test_add_update_delete_in_one_patch(workspace: WorkspaceBackend) -> None:
    put(workspace, "app.py", "def handler():\n    return None\n\nprint('x')\n")
    put(workspace, "old.py", "gone\n")
    out = await apply(
        workspace,
        _patch(
            "*** Add File: docs/notes.md",
            "+# Notes",
            "+second",
            "*** Update File: app.py",
            "@@ def handler():",
            "-    return None",
            "+    return 42",
            "*** Delete File: old.py",
        ),
    )
    assert out == "applied:\n  created docs/notes.md\n  edited app.py\n  deleted old.py"
    assert get(workspace, "docs/notes.md") == "# Notes\nsecond\n"
    assert get(workspace, "app.py") == "def handler():\n    return 42\n\nprint('x')\n"
    assert not workspace.exists("old.py")


async def test_a_move_renames_and_edits(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    await apply(
        workspace, _patch("*** Update File: a.py", "*** Move to: pkg/b.py", "-x = 1", "+x = 2")
    )
    assert not workspace.exists("a.py")
    assert get(workspace, "pkg/b.py") == "x = 2\n"


async def test_a_move_onto_itself_is_an_edit(workspace: WorkspaceBackend) -> None:
    put(workspace, "a.py", "x = 1\n")
    out = await apply(
        workspace, _patch("*** Update File: a.py", "*** Move to: a.py", "-x = 1", "+x = 2")
    )
    assert "edited a.py" in out
    assert get(workspace, "a.py") == "x = 2\n"


async def test_a_hunk_that_does_not_match_changes_nothing(workspace: WorkspaceBackend) -> None:
    put(workspace, "one.py", "a = 1\n")
    put(workspace, "two.py", "b = 1\n")
    out = await apply(
        workspace,
        _patch(
            "*** Update File: one.py",
            "-a = 1",
            "+a = 2",
            "*** Update File: two.py",
            "-not in the file",
            "+b = 2",
        ),
    )
    assert out.startswith("patch not applied")
    assert "two.py: hunk 1 does not match" in out
    assert get(workspace, "one.py") == "a = 1\n", "the first file is untouched too"


async def test_a_failed_write_puts_every_file_back(
    workspace: WorkspaceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(workspace, "one.py", "a = 1\n")
    put(workspace, "two.py", "b = 1\n")
    put(workspace, "gone.py", "bye\n")
    real = workspace.write_bytes

    def flaky(path: str, data: bytes) -> None:
        if path.endswith("two.py") and data == b"b = 2\n":
            raise BackendError("permission_denied", path)
        real(path, data)

    monkeypatch.setattr(workspace, "write_bytes", flaky)
    out = await apply(
        workspace,
        _patch(
            "*** Add File: three.py",
            "+c",
            "*** Delete File: gone.py",
            "*** Update File: one.py",
            "-a = 1",
            "+a = 2",
            "*** Update File: two.py",
            "-b = 1",
            "+b = 2",
        ),
    )
    assert "put back" in out
    assert get(workspace, "one.py") == "a = 1\n"
    assert get(workspace, "gone.py") == "bye\n"
    assert not workspace.exists("three.py")


async def test_the_journal_hook_sees_each_change_before_it_is_written(
    workspace: WorkspaceBackend,
) -> None:
    put(workspace, "one.py", "a = 1\n")
    seen: list[FileChange] = []
    await apply(
        workspace,
        _patch("*** Update File: one.py", "-a = 1", "+a = 2", "*** Add File: new.py", "+n"),
        on_change=seen.append,
    )
    assert [(c.shown, c.before, c.after, c.action) for c in seen] == [
        ("one.py", "a = 1\n", "a = 2\n", "edited"),
        ("new.py", None, "n\n", "created"),
    ]


async def test_a_path_outside_the_workspace_is_refused(workspace: WorkspaceBackend) -> None:
    if workspace.capabilities.root == "/":
        pytest.skip("nothing is outside a namespace rooted at /")
    out = await apply(workspace, _patch("*** Add File: ../evil.py", "+x"))
    assert "escapes the workspace" in out


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (["*** Add File: exists.py", "+x"], "already exists"),
        (["*** Add File: dir", "+x"], "already exists"),
        (["*** Update File: ghost.py", "-x", "+y"], "no such file"),
        (["*** Update File: exists.py", "*** Move to: other.py", "-old", "+new"], "exists"),
        (["*** Update File: exists.py", "-old", "+a", "*** Update File: exists.py", "+b"], "twice"),
        (
            ["*** Add File: new.py", "+n", "*** Update File: exists.py", "*** Move to: new.py"],
            "twice",
        ),
    ],
)
async def test_conflicts_are_refused_before_anything_is_written(
    workspace: WorkspaceBackend, body: list[str], why: str
) -> None:
    put(workspace, "exists.py", "old\n")
    put(workspace, "other.py", "taken\n")
    put(workspace, "dir/inside.py", "x\n")
    out = await apply(workspace, _patch(*body))
    assert out.startswith("patch not applied")
    assert why in out
    assert get(workspace, "exists.py") == "old\n"


async def test_an_empty_added_file(workspace: WorkspaceBackend) -> None:
    await apply(workspace, "*** Begin Patch\n*** Add File: empty.txt\n*** End Patch")
    assert get(workspace, "empty.txt") == ""


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("no envelope", "starts with"),
        ("*** Begin Patch\n*** Add File: a\n+x", "ends with"),
        ("*** Begin Patch\n*** End Patch", "no files"),
        ("*** Begin Patch\n+orphan\n*** End Patch", "expected a file header"),
        ("*** Begin Patch\n*** Add File: a\nno plus\n*** End Patch", "starts with '+'"),
        ("*** Begin Patch\n*** Delete File: a\n+x\n*** End Patch", "takes no body"),
        ("*** Begin Patch\n*** Update File: a\n*** End Patch", "no hunks"),
        ("*** Begin Patch\n*** Update File: a\n*** End of File\n*** End Patch", "outside a hunk"),
        ("*** Begin Patch\n*** Update File: a\n?bad\n*** End Patch", "' ', '-' or '+'"),
    ],
)
def test_malformed_patches_are_explained(text: str, why: str) -> None:
    with pytest.raises(PatchError) as raised:
        parse(text)
    assert why in str(raised.value)


def test_whitespace_drift_still_matches() -> None:
    body = "def f():\n    x = 1   \n    return x\n"
    ops = parse(_patch("*** Update File: f.py", "-  x = 1", "+  x = 2"))
    assert "x = 2" in apply_hunks("f.py", body, ops[0].hunks)


def test_the_anchor_picks_between_identical_blocks() -> None:
    body = "def a():\n    return 1\n\ndef b():\n    return 1\n"
    ops = parse(_patch("*** Update File: f.py", "@@ def b():", "-    return 1", "+    return 2"))
    assert apply_hunks("f.py", body, ops[0].hunks) == (
        "def a():\n    return 1\n\ndef b():\n    return 2\n"
    )


def test_an_anchor_that_is_also_context_is_kept() -> None:
    ops = parse(
        _patch(
            "*** Update File: f.py", "@@ def a():", " def a():", "-    return 1", "+    return 3"
        )
    )
    assert apply_hunks("f.py", "def a():\n    return 1\n", ops[0].hunks) == (
        "def a():\n    return 3\n"
    )


def test_a_missing_anchor_is_explained() -> None:
    ops = parse(_patch("*** Update File: f.py", "@@ def nope():", "-x", "+y"))
    with pytest.raises(PatchError, match="no line matching"):
        apply_hunks("f.py", "x\n", ops[0].hunks)


def test_end_of_file_anchors_at_the_end() -> None:
    ops = parse(_patch("*** Update File: f.py", "-x", "+y", "*** End of File"))
    assert apply_hunks("f.py", "x\nx\n", ops[0].hunks) == "x\ny\n"


def test_a_hunk_with_only_additions_inserts_at_the_cursor() -> None:
    ops = parse(_patch("*** Update File: f.py", "@@ first", "+inserted"))
    assert apply_hunks("f.py", "first\nsecond\n", ops[0].hunks) == "first\ninserted\nsecond\n"


def test_a_file_without_a_trailing_newline_keeps_it_that_way() -> None:
    ops = parse(_patch("*** Update File: f.py", "-a", "+b"))
    assert apply_hunks("f.py", "a", ops[0].hunks) == "b"


def test_an_empty_line_in_a_hunk_is_context() -> None:
    ops = parse("*** Begin Patch\n*** Update File: f.py\n-a\n\n+b\n*** End Patch")
    assert ops[0].hunks[0].old == ["a", ""]
