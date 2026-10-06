# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``notebook_edit``: cells, not escaped JSON. Ported from tulip-code."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.unit.harness.conftest import get, put
from tulip.harness import HarnessConfig, WorkspaceBackend, build_harness


def _notebook(minor: int = 5, with_ids: bool = True) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n", "Intro"]},
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {"tags": ["keep"]},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["4\n"]}],
            "source": ["x = 2\n", "print(x * 2)"],
        },
    ]
    if with_ids:
        cells[0]["id"] = "intro"
        cells[1]["id"] = "calc"
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": minor,
    }


def _write(workspace: WorkspaceBackend, nb: dict[str, Any], indent: int | None = 1) -> None:
    put(workspace, "nb.ipynb", json.dumps(nb, indent=indent) + "\n")


def _load(workspace: WorkspaceBackend) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(get(workspace, "nb.ipynb"))
    return data


async def edit(workspace: WorkspaceBackend, **kwargs: Any) -> str:
    h = build_harness(workspace, tools=["notebook_edit"], config=HarnessConfig())
    out: str = await h.tool("notebook_edit").execute(**{"path": "nb.ipynb", **kwargs})
    return out


async def test_replace_by_id_sets_the_source_and_clears_stale_outputs(
    workspace: WorkspaceBackend,
) -> None:
    _write(workspace, _notebook())
    out = await edit(workspace, cell_id="calc", new_source="y = 3\ny")
    assert out == "replaced cell 1 (id calc) in nb.ipynb"
    cell = _load(workspace)["cells"][1]
    assert cell["source"] == ["y = 3\n", "y"]
    assert cell["outputs"] == [], "outputs of the old code are not results of the new"
    assert cell["execution_count"] is None
    assert cell["metadata"] == {"tags": ["keep"]}


async def test_replace_by_index_and_change_the_type(workspace: WorkspaceBackend) -> None:
    _write(workspace, _notebook())
    await edit(workspace, index=1, new_source="now prose", cell_type="markdown")
    cell = _load(workspace)["cells"][1]
    assert cell["cell_type"] == "markdown"
    assert "outputs" not in cell
    assert "execution_count" not in cell
    await edit(workspace, index=0, new_source="1 + 1", cell_type="code")
    first = _load(workspace)["cells"][0]
    assert first["outputs"] == []
    assert first["execution_count"] is None


async def test_insert_after_a_cell_gets_an_id_on_a_modern_notebook(
    workspace: WorkspaceBackend,
) -> None:
    _write(workspace, _notebook())
    out = await edit(
        workspace, mode="insert", cell_id="intro", cell_type="code", new_source="import os"
    )
    cells = _load(workspace)["cells"]
    assert cells[1]["source"] == ["import os"]
    assert cells[1]["outputs"] == []
    assert cells[1]["id"]
    assert cells[2]["id"] == "calc"
    assert "inserted code cell at index 1" in out


@pytest.mark.parametrize(("index", "where"), [(None, 2), (0, 0), (99, 2), (-5, 0)])
async def test_insert_by_position(
    workspace: WorkspaceBackend, index: int | None, where: int
) -> None:
    _write(workspace, _notebook(minor=4, with_ids=False))
    await edit(workspace, mode="insert", index=index, cell_type="markdown", new_source="note")
    cells = _load(workspace)["cells"]
    assert cells[where]["source"] == ["note"]
    assert "id" not in cells[where], "nbformat 4.4 cells have no id"
    assert "outputs" not in cells[where]


async def test_delete(workspace: WorkspaceBackend) -> None:
    _write(workspace, _notebook())
    assert await edit(workspace, mode="delete", cell_id="intro") == (
        "deleted cell 0 (id intro) in nb.ipynb"
    )
    assert [c["id"] for c in _load(workspace)["cells"]] == ["calc"]


async def test_the_files_own_layout_is_kept(workspace: WorkspaceBackend) -> None:
    """A one-cell change should be a one-cell diff in review."""
    nb = _notebook()
    put(workspace, "nb.ipynb", json.dumps(nb, indent=2))  # no trailing newline
    await edit(workspace, cell_id="intro", new_source="# Title\nIntro")
    assert get(workspace, "nb.ipynb") == json.dumps(nb, indent=2)


async def test_a_compact_notebook_stays_compact(workspace: WorkspaceBackend) -> None:
    put(workspace, "nb.ipynb", json.dumps(_notebook()))
    await edit(workspace, cell_id="calc", new_source="1")
    assert len(get(workspace, "nb.ipynb").splitlines()) == 1


@pytest.mark.parametrize(
    ("kwargs", "said"),
    [
        ({"mode": "rewrite"}, "mode must be one of"),
        ({"cell_type": "sql", "index": 0}, "cell_type must be one of"),
        ({"index": 0, "path": "a.py"}, "not a notebook"),
        ({"index": 0, "path": "missing.ipynb"}, "no such file"),
        ({"cell_id": "nope"}, "no cell with id 'nope' — ids: intro, calc"),
        ({"index": 7}, "index 7 is out of range — the notebook has 2 cell(s)"),
        ({}, "name the cell with cell_id or index"),
        ({"mode": "insert"}, "insert needs cell_type"),
        ({"mode": "insert", "cell_type": "code", "cell_id": "nope"}, "no cell with id"),
    ],
)
async def test_mistakes_are_explained_and_nothing_is_written(
    workspace: WorkspaceBackend, kwargs: dict[str, Any], said: str
) -> None:
    _write(workspace, _notebook())
    before = get(workspace, "nb.ipynb")
    put(workspace, "a.py", "x = 1\n")
    assert said in await edit(workspace, **kwargs)
    assert get(workspace, "nb.ipynb") == before


async def test_a_notebook_without_ids_says_so_when_one_is_asked_for(
    workspace: WorkspaceBackend,
) -> None:
    _write(workspace, _notebook(minor=4, with_ids=False))
    assert await edit(workspace, cell_id="calc", new_source="") == "no cell with id 'calc'"


@pytest.mark.parametrize(
    ("body", "said"), [("{not json", "not valid notebook JSON"), ("[]", "no cell list")]
)
async def test_a_broken_notebook_is_left_alone(
    workspace: WorkspaceBackend, body: str, said: str
) -> None:
    put(workspace, "nb.ipynb", body)
    assert said in await edit(workspace, index=0, new_source="x")
    assert get(workspace, "nb.ipynb") == body


async def test_a_one_line_file_is_written_compactly(workspace: WorkspaceBackend) -> None:
    put(workspace, "nb.ipynb", '{"cells": [], "nbformat": 4, "nbformat_minor": 5}')
    await edit(workspace, mode="insert", cell_type="raw", new_source="r")
    nb = _load(workspace)
    assert nb["cells"][0]["cell_type"] == "raw"
    assert "id" in nb["cells"][0]


async def test_paths_outside_the_workspace_are_refused(workspace: WorkspaceBackend) -> None:
    if workspace.capabilities.root == "/":
        pytest.skip("nothing is outside a namespace rooted at /")
    out = await edit(workspace, path="../elsewhere.ipynb", index=0, new_source="")
    assert "escapes the workspace" in out
