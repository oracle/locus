# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Unit: the llms.txt hook writes Markdown twins, llms.txt and llms-full.txt."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hooks import llms_txt


def _page(src: str, url: str, title: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        file=SimpleNamespace(src_uri=src),
        url=url,
        title=title,
        meta={"description": description} if description else {},
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "mkdocs.yml").write_text("site_name: locus\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "demo.py").write_text("one\ntwo\nthree\nfour\n")
    (tmp_path / "site").mkdir()
    return tmp_path


def _config(repo: Path, nav: list) -> dict:
    return {
        "config_file_path": str(repo / "mkdocs.yml"),
        "site_dir": str(repo / "site"),
        "site_url": "https://docs.example/",
        "site_name": "locus",
        "site_description": "An agent harness.",
        "markdown_extensions": [{"pymdownx.snippets": {"base_path": ["."]}}],
        "nav": nav,
    }


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    llms_txt._pages.clear()


def test_snippets_expand_with_the_line_forms_the_site_uses(repo: Path) -> None:
    roots = [repo]
    assert llms_txt._expand_snippets('--8<-- "examples/demo.py"', roots) == "one\ntwo\nthree\nfour"
    assert llms_txt._expand_snippets('--8<-- "examples/demo.py:2:3"', roots) == "two\nthree"
    assert llms_txt._expand_snippets('--8<-- "examples/demo.py:3:"', roots) == "three\nfour"
    assert llms_txt._expand_snippets('--8<-- "examples/demo.py:3"', roots) == "three\nfour"
    assert llms_txt._expand_snippets('  --8<-- "examples/demo.py::2"', roots) == "  one\n  two"
    # An include the roots cannot resolve is left as written rather than dropped.
    assert llms_txt._expand_snippets('--8<-- "missing.py"', roots) == '--8<-- "missing.py"'


def test_inlined_diagram_becomes_its_title(repo: Path) -> None:
    markdown = (
        'Before.\n\n<figure class="tl-dia">\n<svg><title id="x--title">The agent loop</title>'
        "<rect/></svg>\n</figure>\n\nAfter. <!-- a note -->\n"
    )
    assert (
        llms_txt._plain_markdown(markdown, [repo])
        == "Before.\n\n*Diagram: The agent loop*\n\nAfter.\n"
    )


def test_post_build_writes_twins_index_and_full_file(repo: Path) -> None:
    config = _config(
        repo,
        nav=[
            {"Concepts": [{"Start here": [{"The Agent": "concepts/agent.md"}]}]},
            {"API reference": [{"Agent": "api/agent.md"}]},
        ],
    )
    llms_txt.on_page_markdown(
        "# Welcome\n\nHome body.\n", page=_page("index.md", "", "Home", "Front door"), config=config
    )
    llms_txt.on_page_markdown(
        '# The Agent\n\nIntro.\n\n```python\n--8<-- "examples/demo.py:1:2"\n```\n',
        page=_page("concepts/agent.md", "concepts/agent/", "The Agent", "One class."),
        config=config,
    )
    llms_txt.on_page_markdown(
        "# Agent\n\n::: locus.agent.Agent\n    options:\n      members: true\n",
        page=_page("api/agent.md", "api/agent/", "Agent"),
        config=config,
    )

    llms_txt.on_post_build(config)

    site = repo / "site"
    twin = (site / "concepts/agent/index.md").read_text()
    assert twin.startswith(
        "# The Agent\n\n> One class.\n\nSource: https://docs.example/concepts/agent/\n\n"
    )
    assert twin.count("# The Agent") == 1, "the page's own H1 is kept, not duplicated"
    assert "```python\none\ntwo\n```" in twin

    api_twin = (site / "api/agent/index.md").read_text()
    assert ":::" not in api_twin
    assert "API reference" in api_twin

    index = (site / "llms.txt").read_text()
    assert index.startswith("# Locus\n\n> An agent harness.\n")
    assert "\n## Home\n\n- [Home](https://docs.example/index.md): Front door\n" in index
    assert (
        "\n## Concepts\n\n- [The Agent](https://docs.example/concepts/agent/index.md): One class.\n"
        in index
    )
    assert "\n## API reference\n\n- [Agent](https://docs.example/api/agent/index.md)\n" in index
    assert index.rstrip().endswith(
        "- [llms-full.txt](https://docs.example/llms-full.txt): every guide page in one file"
    )

    full = (site / "llms-full.txt").read_text()
    assert "Home body." in full and "Intro." in full
    assert "locus.agent.Agent" not in full, "API pages are linked from llms.txt, not inlined"
    assert full.index("Home body.") < full.index("Intro."), "nav order, home first"


def test_page_markdown_is_returned_unchanged(repo: Path) -> None:
    config = _config(repo, nav=[])
    markdown = "# T\n\nbody\n"
    assert (
        llms_txt.on_page_markdown(markdown, page=_page("t.md", "t/", "T"), config=config)
        is markdown
    )


def test_facts_tiles_become_a_list_and_the_filter_box_is_dropped(repo: Path) -> None:
    markdown = (
        "Intro.\n\n"
        '<dl class="locus-facts" markdown="block">\n'
        '<div markdown="block"><dt>Python</dt><dd>3.11–3.14</dd></div>\n'
        '<div markdown="block"><dt>Tested SDK</dt><dd markdown="span">`locus-sdk` 3.0.1</dd></div>\n'
        "</dl>\n\n"
        '<div class="notebook-filter">\n'
        '  <input type="search" id="notebook-filter-input" placeholder="Filter examples" />\n'
        "</div>\n\n"
        "## Start here\n"
    )
    assert llms_txt._plain_markdown(markdown, [repo]) == (
        "Intro.\n\n- **Python:** 3.11–3.14\n- **Tested SDK:** `locus-sdk` 3.0.1\n\n## Start here\n"
    )
