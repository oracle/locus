# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the documentation snippet checker.

The checker exists because nothing verified the docs: the README advertised
an import that raised, ``examples/README.md`` imported two hooks from a module
that never contained them, and a quickstart snippet was a ``SyntaxError``.
These tests pin the behaviour, including the two false-positive classes that
would otherwise make the gate noisy enough to be turned off.
"""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from check_doc_snippets import check, check_block, iter_blocks  # noqa: E402


# --------------------------------------------------------------------------
# Syntax — compile, not merely parse
# --------------------------------------------------------------------------


def test_valid_code_passes() -> None:
    assert check_block("from tulip.agent import Agent\nprint(Agent)\n") == []


def test_top_level_await_in_a_complete_snippet_is_reported() -> None:
    """``ast.parse`` accepts this; only ``compile`` rejects it.

    That gap is exactly how a quickstart shipped raising SyntaxError.
    """
    problems = check_block("import asyncio\nfrom tulip.agent import Agent\n\nawait Agent()\n")
    assert [kind for kind, _ in problems] == ["syntax"]
    assert "await" in problems[0][1]


def test_a_bare_async_excerpt_is_not_reported() -> None:
    """Prose legitimately shows this shape to illustrate an API."""
    assert check_block("async for event in agent.run('hi'):\n    print(event)\n") == []


def test_a_bare_await_excerpt_is_not_reported() -> None:
    assert check_block("result = await agent.arun('hi')\n") == []


def test_code_invalid_even_inside_a_coroutine_is_reported() -> None:
    assert [k for k, _ in check_block("f(a=1, positional)\n")] == ["syntax"]


def test_unparseable_code_is_reported() -> None:
    assert [k for k, _ in check_block("def (:\n")] == ["syntax"]


def test_a_complete_async_program_passes() -> None:
    source = (
        "import asyncio\n"
        "from tulip.agent import Agent\n\n"
        "async def main():\n"
        "    await Agent().arun('hi')\n\n"
        "asyncio.run(main())\n"
    )
    assert check_block(source) == []


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------


def test_a_missing_symbol_is_reported() -> None:
    problems = check_block("from tulip.agent import NoSuchThing\n")
    assert [k for k, _ in problems] == ["symbol"]
    assert "NoSuchThing" in problems[0][1]


def test_a_missing_module_is_reported() -> None:
    assert [k for k, _ in check_block("from tulip.nope import thing\n")] == ["import"]


def test_non_tulip_imports_are_ignored() -> None:
    """Only this SDK's surface is verifiable from here."""
    assert check_block("from os.path import nonexistent_thing\n") == []


def test_a_real_export_resolves() -> None:
    assert check_block("from tulip.testing import ScriptedModel\n") == []


# --------------------------------------------------------------------------
# Keyword arguments
# --------------------------------------------------------------------------


def test_an_unknown_keyword_is_reported() -> None:
    problems = check_block(
        "from tulip.evaluation import EvalCase\nEvalCase(name='x', prompt='p', nope=1)\n"
    )
    assert any("nope" in message for _, message in problems)


def test_real_keywords_pass() -> None:
    assert (
        check_block("from tulip.evaluation import EvalCase\nEvalCase(name='x', prompt='p')\n") == []
    )


def test_a_callable_taking_var_keyword_is_not_second_guessed() -> None:
    """``Agent`` sets ``__signature__``, hiding the ``**kwargs`` it accepts.

    Without this the checker would report every documented Agent call.
    """
    assert check_block("from tulip.agent import Agent\nAgent(anything_at_all=1)\n") == []


# --------------------------------------------------------------------------
# Block extraction and opt-outs
# --------------------------------------------------------------------------


def test_only_python_blocks_are_checked(tmp_path: pathlib.Path) -> None:
    md = tmp_path / "d.md"
    md.write_text("```bash\npip install x\n```\n\n```python\nx = 1\n```\n")
    assert [source.strip() for _, source in iter_blocks(md)] == ["x = 1"]


def test_an_inline_skip_marker_exempts_a_block(tmp_path: pathlib.Path) -> None:
    md = tmp_path / "d.md"
    md.write_text("```python\n# docs: skip\nfrom tulip.agent import Nope\n```\n")
    assert list(iter_blocks(md)) == []


def test_a_comment_skip_marker_exempts_the_next_block(tmp_path: pathlib.Path) -> None:
    md = tmp_path / "d.md"
    md.write_text("<!-- docs: skip -->\n```python\nfrom tulip.agent import Nope\n```\n")
    assert list(iter_blocks(md)) == []


def test_a_skip_marker_does_not_silence_the_following_block(
    tmp_path: pathlib.Path,
) -> None:
    """One opt-out must exempt exactly one block, or it hides real defects."""
    md = tmp_path / "d.md"
    md.write_text("<!-- docs: skip -->\n```python\nx = 1\n```\n\n```python\ny = 2\n```\n")
    assert [source.strip() for _, source in iter_blocks(md)] == ["y = 2"]


def test_indented_blocks_are_dedented(tmp_path: pathlib.Path) -> None:
    """A fence inside a list item is indented; checking it raw would fail."""
    md = tmp_path / "d.md"
    md.write_text("- item:\n\n  ```python\n  x = 1\n  ```\n")
    assert [source.strip() for _, source in iter_blocks(md)] == ["x = 1"]


# --------------------------------------------------------------------------
# The repo's own docs must stay clean — this is the regression guard
# --------------------------------------------------------------------------


def test_the_repos_own_markdown_has_no_broken_snippets() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    checked, problems = check([root / "README.md", root / "examples" / "README.md"])
    assert checked > 0, "no snippets found — the extractor is broken, not the docs"
    assert problems == [], "\n".join(str(p) for p in problems)
