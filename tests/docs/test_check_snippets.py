# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Tests for the docs snippet checker.

A stub ``locus`` package is injected into ``sys.modules`` rather than
depending on the real SDK: the test env deliberately does not install it
(``skip-install = true``), and the checker's logic is about resolving and
comparing signatures, not about any particular SDK version.

Each case below corresponds to a real class of drift the checker exists
to catch, or to a real false positive it had to learn to suppress.
"""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel
from scripts.check_snippets import (
    Problem,
    check,
    check_block,
    iter_blocks,
    main,
)


class Widget(BaseModel):
    """Pydantic model: two required fields, extra keys silently ignored."""

    name: str
    provider: str
    tags: dict[str, str] = {}


class Flexible:
    """Sets ``__signature__`` while really accepting ``**kwargs`` — the
    shape that made an earlier version of this checker report eleven
    false positives against ``Agent``."""

    __signature__ = __import__("inspect").Signature(
        [
            __import__("inspect").Parameter(
                "model", __import__("inspect").Parameter.POSITIONAL_OR_KEYWORD
            )
        ]
    )

    def __init__(self, model: str = "", **kwargs: object) -> None:
        self.model = model


def plain(alpha: int = 1, beta: int = 2) -> int:
    """Ordinary function: no ``**kwargs``, so unknown keywords are real."""
    return alpha + beta


@pytest.fixture(autouse=True)
def stub_locus():
    """Install a fake ``locus`` package for the duration of a test."""
    pkg = types.ModuleType("locus")
    pkg.__path__ = []  # marks it as a package
    fake = types.ModuleType("locus.fake")
    fake.Widget = Widget
    fake.Flexible = Flexible
    fake.plain = plain
    sys.modules["locus"] = pkg
    sys.modules["locus.fake"] = fake
    pkg.fake = fake
    yield
    del sys.modules["locus"], sys.modules["locus.fake"]


def write(tmp_path, name, body):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# --------------------------------------------------------------------- blocks


def test_extracts_python_blocks_and_ignores_other_languages(tmp_path):
    page = write(
        tmp_path,
        "p.md",
        "text\n\n```python\nx = 1\n```\n\n```bash\nnot python\n```\n\n```py\ny = 2\n```\n",
    )
    assert [src.strip() for _, src in iter_blocks(page)] == ["x = 1", "y = 2"]


def test_dedents_an_indented_fence(tmp_path):
    page = write(tmp_path, "p.md", "1. step\n\n   ```python\n   x = 1\n   ```\n")
    ((_, source),) = iter_blocks(page)
    assert source.strip() == "x = 1"


def test_html_comment_skips_only_the_next_block(tmp_path):
    page = write(
        tmp_path,
        "p.md",
        "<!-- docs: skip -->\n```python\nfirst = 1\n```\n\n```python\nsecond = 2\n```\n",
    )
    assert [src.strip() for _, src in iter_blocks(page)] == ["second = 2"]


def test_inline_skip_marker(tmp_path):
    page = write(tmp_path, "p.md", "```python\n# docs: skip\nwhatever = 1\n```\n")
    assert list(iter_blocks(page)) == []


# ---------------------------------------------------------------------- checks


def test_syntax_error_is_reported():
    ((kind, message),) = check_block("def f(:\n    pass\n")
    assert kind == "syntax"
    assert message


def test_top_level_await_in_a_pasteable_snippet_is_reported():
    """``ast.parse`` accepts this; only ``compile`` rejects it.

    This exact snippet shipped as the quickstart's headline example (#52) and
    raised ``SyntaxError`` for every reader who pasted it — while this checker
    ran over that file and reported no problems.
    """
    problems = check_block(
        "import asyncio\n"
        "from locus.agent import Agent\n\n"
        "agent = Agent(model='openai:gpt-4o')\n"
        "result = await agent.arun('hi')\n"
        "print(result.message)\n"
    )
    assert [kind for kind, _ in problems] == ["syntax"]
    assert "await" in problems[0][1]


def test_a_fragment_with_undefined_helpers_is_not_reported():
    """The docs are full of these, and they were never pasteable anyway.

    A snippet that wires an Orchestrator to tools it never defines exists to
    show a shape. Rewriting it into ``asyncio.run(main())`` would add ceremony
    to the one thing it communicates — and an earlier discriminator
    ("does it have top-level imports?") flagged 60 of them.
    """
    problems = check_block(
        "from locus.multiagent import Orchestrator\n\n"
        "orch = Orchestrator(specialists=[settlement])\n"
        "await orch.run('dispute 4821')\n"
    )
    # Import findings are expected: this env carries a stub SDK, not the real
    # one. The claim under test is that the *syntax* gate stays quiet.
    assert "syntax" not in [kind for kind, _ in problems]


def test_a_bare_async_excerpt_is_not_reported():
    assert not check_block("async for event in agent.run('hi'):\n    print(event)\n")


def test_a_notebook_style_page_may_await_at_top_level():
    """Pages under docs/ follow the notebook convention: the block compiles inside a coroutine."""
    source = "from locus.fake import plain\n\nresult = await plain(alpha=3)\nprint(result)\n"
    assert check_block(source, notebook_style=True) == []
    assert [k for k, _ in check_block(source)] == ["syntax"], "the README stays strict"


def test_notebook_style_still_reports_code_broken_inside_a_coroutine():
    problems = check_block(
        "from locus.agent import Agent\nf(a=1, positional)\n", notebook_style=True
    )
    assert [k for k, _ in problems] == ["syntax"]


def test_notebook_style_is_decided_by_the_docs_directory():
    from scripts.check_snippets import ROOT, is_notebook_style  # noqa: PLC0415

    assert is_notebook_style(ROOT / "docs" / "concepts" / "events.md")
    assert not is_notebook_style(ROOT / "README.md")
    assert not is_notebook_style(ROOT / "examples" / "README.md")


def test_check_accepts_several_files_and_directories(tmp_path):
    (tmp_path / "a.md").write_text(
        "```python\nfrom locus.fake import plain\nprint(plain(alpha=1))\n```\n"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text(
        "```python\nfrom locus.fake import plain\nprint(plain(beta=1))\n```\n"
    )
    parsed, problems = check([tmp_path / "a.md", sub])
    assert parsed == 2
    assert problems == []


def test_a_complete_async_program_is_not_reported():
    problems = check_block(
        "import asyncio\n"
        "from locus.agent import Agent\n\n"
        "async def main():\n"
        "    print(await Agent().arun('hi'))\n\n"
        "asyncio.run(main())\n"
    )
    assert "syntax" not in [kind for kind, _ in problems]


def test_code_broken_inside_a_coroutine_too_is_still_reported():
    """The excerpt exemption must not become a blanket pass."""
    assert [k for k, _ in check_block("from locus.agent import Agent\n\nf(a=1, positional)\n")] == [
        "syntax"
    ]


def test_every_binding_form_counts_as_defined():
    """A name the snippet binds must not read as "undefined".

    Each case below binds its name through a different AST shape. If one of
    these stopped counting, snippets using it would be misclassified as
    fragments and their top-level ``await`` would stop being reported.
    """
    for label, source in {
        "lambda arg": "f = lambda x: x + 1\nawait f(1)\n",
        "class": "class C:\n    pass\n\nawait C()\n",
        "except name": "try:\n    pass\nexcept ValueError as err:\n    print(err)\nawait err\n",
        "for target": "for item in [1]:\n    pass\nawait item\n",
        "walrus": "if (n := 1):\n    pass\nawait n\n",
        "comprehension": "xs = [y for y in [1]]\nawait xs\n",
        "def arg": "def g(a, *rest, **kw):\n    return a\n\nawait g(1)\n",
    }.items():
        kinds = [kind for kind, _ in check_block(source)]
        assert "syntax" in kinds, f"{label}: bound name was treated as undefined"


def test_a_global_declaration_binds_the_name():
    assert [k for k, _ in check_block("global counter\nawait counter\n")] == ["syntax"]


def test_missing_symbol_is_reported():
    problems = check_block("from locus.fake import Nope\n")
    assert ("symbol", "locus.fake.Nope does not exist") in problems


def test_missing_module_is_reported():
    ((kind, message),) = check_block("from locus.gone import Thing\n")
    assert kind == "import"
    assert "locus.gone" in message


def test_plain_import_of_missing_module_is_reported():
    ((kind, message),) = check_block("import locus.absent\n")
    assert kind == "import"


def test_non_locus_imports_are_ignored():
    assert check_block("import os\nfrom json import dumps\n") == []


def test_good_snippet_is_silent():
    assert check_block("from locus.fake import plain\nplain(alpha=3)\n") == []


def test_unknown_keyword_on_a_plain_callable():
    problems = check_block("from locus.fake import plain\nplain(gamma=1)\n")
    assert ("kwarg", "locus.fake.plain(...) has no parameter 'gamma'") in problems


def test_var_keyword_callable_is_never_flagged():
    """The ``Agent`` false-positive regression."""
    source = "from locus.fake import Flexible\nFlexible(model='m', anything=1, more=2)\n"
    assert check_block(source) == []


def test_pydantic_unknown_field_is_flagged_even_though_ignored():
    source = "from locus.fake import Widget\nWidget(name='a', provider='b', nope=1)\n"
    problems = check_block(source)
    assert any("has no field 'nope'" in m for _, m in problems)


def test_pydantic_missing_required_field():
    source = "from locus.fake import Widget\nWidget(name='a')\n"
    problems = check_block(source)
    assert ("required", "locus.fake.Widget(...) is missing required field 'provider'") in problems


def test_pydantic_positional_or_splat_suppresses_required_check():
    """Both hide fields this checker cannot see, so it must not guess."""
    assert check_block("from locus.fake import Widget\nWidget(**payload)\n") == []
    assert check_block("from locus.fake import Widget\nWidget(data, name='a')\n") == []


def test_aliased_import_is_tracked():
    source = "from locus.fake import plain as go\ngo(gamma=1)\n"
    problems = check_block(source)
    assert any("has no parameter 'gamma'" in m for _, m in problems)


def test_attribute_call_through_a_module_import():
    problems = check_block("import locus.fake\nlocus.fake.plain(gamma=1)\n")
    assert any("has no parameter 'gamma'" in m for _, m in problems)


def test_calls_to_unbound_names_are_ignored():
    assert check_block("something_else(gamma=1)\n") == []


def test_non_callable_attribute_is_ignored():
    assert check_block("from locus.fake import Widget\nWidget.model_fields\n") == []


# ----------------------------------------------------------------------- walk


def test_check_walks_a_tree_and_counts_parsed_blocks(tmp_path):
    write(tmp_path, "ok.md", "```python\nfrom locus.fake import plain\nplain()\n```\n")
    write(tmp_path, "sub/bad.md", "```python\nfrom locus.fake import Nope\n```\n")
    parsed, problems = check(tmp_path)
    assert parsed == 2
    assert [p.kind for p in problems] == ["symbol"]
    assert problems[0].path.endswith("bad.md")


def test_problem_renders_readably():
    text = str(Problem("docs/x.md", 3, "symbol", "locus.a.B does not exist"))
    assert text == "docs/x.md: block 3: [symbol] locus.a.B does not exist"


def test_main_returns_zero_when_clean(tmp_path, capsys):
    write(tmp_path, "p.md", "```python\nx = 1\n```\n")
    assert main(["check_snippets.py", str(tmp_path)]) == 0
    assert "no problems found" in capsys.readouterr().out


def test_main_returns_one_and_explains_the_opt_out(tmp_path, capsys):
    write(tmp_path, "p.md", "```python\nfrom locus.fake import Nope\n```\n")
    assert main(["check_snippets.py", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "1 problem(s)" in out
    assert "docs: skip" in out


def test_main_rejects_a_missing_path(tmp_path, capsys):
    assert main(["check_snippets.py", str(tmp_path / "nope")]) == 2
    assert "no such path" in capsys.readouterr().err


# ------------------------------------------------------- resolution edge cases


def test_unresolvable_attribute_chain_is_ignored():
    """``Widget.missing(...)`` names nothing — no crash, no report."""
    assert check_block("from locus.fake import Widget\nWidget.missing(x=1)\n") == []


def test_star_import_is_skipped():
    assert check_block("from locus.fake import *\n") == []


def test_aliased_module_import_is_tracked():
    source = "import locus.fake as tf\ntf.plain(gamma=1)\n"
    problems = check_block(source)
    assert any("has no parameter 'gamma'" in m for _, m in problems)


def test_non_locus_plain_import_is_ignored():
    assert check_block("import os.path\nos.path.join(a='x')\n") == []


def test_callable_without_an_introspectable_signature_is_tolerated():
    """``inspect.signature`` raises for some C types (``dict`` is one).
    The checker must fall silent rather than crash the whole run."""
    sys.modules["locus.fake"].opaque = dict
    try:
        assert check_block("from locus.fake import opaque\nopaque(anything=1)\n") == []
    finally:
        del sys.modules["locus.fake"].opaque
