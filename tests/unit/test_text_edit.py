# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The edit-matching cascade in :mod:`tulip.tools.text_edit`.

Each strategy gets the near miss it exists for, and the ambiguity it must
refuse. The regression set at the bottom is the shapes real open-model edits
missed on: trailing whitespace, tabs for spaces, CRLF files, quotes escaped
one level too deep.
"""

from __future__ import annotations

import pytest

from tulip.deepagent.backends.protocol import replace_in_file
from tulip.tools.text_edit import (
    DEFAULT_STRATEGIES,
    HINT_MAX_FILE_LINES,
    EditMatchError,
    EditStrategy,
    Hit,
    apply_edit,
    block_anchor,
    closest_region,
    escape_normalized,
    line_trimmed,
    whitespace_normalized,
)


def _error(content: str, old: str, new: str, **kw: bool) -> EditMatchError:
    with pytest.raises(EditMatchError) as info:
        apply_edit(content, old, new, **kw)
    return info.value


# ------------------------------------------------------------------ exact --


def test_an_exact_match_is_the_common_path() -> None:
    out = apply_edit("a = 1\nb = 2\n", "b = 2", "b = 3")
    assert out.content == "a = 1\nb = 3\n"
    assert out.strategy == "exact"
    assert out.exact
    assert out.replacements == 1
    assert out.lines == (2,)
    assert not out.crlf


def test_an_exact_match_in_two_places_is_refused_with_both_lines() -> None:
    err = _error("x = 1\ny = 0\nx = 1\n", "x = 1", "x = 2")
    assert err.reason == "ambiguous"
    assert err.lines == (1, 3)
    assert err.strategy == "exact"
    assert "2 times" in str(err)
    assert "replace_all" in str(err)


def test_replace_all_changes_every_exact_match() -> None:
    out = apply_edit("x = 1\ny = 0\nx = 1\n", "x = 1", "x = 2", replace_all=True)
    assert out.content == "x = 2\ny = 0\nx = 2\n"
    assert out.replacements == 2
    assert out.lines == (1, 3)


def test_an_ambiguous_list_of_lines_is_capped() -> None:
    err = _error("x\n" * 12, "x", "y")
    assert "and 2 more" in str(err)


def test_empty_old_text_is_refused() -> None:
    assert _error("abc", "", "x").reason == "empty"


def test_identical_old_and_new_are_refused() -> None:
    assert _error("abc", "abc", "abc").reason == "unchanged"


# ----------------------------------------------------------- line_trimmed --


def test_line_trimmed_forgives_indentation_and_reindents_new() -> None:
    content = "class A:\n    def f(self):\n        return 1\n"
    out = apply_edit(content, "def f(self):\n    return 1\n", "def f(self):\n    return 2\n")
    assert out.strategy == "line_trimmed"
    assert out.content == "class A:\n    def f(self):\n        return 2\n"


def test_line_trimmed_maps_spaces_onto_tabs_for_deeper_lines_too() -> None:
    content = "def f():\n\treturn 1\n"
    out = apply_edit(
        content,
        "def f():\n    return 1\n",
        "def f():\n    if x:\n        return 2\n    return 1\n",
    )
    assert out.content == "def f():\n\tif x:\n\t\treturn 2\n\treturn 1\n"


def test_line_trimmed_maps_tabs_onto_spaces() -> None:
    content = "def f():\n    return 1\n"
    out = apply_edit(content, "def f():\n\treturn 1", "def f():\n\tif x:\n\t\treturn 2")
    assert out.content == "def f():\n    if x:\n        return 2\n"


def test_line_trimmed_leaves_new_alone_when_the_indent_mapping_is_inconsistent() -> None:
    # One quoted indent ("") standing for two file indents: no safe mapping.
    content = "a\n    b\n"
    out = apply_edit(content, "a\nb", "a\nc")
    assert out.strategy == "line_trimmed"
    assert out.content == "a\nc\n"


def test_line_trimmed_at_end_of_file_without_a_newline_does_not_add_one() -> None:
    out = apply_edit("a\n  b", "b\n", "c\n")
    assert out.content == "a\n  c"


def test_a_crlf_file_with_no_final_newline_gains_no_stray_carriage_return() -> None:
    out = apply_edit("a\r\nb", "a\nb\n", "a\nc\n")
    assert out.content == "a\r\nc"


def test_a_two_space_model_indent_maps_onto_tabs_level_by_level() -> None:
    content = "if a:\n\tif b:\n\t\tx()\n"
    out = apply_edit(content, "  if b:\n    x()", "  if b:\n    x()\n      y()")
    assert out.content == "if a:\n\tif b:\n\t\tx()\n\t\t\ty()\n"


def test_a_line_shallower_than_any_quoted_indent_is_left_as_written() -> None:
    content = "class A:\n\tdef f(self):\n\t\tpass\n"
    out = apply_edit(
        content, "    def f(self):\n        pass", "    def f(self):\n        pass\nx = 1"
    )
    assert out.content == "class A:\n\tdef f(self):\n\t\tpass\nx = 1\n"


def test_line_trimmed_ignores_a_quote_of_only_blank_lines() -> None:
    assert line_trimmed("a\n\nb", "  \n\n", "x") == []


def test_line_trimmed_twice_is_ambiguous_not_a_guess() -> None:
    err = _error("  x = 1\n    x = 1\n", " x = 1 ", "y")
    assert err.reason == "ambiguous"
    assert err.strategy == "line_trimmed"


# -------------------------------------------------- whitespace_normalized --


def test_whitespace_normalized_forgives_spacing_inside_a_line() -> None:
    out = apply_edit("result = call(a, b)\n", "call(a,b)", "call(a, c)")
    assert out.strategy == "whitespace_normalized"
    assert out.content == "result = call(a, c)\n"


def test_whitespace_normalized_forgives_a_reflowed_call() -> None:
    content = "x = f(\n    a,\n    b,\n)\n"
    out = apply_edit(content, "x = f(a, b,)", "x = g(a, b)")
    assert out.content == "x = g(a, b)\n"


def test_whitespace_normalized_does_not_match_inside_a_word() -> None:
    assert whitespace_normalized("maxx = 10\n", "x=1", "y") == []


def test_whitespace_normalized_keeps_separated_words_separated() -> None:
    assert whitespace_normalized("returnx\n", "return x", "y") == []


def test_whitespace_normalized_needs_two_tokens() -> None:
    assert whitespace_normalized("abc", "abc", "y") == []


def test_whitespace_normalized_reindents_lines_after_the_first() -> None:
    content = "if a:\n\tcall(a, b)\n\tdone()\n"
    out = apply_edit(content, "    call(a,b)\n    done()\n", "    call(a, c)\n    done()\n")
    assert out.strategy == "whitespace_normalized"
    assert out.content == "if a:\n\tcall(a, c)\n\tdone()\n"


# ------------------------------------------------------ escape_normalized --


def test_escape_normalized_unescapes_both_old_and_new() -> None:
    out = apply_edit('print("hi")\n', 'print(\\"hi\\")', 'print(\\"bye\\")')
    assert out.strategy == "escape_normalized"
    assert out.content == 'print("bye")\n'


def test_escape_normalized_reads_an_escaped_newline_as_a_line_break() -> None:
    out = apply_edit("a = 1\nb = 2\n", "a = 1\\nb = 2", "a = 1\\nb = 3")
    assert out.content == "a = 1\nb = 3\n"


def test_escape_normalized_skips_a_quote_with_nothing_to_unescape() -> None:
    assert escape_normalized("a", "a", "b") == []


def test_a_file_that_really_contains_escapes_matches_exactly_first() -> None:
    content = 'msg = "say \\"hi\\""\n'
    out = apply_edit(content, 'msg = "say \\"hi\\""', 'msg = "say \\"bye\\""')
    assert out.strategy == "exact"
    assert out.content == 'msg = "say \\"bye\\""\n'


# ----------------------------------------------------------- block_anchor --

_SOURCE = (
    "def f(a):\n"
    "    # compute the total\n"
    "    total = a + 1\n"
    "    return total\n"
    "\n"
    "def g():\n"
    "    pass\n"
)


def test_block_anchor_finds_a_block_with_a_stale_middle_line() -> None:
    old = "def f(a):\n    # compute it\n    total = a + 1\n    return total\n"
    out = apply_edit(_SOURCE, old, "def f(a):\n    return a + 1\n")
    assert out.strategy == "block_anchor"
    assert out.content == "def f(a):\n    return a + 1\n\ndef g():\n    pass\n"


def test_block_anchor_tolerates_a_missing_line() -> None:
    content = "def f(a):\n    x = 1\n    y = 2\n    z = 3\n    w = 4\n    return a\n"
    old = "def f(a):\n    x = 1\n    y = 2\n    w = 4\n    return a"
    out = apply_edit(content, old, "def f(a):\n    return a")
    assert out.strategy == "block_anchor"
    assert out.content == "def f(a):\n    return a\n"


def test_block_anchor_rejects_a_dissimilar_middle() -> None:
    old = "def f(a):\n    completely = different\n    return total\n"
    assert block_anchor(_SOURCE, old, "x") == []


def test_block_anchor_needs_three_lines_and_a_real_head() -> None:
    assert block_anchor(_SOURCE, "def f(a):\n    return total\n", "x") == []
    assert block_anchor("}\nx\n}\n", "}\ny\n}\n", "z") == []


def test_block_anchor_refuses_two_similar_blocks_even_with_replace_all() -> None:
    content = "def f():\n    a = 1\n    return a\n\ndef f():\n    a = 2\n    return a\n"
    old = "def f():\n    a = 3\n    return a\n"
    err = _error(content, old, "x", replace_all=True)
    assert err.reason == "ambiguous"
    assert err.strategy == "block_anchor"
    assert "replace_all" not in str(err)


# ---------------------------------------------------------- the cascade --


def test_the_cascade_stops_at_the_first_strategy_that_matches() -> None:
    # line_trimmed finds two places; whitespace_normalized would find the
    # same two. The cascade must refuse, not fall through to a looser guess.
    content = "  a = 1\n\ta = 1\n"
    assert _error(content, "a = 1 ", "b").strategy == "line_trimmed"


def test_replace_all_never_applies_overlapping_hits() -> None:
    overlapping = EditStrategy("overlap", lambda c, o, n: [Hit(0, 2, n), Hit(1, 3, n)])
    with pytest.raises(EditMatchError) as info:
        apply_edit("aaa", "zz", "b", replace_all=True, strategies=(overlapping,))
    assert info.value.reason == "ambiguous"
    assert "replace_all" not in str(info.value)


def test_a_token_match_may_span_lines_the_quote_joined() -> None:
    # Only whitespace lies between the tokens, so nothing unquoted is replaced.
    out = apply_edit("a\n\n\nb\n", "a b", "c")
    assert out.strategy == "whitespace_normalized"
    assert out.content == "c\n"


def test_duplicate_hits_from_one_strategy_count_once() -> None:
    twice = EditStrategy("twice", lambda c, o, n: [Hit(0, 1, n), Hit(0, 1, n)])
    out = apply_edit("abc", "zzz", "x", strategies=(twice,))
    assert out.content == "xbc"
    assert out.replacements == 1


def test_the_default_order_is_strictest_first() -> None:
    assert [s.name for s in DEFAULT_STRATEGIES] == [
        "exact",
        "line_trimmed",
        "whitespace_normalized",
        "escape_normalized",
        "block_anchor",
    ]


# ------------------------------------------------------------- not found --


def test_not_found_names_the_closest_region_with_line_numbers() -> None:
    content = "import os\n\ndef load(path):\n    with open(path) as fh:\n        return fh.read()\n"
    err = _error(content, "def load(p):\n    with open(p, 'rb') as f:\n", "x")
    assert err.reason == "not_found"
    assert err.hint is not None
    assert "lines 3-4" in err.hint
    assert "     3\tdef load(path):" in err.hint
    assert err.hint in str(err)


def test_not_found_with_nothing_close_says_to_read_again() -> None:
    err = _error("alpha\nbeta\n", "zzzz qqqq", "x")
    assert err.hint is None
    assert "Read the file again" in str(err)


def test_no_hint_for_a_blank_quote_or_a_huge_file() -> None:
    assert closest_region("a\nb\n", "\n\n") is None
    assert closest_region("x\n" * (HINT_MAX_FILE_LINES + 1), "y") is None


def test_a_hint_needs_more_than_a_passing_resemblance() -> None:
    # One similar line among many dissimilar ones scores below the floor.
    content = "def load(path):\n" + "\n".join(f"v{i} = {i}" for i in range(20))
    old = "def load(path):\n" + "\n".join(f"other_{i}()" for i in range(12))
    assert closest_region(content, old) is None


# ---------------------------------------------------- real-world misses --


@pytest.mark.parametrize(
    ("content", "old", "new", "expected", "strategy"),
    [
        pytest.param(
            "x = 1   \ny = 2\t\n",
            "x = 1\ny = 2\n",
            "x = 9\ny = 2\n",
            "x = 9\ny = 2\n",
            "line_trimmed",
            id="trailing-whitespace",
        ),
        pytest.param(
            "def f():\n\tif a:\n\t\treturn 1\n",
            "    if a:\n        return 1",
            "    if a:\n        return 2",
            "def f():\n\tif a:\n\t\treturn 2\n",
            "line_trimmed",
            id="tabs-vs-spaces",
        ),
        pytest.param(
            "a = 1\r\nb = 2\r\nc = 3\r\n",
            "a = 1\nb = 2\n",
            "a = 1\nb = 20\n",
            "a = 1\r\nb = 20\r\nc = 3\r\n",
            "exact",
            id="crlf",
        ),
        pytest.param(
            "def f():\r\n    return 1\r\n",
            "def f():\n  return 1",
            "def f():\n  return 2",
            "def f():\r\n    return 2\r\n",
            "line_trimmed",
            id="crlf-and-indent",
        ),
        pytest.param(
            'log.info("started")\n',
            'log.info(\\"started\\")',
            'log.info(\\"running\\")',
            'log.info("running")\n',
            "escape_normalized",
            id="escaped-quotes",
        ),
        pytest.param(
            "items = [1,2,3]\n",
            "items = [1, 2, 3]",
            "items = [1, 2, 3, 4]",
            "items = [1, 2, 3, 4]\n",
            "whitespace_normalized",
            id="spacing-after-commas",
        ),
    ],
)
def test_real_world_near_misses(
    content: str, old: str, new: str, expected: str, strategy: str
) -> None:
    out = apply_edit(content, old, new)
    assert out.content == expected
    assert out.strategy == strategy


def test_a_crlf_edit_says_it_kept_the_line_endings() -> None:
    assert apply_edit("a\r\nb\r\n", "a\nb", "c\nd").crlf


def test_one_stray_crlf_does_not_make_a_file_crlf() -> None:
    out = apply_edit("a\nb\nc\r\nd\n", "a\nb\n", "x\n")
    assert out.content == "x\nc\r\nd\n"
    assert not out.crlf


# --------------------------------------------------- the backends' wrapper --


def test_replace_in_file_keeps_the_backend_messages() -> None:
    with pytest.raises(ValueError, match="matches 2 times in /f"):
        replace_in_file("/f", "abc abc", "abc", "x")
    with pytest.raises(ValueError, match="old_str not found in /f"):
        replace_in_file("/f", "abc", "zzz qqq", "x")
    with pytest.raises(ValueError, match=r"empty.*\(/f\)"):
        replace_in_file("/f", "abc", "", "x")


def test_replace_in_file_forgives_indentation() -> None:
    assert replace_in_file("/f", "if a:\n    b()\n", "if a:\n  b()", "if a:\n  c()") == (
        "if a:\n    c()\n"
    )


def test_replace_in_file_still_accepts_a_no_op_edit() -> None:
    assert replace_in_file("/f", "abc", "abc", "abc") == "abc"


def test_replace_in_file_carries_the_hint() -> None:
    with pytest.raises(ValueError, match="closest match is lines 1-1"):
        replace_in_file("/f", "value = compute(1)\n", "value = compute(2)", "x")
