# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Unit: the inline-code hook adds <wbr> at separators in long code spans only."""

from __future__ import annotations

from hooks.inline_code_breaks import add_break_opportunities, on_page_content


def test_long_path_gets_breaks_after_separators() -> None:
    html = "<p>Run <code>examples/notebook_55_gsar_typed_grounding.py</code> first.</p>"
    out = add_break_opportunities(html)
    assert (
        "<code>examples/<wbr>notebook_<wbr>55_<wbr>gsar_<wbr>typed_<wbr>grounding.<wbr>py</code>"
        in out
    )


def test_short_code_is_untouched() -> None:
    html = "<p><code>agent.run</code>, <code>Agent(output_schema=...)</code> and <code>execute_handoff(...)</code></p>"
    assert add_break_opportunities(html) == html


def test_code_inside_pre_is_untouched() -> None:
    html = (
        '<div class="highlight"><pre><span></span><code><span class="n">'
        "very_long_identifier_name_in_a_code_block_stays_whole</span></code></pre></div>"
    )
    assert add_break_opportunities(html) == html
    mixed = html + "<p><code>oracleagentmemory.core.OracleAgentMemory</code></p>"
    out = add_break_opportunities(mixed)
    assert out.startswith(html)
    assert "oracleagentmemory.<wbr>core.<wbr>OracleAgentMemory" in out


def test_code_with_attributes_and_member_names() -> None:
    html = (
        '<h3 class="doc doc-heading"><code class="doc-symbol"></code> '
        '<span class="doc doc-object-name doc-function-name">point_unknown_fields_at_model_kwargs</span></h3>'
        '<summary>Source code in <code class="x">src/locus/memory/backends/oracle_versioned.py</code></summary>'
    )
    out = add_break_opportunities(html)
    assert "point_<wbr>unknown_<wbr>fields_<wbr>at_<wbr>model_<wbr>kwargs" in out
    assert "src/<wbr>locus/<wbr>memory/<wbr>backends/<wbr>oracle_<wbr>versioned.<wbr>py" in out


def test_no_break_at_trailing_separator_and_idempotent() -> None:
    html = "<p><code>a_very_long_identifier_ending_with_dot.</code></p>"
    out = add_break_opportunities(html)
    assert out.endswith("dot.</code></p>")
    assert add_break_opportunities(out) == out


def test_hook_entry_point() -> None:
    html = "<p><code>examples/notebook_77_incident_response.py</code></p>"
    assert "<wbr>" in on_page_content(html, page=None, config=None, files=None)
