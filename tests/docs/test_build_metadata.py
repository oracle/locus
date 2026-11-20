# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

from types import SimpleNamespace

import pytest
from hooks import build_metadata


def test_on_config_records_the_installed_sdk_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_metadata, "_sdk_version", lambda: "9.8.7")
    monkeypatch.delenv("LOCUS_SDK_DIR", raising=False)
    config = {"extra": {"generator": False}}

    assert build_metadata.on_config(config) is config
    assert config["extra"] == {"generator": False, "sdk_version": "9.8.7", "sdk_checkout": None}


def test_page_tokens_expand_from_the_build_and_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_metadata, "_provider_table", lambda: "| `openrouter` | OpenRouter |")
    markdown = "SDK {{ locus_sdk_version }}\n\n{{ locus_provider_table }}"

    rendered = build_metadata.on_page_markdown(
        markdown,
        config={"extra": {"sdk_version": "2.16.0"}},
    )

    assert rendered == "SDK 2.16.0\n\n| `openrouter` | OpenRouter |"


def test_page_without_provider_token_does_not_import_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> str:
        raise AssertionError("provider registry should be lazy")

    monkeypatch.setattr(build_metadata, "_provider_table", unexpected)
    rendered = build_metadata.on_page_markdown(
        "SDK {{ locus_sdk_version }}",
        config={"extra": {"sdk_version": "2.16.0"}},
    )

    assert rendered == "SDK 2.16.0"


def test_sdk_version_explains_a_missing_build_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> None:
        raise ModuleNotFoundError("locus")

    monkeypatch.setattr(build_metadata, "import_module", missing)
    with pytest.raises(RuntimeError, match="Install the Locus SDK"):
        build_metadata._sdk_version()


def test_sdk_version_comes_from_the_sdk_imported_by_mkdocstrings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(__version__="3.2.1")
    monkeypatch.setattr(build_metadata, "import_module", lambda _: module)

    assert build_metadata._sdk_version() == "3.2.1"


def test_provider_table_comes_from_the_sdk_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(provider_table=lambda: "generated table")
    monkeypatch.setattr(build_metadata, "import_module", lambda _: module)

    assert build_metadata._provider_table() == "generated table"


# ── rules that keep the site from going stale ────────────────────────────────
SDK_MAIN = "https://github.com/oracle/locus/blob/main/src/locus/tools/executor.py#L21"


def test_example_model_comes_from_one_setting() -> None:
    rendered = build_metadata.on_page_markdown(
        'Agent(model="{{ locus_example_model }}")',
        config={"extra": {"sdk_version": "dev", "example_model": "openai:some-model"}},
    )
    assert rendered == 'Agent(model="openai:some-model")'


def test_example_model_token_without_a_setting_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="extra.example_model"):
        build_metadata.on_page_markdown(
            "{{ locus_example_model }}", config={"extra": {"sdk_version": "dev"}}
        )


def test_event_facts_come_from_the_sdk_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    emit = SimpleNamespace(
        EV_A="agent.started", EV_B="agent.done", EV_C="tool.sandbox.started", NOT_AN_EVENT="x"
    )
    monkeypatch.setattr(build_metadata, "import_module", lambda _: emit)
    rendered = build_metadata.on_page_markdown(
        "{{ locus_event_count }} events across {{ locus_event_prefixes }}",
        config={"extra": {"sdk_version": "dev"}},
    )
    assert rendered == "0+ events across `agent.*`, `tool.*`"


def test_event_count_rounds_down_so_it_stays_true() -> None:
    assert build_metadata._event_count([f"a.{i}" for i in range(68)]) == "60+"


def test_page_without_event_tokens_does_not_import_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(_: str) -> None:
        raise AssertionError("event registry should be lazy")

    monkeypatch.setattr(build_metadata, "import_module", unexpected)
    assert (
        build_metadata.on_page_markdown("plain", config={"extra": {"sdk_version": "dev"}})
        == "plain"
    )


def test_sdk_links_pin_to_the_documented_release(tmp_path) -> None:
    (tmp_path / "src/locus/tools").mkdir(parents=True)
    (tmp_path / "src/locus/tools/executor.py").write_text("")
    pinned = build_metadata.pin_sdk_links(f"[x]({SDK_MAIN})", "2.16.0", tmp_path)
    assert pinned == (
        "[x](https://github.com/oracle/locus/blob/v2.16.0/src/locus/tools/executor.py#L21)"
    )


def test_a_file_missing_at_the_release_stays_on_main(tmp_path) -> None:
    """A page may link a file added to main after the release; pinning it would 404."""
    assert (
        build_metadata.pin_sdk_links(f"[x]({SDK_MAIN})", "2.16.0", tmp_path) == f"[x]({SDK_MAIN})"
    )


def test_links_pin_without_a_checkout_to_check_against() -> None:
    assert "/blob/v2.16.0/" in build_metadata.pin_sdk_links(SDK_MAIN, "2.16.0", None)


def test_a_dev_build_has_no_tag_so_links_stay_on_main() -> None:
    assert build_metadata.pin_sdk_links(SDK_MAIN, "2.17.0.dev3", None) == SDK_MAIN


def test_tree_links_pin_too() -> None:
    link = "https://github.com/oracle/locus/tree/main/examples"
    assert build_metadata.pin_sdk_links(link, "2.16.0", None).endswith("/tree/v2.16.0/examples")


def test_checkout_is_found_from_the_env_then_the_deploy_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCUS_SDK_DIR", str(tmp_path))
    assert build_metadata._sdk_checkout({}) == tmp_path

    monkeypatch.delenv("LOCUS_SDK_DIR")
    (tmp_path / ".sdk").mkdir()
    config = {"config_file_path": str(tmp_path / "mkdocs.yml")}
    assert build_metadata._sdk_checkout(config) == tmp_path / ".sdk"


def test_no_checkout_anywhere_is_none(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LOCUS_SDK_DIR", raising=False)
    assert build_metadata._sdk_checkout({"config_file_path": str(tmp_path / "mkdocs.yml")}) is None


def test_on_config_records_where_the_sdk_checkout_is(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(build_metadata, "_sdk_version", lambda: "2.16.0")
    monkeypatch.setenv("LOCUS_SDK_DIR", str(tmp_path))
    assert build_metadata.on_config({})["extra"]["sdk_checkout"] == str(tmp_path)


def test_page_links_are_pinned_during_the_build(tmp_path) -> None:
    (tmp_path / "src/locus/tools").mkdir(parents=True)
    (tmp_path / "src/locus/tools/executor.py").write_text("")
    rendered = build_metadata.on_page_markdown(
        f"[x]({SDK_MAIN})",
        config={"extra": {"sdk_version": "2.16.0", "sdk_checkout": str(tmp_path)}},
    )
    assert "/blob/v2.16.0/" in rendered


# ── diagrams ─────────────────────────────────────────────────────────────────
SVG = """<?xml version="1.0" encoding="UTF-8"?>
<!-- authoring note -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" role="img" aria-labelledby="t d">
  <title id="t">Loop</title>

  <desc id="d">Think then act</desc>
  <defs><marker id="ah"><path d="M0,0"/></marker></defs>
  <path class="edge" marker-end="url(#ah)" d="M0,0"/>
  <use href="#ah"/>
</svg>
"""


def _diagram_dir(tmp_path, name="loop", body=SVG):
    (tmp_path / "diagrams").mkdir(exist_ok=True)
    (tmp_path / "diagrams" / f"{name}.svg").write_text(body)
    return tmp_path


def test_a_diagram_is_inlined_as_a_themed_figure(tmp_path) -> None:
    out = build_metadata.render_diagram("loop", _diagram_dir(tmp_path))
    assert out.strip().startswith('<figure class="tl-dia">')
    assert "<?xml" not in out and "authoring note" not in out
    assert "\n\n" not in out.strip(), "a blank line would end the raw HTML block"


def test_diagram_ids_are_namespaced_with_every_reference(tmp_path) -> None:
    out = build_metadata.render_diagram("loop", _diagram_dir(tmp_path))
    assert 'id="loop--t"' in out and 'id="loop--ah"' in out
    assert "url(#loop--ah)" in out and 'href="#loop--ah"' in out
    assert 'aria-labelledby="loop--t loop--d"' in out


def test_two_diagrams_on_one_page_do_not_share_ids(tmp_path) -> None:
    docs = _diagram_dir(tmp_path)
    _diagram_dir(tmp_path, "gate")
    rendered = build_metadata.on_page_markdown(
        "{{ locus_diagram loop }}\n\n{{ locus_diagram gate }}",
        config={"extra": {"sdk_version": "dev"}, "docs_dir": str(docs)},
    )
    assert 'id="loop--ah"' in rendered and 'id="gate--ah"' in rendered


def test_a_missing_diagram_fails_the_build(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        build_metadata.render_diagram("nope", tmp_path)


def test_pages_without_diagrams_never_touch_the_disk(monkeypatch) -> None:
    def unexpected(*_a, **_k):
        raise AssertionError("no diagram token, no file read")

    monkeypatch.setattr(build_metadata, "render_diagram", unexpected)
    assert (
        build_metadata.on_page_markdown("text", config={"extra": {"sdk_version": "dev"}}) == "text"
    )


def test_model_tabs_render_the_block_once_per_provider() -> None:
    markdown = (
        "Intro.\n\n{{ locus_model_tabs }}\n\n```python\n"
        'agent = Agent(model="{{ locus_model }}")\n\nprint(agent)\n```\n\nAfter.\n'
    )
    tabs = [{"label": "OCI", "model": "oci:a.b"}, {"label": "OpenAI", "model": "openai:c"}]

    rendered = build_metadata.on_page_markdown(
        markdown, config={"extra": {"sdk_version": "1.0.0", "model_tabs": tabs}}
    )

    assert rendered == (
        "Intro.\n\n"
        '=== "OCI"\n\n    ```python\n    agent = Agent(model="oci:a.b")\n\n    print(agent)\n    ```\n\n'
        '=== "OpenAI"\n\n    ```python\n    agent = Agent(model="openai:c")\n\n    print(agent)\n    ```\n'
        "\n\nAfter.\n"
    )


def test_model_tabs_block_without_the_model_token_is_an_error() -> None:
    markdown = "{{ locus_model_tabs }}\n```python\nx = 1\n```\n"
    with pytest.raises(RuntimeError, match="locus_model"):
        build_metadata.on_page_markdown(
            markdown,
            config={
                "extra": {"sdk_version": "1.0.0", "model_tabs": [{"label": "A", "model": "a:b"}]}
            },
        )


def test_model_tabs_without_config_is_an_error() -> None:
    markdown = '{{ locus_model_tabs }}\n```python\nm = "{{ locus_model }}"\n```\n'
    with pytest.raises(RuntimeError, match="extra.model_tabs"):
        build_metadata.on_page_markdown(markdown, config={"extra": {"sdk_version": "1.0.0"}})


def test_source_macro_quotes_a_symbol_by_name(tmp_path) -> None:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "demo.py").write_text(
        'import os\n\n\n@tool\ndef lookup(order_id: str) -> str:\n    """Look up."""\n    return order_id\n\n\n'
        "class Flow:\n    def build(self):\n        return 1\n\n    def run(self):\n        return 2\n"
    )
    roots = [tmp_path]
    assert build_metadata.render_source("examples/demo.py", "lookup", roots) == (
        '```python\n@tool\ndef lookup(order_id: str) -> str:\n    """Look up."""\n    return order_id\n```'
    )
    assert build_metadata.render_source("examples/demo.py", "Flow.run", roots) == (
        "```python\ndef run(self):\n    return 2\n```"
    )
    with pytest.raises(RuntimeError, match="no function or class named missing"):
        build_metadata.render_source("examples/demo.py", "missing", roots)
    with pytest.raises(RuntimeError, match="does not exist"):
        build_metadata.render_source("examples/nope.py", "lookup", roots)


def test_source_macro_expands_on_a_page(tmp_path) -> None:
    (tmp_path / "mkdocs.yml").write_text("site_name: x\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "demo.py").write_text("def main():\n    return 1\n")
    rendered = build_metadata.on_page_markdown(
        "Before.\n\n{{ locus_source examples/demo.py::main }}\n\nAfter.\n",
        config={
            "extra": {"sdk_version": "1.0.0"},
            "config_file_path": str(tmp_path / "mkdocs.yml"),
        },
    )
    assert rendered == "Before.\n\n```python\ndef main():\n    return 1\n```\n\nAfter.\n"


_GLYPH = (
    '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 56">\n'
    "<title>Three stages</title>\n<!-- authoring note -->\n"
    '<defs><marker id="ah"><path class="arrowhead" d="M0,0 L6,3 L0,6 z"/></marker></defs>\n'
    '<g class="node"><rect x="6" y="21" width="22" height="14" rx="3"/></g>\n'
    '<path class="edge" marker-end="url(#ah)" d="M28,28 L35,28"/>\n</svg>\n'
)


def _glyph_dir(tmp_path, name: str = "pipe"):
    cards = tmp_path / "diagrams" / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    (cards / f"{name}.svg").write_text(_GLYPH)
    return tmp_path


def test_a_card_glyph_is_inlined_on_one_line_and_hidden_from_assistive_tech(tmp_path) -> None:
    out = build_metadata.render_glyph("pipe", _glyph_dir(tmp_path))
    assert out.startswith('<span class="tl-glyph" aria-hidden="true"><svg')
    assert out.endswith("</svg></span>")
    assert "\n" not in out, "the span sits inside a list item's paragraph"
    assert "<?xml" not in out and "authoring note" not in out


def test_card_glyph_ids_are_namespaced(tmp_path) -> None:
    out = build_metadata.render_glyph("pipe", _glyph_dir(tmp_path))
    assert 'id="card-pipe--ah"' in out and "url(#card-pipe--ah)" in out


def test_a_card_glyph_token_expands_inside_a_list_item(tmp_path) -> None:
    docs = _glyph_dir(tmp_path)
    rendered = build_metadata.on_page_markdown(
        "- {{ locus_glyph pipe }} **[Pipelines](x.md)**\n",
        config={"extra": {"sdk_version": "dev"}, "docs_dir": str(docs)},
    )
    assert rendered.startswith('- <span class="tl-glyph" aria-hidden="true"><svg')
    assert rendered.rstrip().endswith("</svg></span> **[Pipelines](x.md)**")


def test_a_missing_card_glyph_fails_the_build(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        build_metadata.render_glyph("nope", tmp_path)


def test_cli_reference_lists_every_command_and_action_from_the_parser() -> None:
    import argparse  # noqa: PLC0415

    from locus.cli import build_parser  # noqa: PLC0415

    rendered = build_metadata.on_page_markdown(
        "{{ locus_cli_reference }}", config={"extra": {"sdk_version": "0.0.0"}}
    )
    parser = build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert len(commands.choices) >= 11
    for name, sub in commands.choices.items():
        assert f"| `locus {name}` |" in rendered
        nested = next((a for a in sub._actions if isinstance(a, argparse._SubParsersAction)), None)
        for action in nested.choices if nested is not None else ():
            assert f"`{action}`:" in rendered, (name, action)


def test_the_command_line_page_carries_the_token_and_is_in_the_nav() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    page = (root / "docs" / "how-to" / "command-line.md").read_text(encoding="utf-8")
    assert "{{ locus_cli_reference }}" in page
    assert "how-to/command-line.md" in (root / "mkdocs.yml").read_text(encoding="utf-8")
