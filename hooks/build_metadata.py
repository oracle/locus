# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Build documentation facts from the SDK used for this exact build.

The deploy workflow installs a tagged SDK checkout before MkDocs runs. Reading
the imported SDK and provider registry here means an older docs build
cannot accidentally display a newer PyPI release, and provider tables cannot
drift from the routes the SDK actually registers.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from importlib import import_module
from pathlib import Path
from typing import Any


SDK_VERSION_TOKEN = "{{ locus_sdk_version }}"  # noqa: S105 — template placeholder, not a secret
PROVIDER_TABLE_TOKEN = "{{ locus_provider_table }}"  # noqa: S105 — template placeholder, not a secret
EXAMPLE_MODEL_TOKEN = "{{ locus_example_model }}"  # noqa: S105 — template placeholder, not a secret
EVENT_COUNT_TOKEN = "{{ locus_event_count }}"  # noqa: S105 — template placeholder, not a secret
EVENT_PREFIXES_TOKEN = "{{ locus_event_prefixes }}"  # noqa: S105 — template placeholder, not a secret
MODEL_TABS_TOKEN = "{{ locus_model_tabs }}"  # noqa: S105 — template placeholder, not a secret
MODEL_TOKEN = "{{ locus_model }}"  # noqa: S105 — template placeholder, not a secret

_SDK_REPO = "https://github.com/oracle/locus/"
#: A link into the SDK repo on its moving ``main`` branch.
_SDK_MAIN_LINK = re.compile(r"https://github\.com/oracle/locus/(blob|tree)/main/([^)\s\"'#>]+)")
#: Only a plain release has a tag to pin to; a dev or local build does not.
_RELEASE = re.compile(r"^\d+\.\d+\.\d+$")

#: ``{{ locus_diagram agent-loop }}`` inlines docs/diagrams/agent-loop.svg.
_DIAGRAM_TOKEN = re.compile(r"\{\{\s*locus_diagram\s+([a-z0-9][a-z0-9-]*)\s*\}\}")
#: ``{{ locus_glyph pipelines }}`` inlines docs/diagrams/cards/pipelines.svg, a
#: card illustration drawn in the diagram vocabulary.
_GLYPH_TOKEN = re.compile(r"\{\{\s*locus_glyph\s+([a-z0-9][a-z0-9-]*)\s*\}\}")
_XML_PROLOG = re.compile(r"<\?xml[^>]*\?>\s*")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_ID_ATTR = re.compile(r'\bid="([^"]+)"')
#: ``{{ locus_source examples/x.py::build_graph }}`` quotes one function or class
#: from an example file, found by name, so the excerpt follows the code when
#: lines move. A dotted name (``Class.method``) reaches into a class.
_SOURCE_TOKEN = re.compile(
    r"^\{\{\s*locus_source\s+(?P<path>[^\s:]+)::(?P<symbol>[A-Za-z_][\w.]*)\s*\}\}[ \t]*$",
    re.MULTILINE,
)
#: ``{{ locus_model_tabs }}`` directly above a fenced block that contains
#: ``{{ locus_model }}`` renders the block once per provider as a tab set.
_MODEL_TABS = re.compile(
    r"^\{\{\s*locus_model_tabs\s*\}\}[ \t]*\n\n?(```[A-Za-z0-9_-]*\n.*?\n```)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _sdk_version() -> str:
    try:
        sdk = import_module("locus")
    except ModuleNotFoundError as exc:  # pragma: no cover - build misconfiguration
        raise RuntimeError(
            "Install the Locus SDK used for this documentation build before running MkDocs"
        ) from exc
    return str(sdk.__version__)


def _provider_table() -> str:
    providers = import_module("locus.models.providers")
    return str(providers.provider_table())


def _events() -> list[str]:
    """Every canonical event name the SDK declares, from its ``EV_*`` constants."""
    emit = import_module("locus.observability.emit")
    return sorted(
        value
        for name, value in vars(emit).items()
        if name.startswith("EV_") and isinstance(value, str)
    )


def _event_count(events: list[str]) -> str:
    """Rounded down to the ten, so "60+" stays true as events are added."""
    return f"{len(events) // 10 * 10}+"


def _event_prefixes(events: list[str]) -> str:
    return ", ".join(f"`{prefix}.*`" for prefix in sorted({e.split(".")[0] for e in events}))


def _sdk_checkout(config: Any) -> Path | None:
    """The SDK repository this build installed, when one is on disk.

    The docs live in the SDK repository, so the checkout is the repository
    root (``mkdocs.yml``'s directory). ``LOCUS_SDK_DIR`` still overrides it,
    and a ``.sdk`` sub-checkout is honoured for builds against a pinned tag.
    """
    candidates = []
    if os.environ.get("LOCUS_SDK_DIR"):
        candidates.append(Path(os.environ["LOCUS_SDK_DIR"]))
    if config.get("config_file_path"):
        root = Path(config["config_file_path"]).parent
        candidates.append(root / ".sdk")
        if (root / "pyproject.toml").is_file():
            candidates.append(root)
    return next((c for c in candidates if c.is_dir()), None)


def pin_sdk_links(markdown: str, version: str, checkout: Path | None) -> str:
    """Point SDK source links at the release this site documents.

    A ``blob/main`` link shows whatever the SDK's main branch holds today, which
    drifts from the documented release, and its ``#L`` line anchors break the
    first time the file changes. A tag never moves. A path that does not exist
    at the release (added to main after it) is left on main rather than turned
    into a 404.
    """
    if not _RELEASE.match(version):
        return markdown

    def pin(match: re.Match[str]) -> str:
        kind, path = match.group(1), match.group(2)
        if checkout is not None and not (checkout / path).exists():
            return match.group(0)
        return f"{_SDK_REPO}{kind}/v{version}/{path}"

    return _SDK_MAIN_LINK.sub(pin, markdown)


def render_diagram(name: str, docs_dir: Path) -> str:
    """Inline one diagram so it themes with the page.

    Diagrams live as SVG under docs/diagrams and style themselves with the
    ``.tl-dia`` classes in locus.css, which read the --tl-* tokens. Inlined,
    they follow the light/dark toggle; as an <img> they could not see it.
    Every id is prefixed with the diagram name so two diagrams on one page
    cannot collide on a marker or a <title>.
    """
    path = docs_dir / "diagrams" / f"{name}.svg"
    if not path.is_file():
        raise RuntimeError(f"{{{{ locus_diagram {name} }}}} names {path}, which does not exist")
    svg = _namespace_ids(_COMMENT.sub("", _XML_PROLOG.sub("", path.read_text())), name)
    # Blank lines would end the raw-HTML block early and hand the rest to Markdown.
    body = "\n".join(line for line in svg.strip().splitlines() if line.strip())
    return f'\n<figure class="tl-dia">\n{body}\n</figure>\n'


def _namespace_ids(svg: str, prefix: str) -> str:
    """Prefix every id in an inlined SVG, and every reference to it."""
    ids = set(_ID_ATTR.findall(svg))
    for old in sorted(ids, key=len, reverse=True):
        new = f"{prefix}--{old}"
        svg = re.sub(rf'\bid="{re.escape(old)}"', f'id="{new}"', svg)
        svg = svg.replace(f"url(#{old})", f"url(#{new})").replace(
            f'href="#{old}"', f'href="#{new}"'
        )
        svg = re.sub(
            r'(aria-(?:labelledby|describedby)="[^"]*)\b' + re.escape(old) + r"\b",
            lambda m: m.group(1) + new,
            svg,
        )
    return svg


def render_glyph(name: str, docs_dir: Path) -> str:
    """Inline one card illustration as a single line of markup.

    Card glyphs live under docs/diagrams/cards and use the diagram classes,
    scoped by ``.tl-glyph`` in locus.css, so they theme with the page like the
    diagrams do. They are decorative: the card title beside them carries the
    meaning, so the span is hidden from assistive technology. The markup is
    one line because the span sits inside a list item's paragraph, where a
    line break would start a new block.
    """
    path = docs_dir / "diagrams" / "cards" / f"{name}.svg"
    if not path.is_file():
        raise RuntimeError(f"{{{{ locus_glyph {name} }}}} names {path}, which does not exist")
    svg = _namespace_ids(_COMMENT.sub("", _XML_PROLOG.sub("", path.read_text())), f"card-{name}")
    one_line = " ".join(part.strip() for part in svg.strip().splitlines() if part.strip())
    return f'<span class="tl-glyph" aria-hidden="true">{one_line}</span>'


def _find_symbol(tree: ast.Module, dotted: str) -> ast.AST | None:
    node: ast.AST = tree
    for name in dotted.split("."):
        body = getattr(node, "body", None)
        if body is None:
            return None
        for child in body:
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name == name
            ):
                node = child
                break
        else:
            return None
    return node


def render_source(path: str, symbol: str, roots: list[Path]) -> str:
    """One function or class from a source file, as a fenced Python block.

    The symbol is looked up by name in the parsed module, so a page keeps
    quoting the right code when the file is edited above it; a symbol that no
    longer exists fails the strict build instead of silently quoting the
    wrong lines. Decorators are included, common indentation is removed.
    """
    file = next((r / path for r in roots if (r / path).is_file()), None)
    if file is None:
        raise RuntimeError(
            f"{{{{ locus_source {path}::{symbol} }}}} names {path}, which does not exist"
        )
    text = file.read_text()
    node = _find_symbol(ast.parse(text), symbol)
    if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        raise RuntimeError(
            f"{{{{ locus_source {path}::{symbol} }}}}: no function or class named {symbol} in {path}"
        )
    first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
    lines = text.splitlines()[first - 1 : node.end_lineno]
    body = textwrap.dedent("\n".join(lines)).rstrip()
    return f"```python\n{body}\n```"


def render_model_tabs(block: str, tabs: list[dict[str, str]]) -> str:
    """One code block, shown once per provider as a linked tab set.

    The page author writes the example a single time with ``{{ locus_model }}``
    where the model string goes; ``extra.model_tabs`` in mkdocs.yml says which
    providers to show and the one example id each. Material's
    ``content.tabs.link`` keeps the chosen provider in sync across the site, so
    a reader who picks OCI once reads every page with OCI model ids.
    """
    if MODEL_TOKEN not in block:
        raise RuntimeError(
            "A {{ locus_model_tabs }} block must contain {{ locus_model }} where the model string goes"
        )
    parts = []
    for tab in tabs:
        body = block.replace(MODEL_TOKEN, tab["model"])
        indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
        parts.append(f'=== "{tab["label"]}"\n\n{indented}\n')
    return "\n".join(parts)


def on_config(config: Any) -> Any:
    extra = dict(config.get("extra", {}))
    extra["sdk_version"] = _sdk_version()
    checkout = _sdk_checkout(config)
    extra["sdk_checkout"] = str(checkout) if checkout else None
    config["extra"] = extra
    return config


def on_page_markdown(markdown: str, *, config: Any, **_: Any) -> str:
    """Expand build facts only where their explicit tokens appear."""
    extra = config["extra"]
    sdk_version = extra["sdk_version"]
    rendered = markdown
    if "locus_source" in rendered:
        root = Path(config["config_file_path"]).parent
        roots = [root]
        rendered = _SOURCE_TOKEN.sub(
            lambda m: render_source(m.group("path"), m.group("symbol"), roots), rendered
        )
    if "locus_model_tabs" in rendered:
        tabs = extra.get("model_tabs")
        if not tabs:
            raise RuntimeError(
                "A page uses {{ locus_model_tabs }} but mkdocs.yml sets no extra.model_tabs"
            )
        rendered = _MODEL_TABS.sub(lambda m: render_model_tabs(m.group(1), tabs), rendered)
    rendered = rendered.replace(SDK_VERSION_TOKEN, sdk_version)
    if PROVIDER_TABLE_TOKEN in rendered:
        rendered = rendered.replace(PROVIDER_TABLE_TOKEN, _provider_table())
    if EXAMPLE_MODEL_TOKEN in rendered:
        if not extra.get("example_model"):
            raise RuntimeError(
                "A page uses {{ locus_example_model }} but mkdocs.yml sets no extra.example_model"
            )
        rendered = rendered.replace(EXAMPLE_MODEL_TOKEN, extra["example_model"])
    if EVENT_COUNT_TOKEN in rendered or EVENT_PREFIXES_TOKEN in rendered:
        events = _events()
        rendered = rendered.replace(EVENT_COUNT_TOKEN, _event_count(events))
        rendered = rendered.replace(EVENT_PREFIXES_TOKEN, _event_prefixes(events))
    if "locus_diagram" in rendered:
        docs_dir = Path(config["docs_dir"])
        rendered = _DIAGRAM_TOKEN.sub(lambda m: render_diagram(m.group(1), docs_dir), rendered)
    if "locus_glyph" in rendered:
        docs_dir = Path(config["docs_dir"])
        rendered = _GLYPH_TOKEN.sub(lambda m: render_glyph(m.group(1), docs_dir), rendered)
    checkout = extra.get("sdk_checkout")
    return pin_sdk_links(rendered, sdk_version, Path(checkout) if checkout else None)
