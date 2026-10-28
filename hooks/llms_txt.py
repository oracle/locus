# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Publish the site as Markdown for people and tools that read Markdown.

Three outputs, all derived from the same page source MkDocs renders:

* ``<page-url>index.md`` — a Markdown twin of every page, which the
  "Copy page" button (docs/javascripts/copy-page.js) fetches and copies.
* ``/llms.txt`` — the index the llms.txt convention asks for: one line per
  page with its title, link and description, grouped by nav section.
* ``/llms-full.txt`` — every guide page in nav order in one file, for tools
  that want the whole site in one request. API reference pages are listed in
  ``llms.txt`` but not inlined: their content is generated from docstrings at
  render time, so the HTML page is the useful form.

The twins are written after the build_metadata hook has expanded the build
facts, so they carry the same SDK version and provider tables the HTML does.
Snippet includes (``--8<--``) are expanded here because the Markdown
extension that handles them runs later, inside the HTML renderer. An inlined
diagram becomes a one-line description taken from the SVG's title. A facts
row (``<dl class="locus-facts">``) becomes a bulleted list of label/value
pairs, and the examples filter box, which only works with the page's script,
is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


#: ``--8<-- "path"``, ``"path:start"``, ``"path:start:end"``, ``"path::end"`` —
#: the forms pymdownx.snippets accepts and this site uses (1-based, inclusive).
_SNIPPET = re.compile(
    r'^(?P<indent>[ \t]*)--8<--\s+"(?P<path>[^":]+)(?::(?P<start>-?\d*)(?::(?P<end>-?\d*))?)?"[ \t]*$',
    re.MULTILINE,
)
_FIGURE = re.compile(r'<figure class="tl-dia">.*?</figure>', re.DOTALL)
_FACTS = re.compile(r'<dl class="locus-facts"[^>]*>(?P<body>.*?)</dl>', re.DOTALL)
_FACT = re.compile(r"<dt[^>]*>(?P<label>.*?)</dt>\s*<dd[^>]*>(?P<value>.*?)</dd>", re.DOTALL)
_FILTER = re.compile(r'\n*<div class="notebook-filter">.*?</div>\n*', re.DOTALL)
_SVG_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL)
_MKDOCSTRINGS = re.compile(r"^:::\s+(\S+).*(?:\n[ \t]+\S.*)*", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Collected in on_page_markdown, written in on_post_build.
_pages: dict[str, dict[str, Any]] = {}


def _snippet_roots(config: Any) -> list[Path]:
    root = Path(config["config_file_path"]).parent
    roots = [root]
    for ext in config.get("markdown_extensions", []):
        if isinstance(ext, dict) and "pymdownx.snippets" in ext:
            for base in ext["pymdownx.snippets"].get("base_path", []):
                roots.append(root / base)
    return roots


def _expand_snippets(markdown: str, roots: list[Path]) -> str:
    def expand(match: re.Match[str]) -> str:
        rel = match.group("path")
        path = next((r / rel for r in roots if (r / rel).is_file()), None)
        if path is None:
            return match.group(0)
        lines = path.read_text().splitlines()
        start = int(match.group("start")) if match.group("start") else 1
        end = int(match.group("end")) if match.group("end") else len(lines)
        indent = match.group("indent")
        return "\n".join(indent + line for line in lines[max(start, 1) - 1 : end])

    return _SNIPPET.sub(expand, markdown)


def _plain_markdown(markdown: str, roots: list[Path]) -> str:
    def figure(match: re.Match[str]) -> str:
        title = _SVG_TITLE.search(match.group(0))
        return f"*Diagram: {title.group(1).strip()}*" if title else "*Diagram*"

    def facts(match: re.Match[str]) -> str:
        return "\n".join(
            f"- **{m['label'].strip()}:** {m['value'].strip()}"
            for m in _FACT.finditer(match["body"])
        )

    text = _FIGURE.sub(figure, markdown)
    text = _FACTS.sub(facts, text)
    text = _FILTER.sub("\n\n", text)
    text = _HTML_COMMENT.sub("", text)
    return _expand_snippets(text, roots).strip() + "\n"


def on_page_markdown(markdown: str, *, page: Any, config: Any, **_: Any) -> str:
    _pages[page.file.src_uri] = {
        "url": page.url,
        "title": page.title,
        "description": (page.meta or {}).get("description", ""),
        "markdown": markdown,
        "is_api": bool(_MKDOCSTRINGS.search(markdown)),
    }
    return markdown


def _walk_nav(nav: Any, section: str | None = None) -> list[tuple[str, str, str]]:
    """``(top-level section, nav title, src_uri)`` for every page in nav order."""
    out: list[tuple[str, str, str]] = []
    for item in nav:
        if isinstance(item, str):
            out.append((section or "", "", item))
            continue
        for title, value in item.items():
            if isinstance(value, str):
                out.append((section or title, title, value))
            else:
                out.extend(_walk_nav(value, section or title))
    return out


def on_post_build(config: Any) -> None:
    site = Path(config["site_dir"])
    site_url = (config.get("site_url") or "/").rstrip("/")
    roots = _snippet_roots(config)
    nav_pages = _walk_nav(config["nav"] or [])
    # The home page is reached from the header, not the nav.
    if "index.md" in _pages and not any(src == "index.md" for _, _, src in nav_pages):
        nav_pages.insert(0, ("", "Home", "index.md"))

    api_note = (
        "This page is the API reference, generated from the package's docstrings "
        "at build time. Read it as HTML at the URL above."
    )
    for info in _pages.values():
        body = _plain_markdown(info["markdown"], roots)
        if info["is_api"]:
            body = _MKDOCSTRINGS.sub("", body).strip() + f"\n\n{api_note}\n"
        page_url = f"{site_url}/{info['url']}"
        # Keep the page's own H1 when it has one; add one only for pages that
        # take their title from the front matter or the nav.
        first, _, rest = body.partition("\n")
        if first.startswith("# "):
            heading, body = first, rest.lstrip("\n")
        else:
            heading = f"# {info['title']}"
        head = f"{heading}\n\n"
        if info["description"]:
            head += f"> {info['description']}\n\n"
        head += f"Source: {page_url}\n\n"
        out = site / info["url"] / "index.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(head + body)

    lines = [
        "# Locus",
        "",
        f"> {config.get('site_description', '').strip()}",
        "",
        "Locus is a Python SDK. Each link below is the Markdown twin of a page on "
        f"{site_url}/; drop `index.md` from a link for the HTML page. "
        "`llms-full.txt` holds every guide page in one file.",
        "",
    ]
    full: list[str] = []
    current = None
    for section, title, src in nav_pages:
        info = _pages.get(src)
        if info is None:
            continue
        if section != current:
            current = section
            lines += ["", f"## {section or 'Home'}", ""]
        link = f"{site_url}/{info['url']}index.md"
        desc = f": {info['description']}" if info["description"] else ""
        lines.append(f"- [{title or info['title']}]({link}){desc}")
        if not info["is_api"]:
            full.append((site / info["url"] / "index.md").read_text())
    lines += [
        "",
        "## Optional",
        "",
        f"- [llms-full.txt]({site_url}/llms-full.txt): every guide page in one file",
        "",
    ]
    (site / "llms.txt").write_text("\n".join(lines))
    (site / "llms-full.txt").write_text("\n\n---\n\n".join(full))
