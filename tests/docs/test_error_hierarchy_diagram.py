# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The error-hierarchy diagram shows every ``LocusError`` subclass with its ``kind``.

``docs/concepts/errors.md`` embeds ``docs/diagrams/error-hierarchy.svg`` as
the hierarchy and tells readers to key dashboards on ``kind``. The diagram
writes each class name and, under it, its ``kind``, as consecutive text
nodes. This test reads those pairs and compares them with the classes
``locus.core.errors`` defines; a subclass defined in another module has to be
named in the page's prose instead.
"""

from __future__ import annotations

import inspect
import itertools
import re
from pathlib import Path

import locus.core.errors as errors_module
from locus.core.errors import LocusError


ROOT = Path(__file__).resolve().parents[2]
DIAGRAM = ROOT / "docs" / "diagrams" / "error-hierarchy.svg"
PAGE = ROOT / "docs" / "concepts" / "errors.md"
TEXT_NODE = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
TAGS = re.compile(r"<[^>]+>")
CLASS_NAME = re.compile(r"^[A-Z]\w*Error$")


def diagram_pairs() -> dict[str, str]:
    """Class name to the ``kind`` written under it."""
    texts = [
        TAGS.sub("", t).strip() for t in TEXT_NODE.findall(DIAGRAM.read_text(encoding="utf-8"))
    ]
    pairs: dict[str, str] = {}
    for name, kind in itertools.pairwise(texts):
        if CLASS_NAME.match(name) and re.fullmatch(r"[a-z_]+", kind):
            pairs[name] = kind
    return pairs


def _subclasses(cls: type[LocusError]) -> set[type[LocusError]]:
    found: set[type[LocusError]] = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _subclasses(sub)
    return found


def code_pairs() -> tuple[dict[str, str], dict[str, str]]:
    """(classes in ``locus.core.errors``, classes defined elsewhere), each name to ``kind``."""
    inside: dict[str, str] = {LocusError.__name__: LocusError.kind}
    elsewhere: dict[str, str] = {}
    for cls in _subclasses(LocusError):
        target = inside if inspect.getmodule(cls) is errors_module else elsewhere
        target[cls.__name__] = cls.kind
    return inside, elsewhere


def test_the_diagram_names_every_class_in_the_module_with_its_kind() -> None:
    inside, _ = code_pairs()
    drawn = diagram_pairs()
    assert len(drawn) > 10
    assert drawn == inside, {
        "in code, not drawn": sorted(set(inside) - set(drawn)),
        "drawn, not in code": sorted(set(drawn) - set(inside)),
        "kind differs": sorted(n for n in set(drawn) & set(inside) if drawn[n] != inside[n]),
    }


def test_subclasses_defined_elsewhere_are_named_on_the_page() -> None:
    # Import the modules that define LocusError subclasses outside core/errors.
    import locus.tools.integrity  # noqa: F401, PLC0415

    _, elsewhere = code_pairs()
    page = PAGE.read_text(encoding="utf-8")
    assert "ToolCallIntegrityError" in elsewhere
    missing = sorted(name for name in elsewhere if name not in page)
    assert missing == []


def test_every_kind_is_unique_among_classes_that_declare_their_own() -> None:
    """``kind`` is the stable key dashboards use; two classes sharing one by accident would merge their metrics."""
    inside, _ = code_pairs()
    own = {
        name: kind for name, kind in inside.items() if "kind" in vars(getattr(errors_module, name))
    }
    kinds = list(own.values())
    assert len(kinds) == len(set(kinds)), sorted(k for k in kinds if kinds.count(k) > 1)
