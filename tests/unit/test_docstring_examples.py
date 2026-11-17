# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The ``Example:`` blocks on the store and registry docstrings run as written.

The API reference renders these docstrings verbatim, so an example that has
drifted from the code is a reader's first failed copy-paste. Each block is
pulled out of the docstring and executed. The registry examples run against a
copy of the registry, so the names they register never reach another test.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import textwrap
from typing import Any

import pytest

from locus.memory import registry as checkpointer_registry
from locus.memory.store import BaseStore, InMemoryStore
from locus.models import registry as model_registry


_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def example_code(obj: Any) -> str:
    """The Python block under the object's ``Example:`` section."""
    doc = inspect.getdoc(obj) or ""
    _, marker, tail = doc.partition("Example:")
    assert marker, f"{obj.__qualname__} has no Example: section"
    match = _FENCE.search(tail)
    assert match, f"{obj.__qualname__}'s Example: has no python block"
    return textwrap.dedent(match.group(1))


def run_example(obj: Any) -> None:
    """Execute the block as the body of a coroutine, so its ``await`` lines run."""
    body = textwrap.indent(example_code(obj), "    ")
    source = f"async def _example() -> None:\n{body}\n"
    namespace: dict[str, Any] = {}
    exec(compile(source, f"<{obj.__qualname__} example>", "exec"), namespace)  # noqa: S102
    asyncio.run(namespace["_example"]())


@pytest.fixture
def isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copies of both registries, restored afterwards."""
    monkeypatch.setattr(model_registry, "_PROVIDERS", dict(model_registry._PROVIDERS))
    monkeypatch.setattr(
        checkpointer_registry, "_CHECKPOINTERS", dict(checkpointer_registry._CHECKPOINTERS)
    )


@pytest.mark.parametrize(
    "documented",
    [
        BaseStore,
        InMemoryStore,
        model_registry.register_provider,
        checkpointer_registry.list_checkpointers,
    ],
    ids=lambda obj: obj.__qualname__,
)
def test_the_example_runs_as_written(documented: Any, isolated_registries: None) -> None:
    run_example(documented)


def test_registry_examples_leave_no_names_behind() -> None:
    """What the examples register stays in the copy the fixture handed them."""
    assert "mine" not in model_registry.list_providers()
    assert "audited" not in checkpointer_registry.list_checkpointers()
