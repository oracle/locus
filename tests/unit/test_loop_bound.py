# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Async clients that must not outlive the loop that built them.

`redis.asyncio`, `asyncpg`, `httpx` — and therefore `openai` and `anthropic` —
bind a connection pool to the loop running when the client is created. Cache
one on `self` and it works exactly once per process.

The Redis version of this bug was reported and fixed; it then turned out to be
live in ten more places, including both native model clients. On those the
failure is worse than Redis's ``Event loop is closed``:

    APIConnectionError: Connection error.

which reads as a provider outage. And it is *intermittent* — the second loop
fails, the third succeeds, because the failed request evicts the dead
connection. That profile gets written off as a flaky network.

The pattern had been hand-copied three times before this, with three different
spellings of the cache key, which is how copies drift. These tests cover the
one shared implementation and a guard that fails when a new client is cached
without it.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from tulip.core.loop_bound import loop_bound, loop_key_for


SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tulip"


class _Owner:
    """Stands in for a backend or model that caches a client."""

    def __init__(self) -> None:
        self._client: object | None = None
        self._client_loop: object | None = None
        self.builds = 0

    def build(self) -> object:
        self.builds += 1
        return object()

    def get(self) -> object:
        return loop_bound(self, "_client", self.build)


# --------------------------------------------------------------------------
# The behaviour the bug needed
# --------------------------------------------------------------------------


def test_a_new_loop_gets_a_new_client() -> None:
    """The regression itself, in miniature."""
    owner = _Owner()

    first = asyncio.run(_get(owner))
    second = asyncio.run(_get(owner))

    assert first is not second
    assert owner.builds == 2


async def _get(owner: _Owner) -> object:
    return owner.get()


def test_one_loop_builds_exactly_one_client() -> None:
    """Every real deployment is this case; it must stay as cheap as before."""
    owner = _Owner()

    async def many() -> None:
        for _ in range(5):
            owner.get()

    asyncio.run(many())

    assert owner.builds == 1


@pytest.mark.asyncio
async def test_a_caller_supplied_client_is_never_discarded() -> None:
    """Assigning the attribute means the caller owns the lifecycle.

    Tests inject a double this way. A cache that threw it out would quietly
    send the test at a real network.
    """
    owner = _Owner()
    injected = object()
    owner._client = injected

    assert owner.get() is injected
    assert owner.builds == 0


@pytest.mark.asyncio
async def test_the_loop_is_recorded_where_the_helper_looks_for_it() -> None:
    owner = _Owner()
    owner.get()

    assert getattr(owner, loop_key_for("_client")) is asyncio.get_running_loop()


def test_it_works_outside_a_running_loop() -> None:
    """Several of these caches sit behind a plain synchronous property.

    With no loop running there is nothing to compare against, so the helper
    must hand back a client rather than raise ``RuntimeError``.
    """
    owner = _Owner()

    assert owner.get() is not None
    assert owner.builds == 1


def test_a_client_built_outside_a_loop_is_adopted_by_the_first_one() -> None:
    """It records no loop, so the first async use must not throw it away."""
    owner = _Owner()
    built_sync = owner.get()

    assert asyncio.run(_get(owner)) is built_sync
    assert owner.builds == 1


# --------------------------------------------------------------------------
# The guard: no eleventh instance
# --------------------------------------------------------------------------

#: Cached under these names, an async client is loop-bound in practice.
_CACHE_ATTRS = ("_client", "_pool", "_session", "_engine")


def _lazily_cached_without_loop_binding() -> list[str]:
    """Files that rebuild a cached client with no loop key in sight."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "loop_bound.py":
            continue
        text = path.read_text()
        if "loop_bound(" in text or "get_running_loop()" in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # `if self._client is None:` guarding a lazy build.
            if not isinstance(node, ast.Compare) or not isinstance(node.ops[0], ast.Is):
                continue
            left, right = node.left, node.comparators[0]
            if not (isinstance(right, ast.Constant) and right.value is None):
                continue
            if (
                isinstance(left, ast.Attribute)
                and isinstance(left.value, ast.Name)
                and left.value.id == "self"
                and left.attr in _CACHE_ATTRS
            ):
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno} caches {left.attr}")
    return offenders


def test_the_guard_scans_the_package() -> None:
    """A guard that silently matches nothing passes for the wrong reason."""
    assert len(list(SRC.rglob("*.py"))) > 100


def test_no_async_client_is_cached_without_a_loop_key() -> None:
    """This is what stops the eleventh from being written the old way.

    A lazily-cached client that never consults the running loop is the exact
    shape of the bug — and it is invisible in review, because the code reads
    like ordinary memoisation.
    """
    offenders = _lazily_cached_without_loop_binding()

    assert not offenders, "\n  ".join(
        [
            "these cache an async client without keying it on the event loop; "
            "use tulip.core.loop_bound.loop_bound:",
            *offenders,
        ]
    )
