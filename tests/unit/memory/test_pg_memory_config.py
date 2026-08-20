# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`PgMemory`'s construction-time width rules.

No database is touched here — these pin the arithmetic that decides whether a
store *can* have an ANN index at all, which is what the default ``dim`` got
wrong (1024 → a 2048-wide ``[cos φ, sin φ]`` column → over pgvector's 2000).
The live behaviour is in ``tests/integration/test_pg_memory.py``.
"""

from __future__ import annotations

import pytest

from tulip.memory.store_backends.postgresql import PGVECTOR_ANN_MAX_DIM, PgMemory


_DSN = "postgresql://u:p@127.0.0.1:5432/db"  # never connected to


class _FakeEmbedder:
    """Just the ``dimension`` attribute PgMemory reads at construction."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


def test_default_dim_fits_under_the_pgvector_ann_limit() -> None:
    store = PgMemory(_DSN)
    assert store._vdim == 2 * store._dim
    assert store._vdim <= PGVECTOR_ANN_MAX_DIM
    assert store._ann is True
    assert store.capabilities.embedding_dimension == store._vdim


def test_explicit_dim_over_the_limit_fails_loudly_at_construction() -> None:
    """dim=1024 doubles to 2048 — the exact defect. Name both numbers."""
    with pytest.raises(ValueError) as exc:
        PgMemory(_DSN, dim=1024)
    msg = str(exc.value)
    assert "2048" in msg  # the real column width
    assert "2000" in msg  # the real pgvector limit
    assert "hnsw" in msg.lower()


def test_dim_exactly_at_the_limit_is_accepted() -> None:
    store = PgMemory(_DSN, dim=PGVECTOR_ANN_MAX_DIM // 2)
    assert store._vdim == PGVECTOR_ANN_MAX_DIM
    assert store._ann is True


def test_non_positive_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="dim must be >= 1"):
        PgMemory(_DSN, dim=0)


def test_wide_embedder_is_allowed_but_warns_that_there_is_no_ann_index() -> None:
    """A 3072-dim model (text-embedding-3-large) is legitimate; the silent
    sequential scan it implies is not."""
    with pytest.warns(RuntimeWarning, match="WITHOUT an ANN index"):
        store = PgMemory(_DSN, embedder=_FakeEmbedder(3072))  # type: ignore[arg-type]
    assert store._vdim == 3072
    assert store._ann is False


def test_embedder_within_the_limit_keeps_the_index_and_warns_nothing() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store = PgMemory(_DSN, embedder=_FakeEmbedder(1536))  # type: ignore[arg-type]
    assert store._ann is True
    assert store.capabilities.semantic_search is True


class TestDriverIsCheckedAtConstruction:
    """asyncpg is optional, and it is imported lazily inside the pool builder.

    Before this check, a missing driver surfaced as a bare
    ``ModuleNotFoundError`` raised from inside a coroutine on first *use* —
    long after the mistake, with nothing saying which package to install.
    Construction is where a caller mistake belongs.
    """

    def test_a_missing_driver_fails_at_construction_with_the_fix(self, monkeypatch):
        import importlib.util as _util

        real = _util.find_spec

        def _no_asyncpg(name, *a, **k):
            return None if name == "asyncpg" else real(name, *a, **k)

        monkeypatch.setattr(_util, "find_spec", _no_asyncpg)
        with pytest.raises(ImportError) as excinfo:
            PgMemory(_DSN)
        message = str(excinfo.value)
        assert "asyncpg" in message
        assert "tulip-agents[postgresql]" in message, "say which extra installs it"

    def test_construction_succeeds_when_the_driver_is_present(self):
        # No connection is made at construction, so this only proves the check
        # does not reject a correctly installed environment.
        assert PgMemory(_DSN) is not None
