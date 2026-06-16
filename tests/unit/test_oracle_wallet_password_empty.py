# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Regression tests for issue #289 — empty-string ``wallet_password``.

In Pydantic v2 ``SecretStr("")`` is *falsy*, so a truthy guard
(``if cfg.wallet_password:``) silently drops an explicit
``wallet_password=""`` before it reaches ``oracledb.create_pool_async``.
But ``""`` is the documented python-oracledb idiom for an auto-login
(``cwallet.sso``) wallet — dropping it forces the driver onto the
encrypted ``ewallet.pem`` path, which fails at connect time with
``DPY-6005`` / ``OSError: [Errno 22]``.

Every Oracle pool builder in Locus shared this guard. These tests pin
the contract for all of them: an explicit ``""`` is forwarded, an
omitted password (``None``) is not.

The tests stub ``oracledb`` and capture the kwargs each ``_get_pool``
sends to ``create_pool_async`` — no real database required.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# oracledb stub that records create_pool_async kwargs
# ---------------------------------------------------------------------------


class _StubPool:
    async def close(self) -> None:  # pragma: no cover - never acquired here
        return None


def _install_capturing_oracledb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub ``oracledb`` and return a list that collects pool kwargs."""
    calls: list[dict[str, Any]] = []

    def fake_create_pool_async(*args: Any, **kwargs: Any) -> _StubPool:
        calls.append(kwargs)
        return _StubPool()

    def fake_makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"{host}:{port}/{service_name}"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = fake_create_pool_async  # type: ignore[attr-defined]
    fake.makedsn = fake_makedsn  # type: ignore[attr-defined]
    fake.AsyncConnectionPool = _StubPool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return calls


# ---------------------------------------------------------------------------
# Every Locus Oracle pool builder, constructed with a wallet location set
# ---------------------------------------------------------------------------

_COMMON = {"dsn": "mydb_high", "user": "MYAPP", "password": "pw", "wallet_location": "/w"}  # noqa: S106


def _factories() -> list[tuple[str, Callable[..., Any]]]:
    """Return (name, builder) pairs. Imports are deferred so the stub is in
    place when each module first touches ``oracledb``."""
    from locus.memory.backends.oracle import OracleBackend
    from locus.memory.backends.oracle_versioned import OracleCheckpointSaver
    from locus.memory.store_backends.oracle import OracleStore
    from locus.rag.chunkers.oracle_indb import OracleInDBChunker
    from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings
    from locus.rag.loaders.oracle import OracleADBLoader
    from locus.rag.stores.oracle import OracleVectorStore

    return [
        ("OracleBackend", lambda **kw: OracleBackend(**_COMMON, **kw)),
        ("OracleCheckpointSaver", lambda **kw: OracleCheckpointSaver(**_COMMON, **kw)),
        ("OracleStore", lambda **kw: OracleStore(**_COMMON, **kw)),
        (
            "OracleInDBEmbeddings",
            lambda **kw: OracleInDBEmbeddings(model_name="m", dimension=8, **_COMMON, **kw),
        ),
        ("OracleInDBChunker", lambda **kw: OracleInDBChunker(**_COMMON, **kw)),
        ("OracleVectorStore", lambda **kw: OracleVectorStore(**_COMMON, **kw)),
        (
            "OracleADBLoader",
            lambda **kw: OracleADBLoader(
                sql="SELECT id, body FROM docs",
                content_column="body",
                **_COMMON,
                **kw,
            ),
        ),
    ]


def _ids() -> list[str]:
    return [name for name, _ in _factories()]


@pytest.mark.asyncio
@pytest.mark.parametrize("idx", range(7), ids=_ids())
async def test_empty_wallet_password_is_forwarded(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """issue #289: ``wallet_password=""`` must reach create_pool_async."""
    calls = _install_capturing_oracledb(monkeypatch)
    _name, build = _factories()[idx]

    backend = build(wallet_password="")
    await backend._get_pool()

    assert len(calls) == 1
    assert "wallet_password" in calls[0], (
        "empty-string wallet_password was dropped (truthy-guard regression)"
    )
    assert calls[0]["wallet_password"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("idx", range(7), ids=_ids())
async def test_omitted_wallet_password_is_not_sent(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No wallet password supplied (None) → kwarg must be absent."""
    calls = _install_capturing_oracledb(monkeypatch)
    _name, build = _factories()[idx]

    backend = build()  # wallet_password defaults to None
    await backend._get_pool()

    assert len(calls) == 1
    assert "wallet_password" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("idx", range(7), ids=_ids())
async def test_real_wallet_password_is_forwarded(idx: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty wallet password is forwarded unchanged."""
    calls = _install_capturing_oracledb(monkeypatch)
    _name, build = _factories()[idx]

    backend = build(wallet_password="hunter2")  # noqa: S106
    await backend._get_pool()

    assert len(calls) == 1
    assert calls[0]["wallet_password"] == "hunter2"  # noqa: S105
