# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Issue #347 — TLS-only Autonomous Database connections and ``config_dir``.

An Autonomous Database created without mutual TLS has no wallet: the client
connects over one-way TLS with ``dsn`` plus ``user`` / ``password``. Before
the fix, the only way to resolve a ``tnsnames.ora`` alias was to pass the
directory as ``wallet_location``, which python-oracledb takes as a promise
that a wallet file lives there — and fails with ``DPY-6005`` when none does.

Every Oracle pool builder now takes an optional ``config_dir`` and passes
``wallet_location`` only when a wallet is given. These tests pin the four
combinations on each builder, the sync twins' passthrough, and the two
``oracle://`` spec parsers reading ``ORACLE_CONFIG_DIR`` / ``TNS_ADMIN``.

The tests stub ``oracledb`` and capture the kwargs each pool builder sends
to ``create_pool_async`` / ``create_pool`` — no database required.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest


_DESCRIPTOR = (
    "(description=(retry_count=20)(retry_delay=3)"
    "(address=(protocol=tcps)(port=1522)(host=db.example))"
    "(connect_data=(service_name=svc_low.example))"
    "(security=(ssl_server_dn_match=yes)))"
)


class _StubPool:
    def acquire(self) -> Any:  # pragma: no cover - never acquired here
        raise AssertionError("the stub pool is never acquired")

    async def close(self) -> None:  # pragma: no cover - never closed here
        return None


def _install_capturing_oracledb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub ``oracledb`` and return the list that collects pool kwargs."""
    calls: list[dict[str, Any]] = []

    def fake_create_pool(*args: Any, **kwargs: Any) -> _StubPool:
        calls.append(kwargs)
        return _StubPool()

    def fake_makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"{host}:{port}/{service_name}"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = fake_create_pool  # type: ignore[attr-defined]
    fake.create_pool = fake_create_pool  # type: ignore[attr-defined]
    fake.makedsn = fake_makedsn  # type: ignore[attr-defined]
    fake.AsyncConnectionPool = _StubPool  # type: ignore[attr-defined]
    fake.ConnectionPool = _StubPool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return calls


# ---------------------------------------------------------------------------
# Every Oracle pool builder in Locus
# ---------------------------------------------------------------------------

_CREDENTIALS = {"user": "MYAPP", "password": "pw"}  # noqa: S106


def _await_pool(obj: Any) -> None:
    asyncio.run(obj._get_pool())


def _call_build_pool(obj: Any) -> None:
    obj._build_pool()


def _builders() -> list[tuple[str, Callable[..., Any], Callable[[Any], None]]]:
    """``(name, constructor, build_pool)`` for every Oracle pool builder.

    Imports are deferred so the stub is in place when each module first
    touches ``oracledb``.
    """
    from locus.control.approvals_oracle import OracleApprovals
    from locus.control.audit_oracle import OracleAuditTrail
    from locus.evaluation.stores import OracleReportStore
    from locus.memory.backends.oracle import OracleBackend
    from locus.memory.backends.oracle_versioned import OracleCheckpointSaver
    from locus.memory.store_backends.oracle import OracleStore
    from locus.rag.chunkers.oracle_indb import OracleInDBChunker
    from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings
    from locus.rag.loaders.oracle import OracleADBLoader
    from locus.rag.stores.oracle import OracleVectorStore

    return [
        ("OracleBackend", OracleBackend, _await_pool),
        ("OracleCheckpointSaver", OracleCheckpointSaver, _await_pool),
        ("OracleStore", OracleStore, _await_pool),
        ("OracleVectorStore", OracleVectorStore, _await_pool),
        (
            "OracleADBLoader",
            lambda **kw: OracleADBLoader(
                sql="SELECT id, body FROM docs", content_column="body", **kw
            ),
            _await_pool,
        ),
        ("OracleInDBChunker", OracleInDBChunker, _await_pool),
        (
            "OracleInDBEmbeddings",
            lambda **kw: OracleInDBEmbeddings(model_name="m", dimension=8, **kw),
            _await_pool,
        ),
        ("OracleApprovals", OracleApprovals, _call_build_pool),
        ("OracleAuditTrail", OracleAuditTrail, _call_build_pool),
        ("OracleReportStore", OracleReportStore, _call_build_pool),
    ]


def _ids() -> list[str]:
    return [name for name, _, _ in _builders()]


def _pool_kwargs(idx: int, monkeypatch: pytest.MonkeyPatch, **connection: Any) -> dict[str, Any]:
    calls = _install_capturing_oracledb(monkeypatch)
    _name, construct, build_pool = _builders()[idx]
    build_pool(construct(**_CREDENTIALS, **connection))
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("idx", range(10), ids=_ids())
def test_wallet_only_doubles_as_the_tns_directory(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) Wallet only — today's behaviour: the wallet directory resolves the alias too."""
    kwargs = _pool_kwargs(idx, monkeypatch, dsn="mydb_low", wallet_location="/w")
    assert kwargs["dsn"] == "mydb_low"
    assert kwargs["config_dir"] == "/w"
    assert kwargs["wallet_location"] == "/w"
    assert "wallet_password" not in kwargs


@pytest.mark.parametrize("idx", range(10), ids=_ids())
def test_config_dir_only_sends_no_wallet(idx: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) TLS-only database, alias in a tnsnames.ora: ``config_dir`` and nothing wallet-shaped."""
    kwargs = _pool_kwargs(idx, monkeypatch, dsn="mydb_low", config_dir="/tns")
    assert kwargs["dsn"] == "mydb_low"
    assert kwargs["config_dir"] == "/tns"
    assert "wallet_location" not in kwargs, (
        "a config_dir without a wallet must not be sent as wallet_location: "
        "python-oracledb would look for ewallet.pem / cwallet.sso there (DPY-6005)"
    )
    assert "wallet_password" not in kwargs


@pytest.mark.parametrize("idx", range(10), ids=_ids())
def test_both_keep_the_wallet_and_the_separate_tns_directory(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) Both — the aliases live outside the wallet directory."""
    kwargs = _pool_kwargs(
        idx,
        monkeypatch,
        dsn="mydb_low",
        config_dir="/tns",
        wallet_location="/w",
        wallet_password="",
    )
    assert kwargs["config_dir"] == "/tns"
    assert kwargs["wallet_location"] == "/w"
    assert kwargs["wallet_password"] == ""


@pytest.mark.parametrize("idx", range(10), ids=_ids())
def test_a_full_connect_descriptor_needs_neither(idx: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) Neither — the descriptor from the console goes to the driver verbatim."""
    kwargs = _pool_kwargs(idx, monkeypatch, dsn=_DESCRIPTOR)
    assert kwargs["dsn"] == _DESCRIPTOR
    assert "config_dir" not in kwargs
    assert "wallet_location" not in kwargs
    assert "wallet_password" not in kwargs


# ---------------------------------------------------------------------------
# Sync twins pass config_dir through
# ---------------------------------------------------------------------------


def test_sync_wrappers_pass_config_dir_through() -> None:
    from locus.memory.backends.oracle_sync import OracleSyncBackend, OracleSyncCheckpointSaver
    from locus.memory.store_backends.oracle_sync import OracleSyncStore
    from locus.rag.chunkers.oracle_sync import OracleSyncInDBChunker
    from locus.rag.embeddings.oracle_sync import OracleSyncInDBEmbeddings
    from locus.rag.loaders.oracle_sync import OracleSyncADBLoader

    common = {"dsn": "mydb_low", "config_dir": "/tns", **_CREDENTIALS}
    assert OracleSyncBackend(**common)._async.config.config_dir == "/tns"
    assert OracleSyncCheckpointSaver(**common)._async.config.config_dir == "/tns"
    assert OracleSyncStore(**common)._async.config.config_dir == "/tns"
    assert OracleSyncInDBChunker(**common)._async.oracle_config.config_dir == "/tns"
    embeddings = OracleSyncInDBEmbeddings(model_name="m", dimension=8, **common)
    assert embeddings._async._cfg.config_dir == "/tns"
    loader = OracleSyncADBLoader(sql="SELECT id, body FROM docs", content_column="body", **common)
    assert loader._async.config.config_dir == "/tns"


# ---------------------------------------------------------------------------
# oracle:// specs: config_dir= option, ORACLE_CONFIG_DIR, TNS_ADMIN
# ---------------------------------------------------------------------------

_ORACLE_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_CONFIG_DIR",
    "ORACLE_WALLET",
    "ORACLE_WALLET_PASSWORD",
    "TNS_ADMIN",
)


@pytest.fixture
def clean_oracle_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _ORACLE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_approval_spec_takes_config_dir_and_round_trips_it(
    clean_oracle_env: pytest.MonkeyPatch,
) -> None:
    from locus.control.approvals_oracle import OracleApprovals
    from locus.control.approvals_spec import approval_store

    store = approval_store("oracle://U:pw@/?dsn=db_low&config_dir=/tns&table=T")
    assert isinstance(store, OracleApprovals)
    assert store.config.config_dir == "/tns"
    assert store.config.wallet_location is None
    assert store.spec == "oracle://U@/?dsn=db_low&config_dir=/tns&table=T"
    reopened = approval_store(store.spec)
    assert isinstance(reopened, OracleApprovals)
    assert reopened.config.config_dir == "/tns"


def test_approval_spec_reads_oracle_config_dir_from_the_environment(
    clean_oracle_env: pytest.MonkeyPatch,
) -> None:
    from locus.control.approvals_oracle import OracleApprovals
    from locus.control.approvals_spec import approval_store

    clean_oracle_env.setenv("ORACLE_DSN", "env_low")
    clean_oracle_env.setenv("ORACLE_CONFIG_DIR", "/env/tns")
    store = approval_store("oracle://")
    assert isinstance(store, OracleApprovals)
    assert (store.config.dsn, store.config.config_dir) == ("env_low", "/env/tns")
    assert store.config.wallet_location is None

    # The spec's own option wins over the environment.
    explicit = approval_store("oracle://?config_dir=/spec/tns")
    assert isinstance(explicit, OracleApprovals)
    assert explicit.config.config_dir == "/spec/tns"

    # A wallet keeps resolving its own aliases: an ambient ORACLE_CONFIG_DIR,
    # set for some other database, is not applied over it (like TNS_ADMIN).
    clean_oracle_env.setenv("ORACLE_WALLET", "/env/wallet")
    both = approval_store("oracle://")
    assert isinstance(both, OracleApprovals)
    assert (both.config.config_dir, both.config.wallet_location) == (None, "/env/wallet")

    # The spec's own option still wins over the wallet's directory.
    explicit_both = approval_store("oracle://?config_dir=/spec/tns")
    assert isinstance(explicit_both, OracleApprovals)
    assert explicit_both.config.config_dir == "/spec/tns"
    assert explicit_both.config.wallet_location == "/env/wallet"


def test_approval_spec_round_trips_directories_with_url_characters(
    clean_oracle_env: pytest.MonkeyPatch,
) -> None:
    from locus.control.approvals_oracle import OracleApprovals
    from locus.control.approvals_spec import approval_store

    store = OracleApprovals(
        dsn="db_low",
        user="U",
        password="pw",  # noqa: S106
        config_dir="/srv/tns+prod 1",
        wallet_location="/srv/w&b",
    )
    reopened = approval_store(store.spec)
    assert isinstance(reopened, OracleApprovals)
    assert reopened.config.config_dir == "/srv/tns+prod 1"
    assert reopened.config.wallet_location == "/srv/w&b"
    assert reopened.config.dsn == "db_low"


def test_approval_spec_falls_back_to_tns_admin_only_without_a_wallet(
    clean_oracle_env: pytest.MonkeyPatch,
) -> None:
    from locus.control.approvals_oracle import OracleApprovals
    from locus.control.approvals_spec import approval_store

    clean_oracle_env.setenv("ORACLE_DSN", "env_low")
    clean_oracle_env.setenv("TNS_ADMIN", "/env/tns-admin")
    store = approval_store("oracle://")
    assert isinstance(store, OracleApprovals)
    assert store.config.config_dir == "/env/tns-admin"

    # A wallet keeps resolving its own aliases: TNS_ADMIN is left to the driver.
    clean_oracle_env.setenv("ORACLE_WALLET", "/env/wallet")
    with_wallet = approval_store("oracle://")
    assert isinstance(with_wallet, OracleApprovals)
    assert with_wallet.config.config_dir is None
    assert with_wallet.config.wallet_location == "/env/wallet"


def test_report_store_spec_reads_config_dir(clean_oracle_env: pytest.MonkeyPatch) -> None:
    from locus.evaluation.stores import OracleReportStore, report_store

    explicit = report_store("oracle://u:pw@/?dsn=db_low&config_dir=/tns&table=reports")
    assert isinstance(explicit, OracleReportStore)
    assert explicit.config.config_dir == "/tns"
    assert explicit.config.wallet_location is None

    clean_oracle_env.setenv("ORACLE_DSN", "env_low")
    clean_oracle_env.setenv("ORACLE_CONFIG_DIR", "/env/tns")
    from_env = report_store("oracle://?table=LOCUS_EVAL_REPORTS")
    assert isinstance(from_env, OracleReportStore)
    assert (from_env.config.dsn, from_env.config.config_dir) == ("env_low", "/env/tns")

    clean_oracle_env.delenv("ORACLE_CONFIG_DIR")
    clean_oracle_env.setenv("TNS_ADMIN", "/env/tns-admin")
    from_tns_admin = report_store("oracle://?table=LOCUS_EVAL_REPORTS")
    assert isinstance(from_tns_admin, OracleReportStore)
    assert from_tns_admin.config.config_dir == "/env/tns-admin"


def test_report_store_spec_carries_config_dir_and_wallet_and_round_trips(
    clean_oracle_env: pytest.MonkeyPatch,
) -> None:
    from locus.evaluation.stores import OracleReportStore, report_store

    store = OracleReportStore(
        dsn="db_low",
        user="u",
        password="pw",  # noqa: S106
        config_dir="/tns dir",
        wallet_location="/w+1",
        schema_name="S",
        table_name="R",
    )
    assert "config_dir=/tns%20dir" in store.spec
    assert "wallet=/w%2B1" in store.spec
    reopened = report_store(store.spec)
    assert isinstance(reopened, OracleReportStore)
    assert (reopened.config.config_dir, reopened.config.wallet_location) == ("/tns dir", "/w+1")
    assert (reopened.config.schema_name, reopened.config.table_name) == ("S", "R")
