# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Every approval store passes the same conformance suite; the database ones over fakes.

The in-memory and file stores run as they are. ``PostgresApprovals``,
``RedisApprovals`` and ``OracleApprovals`` run over driver fakes that answer
exactly the statements the stores issue (``tests/unit/approval_store_fakes.py``),
so the state machine, the sync-over-async plumbing and the SQL shape are pinned
with no service; ``tests/integration/test_approval_stores.py`` runs the same
suite against live servers. The store specs ``locus approvals`` opens are
checked here too.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from locus.control import (
    ApprovalAuthority,
    ApprovalRecord,
    ApproverRule,
    FileApprovals,
    InMemoryApprovals,
    OracleApprovals,
    PostgresApprovals,
    RedisApprovals,
    approval_store,
    store_spec,
)
from locus.control.approvals_db import record_from_json, record_to_json, without_password
from tests.unit.approval_store_conformance import ApprovalStoreConformance
from tests.unit.approval_store_fakes import (
    FakeRedisServer,
    FakeServer,
    install_asyncpg,
    install_oracledb,
    install_redis,
    records_in,
)


Maker = Callable[[Any], Any]


@pytest.fixture
def closing() -> Iterator[list[Any]]:
    """Stores to close when the test ends, so their loop threads do not pile up."""
    opened: list[Any] = []
    yield opened
    for store in opened:
        store.close()


class TestInMemoryApprovals(ApprovalStoreConformance):
    @pytest.fixture
    def make_store(self) -> Maker:
        return InMemoryApprovals

    @pytest.fixture
    def reopen(self) -> Callable[[Any], Any]:
        return lambda store: store  # one process only: the same object is the store


class TestFileApprovals(ApprovalStoreConformance):
    @pytest.fixture
    def make_store(self, tmp_path: Path) -> Maker:
        return lambda authority: FileApprovals(tmp_path / f"{uuid.uuid4().hex}.json", authority)

    @pytest.fixture
    def reopen(self) -> Callable[[Any], Any]:
        return lambda store: FileApprovals(store.path, store.authority)


@pytest.fixture
def pg_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer()
    install_asyncpg(monkeypatch, server)
    return server


class TestPostgresApprovals(ApprovalStoreConformance):
    @pytest.fixture
    def make_store(self, pg_server: FakeServer, closing: list[Any]) -> Maker:
        def make(authority: Any) -> PostgresApprovals:
            store = PostgresApprovals(
                dsn="postgresql://locus:secret@db.internal:5432/locus",
                table_name=f"approvals_{uuid.uuid4().hex[:8]}",
                authority=authority,
            )
            closing.append(store)
            return store

        return make

    @pytest.fixture
    def reopen(self, closing: list[Any]) -> Callable[[Any], Any]:
        def again(store: PostgresApprovals) -> PostgresApprovals:
            other = PostgresApprovals(
                dsn=store.dsn, table_name=store.table_name, authority=store.authority
            )
            closing.append(other)
            return other

        return again

    def test_rows_are_locked_while_a_decision_is_made(
        self, pg_server: FakeServer, make_store: Maker
    ) -> None:
        store = make_store(None)
        approval_id = store.submit("svc", "t", {"a": 1})
        store.decide(approval_id, "approved", by="alice")
        store.consume(approval_id)

        locked = [s for s in pg_server.statements if "FOR UPDATE" in s]
        assert any("WHERE digest = $1" in s for s in locked), "submit locks the call's records"
        assert sum("WHERE approval_id = $1" in s for s in locked) == 2, "decide and consume"
        assert any("ON CONFLICT (approval_id) DO NOTHING" in s for s in pg_server.statements)
        [record] = records_in(pg_server, f"public.{store.table_name}")
        assert record["status"] == "consumed"

    def test_the_table_is_created_once_and_only_when_missing(
        self, pg_server: FakeServer, make_store: Maker
    ) -> None:
        store = make_store(None)
        store.pending()
        store.pending()
        ddl = [s for s in pg_server.statements if s.strip().startswith("CREATE")]
        assert [s.split()[1] for s in ddl] == ["SCHEMA", "TABLE", "INDEX", "INDEX"]
        assert any("CREATE INDEX IF NOT EXISTS idx_" in s for s in ddl)

        pg_server.statements.clear()
        PostgresApprovals(dsn=store.dsn, table_name=store.table_name).pending()
        assert not [s for s in pg_server.statements if s.strip().startswith("CREATE")], (
            "an existing table needs no DDL, so a role without CREATE works"
        )

    def test_the_spec_names_the_table_and_hides_the_password(self, make_store: Maker) -> None:
        store = make_store(None)
        assert store.spec == (
            f"postgresql://locus@db.internal:5432/locus?table=public.{store.table_name}"
        )
        assert "secret" not in repr(store)
        by_parts = PostgresApprovals(host="h", port=5433, database="d", user="u", password="pw")  # noqa: S106
        assert by_parts.spec == "postgresql://h:5433/d?table=public.locus_approvals"

    def test_the_pool_is_built_from_the_dsn_or_the_parts(
        self, monkeypatch: pytest.MonkeyPatch, closing: list[Any]
    ) -> None:
        calls = install_asyncpg(monkeypatch, FakeServer())
        dsn = PostgresApprovals(dsn="postgresql://u:p@h/d", max_pool_size=3)
        parts = PostgresApprovals(host="h", port=5433, database="d", user="u", password="pw")  # noqa: S106
        closing.extend([dsn, parts])

        dsn.pending()
        parts.pending()

        assert calls[0] == {"dsn": "postgresql://u:p@h/d", "min_size": 1, "max_size": 3}
        assert calls[1] == {
            "dsn": None,
            "host": "h",
            "port": 5433,
            "database": "d",
            "user": "u",
            "password": "pw",
            "min_size": 1,
            "max_size": 5,
        }

    def test_identifiers_are_validated(self) -> None:
        with pytest.raises(ValueError, match="table_name"):
            PostgresApprovals(table_name="approvals; DROP TABLE x")
        with pytest.raises(ValueError, match="schema_name"):
            PostgresApprovals(schema_name="a.b")

    def test_a_missing_driver_says_what_to_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "asyncpg", None)
        store = PostgresApprovals()
        with pytest.raises(ImportError, match=r"locus-sdk\[postgresql\]"):
            store.pending()
        store.close()


@pytest.fixture
def redis_server(monkeypatch: pytest.MonkeyPatch) -> FakeRedisServer:
    server = FakeRedisServer()
    install_redis(monkeypatch, server)
    return server


class TestRedisApprovals(ApprovalStoreConformance):
    @pytest.fixture
    def make_store(self, redis_server: FakeRedisServer, closing: list[Any]) -> Maker:
        def make(authority: Any) -> RedisApprovals:
            store = RedisApprovals(
                "redis://:secret@cache.internal:6379/2",
                prefix=f"t:{uuid.uuid4().hex[:8]}:",
                authority=authority,
            )
            closing.append(store)
            return store

        return make

    @pytest.fixture
    def reopen(self, closing: list[Any]) -> Callable[[Any], Any]:
        def again(store: RedisApprovals) -> RedisApprovals:
            other = RedisApprovals(store.url, prefix=store.prefix, authority=store.authority)
            closing.append(other)
            return other

        return again

    def test_every_unit_of_work_holds_the_store_lock(
        self, redis_server: FakeRedisServer, make_store: Maker
    ) -> None:
        store = make_store(None)
        approval_id = store.submit("svc", "t", {"a": 1})
        store.decide(approval_id, "approved", by="alice")
        store.consume(approval_id)
        store.pending()

        locks = [c for c in redis_server.commands if c.startswith("LOCK")]
        assert len(locks) == 4
        assert locks[0] == f"LOCK {store.prefix}lock ['blocking_timeout', 'timeout']"
        assert f"SET {store.prefix}record:{approval_id} NX" in redis_server.commands
        assert redis_server.sets[f"{store.prefix}pending"] == set()
        assert redis_server.sets[f"{store.prefix}digest:{store.get(approval_id).digest}"] == {
            approval_id
        }

    def test_the_spec_names_the_prefix_and_hides_the_password(
        self, monkeypatch: pytest.MonkeyPatch, make_store: Maker
    ) -> None:
        store = make_store(None)
        assert store.spec == f"redis://cache.internal:6379/2?prefix={store.prefix}"
        assert "secret" not in repr(store)

    def test_the_client_is_built_from_the_url(
        self, monkeypatch: pytest.MonkeyPatch, closing: list[Any]
    ) -> None:
        calls = install_redis(monkeypatch, FakeRedisServer())
        store = RedisApprovals("redis://h:1/0", db=3)
        closing.append(store)
        store.pending()
        assert calls == [{"url": "redis://h:1/0", "db": 3, "decode_responses": True}]

    def test_a_missing_driver_says_what_to_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "redis.asyncio", None)
        store = RedisApprovals()
        with pytest.raises(ImportError, match=r"locus-sdk\[redis\]"):
            store.pending()
        store.close()


@pytest.fixture
def oracle_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer()
    install_oracledb(monkeypatch, server)
    return server


class TestOracleApprovals(ApprovalStoreConformance):
    @pytest.fixture
    def make_store(self, oracle_server: FakeServer, closing: list[Any]) -> Maker:
        def make(authority: Any) -> OracleApprovals:
            store = OracleApprovals(
                dsn="locusdb_low",
                user="LOCUS",
                password="secret",  # noqa: S106 - a fake
                wallet_location="/opt/wallet",
                table_name=f"approvals_{uuid.uuid4().hex[:8]}",
                authority=authority,
            )
            closing.append(store)
            return store

        return make

    @pytest.fixture
    def reopen(self, closing: list[Any]) -> Callable[[Any], Any]:
        def again(store: OracleApprovals) -> OracleApprovals:
            other = OracleApprovals(
                dsn=store.config.dsn,
                user=store.config.user,
                password=store.config.password,
                wallet_location=store.config.wallet_location,
                table_name=store.config.table_name,
                authority=store.authority,
            )
            closing.append(other)
            return other

        return again

    def test_rows_are_locked_and_each_unit_of_work_commits(
        self, monkeypatch: pytest.MonkeyPatch, closing: list[Any]
    ) -> None:
        server = FakeServer()
        _, pools = install_oracledb(monkeypatch, server)
        store = OracleApprovals(dsn="x", user="u", password="p", table_name="appr")  # noqa: S106
        closing.append(store)

        approval_id = store.submit("svc", "t", {"a": 1})
        store.decide(approval_id, "approved", by="alice")
        store.consume(approval_id)
        with pytest.raises(KeyError):
            store.consume("appr-nope")

        locked = [s for s in server.statements if "FOR UPDATE" in s]
        assert any("WHERE digest = :digest" in s for s in locked)
        assert sum("WHERE approval_id = :approval_id" in s for s in locked) == 3
        [pool] = pools
        # DDL commit + 3 units of work; the failed consume rolled back instead.
        assert sum(c.commits for c in pool.connections) == 4
        assert sum(c.rollbacks for c in pool.connections) == 1
        [record] = records_in(server, "appr")
        assert record["status"] == "consumed"

    def test_the_table_is_created_once_and_only_when_missing(
        self, oracle_server: FakeServer, make_store: Maker
    ) -> None:
        store = make_store(None)
        store.pending()
        store.pending()
        ddl = [s for s in oracle_server.statements if s.strip().startswith("CREATE")]
        assert [s.split()[1] for s in ddl] == ["TABLE", "INDEX", "INDEX"]
        assert "CHECK (record IS JSON)" in ddl[0]

        oracle_server.statements.clear()
        OracleApprovals(dsn="locusdb_low", table_name=store.config.table_name).pending()
        assert not [s for s in oracle_server.statements if s.strip().startswith("CREATE")]

    def test_the_pool_is_built_like_the_checkpointers(
        self, monkeypatch: pytest.MonkeyPatch, closing: list[Any]
    ) -> None:
        calls, _ = install_oracledb(monkeypatch, FakeServer())
        wallet = OracleApprovals(
            dsn="db_low",
            user="U",
            password="pw",  # noqa: S106 - a fake
            wallet_location="/w",
            wallet_password="",
            max_pool_size=2,
        )
        parts = OracleApprovals(host="h", port=1522, service_name="svc", user="U", password="pw")  # noqa: S106
        closing.extend([wallet, parts])

        wallet.pending()
        parts.pending()

        assert calls[0] == {
            "user": "U",
            "password": "pw",
            "dsn": "db_low",
            "min": 1,
            "max": 2,
            "config_dir": "/w",
            "wallet_location": "/w",
            "wallet_password": "",
        }
        assert calls[1]["dsn"] == "(HOST=h)(PORT=1522)(SERVICE=svc)"
        assert "wallet_location" not in calls[1]

    def test_the_spec_names_the_connection_without_the_password(self, make_store: Maker) -> None:
        store = make_store(None)
        assert store.spec == (
            f"oracle://LOCUS@/?dsn=locusdb_low&wallet=/opt/wallet&table={store.config.table_name}"
        )
        assert "secret" not in repr(store)
        parts = OracleApprovals(host="h", port=1522, service_name="svc", user="U")
        assert parts.spec == "oracle://U@h:1522/svc?table=locus_approvals"

    def test_a_record_the_driver_already_parsed_is_read(
        self, monkeypatch: pytest.MonkeyPatch, closing: list[Any]
    ) -> None:
        """Oracle AI Database 26ai hands a ``CLOB CHECK (IS JSON)`` column back as a dict."""
        install_oracledb(monkeypatch, FakeServer(parsed_json=True))
        store = OracleApprovals(dsn="x", user="u", password="p", table_name="appr")  # noqa: S106
        closing.append(store)

        approval_id = store.submit("svc", "t", {"a": 1})
        record = store.decide(approval_id, "approved", by="alice")

        assert (record.status, record.arguments) == ("approved", {"a": 1})
        assert store.consume(approval_id).status == "consumed"
        assert store.pending() == []

    def test_a_missing_driver_says_what_to_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "oracledb", None)
        store = OracleApprovals(dsn="x")
        with pytest.raises(ImportError, match=r"locus-sdk\[oracledb\]"):
            store.pending()
        store.close()


# ---------------------------------------------------------------------------
# Records on the wire, and the specs that name a store
# ---------------------------------------------------------------------------


def test_a_record_survives_the_json_round_trip() -> None:
    record = ApprovalRecord(
        approval_id="appr-1",
        digest="d",
        principal="svc",
        tool="t",
        arguments={"n": 1.5, "items": ["a", "b"]},
        labels=["x"],
        approvals=[{"by": "alice", "at": "now", "verdict": "approved"}],
        context={"policy_version": "v1"},
        approved_arguments={"n": 1.0, "items": []},
    )
    assert record_from_json(record_to_json(record)) == record
    assert record_from_json(json.loads(record_to_json(record))) == record


def test_passwords_are_removed_from_urls() -> None:
    assert without_password("postgresql://u:pw@h:5432/d?x=1") == "postgresql://u@h:5432/d?x=1"
    assert without_password("redis://:pw@h:6379/0") == "redis://h:6379/0"
    assert without_password("redis://h:6379") == "redis://h:6379"


class TestApprovalStoreSpecs:
    def test_memory_and_file(self, tmp_path: Path) -> None:
        assert isinstance(approval_store("memory://"), InMemoryApprovals)
        file_store = approval_store(f"file://{tmp_path / 'a.json'}")
        assert isinstance(file_store, FileApprovals)
        assert file_store.path == tmp_path / "a.json"
        bare = approval_store(str(tmp_path / "b.json"))
        assert isinstance(bare, FileApprovals)
        assert store_spec(bare) == f"file://{tmp_path / 'b.json'}"
        assert store_spec(InMemoryApprovals()) == "memory://"
        assert store_spec(object()) is None

    def test_an_authority_rides_along(self) -> None:
        authority = ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"a"})),))
        assert approval_store("memory://", authority).authority is authority

    def test_postgresql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = approval_store("postgresql://u:pw@h:5433/d?sslmode=require&table=ops.approvals")
        assert isinstance(store, PostgresApprovals)
        assert store.dsn == "postgresql://u:pw@h:5433/d?sslmode=require"
        assert (store.schema_name, store.table_name) == ("ops", "approvals")
        assert approval_store("postgres://u@h/d").table_name == "locus_approvals"

        with pytest.raises(ValueError, match="POSTGRES_HOST"):
            approval_store("postgresql://")
        monkeypatch.setenv("POSTGRES_HOST", "pg.internal")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_DB", "locus_test")
        monkeypatch.setenv("POSTGRES_USER", "tester")
        monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
        from_env = approval_store("postgresql://?table=approvals")
        assert isinstance(from_env, PostgresApprovals)
        assert from_env.dsn is None
        assert (from_env.host, from_env.port, from_env.database, from_env.user) == (
            "pg.internal",
            5433,
            "locus_test",
            "tester",
        )
        assert from_env.table_name == "approvals"

    def test_redis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = approval_store("rediss://:pw@h:6380/1?prefix=ops:appr:")
        assert isinstance(store, RedisApprovals)
        assert (store.url, store.prefix) == ("rediss://:pw@h:6380/1", "ops:appr:")

        with pytest.raises(ValueError, match="REDIS_URL"):
            approval_store("redis://")
        monkeypatch.setenv("REDIS_URL", "redis://cache:6379")
        from_env = approval_store("redis://")
        assert isinstance(from_env, RedisApprovals)
        assert (from_env.url, from_env.prefix) == ("redis://cache:6379", "locus:approvals:")

    def test_oracle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = approval_store("oracle://U:pw@h:1522/svc?wallet=/w&table=S.T")
        assert isinstance(store, OracleApprovals)
        config = store.config
        assert (config.user, config.password.get_secret_value()) == ("U", "pw")
        assert (config.host, config.port, config.service_name) == ("h", 1522, "svc")
        assert config.wallet_location == "/w"
        assert (config.schema_name, config.table_name) == ("S", "T")

        alias = approval_store("oracle://U:pw@/?dsn=db_low")
        assert isinstance(alias, OracleApprovals)
        assert alias.config.dsn == "db_low"

        with pytest.raises(ValueError, match="dsn="):
            approval_store("oracle://")
        monkeypatch.setenv("ORACLE_DSN", "env_low")
        monkeypatch.setenv("ORACLE_USER", "ENVUSER")
        monkeypatch.setenv("ORACLE_PASSWORD", "envpw")
        monkeypatch.setenv("ORACLE_WALLET", "/env/wallet")
        from_env = approval_store("oracle://")
        assert isinstance(from_env, OracleApprovals)
        assert (from_env.config.dsn, from_env.config.user) == ("env_low", "ENVUSER")
        assert from_env.config.password.get_secret_value() == "envpw"
        assert from_env.config.wallet_location == "/env/wallet"

    def test_the_spec_can_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCUS_APPROVALS_STORE", raising=False)
        with pytest.raises(ValueError, match="LOCUS_APPROVALS_STORE"):
            approval_store()
        monkeypatch.setenv("LOCUS_APPROVALS_STORE", "memory://")
        assert isinstance(approval_store(), InMemoryApprovals)

    def test_an_unknown_scheme_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown approval store scheme 'ftp'"):
            approval_store("ftp://somewhere")
