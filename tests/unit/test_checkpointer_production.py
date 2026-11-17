# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The production settings every checkpointer takes: ttl, encryption at rest, scope.

The shared behaviours run in ``checkpointer_time_travel_conformance.py``
(``ProductionSettingsConformance``) over every backend. This file covers
the pieces on their own — :mod:`locus.memory.scope`,
:mod:`locus.memory.encryption`, :mod:`locus.memory.expiry`, the base class
hooks — and what only a look at each backend's storage can show: that the
bytes at rest are ciphertext with the listing fields in the clear, that an
expiry is recorded where the backend keeps it, and that a scope is a prefix
of the stored key.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory import CheckpointEncryptionError, CheckpointScope, EncryptedSerializer
from locus.memory.backends import FileCheckpointer, MemoryCheckpointer, StorageBackendAdapter
from locus.memory.backends.http import HTTPCheckpointer
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.encryption import is_envelope
from locus.memory.expiry import expiry_time, is_expired, parse_ttl, ttl_seconds
from locus.memory.retention import RetainedCheckpointer
from locus.memory.scope import current_scope
from tests.unit.test_checkpointer_time_travel import (
    _DictStore,
    _FakeObjectStorage,
    _IndexedDictStore,
    _state,
)


SENSITIVE = "the customer's card ends in 4242"
KEY = secrets.token_bytes(32)


def _secret_state() -> AgentState:
    return AgentState(agent_id="prod").with_message(Message.user(SENSITIVE))


def _contains_secret(blob: bytes | str) -> bool:
    text = blob.decode("utf-8", "replace") if isinstance(blob, bytes) else blob
    return SENSITIVE in text


# ---------------------------------------------------------------------------
# CheckpointScope
# ---------------------------------------------------------------------------


class TestCheckpointScope:
    def test_prefix_forms(self) -> None:
        assert CheckpointScope().prefix == ""
        assert CheckpointScope().is_empty
        assert CheckpointScope(app_id="billing").prefix == "app=billing/"
        assert CheckpointScope(user_id="u-1").prefix == "user=u-1/"
        both = CheckpointScope(app_id="billing", user_id="u-1")
        assert both.prefix == "app=billing/user=u-1/"
        assert both.key("case-4821") == "app=billing/user=u-1/case-4821"
        assert str(both) == "app=billing/user=u-1"
        assert str(CheckpointScope()) == "<unscoped>"

    @pytest.mark.parametrize("bad", ["", "a/b", "a*b", "a b", "a%b", "-x", 7])
    def test_ids_are_validated(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="app_id"):
            CheckpointScope(app_id=bad)
        with pytest.raises(ValueError, match="user_id"):
            CheckpointScope(user_id=bad)

    def test_ids_may_carry_the_usual_identifier_characters(self) -> None:
        scope = CheckpointScope(app_id="svc.billing:v2", user_id="ann@example.com")
        assert scope.key("t") == "app=svc.billing:v2/user=ann@example.com/t"

    def test_owns_unkey_and_strip(self) -> None:
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        assert scope.owns("app=billing/user=u-1/t")
        assert not scope.owns("app=billing/user=u-2/t")
        assert not scope.owns("t")
        assert scope.unkey("app=billing/user=u-1/t") == "t"
        assert scope.unkey("t") is None
        stored = ["t", "app=billing/user=u-1/a", "app=billing/user=u-2/b", "user=u-1/c"]
        assert scope.strip(stored) == ["a"]
        empty = CheckpointScope()
        assert empty.strip(stored) == ["t"], "the unscoped namespace owns only bare ids"
        assert not empty.owns("app=billing"), "a bare scope segment is not an unscoped thread"
        assert CheckpointScope(user_id="u-1").strip(stored) == ["c"]

    def test_active_sets_the_current_scope_for_the_block_only(self) -> None:
        scope = CheckpointScope(app_id="billing")
        assert current_scope() is None
        with scope.active():
            assert current_scope() == scope
            with CheckpointScope(app_id="other").active():
                assert current_scope() == CheckpointScope(app_id="other")
            assert current_scope() == scope
        assert current_scope() is None

    async def test_active_is_context_local_across_tasks(self) -> None:
        seen: list[CheckpointScope | None] = []

        async def inside() -> None:
            with CheckpointScope(app_id="a").active():
                await asyncio.sleep(0.01)
                seen.append(current_scope())

        async def outside() -> None:
            await asyncio.sleep(0.005)
            seen.append(current_scope())

        await asyncio.gather(inside(), outside())
        assert seen == [None, CheckpointScope(app_id="a")]


# ---------------------------------------------------------------------------
# EncryptedSerializer
# ---------------------------------------------------------------------------


class TestEncryptedSerializer:
    def test_needs_exactly_one_key_source_of_32_bytes(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            EncryptedSerializer()
        with pytest.raises(ValueError, match="exactly one"):
            EncryptedSerializer(KEY, key_provider=lambda: KEY)
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptedSerializer(b"short")
        with pytest.raises(ValueError, match="previous key"):
            EncryptedSerializer(KEY, previous_keys=[b"short"])

    def test_envelope_shape_keeps_the_iteration_and_hides_the_rest(self) -> None:
        serializer = EncryptedSerializer(KEY)
        data = _secret_state().with_iteration(3).to_checkpoint()

        envelope = serializer.encrypt(data)

        assert set(envelope) == {"v", "alg", "kid", "nonce", "ct", "iteration"}
        assert (envelope["v"], envelope["alg"]) == (1, "A256GCM")
        assert envelope["iteration"] == 3
        assert envelope["kid"] == serializer.key_id
        assert len(envelope["kid"]) == 8
        assert not _contains_secret(json.dumps(envelope))
        assert is_envelope(envelope)
        assert not is_envelope(data)
        assert serializer.decrypt(envelope) == data
        assert serializer.encrypt(data)["ct"] != envelope["ct"], "a fresh nonce every time"

    def test_plaintext_passes_through_decrypt(self) -> None:
        data = {"iteration": 1, "messages": []}
        assert EncryptedSerializer(KEY).decrypt(data) == data

    def test_a_wrong_key_fails_without_revealing_anything(self) -> None:
        envelope = EncryptedSerializer(KEY).encrypt({"iteration": 0})
        with pytest.raises(CheckpointEncryptionError) as failure:
            EncryptedSerializer(secrets.token_bytes(32)).decrypt(envelope)
        message = str(failure.value)
        assert envelope["kid"] in message
        assert KEY.hex() not in message

    def test_previous_keys_read_old_envelopes_by_their_kid(self) -> None:
        old, new = secrets.token_bytes(32), secrets.token_bytes(32)
        envelope = EncryptedSerializer(old).encrypt({"iteration": 2})
        rotated = EncryptedSerializer(new, previous_keys=[old])
        assert rotated.decrypt(envelope) == {"iteration": 2}
        # Without a kid every key is tried in turn.
        anonymous = {k: v for k, v in envelope.items() if k != "kid"}
        assert rotated.decrypt(anonymous) == {"iteration": 2}

    def test_key_provider_is_called_for_each_operation(self) -> None:
        calls: list[str] = []

        def provider() -> bytes:
            calls.append("fetch")
            return KEY

        serializer = EncryptedSerializer(key_provider=provider)
        envelope = serializer.encrypt({"iteration": 0})
        serializer.decrypt(envelope)
        assert calls == ["fetch", "fetch"]
        with pytest.raises(ValueError, match="provider returned"):
            EncryptedSerializer(key_provider=lambda: b"short").encrypt({})

    def test_an_unknown_envelope_version_is_refused(self) -> None:
        envelope = EncryptedSerializer(KEY).encrypt({"iteration": 0})
        envelope["v"] = 99
        with pytest.raises(CheckpointEncryptionError, match="version"):
            EncryptedSerializer(KEY).decrypt(envelope)

    def test_repr_never_shows_key_material(self) -> None:
        text = repr(EncryptedSerializer(KEY, previous_keys=[secrets.token_bytes(32)]))
        assert "EncryptedSerializer(kid=" in text
        assert "previous_keys=1" in text
        assert KEY.hex() not in text
        assert repr(KEY) not in text


# ---------------------------------------------------------------------------
# expiry helpers and the base class hooks
# ---------------------------------------------------------------------------


class TestExpiryHelpers:
    def test_parse_ttl(self) -> None:
        assert parse_ttl(None) is None
        assert parse_ttl(90) == timedelta(seconds=90)
        assert parse_ttl(0.5) == timedelta(milliseconds=500)
        assert parse_ttl(timedelta(days=1)) == timedelta(days=1)
        with pytest.raises(ValueError, match="positive"):
            parse_ttl(0)
        with pytest.raises(ValueError, match="positive"):
            parse_ttl(timedelta(seconds=-1))
        with pytest.raises(TypeError, match="seconds or a timedelta"):
            parse_ttl(True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            parse_ttl("1h")  # type: ignore[arg-type]

    def test_ttl_seconds_rounds_up_to_a_whole_second(self) -> None:
        assert ttl_seconds(timedelta(milliseconds=200)) == 1
        assert ttl_seconds(timedelta(seconds=1.5)) == 2
        assert ttl_seconds(timedelta(hours=1)) == 3600

    def test_expiry_time_and_is_expired(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert expiry_time(None, now) is None
        assert expiry_time(timedelta(minutes=5), now) == datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        assert not is_expired(None)
        assert not is_expired("not a time")
        assert is_expired("2000-01-01T00:00:00+00:00")
        assert not is_expired("2999-01-01T00:00:00+00:00")
        assert is_expired(now, now), "the moment itself counts as expired"
        assert not is_expired(now + timedelta(seconds=1), now)


class _Bare(BaseCheckpointer):
    """Only the abstract methods: the base defaults are what it has."""

    async def save(
        self, state: Any, thread_id: str, checkpoint_id: Any = None, metadata: Any = None
    ) -> str:  # noqa: E501
        return "x"

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        return None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        return []


class TestBaseHooks:
    def test_configure_validates_its_arguments(self) -> None:
        cp = _Bare()
        with pytest.raises(TypeError, match="EncryptedSerializer"):
            cp._configure(encryption="not a serializer")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="CheckpointScope"):
            cp._configure(scope="billing")  # type: ignore[arg-type]
        cp._configure(ttl=5, scope=CheckpointScope())
        assert cp.ttl == timedelta(seconds=5)
        assert cp.scope is None, "an empty scope is no scope"

    def test_scoped_prefers_the_active_scope(self) -> None:
        cp = _Bare()
        cp._configure(scope=CheckpointScope(app_id="a"))
        assert cp._scoped("t") == "app=a/t"
        with CheckpointScope(app_id="b").active():
            assert cp._scoped("t") == "app=b/t"
            assert cp._listing_scope(None) == CheckpointScope(app_id="b")
        assert cp._listing_scope(CheckpointScope(user_id="u")) == CheckpointScope(user_id="u")
        assert _Bare()._listing_scope(None) == CheckpointScope()

    def test_decode_refuses_an_envelope_without_a_key(self) -> None:
        envelope = EncryptedSerializer(KEY).encrypt(AgentState().to_checkpoint())
        with pytest.raises(CheckpointEncryptionError, match="give this checkpointer encryption="):
            _Bare()._decode_state(envelope)
        plain = _Bare()._decode_state(AgentState().with_iteration(4).to_checkpoint())
        assert plain.iteration == 4

    async def test_expire_is_not_available_by_default(self) -> None:
        with pytest.raises(NotImplementedError, match="expire"):
            await _Bare().expire()
        with pytest.raises(NotImplementedError):
            await HTTPCheckpointer("http://cp.local").expire()


# ---------------------------------------------------------------------------
# What is at rest, per backend
# ---------------------------------------------------------------------------


class TestAtRestMemory:
    async def test_storage_holds_ciphertext_beside_plain_listing_fields(self) -> None:
        cp = MemoryCheckpointer(encryption=EncryptedSerializer(KEY), ttl=60)
        await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "4821"})

        stored = cp._storage["t"]["cp-0"]
        assert is_envelope(stored.data)
        assert not _contains_secret(json.dumps(stored.data))
        assert stored.iteration == 2
        assert stored.metadata == {"case": "4821"}
        assert stored.expires_at is not None
        assert stored.expires_at > datetime.now(UTC)
        loaded = await cp.load("t")
        assert loaded is not None
        assert _contains_secret(str(loaded.messages[0].content))

    async def test_scope_is_a_prefix_of_the_stored_key(self) -> None:
        cp = MemoryCheckpointer(scope=CheckpointScope(app_id="billing", user_id="u-1"))
        await cp.save(AgentState(), "t")
        assert cp.get_thread_ids() == ["app=billing/user=u-1/t"]
        assert cp.get_checkpoint_count("t") == 1
        assert await MemoryCheckpointer().list_threads() == []


class TestAtRestFile:
    async def test_file_holds_ciphertext_and_a_readable_header(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(tmp_path, encryption=EncryptedSerializer(KEY), ttl=60)
        await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "4821"})

        raw = (tmp_path / "t" / "cp-0.json").read_bytes()
        assert not _contains_secret(raw)
        header = json.loads(raw)
        assert header["iteration"] == 2
        assert header["metadata"] == {"case": "4821"}
        assert header["expires_at"] is not None
        assert is_envelope(header["state"])
        [info] = await cp.list("t")
        assert (info.iteration, info.metadata) == (2, {"case": "4821"})

    async def test_expired_files_are_skipped_on_every_read_and_swept(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(tmp_path, ttl=0.05)
        await cp.save(_state(0), "t", "cp-0")
        await cp.save(_state(1), "other", "cp-1")
        plain = FileCheckpointer(tmp_path)
        await plain.save(_state(2), "kept", "cp-2")
        await asyncio.sleep(0.1)

        assert await plain.load("t") is None, "a reader without a ttl honours the header"
        assert await plain.list_threads() == ["kept"]
        assert [e["thread_id"] for e in await plain.list_with_metadata()] == ["kept"]
        assert await plain.get_disk_usage("t") > 0
        assert await plain.expire() == 2
        assert not (tmp_path / "t").exists()
        assert (tmp_path / "kept").exists()

    async def test_a_scoped_thread_has_its_own_directory(self, tmp_path: Path) -> None:
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        scoped = FileCheckpointer(tmp_path, scope=scope)
        plain = FileCheckpointer(tmp_path)
        await scoped.save(_state(0), "t", "cp-0")
        await plain.save(_state(1), "t", "cp-1")

        assert sorted(p.name for p in tmp_path.iterdir()) == ["app_billing_user_u-1_t", "t"]
        assert await plain.list_threads() == ["t"]
        assert await plain.list_threads(scope=scope) == ["t"]
        assert await scoped.list_threads() == ["t"]
        assert await scoped.list_threads(scope=CheckpointScope()) == ["t"]
        assert await scoped.load("t", "cp-1") is None
        assert await scoped.get_disk_usage("t") > 0


class _LikeStore(_DictStore):
    """A store whose ``list_threads`` speaks SQL ``LIKE``, as the SQL backends do."""

    async def list_threads(
        self, pattern: str = "%", limit: int = 100, offset: int = 0
    ) -> list[str]:
        import re

        regex, chars = "^", iter(pattern)
        for char in chars:
            if char == "\\":
                regex += re.escape(next(chars, ""))
            elif char == "%":
                regex += ".*"
            elif char == "_":
                regex += "."
            else:
                regex += re.escape(char)
        return [k for k in self.rows if re.match(regex + "$", k)][:limit]


class _ExpiringStore(_DictStore):
    """A store whose ``save`` takes ``expires_at`` and that sweeps itself, as the SQL backends do."""

    def __init__(self) -> None:
        super().__init__()
        self.expiries: dict[str, datetime | None] = {}
        self.swept = 0

    async def save(
        self, key: str, data: dict[str, Any], expires_at: datetime | None = None
    ) -> None:
        await super().save(key, data)
        self.expiries[key] = expires_at

    async def expire(self) -> int:
        self.swept += 1
        return 0


class _RedisLikeConfig:
    def __init__(self) -> None:
        self.ttl_seconds: int | None = None


class _RedisLikeStore(_DictStore):
    def __init__(self) -> None:
        super().__init__()
        self.config = _RedisLikeConfig()


class TestAtRestAdapter:
    async def test_documents_hold_ciphertext_beside_the_bookkeeping_fields(self) -> None:
        store = _DictStore()
        cp = StorageBackendAdapter(store, encryption=EncryptedSerializer(KEY), ttl=60)
        await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "4821"})

        document = store.rows["t:cp-0"]
        assert not _contains_secret(json.dumps(document))
        assert document["_checkpoint_id"] == "cp-0"
        assert document["_metadata"] == {"case": "4821"}
        assert document["_expires_at"]
        assert document["iteration"] == 2
        assert is_envelope({k: v for k, v in document.items() if not k.startswith("_")})
        [entry] = store.rows["t:_checkpoints"]["checkpoints"]
        assert (entry["iteration"], entry["metadata"]) == (2, {"case": "4821"})
        assert entry["expires_at"] == document["_expires_at"]
        assert not _contains_secret(json.dumps(store.rows["t:latest"]))

    async def test_expiry_is_handed_to_a_store_that_takes_it_and_swept_with_it(self) -> None:
        store = _ExpiringStore()
        cp = StorageBackendAdapter(store, ttl=0.05)
        await cp.save(_state(0), "t", "cp-0")
        assert store.expiries["t:cp-0"] is not None
        assert store.expiries["t:latest"] == store.expiries["t:cp-0"]
        assert await cp.exists("t")
        await asyncio.sleep(0.1)
        assert await cp.exists("t") is False, "with a ttl, exists reads the document"
        assert await cp.expire() == 1
        assert store.swept == 1
        assert "t:cp-0" not in store.rows
        assert "t:latest" not in store.rows

    async def test_the_ttl_and_a_store_s_native_ttl_agree(self) -> None:
        store = _RedisLikeStore()
        cp = StorageBackendAdapter(store, ttl=timedelta(milliseconds=1500))
        assert store.config.ttl_seconds == 2, "whole seconds, rounded up, for the store's EXPIRE"
        assert cp.ttl == timedelta(milliseconds=1500)

        native = _RedisLikeStore()
        native.config.ttl_seconds = 30
        assert StorageBackendAdapter(native).ttl == timedelta(seconds=30)
        assert StorageBackendAdapter(native, ttl=5).ttl == timedelta(seconds=5)
        assert native.config.ttl_seconds == 30, "an explicit native ttl is left alone"

    async def test_list_threads_translates_a_glob_for_a_like_store(self) -> None:
        store = _LikeStore()
        scope = CheckpointScope(app_id="billing", user_id="u_1")
        await StorageBackendAdapter(store, scope=scope).save(_state(0), "case-1", "cp-0")
        await StorageBackendAdapter(store).save(_state(0), "case-2", "cp-0")

        scoped = await StorageBackendAdapter(store, scope=scope).list_threads(pattern="case-*")
        assert {k.split(":")[0] for k in scoped} == {"case-1"}
        plain = await StorageBackendAdapter(store).list_threads(pattern="case-2*")
        assert {k.split(":")[0] for k in plain} == {"case-2"}
        assert await StorageBackendAdapter(store, ttl=1).expire() == 0

    async def test_a_native_index_store_is_swept_through_its_latest_keys(self) -> None:
        store = _IndexedDictStore()
        cp = StorageBackendAdapter(store, ttl=0.05)
        await cp.save(_state(0), "t", "cp-0")
        await asyncio.sleep(0.1)
        assert await cp.expire() == 1
        assert store.index["t"] == {}
        assert store.rows == {}

    async def test_a_store_without_a_listing_cannot_be_swept(self) -> None:
        class _Minimal:
            def __init__(self) -> None:
                self.rows: dict[str, Any] = {}

            async def save(self, key: str, data: dict[str, Any]) -> None:
                self.rows[key] = data

            async def load(self, key: str) -> dict[str, Any] | None:
                return self.rows.get(key)

            async def delete(self, key: str) -> bool:
                return self.rows.pop(key, None) is not None

            async def exists(self, key: str) -> bool:
                return key in self.rows

        cp = StorageBackendAdapter(_Minimal(), ttl=0.05)
        await cp.save(_state(0), "t", "cp-0")
        await asyncio.sleep(0.1)
        assert await cp.expire() == 0
        assert await cp.load("t") is None, "reads still filter"


moto = pytest.importorskip("moto", reason="S3 at-rest checks run on moto")


class TestAtRestS3:
    @pytest.fixture(autouse=True)
    def _mock_aws(self) -> Any:
        pytest.importorskip("boto3")
        with moto.mock_aws():
            yield

    def _backend(self, **settings: Any) -> Any:
        from locus.memory.backends.s3 import S3Backend

        return S3Backend(bucket="locus-prod", region_name="us-east-1", **settings)

    async def test_objects_hold_ciphertext_and_plain_metadata(self) -> None:
        cp = self._backend(encryption=EncryptedSerializer(KEY), ttl=60)
        await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "4821"})

        client = cp._get_client()
        payload = client.get_object(Bucket="locus-prod", Key="locus/checkpoints/t/cp-0.json")
        body = payload["Body"].read()
        assert not _contains_secret(body)
        assert is_envelope(json.loads(body)["state"])
        assert json.loads(body)["expires_at"]
        meta = client.get_object(Bucket="locus-prod", Key="locus/checkpoints/t/cp-0.meta.json")
        meta_doc = json.loads(meta["Body"].read())
        assert (meta_doc["iteration"], meta_doc["metadata"]) == (2, {"case": "4821"})
        assert meta_doc["expires_at"]

    async def test_expire_deletes_the_pair_and_repoints_latest(self) -> None:
        cp = self._backend(ttl=0.05)
        await cp.save(_state(0), "t", "cp-0")
        plain = self._backend()
        await asyncio.sleep(0.1)
        await plain.save(_state(1), "t", "cp-1")
        await cp.save(_state(2), "u", "cp-2")
        await asyncio.sleep(0.1)

        assert await plain.list_checkpoints("t") == ["cp-1", "cp-0"], "no ttl: the listing is raw"
        assert await plain.load("t", "cp-0") is None, "but an expired payload never loads"
        assert [i.checkpoint_id for i in await plain.list("t")] == ["cp-1"]
        assert await cp.expire() == 2
        assert await plain.list_checkpoints("t") == ["cp-1"]
        assert await plain.load("t") is not None
        assert await plain.load("u") is None
        assert await plain.list_threads() == ["t"]
        assert [m["checkpoint_id"] for m in await plain.list_with_metadata()] == ["cp-1"]
        assert await plain.get_metadata("t", "cp-0") is None
        assert await cp.expire() == 0

    async def test_scope_is_an_object_key_prefix(self) -> None:
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        scoped = self._backend(scope=scope)
        plain = self._backend()
        await scoped.save(_state(0), "t", "cp-0")
        await plain.save(_state(1), "t", "cp-1")

        keys = {
            o["Key"] for o in scoped._get_client().list_objects_v2(Bucket="locus-prod")["Contents"]
        }
        assert "locus/checkpoints/app=billing/user=u-1/t/cp-0.json" in keys
        assert "locus/checkpoints/t/cp-1.json" in keys
        assert await plain.list_threads() == ["t"]
        assert await plain.list_threads(scope=scope) == ["t"]
        assert await scoped.list_threads() == ["t"]
        assert await scoped.load("t", "cp-1") is None
        assert await scoped.copy_thread("t", "t-copy")
        assert await scoped.load("t-copy") is not None
        assert await plain.load("t-copy") is None
        assert await scoped.get_metadata("t", "cp-0") == {}


class TestAtRestOCIBucket:
    def _backend(self, client: _FakeObjectStorage, **settings: Any) -> Any:
        from locus.memory.backends.oci_bucket import OCIBucketBackend

        backend = OCIBucketBackend(
            bucket_name="b", namespace="n", retry_strategy=object(), **settings
        )
        backend._client = client  # type: ignore[assignment]
        return backend

    async def test_objects_hold_ciphertext_and_plain_metadata(self) -> None:
        client = _FakeObjectStorage()
        cp = self._backend(client, encryption=EncryptedSerializer(KEY), ttl=60)
        await cp.save(_secret_state().with_iteration(2), "t", "cp-0", metadata={"case": "4821"})

        body, _ = client.objects["locus/checkpoints/t/cp-0.json"]
        assert not _contains_secret(body)
        assert is_envelope(json.loads(body)["state"])
        meta = json.loads(client.objects["locus/checkpoints/t/cp-0.meta.json"][0])
        assert (meta["iteration"], meta["metadata"]) == (2, {"case": "4821"})
        assert meta["expires_at"] == json.loads(body)["expires_at"]

    async def test_expire_deletes_the_pair_and_clears_latest(self) -> None:
        client = _FakeObjectStorage()
        cp = self._backend(client, ttl=0.05)
        plain = self._backend(client)
        await cp.save(_state(0), "t", "cp-0")
        await asyncio.sleep(0.1)
        await plain.save(_state(1), "t", "cp-1")
        await cp.save(_state(2), "u", "cp-2")
        await asyncio.sleep(0.1)

        assert await plain.load("t", "cp-0") is None
        assert await plain.exists("t", "cp-0") is False
        assert await plain.get_metadata("t", "cp-0") is None
        assert [i.checkpoint_id for i in await plain.list("t")] == ["cp-1"]
        assert await cp.expire() == 2
        assert "locus/checkpoints/u/_latest" not in client.objects
        assert await plain.load("u") is None
        assert await plain.load("t") is not None
        assert await cp.expire() == 0

    async def test_scope_is_an_object_key_prefix(self) -> None:
        client = _FakeObjectStorage()
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        scoped = self._backend(client, scope=scope)
        plain = self._backend(client)
        await scoped.save(_state(0), "t", "cp-0")
        await plain.save(_state(1), "t", "cp-1")

        assert "locus/checkpoints/app=billing/user=u-1/t/cp-0.json" in client.objects
        assert await plain.list_threads() == ["t"]
        assert await plain.list_threads(scope=scope) == ["t"]
        assert await scoped.list_threads() == ["t"]
        assert await scoped.copy_thread("t", "t-copy")
        assert await scoped.load("t-copy") is not None
        assert await plain.load("t-copy") is None
        assert [m["thread_id"] for m in await plain.list_with_metadata()] == ["t"]


class TestAtRestHTTP:
    async def test_the_service_receives_ciphertext_under_the_scoped_thread(self) -> None:
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        cp = HTTPCheckpointer(
            "http://cp.local", encryption=EncryptedSerializer(KEY), ttl=60, scope=scope
        )
        with respx.mock(base_url="http://cp.local") as mock:
            post = mock.post("/threads/app%3Dbilling%2Fuser%3Du-1%2Ft/checkpoints").mock(
                return_value=httpx.Response(200, json={"checkpoint_id": "cp-0"})
            )
            assert await cp.save(_secret_state(), "t", "cp-0") == "cp-0"
            sent = json.loads(post.calls[0].request.content)
            assert not _contains_secret(json.dumps(sent))
            assert is_envelope(sent["state"])
            assert sent["expires_at"]
            assert sent["thread_id"] == "app=billing/user=u-1/t"

            mock.get("/threads/app%3Dbilling%2Fuser%3Du-1%2Ft/checkpoints/cp-0").mock(
                return_value=httpx.Response(200, json=sent)
            )
            loaded = await cp.load("t", "cp-0")
            assert loaded is not None
            assert _contains_secret(str(loaded.messages[0].content))

            expired = {**sent, "expires_at": "2000-01-01T00:00:00+00:00"}
            mock.get("/threads/app%3Dbilling%2Fuser%3Du-1%2Ft/checkpoints/old").mock(
                return_value=httpx.Response(200, json=expired)
            )
            assert await cp.load("t", "old") is None
            mock.get("/threads/app%3Dbilling%2Fuser%3Du-1%2Ft/checkpoints").mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {"checkpoint_id": "cp-0"},
                        {"checkpoint_id": "old", "expires_at": "2000-01-01T00:00:00+00:00"},
                    ],
                )
            )
            assert await cp.list_checkpoints("t") == ["cp-0"]
            mock.delete("/threads/app%3Dbilling%2Fuser%3Du-1%2Ft/checkpoints").mock(
                return_value=httpx.Response(204)
            )
            assert await cp.delete("t")
        await cp.close()


class TestRetainedForwarding:
    async def test_the_wrapper_reports_and_forwards_its_inner_settings(self) -> None:
        scope = CheckpointScope(app_id="billing")
        inner = MemoryCheckpointer(ttl=0.05, encryption=EncryptedSerializer(KEY), scope=scope)
        cp = RetainedCheckpointer(inner)
        assert (cp.ttl, cp.encryption, cp.scope) == (inner.ttl, inner.encryption, scope)
        await cp.save(_state(0), "t", "cp-0")
        assert await cp.list_threads() == ["t"]
        assert await cp.list_threads(scope=CheckpointScope()) == []
        await asyncio.sleep(0.1)
        assert await cp.expire() == 1


def test_active_reset_from_another_context_clears_instead_of_raising() -> None:
    """A generator driven from two tasks may exit the block in a context it did not enter."""
    import contextvars

    scope = CheckpointScope(app_id="billing")
    block = scope.active()
    contextvars.copy_context().run(block.__enter__)
    with scope.active():
        block.__exit__(None, None, None)  # the token belongs to the other context
        assert current_scope() is None
