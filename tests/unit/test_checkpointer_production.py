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
from locus.memory.encryption import associated_data, is_envelope
from locus.memory.expiry import expiry_time, is_expired, parse_ttl, ttl_seconds
from locus.memory.retention import RetainedCheckpointer
from locus.memory.scope import StoredThreadId, current_scope
from locus.memory.scoped import ScopedCheckpointer, scoped_checkpointer
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
        # Each segment is percent-encoded in the stored id, and decoded on the way back.
        assert scope.key("t") == "app=svc.billing%3Av2/user=ann%40example.com/t"
        assert scope.unkey("app=svc.billing%3Av2/user=ann%40example.com/t") == "t"
        assert isinstance(scope.key("t"), StoredThreadId)

    def test_no_thread_id_can_forge_the_scope_grammar(self) -> None:
        app = CheckpointScope(app_id="billing")
        user = CheckpointScope(app_id="billing", user_id="victim")
        stored = user.key("t")
        assert stored == "app=billing/user=victim/t"
        # The app-only scope stores the lookalike as its own encoded thread...
        assert app.key("user=victim/t") == "app=billing/user%3Dvictim%2Ft"
        assert app.unkey("app=billing/user%3Dvictim%2Ft") == "user=victim/t"
        # ...and never resolves the user's thread, which has a segment too many.
        assert app.unkey(stored) is None
        assert app.strip([stored, "app=billing/t", "app=billing/user=u/x"]) == ["t"]
        # A thread id with a slash or an equals sign is one encoded segment.
        assert user.key("a/b=c") == "app=billing/user=victim/a%2Fb%3Dc"
        assert user.unkey("app=billing/user=victim/a%2Fb%3Dc") == "a/b=c"

    def test_the_unscoped_namespace_never_resolves_a_scoped_id(self) -> None:
        empty = CheckpointScope()
        # A bare id is stored as it is; a scoped stored id is not a thread of it.
        assert empty.key("t") == "t"
        assert empty.unkey("app=billing/user=victim/t") is None
        assert empty.unkey("user=victim/t") is None
        # An id that could pass for a scope marker is encoded whole, and decoded back.
        assert empty.key("app=billing/user=victim/t") == "app%3Dbilling%2Fuser%3Dvictim%2Ft"
        assert empty.unkey("app%3Dbilling%2Fuser%3Dvictim%2Ft") == "app=billing/user=victim/t"
        assert empty.key("app%3Dx") == "app%253Dx"
        assert empty.unkey("app%253Dx") == "app%3Dx"
        assert empty.key("apple") == "apple"
        assert empty.unkey("apple") == "apple"

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

    def test_plaintext_is_refused_unless_allowed(self, caplog: pytest.LogCaptureFixture) -> None:
        data = {"iteration": 1, "messages": []}
        with pytest.raises(CheckpointEncryptionError, match="allow_plaintext=True"):
            EncryptedSerializer(KEY).decrypt(data)
        with caplog.at_level("WARNING"):
            assert EncryptedSerializer(KEY, allow_plaintext=True).decrypt(data) == data
        [record] = [r for r in caplog.records if "plaintext" in r.getMessage()]
        assert record.exc_info is None, "one line, no traceback"

    def test_an_envelope_is_bound_to_its_associated_data(self) -> None:
        serializer = EncryptedSerializer(KEY)
        data = _secret_state().with_iteration(3).to_checkpoint()
        aad = associated_data("app=billing/user=u-1/t", "cp-1", 3)
        envelope = serializer.encrypt(data, aad=aad)
        assert serializer.decrypt(envelope, aad=aad) == data
        # Moved to another thread or checkpoint: refused.
        with pytest.raises(CheckpointEncryptionError, match="moved or edited"):
            serializer.decrypt(envelope, aad=associated_data("app=billing/user=u-2/t", "cp-1", 3))
        with pytest.raises(CheckpointEncryptionError):
            serializer.decrypt(envelope, aad=associated_data("app=billing/user=u-1/t", "cp-2", 3))
        # The plaintext iteration is part of it: edited, refused.
        with pytest.raises(CheckpointEncryptionError):
            serializer.decrypt(envelope, aad=associated_data("app=billing/user=u-1/t", "cp-1", 999))
        # Rotation keeps the binding.
        rotated = EncryptedSerializer(secrets.token_bytes(32), previous_keys=[KEY])
        assert rotated.decrypt(envelope, aad=aad) == data

    def test_a_malformed_envelope_is_an_encryption_error(self) -> None:
        serializer = EncryptedSerializer(KEY)
        envelope = serializer.encrypt({"iteration": 1})
        for field, value in (("nonce", "AAAA"), ("nonce", "not base64!"), ("ct", "####")):
            broken = {**envelope, field: value}
            with pytest.raises(CheckpointEncryptionError, match="malformed"):
                serializer.decrypt(broken)
        # Sealed bytes that are not a JSON object are malformed too.
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        for plaintext in (b"not json", b"[1, 2]"):
            nonce = secrets.token_bytes(12)
            sealed = {
                **envelope,
                "nonce": base64.b64encode(nonce).decode(),
                "ct": base64.b64encode(AESGCM(KEY).encrypt(nonce, plaintext, None)).decode(),
            }
            with pytest.raises(CheckpointEncryptionError, match="malformed"):
                serializer.decrypt(sealed)

    def test_repr_does_not_call_the_key_provider(self) -> None:
        calls: list[str] = []

        def provider() -> bytes:
            calls.append("fetch")
            return KEY

        text = repr(EncryptedSerializer(key_provider=provider))
        assert calls == []
        assert "key_provider" in text
        assert KEY.hex() not in text

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
            _Bare()._decode_state(envelope, "t", "cp-0")
        plain = _Bare()._decode_state(AgentState().with_iteration(4).to_checkpoint(), "t", "cp-0")
        assert plain.iteration == 4

    def test_encode_binds_the_envelope_to_thread_checkpoint_and_iteration(self) -> None:
        cp = _Bare()
        cp._configure(encryption=EncryptedSerializer(KEY))
        envelope = cp._encode_state(AgentState().with_iteration(2), "app=a/t", "cp-1")
        assert cp._decode_state(envelope, "app=a/t", "cp-1").iteration == 2
        with pytest.raises(CheckpointEncryptionError):
            cp._decode_state(envelope, "app=b/t", "cp-1")
        with pytest.raises(CheckpointEncryptionError):
            cp._decode_state(envelope, "app=a/t", "cp-2")
        with pytest.raises(CheckpointEncryptionError):
            cp._decode_state({**envelope, "iteration": 99}, "app=a/t", "cp-1")

    def test_decode_refuses_plaintext_with_encryption_unless_allowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plain = AgentState().with_iteration(1).to_checkpoint()
        strict = _Bare()
        strict._configure(encryption=EncryptedSerializer(KEY))
        with pytest.raises(CheckpointEncryptionError, match="plaintext"):
            strict._decode_state(plain, "t", "cp-0")
        migrating = _Bare()
        migrating._configure(encryption=EncryptedSerializer(KEY, allow_plaintext=True))
        with caplog.at_level("WARNING"):
            assert migrating._decode_state(plain, "t", "cp-0").iteration == 1
        [record] = [r for r in caplog.records if "plaintext" in r.getMessage()]
        assert "cp-0" in record.getMessage()
        assert record.exc_info is None

    def test_scoped_passes_a_stored_id_through(self) -> None:
        cp = _Bare()
        cp._configure(scope=CheckpointScope(app_id="a"))
        stored = CheckpointScope(app_id="b").key("t")
        assert cp._scoped(stored) == "app=b/t", "a stored id is not keyed again"
        assert cp._scoped("t") == "app=a/t"
        assert cp.with_scope(None) is cp
        view = cp.with_scope(CheckpointScope(app_id="b"))
        assert isinstance(view, ScopedCheckpointer)
        assert view.scope == CheckpointScope(app_id="b")
        assert cp.with_scope(CheckpointScope()).scope is None, "the empty scope is unscoped"

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

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "app%3Dbilling%2Fuser%3Du-1%2Ft",
            "t",
        ]
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


# ---------------------------------------------------------------------------
# The scope grammar, on the backends
# ---------------------------------------------------------------------------


def _memory_store() -> Any:
    shared: dict[str, Any] = {}

    def make(**settings: Any) -> MemoryCheckpointer:
        cp = MemoryCheckpointer(**settings)
        cp._storage = shared
        return cp

    return make


def _adapter_store() -> Any:
    store = _DictStore()
    return lambda **settings: StorageBackendAdapter(store, **settings)


def _file_store(tmp_path: Path) -> Any:
    return lambda **settings: FileCheckpointer(tmp_path / "grammar", **settings)


@pytest.mark.parametrize("make_store", [_memory_store, _adapter_store, _file_store])
class TestScopeGrammarOnBackends:
    """No thread id reaches another scope's thread, on a dict, the adapter and files."""

    @pytest.fixture
    def make(self, make_store: Any, tmp_path: Path) -> Any:
        return make_store(tmp_path) if make_store is _file_store else make_store()

    async def test_an_app_scope_cannot_read_a_user_thread_through_its_id(self, make: Any) -> None:
        victim = make(scope=CheckpointScope(app_id="billing", user_id="victim"))
        await victim.save(_secret_state(), "t", "cp-1")
        app_only = make(scope=CheckpointScope(app_id="billing"))
        assert await app_only.load("user=victim/t") is None
        assert await app_only.list_checkpoints("user=victim/t") == []
        assert await app_only.exists("user=victim/t") is False
        # Written by the app scope, the lookalike is the app's own thread, not the victim's.
        await app_only.save(_state(0), "user=victim/t", "cp-2")
        assert await victim.list_checkpoints("t") == ["cp-1"]
        loaded = await app_only.load("user=victim/t")
        assert loaded is not None
        assert loaded.iteration == 0

    async def test_an_app_scope_lists_only_app_level_threads(self, make: Any) -> None:
        await make(scope=CheckpointScope(app_id="billing", user_id="victim")).save(_state(0), "t")
        app_only = make(scope=CheckpointScope(app_id="billing"))
        await app_only.save(_state(0), "shared")
        assert {t.split(":", 1)[0] for t in await app_only.list_threads()} == {"shared"}
        assert {
            t.split(":", 1)[0]
            for t in await app_only.list_threads(
                scope=CheckpointScope(app_id="billing", user_id="victim")
            )
        } == {"t"}

    async def test_an_unscoped_instance_never_resolves_a_scoped_id(self, make: Any) -> None:
        await make(scope=CheckpointScope(app_id="billing", user_id="victim")).save(
            _secret_state(), "t", "cp-1"
        )
        unscoped = make()
        assert await unscoped.load("app=billing/user=victim/t") is None
        assert await unscoped.list_checkpoints("app=billing/user=victim/t") == []
        assert await unscoped.list_threads() == []
        # The lookalike id is a thread of its own in the unscoped namespace and round-trips.
        await unscoped.save(_state(2), "app=billing/user=victim/t", "cp-x")
        loaded = await unscoped.load("app=billing/user=victim/t")
        assert loaded is not None
        assert loaded.iteration == 2
        assert {t.split(":", 1)[0] for t in await unscoped.list_threads()} == {
            "app=billing/user=victim/t"
        }
        victim = make(scope=CheckpointScope(app_id="billing", user_id="victim"))
        assert await victim.list_checkpoints("t") == ["cp-1"], "the victim's thread is untouched"


# ---------------------------------------------------------------------------
# File backend: collision-free paths, header-only expiry
# ---------------------------------------------------------------------------


class TestFilePaths:
    async def test_distinct_user_ids_get_distinct_directories(self, tmp_path: Path) -> None:
        dot = FileCheckpointer(
            tmp_path, scope=CheckpointScope(app_id="billing", user_id="ann.smith")
        )
        underscore = FileCheckpointer(
            tmp_path, scope=CheckpointScope(app_id="billing", user_id="ann_smith")
        )
        await dot.save(_secret_state(), "t", "cp-a")
        assert await underscore.load("t") is None
        assert await underscore.list_checkpoints("t") == []
        await underscore.save(_state(0), "t", "cp-b")
        assert len(list(tmp_path.iterdir())) == 2
        assert await dot.list_checkpoints("t") == ["cp-a"]
        at, colon = (
            FileCheckpointer(tmp_path, scope=CheckpointScope(user_id="u@1")),
            FileCheckpointer(tmp_path, scope=CheckpointScope(user_id="u:1")),
        )
        await at.save(_state(0), "x", "cp-c")
        assert await colon.load("x") is None

    async def test_an_unscoped_literal_id_cannot_alias_a_scoped_thread(
        self, tmp_path: Path
    ) -> None:
        scoped = FileCheckpointer(
            tmp_path, scope=CheckpointScope(app_id="billing", user_id="ann_smith")
        )
        await scoped.save(_secret_state(), "t", "cp-a")
        plain = FileCheckpointer(tmp_path)
        assert await plain.load("app_billing_user_ann_smith_t") is None
        assert await plain.load("app%3Dbilling%2Fuser%3Dann_smith%2Ft") is None
        assert await plain.list_threads() == []

    async def test_a_thread_written_under_the_old_lossy_name_is_still_found(
        self, tmp_path: Path
    ) -> None:
        cp = FileCheckpointer(tmp_path)
        await cp.save(_state(1), "a#1", "cp-0")
        new_dir = tmp_path / "a%231"
        assert new_dir.is_dir()
        # As an earlier release laid it out: the directory named ``a_1``.
        new_dir.rename(tmp_path / "a_1")
        loaded = await cp.load("a#1")
        assert loaded is not None
        assert loaded.iteration == 1
        assert await cp.list_threads() == ["a#1"]
        await cp.save(_state(2), "a#1", "cp-1")
        assert not new_dir.exists(), "the thread stays in the directory that records it"
        assert await cp.list_checkpoints("a#1") == ["cp-1", "cp-0"]
        # A different id that sanitised to the same name is not read from it.
        assert await cp.load("a_1") is None
        assert await cp.load("a%1") is None

    async def test_metadata_mentioning_expires_at_is_not_an_expiry(self, tmp_path: Path) -> None:
        cp = FileCheckpointer(tmp_path)
        await cp.save(_state(0), "t", "cp-1", metadata={"expires_at": "2000-01-01T00:00:00+00:00"})
        await cp.save(
            _state(1),
            "t",
            "cp-2",
            metadata={"token_expires_at": "x", "expires_at": "2099-01-01T00:00:00+00:00"},
        )
        token = json.dumps({"access_token": "abc", "expires_at": "2001-01-01T00:00:00+00:00"})
        await cp.save(AgentState(agent_id="p").with_message(Message.user(token)), "u", "cp-9")
        # A file from before the header carried ``expires_at`` at all.
        path = tmp_path / "u" / "cp-9.json"
        data = json.loads(path.read_text())
        data.pop("expires_at")
        path.write_text(json.dumps(data, indent=2))

        assert await cp.list_checkpoints("t") == ["cp-2", "cp-1"]
        assert [i.checkpoint_id for i in await cp.list("t")] == ["cp-2", "cp-1"]
        assert await cp.load("t", "cp-1") is not None
        assert await cp.load("u") is not None
        assert await cp.list_threads() == ["u", "t"]
        assert await cp.expire() == 0
        assert sorted(p.name for p in (tmp_path / "t").glob("*.json")) == ["cp-1.json", "cp-2.json"]
        assert (tmp_path / "u" / "cp-9.json").exists()
        # The header's own field still expires a file.
        expiring = FileCheckpointer(tmp_path, ttl=0.05)
        await expiring.save(_state(0), "v", "cp-0", metadata={"expires_at": "2099-01-01"})
        await asyncio.sleep(0.1)
        assert await cp.load("v") is None
        assert await cp.expire() == 1


class TestFileHeaderReads:
    """The head-of-file readers: the recorded thread id, the expiry, and files that are not there."""

    async def test_readers_honour_the_recorded_thread_and_the_expiry(self, tmp_path: Path) -> None:
        import shutil

        cp = FileCheckpointer(tmp_path)
        await cp.save(_state(0), "a#1", "cp-0")
        path = tmp_path / "a%231" / "cp-0.json"
        assert cp._read_header(path, "a#1") is not None
        assert cp._read_header(path, "a_1") is None, "recorded as a#1, so not a_1's"
        assert cp._read_info(path, "a#1", "a#1") is not None
        assert cp._read_info(path, "a_1", "a_1") is None
        missing = tmp_path / "missing.json"
        assert cp._read_header(missing) is None
        assert cp._read_info(missing, "t") is None
        assert cp._file_expired(missing) is False

        # An older layout: no step fields ahead of the state, and no expiry field at all.
        data = json.loads(path.read_text())
        for field in ("iteration", "size", "expires_at"):
            data.pop(field)
        old = tmp_path / "a%231" / "old.json"
        old.write_text(json.dumps(data))
        assert cp._read_header(old, "a#1") == ("cp-0", data["created_at"])
        assert cp._read_header(old, "a_1") is None
        assert cp._read_info(old, "a#1", "a#1") is not None
        assert cp._read_info(old, "a_1", "a_1") is None
        assert cp._file_expired(old) is False

        # The same layout with a top-level expiry that has passed: found by the full parse.
        expired = tmp_path / "a%231" / "expired.json"
        expired.write_text(json.dumps({**data, "expires_at": "2000-01-01T00:00:00+00:00"}))
        assert cp._read_header(expired, "a#1") is None
        assert cp._read_info(expired, "a#1", "a#1") is None
        assert cp._file_expired(expired) is True

        # A file recording another thread, found under this thread's directory.
        (tmp_path / "x").mkdir()
        shutil.copy(path, tmp_path / "x" / "cp-0.json")
        assert await cp.load("x", "cp-0") is None
        assert await cp.list_checkpoints("x") == []


def test_parameter_default_of_an_unsignaturable_callable_is_none() -> None:
    from locus.memory.backends.adapters import _parameter_default

    assert _parameter_default(object(), "pattern") is None
    assert _parameter_default(lambda pattern="%": None, "pattern") == "%"
    assert _parameter_default(lambda pattern="%": None, "missing") is None


class _Recording(BaseCheckpointer):
    """Every surface method, recording what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    @property
    def capabilities(self) -> Any:
        from locus.core.protocols import CheckpointerCapabilities

        return CheckpointerCapabilities(
            search=True,
            metadata_query=True,
            vacuum=True,
            branching=True,
            list_threads=True,
            list_with_metadata=True,
        )

    async def save(
        self, state: Any, thread_id: str, checkpoint_id: Any = None, metadata: Any = None
    ) -> str:
        self.calls.append(("save", (thread_id, checkpoint_id)))
        return checkpoint_id or "x"

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        self.calls.append(("load", (thread_id, checkpoint_id)))
        return None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        self.calls.append(("list_checkpoints", (thread_id,)))
        return []

    async def close(self) -> None:
        self.calls.append(("close", ()))

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self.calls.append(("search", (query, limit)))
        return []

    async def query_by_metadata(
        self, key: str, value: Any, limit: int = 100
    ) -> list[dict[str, Any]]:
        self.calls.append(("query_by_metadata", (key, value, limit)))
        return []

    async def get_metadata(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        self.calls.append(("get_metadata", (thread_id, checkpoint_id)))
        return {}

    async def vacuum(self, older_than_days: int = 30) -> int:
        self.calls.append(("vacuum", (older_than_days,)))
        return 0

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        self.calls.append(("copy_thread", (source_thread_id, dest_thread_id)))
        return True

    async def expire(self) -> int:
        self.calls.append(("expire", ()))
        return 0

    async def list_with_metadata(self, limit: int = 100) -> list[dict[str, Any]]:
        self.calls.append(("list_with_metadata", (limit,)))
        return []


class TestScopedViewForwarding:
    async def test_every_surface_method_forwards_with_the_stored_id(self) -> None:
        inner = _Recording()
        view = inner.with_scope(CheckpointScope(app_id="a"))
        assert view.capabilities.search
        await view.search("q", limit=3)
        await view.query_by_metadata("k", "v", limit=4)
        assert await view.get_metadata("t", "cp-0") == {}
        assert await view.vacuum(7) == 0
        assert await view.copy_thread("t", "u")
        assert await view.expire() == 0
        assert await view.list_with_metadata(limit=9) == []
        assert await view.exists("t") is False, "the base default over the view's listing"
        await view.close()
        assert inner.calls == [
            ("search", ("q", 3)),
            ("query_by_metadata", ("k", "v", 4)),
            ("get_metadata", ("app=a/t", "cp-0")),
            ("vacuum", (7,)),
            ("copy_thread", ("app=a/t", "app=a/u")),
            ("expire", ()),
            ("list_with_metadata", (9,)),
            ("list_checkpoints", ("app=a/t",)),
            ("close", ()),
        ]
        with CheckpointScope(app_id="block").active():
            assert view._effective_scope() == CheckpointScope(app_id="a"), "pinned"
        other = view.with_scope(CheckpointScope(app_id="b"))
        assert isinstance(other, ScopedCheckpointer)
        assert other.inner is inner
        with pytest.raises(TypeError, match="CheckpointScope"):
            ScopedCheckpointer(inner, "a")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adapter: exists() and load() against expired documents
# ---------------------------------------------------------------------------


class TestAdapterExpiredDocuments:
    async def test_exists_filters_an_expired_document_without_a_ttl(self) -> None:
        store = _DictStore()
        await StorageBackendAdapter(store, ttl=0.05).save(_state(0), "s", "cp-1")
        plain = StorageBackendAdapter(store)
        assert await plain.exists("s", "cp-1")
        assert await plain.exists("s")
        await asyncio.sleep(0.1)
        assert "s:cp-1" in store.rows, "the store still holds it"
        assert await plain.exists("s", "cp-1") is False
        assert await plain.exists("s") is False
        assert await plain.list_checkpoints("s") == []

    async def test_load_falls_back_to_the_newest_live_checkpoint_when_latest_expired(
        self,
    ) -> None:
        store = _DictStore()
        long_lived = StorageBackendAdapter(store, ttl=600)
        short_lived = StorageBackendAdapter(store, ttl=0.05)
        await long_lived.save(_state(1), "m", "cp-1")
        await asyncio.sleep(0.002)
        await short_lived.save(_state(2), "m", "cp-2")
        latest = await long_lived.load("m")
        assert latest is not None
        assert latest.iteration == 2
        await asyncio.sleep(0.1)
        assert await long_lived.list_checkpoints("m") == ["cp-1"]
        fallen_back = await long_lived.load("m")
        assert fallen_back is not None
        assert fallen_back.iteration == 1, "the newest live checkpoint, not None"
        assert await long_lived.exists("m")
        assert await StorageBackendAdapter(store).load("m") is not None


# ---------------------------------------------------------------------------
# The run's scope is explicit: a view, never a context variable
# ---------------------------------------------------------------------------


class TestScopedRuns:
    async def test_interleaved_scoped_runs_keep_their_own_scope(self) -> None:
        from tests.unit.checkpointer_time_travel_conformance import _agent

        cp = MemoryCheckpointer()
        agent = _agent(cp)
        a, b = CheckpointScope(app_id="A"), CheckpointScope(app_id="B")
        first = agent.run("hello from A", thread_id="t", scope=a)
        second = agent.run("hello from B", thread_id="t", scope=b)
        await anext(first)
        await anext(second)
        async for _ in first:
            pass
        async for _ in second:
            pass
        assert sorted(cp.get_thread_ids()) == ["app=A/t", "app=B/t"]
        assert current_scope() is None, "the context is untouched"
        loaded_a, loaded_b = await cp.with_scope(a).load("t"), await cp.with_scope(b).load("t")
        assert loaded_a is not None
        assert loaded_b is not None
        assert str(loaded_a.messages[-2].content) == "hello from A"
        assert str(loaded_b.messages[-2].content) == "hello from B"

    async def test_run_scope_none_inherits_and_the_empty_scope_is_unscoped(self) -> None:
        from tests.unit.checkpointer_time_travel_conformance import _agent

        instance = CheckpointScope(app_id="inst")
        cp = MemoryCheckpointer(scope=instance)
        agent = _agent(cp)
        await agent.arun("inherited", thread_id="t")
        await agent.arun("unscoped", thread_id="t", scope=CheckpointScope())
        assert sorted(cp.get_thread_ids()) == ["app=inst/t", "t"]
        inherited = await cp.load("t")
        assert inherited is not None
        assert str(inherited.messages[-2].content) == "inherited"
        unscoped = await cp.with_scope(CheckpointScope()).load("t")
        assert unscoped is not None
        assert str(unscoped.messages[-2].content) == "unscoped"
        assert await MemoryCheckpointer().list_threads() == []

    async def test_an_active_block_is_resolved_once_when_the_run_starts(self) -> None:
        from tests.unit.checkpointer_time_travel_conformance import _agent

        cp = MemoryCheckpointer()
        agent = _agent(cp)
        with CheckpointScope(app_id="block").active():
            stream = agent.run("inside", thread_id="t")
            await anext(stream)
        # Consumed outside the block: the run keeps the scope it started under.
        async for _ in stream:
            pass
        assert cp.get_thread_ids() == ["app=block/t"]
        assert current_scope() is None

    async def test_resume_continues_under_the_paused_run_s_scope(self) -> None:
        from tests.unit.checkpointer_time_travel_conformance import _agent

        cp = MemoryCheckpointer()
        agent = _agent(cp, asks=True)
        scope = CheckpointScope(app_id="billing", user_id="u-1")
        paused = [e async for e in agent.run("book it", thread_id="t", scope=scope)]
        assert paused
        assert cp.get_thread_ids() == ["app=billing/user=u-1/t"]
        resumed = [e async for e in agent.resume("option b", thread_id="t")]
        assert resumed
        assert cp.get_thread_ids() == ["app=billing/user=u-1/t"], "no unscoped copy"
        state = await cp.with_scope(scope).load("t")
        assert state is not None
        assert state.messages[-1].role.value == "assistant"

    async def test_the_view_keys_every_call_and_reports_the_caller_s_thread(self) -> None:
        inner = MemoryCheckpointer(scope=CheckpointScope(app_id="inst"))
        view = inner.with_scope(CheckpointScope(app_id="run", user_id="u"))
        await view.save(_state(0), "t", "cp-0", metadata={"n": 0})
        await asyncio.sleep(0.002)
        await view.save(_state(1), "t", "cp-1")
        assert inner.get_thread_ids() == ["app=run/user=u/t"]
        assert await view.list_checkpoints("t") == ["cp-1", "cp-0"]
        assert [(i.thread_id, i.checkpoint_id) for i in await view.list("t")] == [
            ("t", "cp-1"),
            ("t", "cp-0"),
        ]
        assert await view.exists("t", "cp-0")
        assert await view.list_threads() == ["t"]
        assert await inner.list_threads() == []
        forked = await view.fork("t", "cp-0", new_thread_id="t-alt")
        assert forked == "t-alt"
        assert "app=run/user=u/t-alt" in inner.get_thread_ids()
        assert (view.ttl, view.encryption, view.capabilities) == (
            inner.ttl,
            inner.encryption,
            inner.capabilities,
        )
        assert view.deletes_single_checkpoints is inner.deletes_single_checkpoints
        assert await view.delete("t", "cp-1")
        assert await view.list_checkpoints("t") == ["cp-0"]
        assert view.with_scope(None) is inner
        assert "ScopedCheckpointer" in repr(view)
        assert "app=run/user=u" in repr(view)

    async def test_a_retained_wrapper_is_viewed_through_its_inner(self) -> None:
        inner = MemoryCheckpointer()
        view = scoped_checkpointer(RetainedCheckpointer(inner), CheckpointScope(app_id="a"))
        await view.save(_state(0), "t", "cp-0")
        assert inner.get_thread_ids() == ["app=a/t"]
        assert scoped_checkpointer(inner, None) is inner

    async def test_a_duck_typed_checkpointer_receives_the_stored_id(self) -> None:
        class _Duck:
            def __init__(self) -> None:
                self.saved: list[str] = []

            async def save(self, state: Any, thread_id: str, *_: Any) -> str:
                self.saved.append(thread_id)
                return "x"

            async def load(self, thread_id: str, checkpoint_id: Any = None) -> Any:
                return None

            async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
                return []

        duck = _Duck()
        view = scoped_checkpointer(duck, CheckpointScope(app_id="a"))
        assert isinstance(view, ScopedCheckpointer)
        await view.save(AgentState(), "t")
        assert duck.saved == ["app=a/t"]
        assert await view.exists("t") is False


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

    async def test_load_falls_back_when_latest_names_an_expired_checkpoint(self) -> None:
        long_lived = self._backend(ttl=600)
        short_lived = self._backend(ttl=0.05)
        await long_lived.save(_state(1), "t", "cp-1")
        await asyncio.sleep(0.002)
        await short_lived.save(_state(2), "t", "cp-2")
        await asyncio.sleep(0.1)
        assert await long_lived.list_checkpoints("t") == ["cp-1"]
        loaded = await long_lived.load("t")
        assert loaded is not None
        assert loaded.iteration == 1, "the newest live checkpoint, though _latest names cp-2"
        plain = self._backend()
        fallen_back = await plain.load("t")
        assert fallen_back is not None
        assert fallen_back.iteration == 1

    async def test_copy_thread_reseals_encrypted_checkpoints_under_the_new_thread(self) -> None:
        cp = self._backend(encryption=EncryptedSerializer(KEY))
        await cp.save(_secret_state().with_iteration(1), "t", "cp-0", metadata={"k": "v"})
        assert await cp.copy_thread("t", "t-copy")
        copied = await cp.load("t-copy")
        assert copied is not None
        assert copied.iteration == 1
        assert _contains_secret(str(copied.messages[0].content))
        assert await cp.get_metadata("t-copy", "cp-0") == {"k": "v"}
        # Bytes moved by hand do not load: the envelope is bound to its thread.
        client = cp._get_client()
        body = client.get_object(Bucket="locus-prod", Key="locus/checkpoints/t/cp-0.json")["Body"]
        client.put_object(
            Bucket="locus-prod", Key="locus/checkpoints/moved/cp-0.json", Body=body.read()
        )
        with pytest.raises(CheckpointEncryptionError):
            await cp.load("moved", "cp-0")


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

    async def test_load_falls_back_when_latest_names_an_expired_checkpoint(self) -> None:
        client = _FakeObjectStorage()
        long_lived = self._backend(client, ttl=600)
        short_lived = self._backend(client, ttl=0.05)
        await long_lived.save(_state(1), "t", "cp-1")
        await asyncio.sleep(0.002)
        await short_lived.save(_state(2), "t", "cp-2")
        await asyncio.sleep(0.1)
        loaded = await long_lived.load("t")
        assert loaded is not None
        assert loaded.iteration == 1
        plain = self._backend(client)
        fallen_back = await plain.load("t")
        assert fallen_back is not None
        assert fallen_back.iteration == 1

    async def test_copy_thread_reseals_encrypted_checkpoints_under_the_new_thread(self) -> None:
        client = _FakeObjectStorage()
        cp = self._backend(client, encryption=EncryptedSerializer(KEY))
        await cp.save(_secret_state().with_iteration(1), "t", "cp-0")
        assert await cp.copy_thread("t", "t-copy")
        copied = await cp.load("t-copy")
        assert copied is not None
        assert _contains_secret(str(copied.messages[0].content))
        client.objects["locus/checkpoints/moved/cp-0.json"] = client.objects[
            "locus/checkpoints/t/cp-0.json"
        ]
        with pytest.raises(CheckpointEncryptionError):
            await cp.load("moved", "cp-0")


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


def test_active_resets_with_its_own_token_and_never_clears_another_block() -> None:
    """Leaving a block in a context that did not enter it raises; the caller's scope is kept."""
    import contextvars

    scope = CheckpointScope(app_id="billing")
    block = scope.active()
    contextvars.copy_context().run(block.__enter__)
    mine = CheckpointScope(app_id="mine")
    with mine.active():
        with pytest.raises(ValueError, match="different Context"):
            block.__exit__(None, None, None)  # the token belongs to the other context
        assert current_scope() == mine, "never cleared to None under the caller"
    assert current_scope() is None
