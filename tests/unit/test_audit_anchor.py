# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Anchoring the chain head in object storage and checking the chain against it."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.control import (
    Anchor,
    AnchorError,
    AnchorStore,
    AuditTrail,
    OCIBucketAnchorStore,
    S3AnchorStore,
    anchor_periodically,
    anchor_trail,
    latest_anchor,
    verify_anchor,
)
from locus.control.audit import _GENESIS
from locus.control.audit_anchor import DEFAULT_PREFIX


class _MemoryStore:
    """An ``AnchorStore`` in a dict, with configurable retention rules."""

    def __init__(self, rules: list[str] | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.rules = rules or []
        self.puts = 0

    def put(self, name: str, body: bytes) -> None:
        self.objects[name] = body
        self.puts += 1

    def get(self, name: str) -> bytes | None:
        return self.objects.get(name)

    def list_names(self, prefix: str) -> list[str]:
        return sorted(n for n in self.objects if n.startswith(prefix))

    def retention_rules(self) -> list[str]:
        return list(self.rules)


def _trail(n: int) -> AuditTrail:
    trail = AuditTrail(clock=lambda: "2026-11-16T00:00:00+00:00")
    for i in range(n):
        trail.record("gate", {"n": i})
    return trail


def test_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(_MemoryStore(), AnchorStore)


# ---------------------------------------------------------------------------
# anchor_trail
# ---------------------------------------------------------------------------


def test_anchor_writes_the_head_as_json() -> None:
    trail = _trail(3)
    store = _MemoryStore()
    anchor = anchor_trail(trail, store)
    assert anchor is not None
    assert anchor.name == f"{DEFAULT_PREFIX}00000000000000000003-{trail.head[:16]}.json"
    body = json.loads(store.objects[anchor.name])
    assert body == {"records": 3, "seq": 2, "head": trail.head, "ts": anchor.ts, "version": 1}
    assert latest_anchor(store) == anchor


def test_anchor_of_an_empty_trail_is_the_genesis() -> None:
    anchor = anchor_trail(AuditTrail(), _MemoryStore())
    assert anchor is not None
    assert (anchor.records, anchor.seq, anchor.head) == (0, -1, _GENESIS)


def test_anchor_of_an_empty_chain_verifies_after_records_arrive() -> None:
    trail = AuditTrail()
    store = _MemoryStore()
    anchor_trail(trail, store)
    trail.record("gate", {})
    assert verify_anchor(trail, store).ok


def test_unchanged_head_is_not_written_twice() -> None:
    trail = _trail(2)
    store = _MemoryStore()
    first = anchor_trail(trail, store)
    again = anchor_trail(trail, store)
    assert again == first
    assert store.puts == 1


def test_every_gates_on_growth_since_the_latest_anchor() -> None:
    trail = _trail(2)
    store = _MemoryStore()
    assert anchor_trail(trail, store, every=5) is not None  # the first anchor always lands
    trail.record("gate", {"n": 2})
    assert anchor_trail(trail, store, every=5) is None
    for i in range(4):
        trail.record("gate", {"n": 3 + i})
    anchor = anchor_trail(trail, store, every=5)
    assert anchor is not None
    assert anchor.records == 7
    assert store.puts == 2


def test_latest_anchor_is_the_greatest_name() -> None:
    trail = _trail(1)
    store = _MemoryStore()
    anchor_trail(trail, store)
    for i in range(1, 12):
        trail.record("gate", {"n": i})
        anchor_trail(trail, store)
    latest = latest_anchor(store)
    assert latest is not None
    assert latest.records == 12
    assert len(store.objects) == 12


def test_custom_prefix() -> None:
    trail = _trail(1)
    store = _MemoryStore()
    anchor = anchor_trail(trail, store, prefix="anchors/agent-7/")
    assert anchor is not None
    assert anchor.name.startswith("anchors/agent-7/")
    assert latest_anchor(store) is None
    assert latest_anchor(store, prefix="anchors/agent-7/") == anchor


def test_refuses_a_bucket_without_a_retention_rule() -> None:
    store = _MemoryStore()
    with pytest.raises(AnchorError, match="no retention rule"):
        anchor_trail(_trail(1), store, require_retention_rule=True)
    assert store.puts == 0


def test_anchors_into_a_bucket_with_a_retention_rule() -> None:
    store = _MemoryStore(rules=["keep: 30 DAYS (locked)"])
    assert anchor_trail(_trail(1), store, require_retention_rule=True) is not None


# ---------------------------------------------------------------------------
# verify_anchor
# ---------------------------------------------------------------------------


def test_verify_passes_and_keeps_passing_as_the_trail_grows() -> None:
    trail = _trail(3)
    store = _MemoryStore()
    anchor = anchor_trail(trail, store)
    report = verify_anchor(trail, store)
    assert report.ok
    assert report.anchor == anchor
    trail.record("gate", {"n": 3})
    assert verify_anchor(trail, store).ok


def test_verify_without_an_anchor_fails() -> None:
    report = verify_anchor(_trail(1), _MemoryStore())
    assert not report.ok
    assert report.anchor is None
    assert "no anchor" in report.problem


def test_truncation_below_the_anchor_is_caught() -> None:
    trail = _trail(4)
    store = _MemoryStore()
    anchor_trail(trail, store)
    shorter = AuditTrail.from_records(trail.records()[:2])
    assert shorter.verify()  # the chain alone cannot see it
    report = verify_anchor(shorter, store)
    assert not report.ok
    assert "cut off" in report.problem


def test_rebuilt_chain_is_caught() -> None:
    trail = _trail(3)
    store = _MemoryStore()
    anchor_trail(trail, store)
    rebuilt = AuditTrail(clock=lambda: "2026-11-16T00:00:00+00:00")
    for i in range(3):
        rebuilt.record("gate", {"n": i, "edited": True})
    assert rebuilt.verify()
    report = verify_anchor(rebuilt, store)
    assert not report.ok
    assert "rebuilt" in report.problem


def test_edited_record_is_reported_from_the_chain_check() -> None:
    trail = _trail(3)
    store = _MemoryStore()
    anchor_trail(trail, store)
    records = trail.records()
    bad = records[1]
    records[1] = type(bad)(**{**bad.__dict__, "payload": {"n": 99}})
    report = verify_anchor(AuditTrail.from_records(records), store)
    assert not report.ok
    assert "changed after it was written" in report.problem


def test_verify_forwards_keys() -> None:
    pytest.importorskip("cryptography")
    from locus.control import Ed25519Signer  # noqa: PLC0415

    signer = Ed25519Signer.generate(key_id="k1")
    trail = AuditTrail(signer=signer)
    trail.record("gate", {})
    store = _MemoryStore()
    anchor_trail(trail, store)
    assert verify_anchor(trail, store, keys={"k1": signer.public_key_pem()}).ok
    assert not verify_anchor(trail, store, keys={"k2": signer.public_key_pem()}).ok


def test_anchor_json_round_trip() -> None:
    anchor = Anchor(records=5, seq=4, head="ab" * 32, ts="2026-11-16T00:00:00+00:00")
    assert Anchor.from_json(anchor.to_json(), name="x") == Anchor(
        records=5, seq=4, head="ab" * 32, ts="2026-11-16T00:00:00+00:00", name="x"
    )


# ---------------------------------------------------------------------------
# anchor_periodically
# ---------------------------------------------------------------------------


async def test_periodic_anchoring_writes_on_the_interval_and_on_stop() -> None:
    trail = _trail(1)
    store = _MemoryStore()
    stop = asyncio.Event()
    task = asyncio.create_task(
        anchor_periodically(trail, store, interval_seconds=0.01, every=2, stop=stop)
    )
    for _ in range(20):
        await asyncio.sleep(0.01)
        if store.puts:
            break
    assert store.puts == 1  # the first anchor
    trail.record("gate", {"n": 1})  # one new record: below `every`
    await asyncio.sleep(0.05)
    assert store.puts == 1
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    latest = latest_anchor(store)
    assert latest is not None
    assert latest.records == 2  # `stop` anchors the final head regardless of `every`


async def test_periodic_anchoring_surfaces_a_refusal() -> None:
    store = _MemoryStore()
    with pytest.raises(AnchorError):
        await anchor_periodically(
            _trail(1), store, interval_seconds=0.001, require_retention_rule=True
        )


# ---------------------------------------------------------------------------
# OCI Object Storage store (client double)
# ---------------------------------------------------------------------------


def _oci_store(client: MagicMock) -> OCIBucketAnchorStore:
    return OCIBucketAnchorStore("anchors", "ns", retry_strategy="RETRY", _client=client)


def test_oci_put_and_get() -> None:
    client = MagicMock()
    client.get_object.return_value.data.content = b'{"a":1}'
    store = _oci_store(client)
    store.put("k", b'{"a":1}')
    kwargs = client.put_object.call_args.kwargs
    assert kwargs["namespace_name"] == "ns"
    assert kwargs["bucket_name"] == "anchors"
    assert kwargs["object_name"] == "k"
    assert kwargs["put_object_body"] == b'{"a":1}'
    assert kwargs["retry_strategy"] == "RETRY"
    assert store.get("k") == b'{"a":1}'


def test_oci_get_missing_object_is_none() -> None:
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("ObjectNotFound: 404")
    assert _oci_store(client).get("k") is None
    client.get_object.side_effect = RuntimeError("NotAuthenticated")
    with pytest.raises(RuntimeError, match="NotAuthenticated"):
        _oci_store(client).get("k")


def test_oci_list_follows_pagination() -> None:
    client = MagicMock()
    page1 = MagicMock()
    page1.data.objects = [MagicMock(name="o1"), MagicMock(name="o2")]
    page1.data.objects[0].name = "p/a"
    page1.data.objects[1].name = "p/b"
    page1.data.next_start_with = "p/b"
    page2 = MagicMock()
    page2.data.objects = [MagicMock()]
    page2.data.objects[0].name = "p/c"
    page2.data.next_start_with = None
    client.list_objects.side_effect = [page1, page2]
    assert _oci_store(client).list_names("p/") == ["p/a", "p/b", "p/c"]
    assert client.list_objects.call_args_list[1].kwargs["start"] == "p/b"


def test_oci_retention_rules_are_described() -> None:
    client = MagicMock()
    timed = MagicMock()
    timed.display_name = "keep"
    timed.duration.time_amount = 30
    timed.duration.time_unit = "DAYS"
    timed.time_rule_locked = "2026-12-01T00:00:00Z"
    forever = MagicMock()
    forever.display_name = "forever"
    forever.duration = None
    forever.time_rule_locked = None
    client.list_retention_rules.return_value.data.items = [timed, forever]
    assert _oci_store(client).retention_rules() == ["keep: 30 DAYS (locked)", "forever: indefinite"]
    client.list_retention_rules.return_value.data.items = []
    assert _oci_store(client).retention_rules() == []


def test_oci_end_to_end_with_the_double() -> None:
    objects: dict[str, bytes] = {}
    client = MagicMock()

    def put_object(**kw: Any) -> None:
        objects[kw["object_name"]] = kw["put_object_body"]

    def get_object(**kw: Any) -> Any:
        if kw["object_name"] not in objects:
            raise RuntimeError("ObjectNotFound")
        resp = MagicMock()
        resp.data.content = objects[kw["object_name"]]
        return resp

    def list_objects(**kw: Any) -> Any:
        resp = MagicMock()
        resp.data.objects = []
        for name in sorted(objects):
            if name.startswith(kw["prefix"]):
                o = MagicMock()
                o.name = name
                resp.data.objects.append(o)
        resp.data.next_start_with = None
        return resp

    client.put_object.side_effect = put_object
    client.get_object.side_effect = get_object
    client.list_objects.side_effect = list_objects
    client.list_retention_rules.return_value.data.items = []
    store = _oci_store(client)
    trail = _trail(2)
    with pytest.raises(AnchorError):
        anchor_trail(trail, store, require_retention_rule=True)
    anchor = anchor_trail(trail, store)
    assert anchor is not None
    assert verify_anchor(trail, store).ok
    assert not verify_anchor(AuditTrail.from_records(trail.records()[:1]), store).ok


def test_oci_store_builds_a_client_from_the_bucket_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.memory.backends import oci_bucket  # noqa: PLC0415

    seen: list[Any] = []
    monkeypatch.setattr(
        oci_bucket, "build_object_storage_client", lambda cfg: seen.append(cfg) or MagicMock()
    )
    store = OCIBucketAnchorStore(
        "b", "ns", profile_name="P", auth_type="security_token", region="r"
    )
    store._get_client()
    assert seen[0].bucket_name == "b"
    assert seen[0].namespace == "ns"
    assert seen[0].profile_name == "P"
    assert seen[0].auth_type == "security_token"
    assert seen[0].region == "r"


# ---------------------------------------------------------------------------
# S3-compatible store
# ---------------------------------------------------------------------------


def test_s3_store_against_object_lock() -> None:
    boto3 = pytest.importorskip("boto3")
    mock_aws = pytest.importorskip("moto").mock_aws
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="plain")
        client.create_bucket(Bucket="locked", ObjectLockEnabledForBucket=True)
        client.put_object_lock_configuration(
            Bucket="locked",
            ObjectLockConfiguration={
                "ObjectLockEnabled": "Enabled",
                "Rule": {"DefaultRetention": {"Mode": "COMPLIANCE", "Days": 1}},
            },
        )
        plain = S3AnchorStore("plain", _client=client)
        locked = S3AnchorStore("locked", _client=client)
        assert plain.retention_rules() == []
        assert locked.retention_rules() == ["object-lock: COMPLIANCE 1 DAYS"]

        trail = _trail(3)
        with pytest.raises(AnchorError):
            anchor_trail(trail, plain, require_retention_rule=True)
        anchor = anchor_trail(trail, locked, require_retention_rule=True)
        assert anchor is not None
        assert locked.get(anchor.name) == anchor.to_json()
        assert locked.get("missing") is None
        assert locked.list_names(DEFAULT_PREFIX) == [anchor.name]
        assert verify_anchor(trail, locked).ok
        assert not verify_anchor(AuditTrail.from_records(trail.records()[:2]), locked).ok


def test_s3_list_follows_continuation_tokens() -> None:
    client = MagicMock()
    client.list_objects_v2.side_effect = [
        {"Contents": [{"Key": "p/a"}], "IsTruncated": True, "NextContinuationToken": "t"},
        {"Contents": [{"Key": "p/b"}], "IsTruncated": False},
    ]
    assert S3AnchorStore("b", _client=client).list_names("p/") == ["p/a", "p/b"]
    assert client.list_objects_v2.call_args_list[1].kwargs["ContinuationToken"] == "t"


def test_s3_store_without_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ImportError, match="boto3 is not installed"):
        S3AnchorStore("b")._get_client()
