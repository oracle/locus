# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Anchoring the audit chain's head in object storage the agent cannot reach.

A hash chain cannot see its own tail: drop records off the end and what is
left still verifies. :meth:`AuditTrail.verify(expected_head=…)
<locus.control.audit.AuditTrail.verify>` closes that gap when the head is kept
somewhere else. This module keeps it there — as a small JSON object in an OCI
Object Storage bucket or any S3-compatible bucket::

    store = OCIBucketAnchorStore(bucket_name="audit-anchors", namespace="…")
    anchor_trail(trail, store)  # writes {records, seq, head, ts}
    ...
    verify_anchor(trail, store).ok  # the chain still reaches that head

An anchor object is written once and never rewritten: its name carries the
record count, so the latest anchor is the greatest name under the prefix, and a
bucket **retention rule** (OCI) or **Object Lock** (S3) can forbid deleting or
overwriting it for as long as the policy says. The rule is the bucket's
configuration, not the SDK's; pass ``require_retention_rule=True`` to refuse to
anchor into a bucket that has none, so a misconfigured bucket fails loudly
instead of looking anchored.

:func:`anchor_periodically` is the loop a server or runner calls to anchor
every ``every`` records (or whenever the head moved) at an interval.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from locus.control.audit import AuditTrail

#: Prefix anchor objects are written under when a store is not told otherwise.
DEFAULT_PREFIX = "locus/audit-anchors/"

#: Anchor object format, for a reader that meets a newer or older one.
ANCHOR_VERSION = 1


class AnchorError(RuntimeError):
    """Anchoring was refused — for example the bucket has no retention rule."""


@dataclass(frozen=True)
class Anchor:
    """The chain head as written to the store.

    Attributes:
        records: How many records the chain had.
        seq: ``seq`` of the last record (``-1`` for an empty chain).
        head: Hash of the last record — the genesis anchor when empty.
        ts: When the anchor was written (UTC, ISO 8601).
        name: The object name it was written under.
    """

    records: int
    seq: int
    head: str
    ts: str
    name: str = ""

    def to_json(self) -> bytes:
        data = {k: v for k, v in asdict(self).items() if k != "name"}
        data["version"] = ANCHOR_VERSION
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, body: bytes, *, name: str = "") -> Anchor:
        data = json.loads(body.decode("utf-8"))
        return cls(
            records=int(data["records"]),
            seq=int(data["seq"]),
            head=str(data["head"]),
            ts=str(data["ts"]),
            name=name,
        )


@dataclass(frozen=True)
class AnchorReport:
    """Whether the chain still reaches its latest anchor, and if not, why.

    Attributes:
        ok: Whether the check passed.
        anchor: The anchor that was checked, or ``None`` when the store had none.
        problem: What failed, in a sentence; empty when ``ok``.
    """

    ok: bool
    anchor: Anchor | None
    problem: str = ""


@runtime_checkable
class AnchorStore(Protocol):
    """Where anchors go. Object storage with a retention policy is the intended target."""

    def put(self, name: str, body: bytes) -> None:
        """Write ``body`` under ``name``; must not silently overwrite a different body."""
        ...

    def get(self, name: str) -> bytes | None:
        """The object's bytes, or ``None`` when there is no such object."""
        ...

    def list_names(self, prefix: str) -> list[str]:
        """Every object name under ``prefix``."""
        ...

    def retention_rules(self) -> list[str]:
        """One line per retention rule on the bucket; empty when it has none."""
        ...


def _anchor_name(prefix: str, records: int, head: str) -> str:
    # Zero-padded count first so the greatest name is the newest anchor; the
    # head's prefix second so two anchors of different chains never collide.
    return f"{prefix}{records:020d}-{head[:16]}.json"


def latest_anchor(store: AnchorStore, *, prefix: str = DEFAULT_PREFIX) -> Anchor | None:
    """The newest anchor under ``prefix``, or ``None`` when there is none."""
    names = [n for n in store.list_names(prefix) if n.endswith(".json")]
    if not names:
        return None
    name = max(names)
    body = store.get(name)
    if body is None:
        return None
    return Anchor.from_json(body, name=name)


def anchor_trail(
    trail: AuditTrail,
    store: AnchorStore,
    *,
    every: int | None = None,
    require_retention_rule: bool = False,
    prefix: str = DEFAULT_PREFIX,
) -> Anchor | None:
    """Write the trail's current head to ``store``.

    Args:
        trail: The trail to anchor; any :class:`~locus.control.audit.AuditTrail`.
        store: An :class:`OCIBucketAnchorStore`, :class:`S3AnchorStore`, or
            anything else that satisfies :class:`AnchorStore`.
        every: Anchor only once the chain has grown by at least this many
            records since the latest anchor. ``None`` anchors whenever the
            head has moved.
        require_retention_rule: Refuse (``AnchorError``) when the bucket has
            no retention rule or Object Lock configuration, so a bucket that
            can be emptied is never mistaken for an anchor.
        prefix: Object-name prefix the anchors live under.

    Returns:
        The anchor written, the existing one when the head has not moved, or
        ``None`` when ``every`` says it is not time yet.
    """
    if require_retention_rule and not store.retention_rules():
        raise AnchorError(
            "the bucket has no retention rule: an anchor there could be deleted; "
            "add a retention rule (OCI) or Object Lock (S3) or pass require_retention_rule=False"
        )
    latest = latest_anchor(store, prefix=prefix)
    records = len(trail)
    head = trail.head
    if latest is not None:
        if latest.records == records and latest.head == head:
            return latest
        if every is not None and records - latest.records < every:
            return None
    anchor = Anchor(
        records=records,
        seq=records - 1,
        head=head,
        ts=datetime.now(UTC).isoformat(),
        name=_anchor_name(prefix, records, head),
    )
    store.put(anchor.name, anchor.to_json())
    return anchor


def verify_anchor(
    trail: AuditTrail,
    store: AnchorStore,
    *,
    keys: Mapping[str, bytes | str] | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> AnchorReport:
    """Check that the chain still reaches the latest anchor.

    Passes when the chain is intact, holds at least as many records as the
    anchor names, and the record at the anchor's ``seq`` carries the anchored
    hash. Records appended after the anchor are fine; records removed before
    it, or any edit, are not. ``keys`` is forwarded to
    :meth:`~locus.control.audit.AuditTrail.check` to also demand signatures.
    """
    anchor = latest_anchor(store, prefix=prefix)
    if anchor is None:
        return AnchorReport(ok=False, anchor=None, problem="no anchor found in the store")
    report = trail.check(keys=keys)
    if not report.ok:
        return AnchorReport(ok=False, anchor=anchor, problem=report.problem)
    records = trail.records()
    if len(records) < anchor.records:
        return AnchorReport(
            ok=False,
            anchor=anchor,
            problem=(
                f"the chain holds {len(records)} record(s) but was anchored at "
                f"{anchor.records}: records were cut off"
            ),
        )
    if anchor.seq < 0:
        return AnchorReport(ok=True, anchor=anchor)  # an empty chain was anchored; nothing to cut
    if records[anchor.seq].hash != anchor.head:
        return AnchorReport(
            ok=False,
            anchor=anchor,
            problem=f"record {anchor.seq} does not carry the anchored hash: the chain was rebuilt",
        )
    return AnchorReport(ok=True, anchor=anchor)


async def anchor_periodically(
    trail: AuditTrail,
    store: AnchorStore,
    *,
    interval_seconds: float,
    every: int | None = None,
    require_retention_rule: bool = False,
    prefix: str = DEFAULT_PREFIX,
    stop: asyncio.Event | None = None,
) -> None:
    """Anchor on a schedule until cancelled or ``stop`` is set.

    Every ``interval_seconds`` it calls :func:`anchor_trail` on a worker
    thread (the object-storage clients are synchronous). A failed write is
    not swallowed: the exception ends the loop, which is what a supervisor
    should see. A final anchor is written when ``stop`` is set, so a clean
    shutdown leaves the last head behind.
    """
    stop = stop or asyncio.Event()
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass
        await asyncio.to_thread(
            lambda: anchor_trail(
                trail,
                store,
                every=None if stop.is_set() else every,
                require_retention_rule=require_retention_rule,
                prefix=prefix,
            )
        )


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class OCIBucketAnchorStore:
    """An :class:`AnchorStore` on an OCI Object Storage bucket.

    Authentication follows the OCI checkpointer: ``profile_name`` from
    ``config_file`` with ``auth_type`` ``api_key`` (default), ``security_token``,
    ``instance_principal`` or ``resource_principal``. ``retention_rules()``
    lists the bucket's retention rules, which is what
    ``anchor_trail(require_retention_rule=True)`` consults.

    Args:
        bucket_name: The bucket.
        namespace: The bucket's Object Storage namespace.
        profile_name: Profile in ``config_file`` (default ``DEFAULT``).
        config_file: OCI config file (default ``~/.oci/config``).
        auth_type: ``api_key``, ``security_token``, ``instance_principal`` or
            ``resource_principal``.
        region: Region override for the profile.
        retry_strategy: An ``oci.retry`` strategy; the SDK default when omitted.
        _client: An ``ObjectStorageClient`` to use instead of building one — the
            seam tests inject a double through.
    """

    def __init__(
        self,
        bucket_name: str,
        namespace: str,
        *,
        profile_name: str = "DEFAULT",
        config_file: str = "~/.oci/config",
        auth_type: str = "api_key",
        region: str | None = None,
        retry_strategy: Any = None,
        _client: Any = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.namespace = namespace
        self._profile_name = profile_name
        self._config_file = config_file
        self._auth_type = auth_type
        self._region = region
        self._retry_strategy = retry_strategy
        self._client: Any = _client

    def _get_client(self) -> Any:
        # A synchronous client: not bound to an event loop, so cached plainly.
        if self._client is not None:
            return self._client
        from locus.memory.backends.oci_bucket import (  # noqa: PLC0415
            OCIBucketConfig,
            build_object_storage_client,
        )

        self._client = build_object_storage_client(
            OCIBucketConfig(
                bucket_name=self.bucket_name,
                namespace=self.namespace,
                profile_name=self._profile_name,
                config_file=self._config_file,
                auth_type=self._auth_type,
                region=self._region,
            )
        )
        return self._client

    def _retry(self) -> dict[str, Any]:
        if self._retry_strategy is not None:
            return {"retry_strategy": self._retry_strategy}
        try:
            import oci  # noqa: PLC0415
        except ImportError:  # pragma: no cover - depends on install extras
            return {}
        return {"retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY}

    def put(self, name: str, body: bytes) -> None:
        self._get_client().put_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket_name,
            object_name=name,
            put_object_body=body,
            content_type="application/json",
            **self._retry(),
        )

    def get(self, name: str) -> bytes | None:
        try:
            response = self._get_client().get_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket_name,
                object_name=name,
                **self._retry(),
            )
        except Exception as exc:
            if "ObjectNotFound" in str(exc) or "404" in str(exc):
                return None
            raise
        content: bytes = response.data.content
        return content

    def list_names(self, prefix: str) -> list[str]:
        client = self._get_client()
        names: list[str] = []
        start: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "namespace_name": self.namespace,
                "bucket_name": self.bucket_name,
                "prefix": prefix,
                "fields": "name",
                "limit": 1000,
                **self._retry(),
            }
            if start:
                kwargs["start"] = start
            data = client.list_objects(**kwargs).data
            names.extend(o.name for o in data.objects or [])
            start = getattr(data, "next_start_with", None)
            if not start:
                return names

    def retention_rules(self) -> list[str]:
        data = (
            self._get_client()
            .list_retention_rules(
                namespace_name=self.namespace, bucket_name=self.bucket_name, **self._retry()
            )
            .data
        )
        lines: list[str] = []
        for rule in data.items or []:
            duration = getattr(rule, "duration", None)
            span = (
                f"{duration.time_amount} {duration.time_unit}"
                if duration is not None
                else "indefinite"
            )
            locked = " (locked)" if getattr(rule, "time_rule_locked", None) else ""
            lines.append(f"{rule.display_name}: {span}{locked}")
        return lines


class S3AnchorStore:
    """An :class:`AnchorStore` on an S3-compatible bucket.

    ``retention_rules()`` reports the bucket's Object Lock default retention,
    the S3 equivalent of a retention rule. Pass ``endpoint_url`` for a
    non-AWS S3-compatible service.

    Args:
        bucket: The bucket.
        endpoint_url: Endpoint of a non-AWS S3-compatible service.
        region_name: Region, when the endpoint needs one.
        aws_access_key_id: Explicit access key; the default credential chain
            applies when omitted.
        aws_secret_access_key: Secret key, paired with ``aws_access_key_id``.
        _client: A pre-built S3 client to use instead — the seam tests inject a
            double through.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        _client: Any = None,
    ) -> None:
        self.bucket = bucket
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._client: Any = _client

    def _get_client(self) -> Any:
        # A synchronous client: not bound to an event loop, so cached plainly.
        if self._client is not None:
            return self._client
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                'boto3 is not installed. Install with: pip install "locus-sdk[s3]"'
            ) from exc
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        return self._client

    def put(self, name: str, body: bytes) -> None:
        self._get_client().put_object(
            Bucket=self.bucket, Key=name, Body=body, ContentType="application/json"
        )

    def get(self, name: str) -> bytes | None:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code", "") in ("NoSuchKey", "404"):
                return None
            raise
        data: bytes = response["Body"].read()
        return data

    def list_names(self, prefix: str) -> list[str]:
        client = self._get_client()
        names: list[str] = []
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        while True:
            response = client.list_objects_v2(**kwargs)
            names.extend(obj["Key"] for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return names
            kwargs["ContinuationToken"] = response.get("NextContinuationToken")

    def retention_rules(self) -> list[str]:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            response = self._get_client().get_object_lock_configuration(Bucket=self.bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("ObjectLockConfigurationNotFoundError", "NoSuchObjectLockConfiguration"):
                return []
            raise
        config = response.get("ObjectLockConfiguration", {})
        if config.get("ObjectLockEnabled") != "Enabled":
            return []
        retention = config.get("Rule", {}).get("DefaultRetention")
        if not retention:
            return ["object-lock: enabled, no default retention"]
        span = retention.get("Days") or retention.get("Years")
        unit = "DAYS" if retention.get("Days") else "YEARS"
        return [f"object-lock: {retention.get('Mode', '')} {span} {unit}"]


__all__ = [
    "ANCHOR_VERSION",
    "DEFAULT_PREFIX",
    "Anchor",
    "AnchorError",
    "AnchorReport",
    "AnchorStore",
    "OCIBucketAnchorStore",
    "S3AnchorStore",
    "anchor_periodically",
    "anchor_trail",
    "latest_anchor",
    "verify_anchor",
]
