# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""S3-compatible object-storage checkpointer.

``S3Backend`` implements :class:`BaseCheckpointer` directly, so it can be
passed to ``Agent(checkpointer=...)`` without any adapter glue. It speaks
the S3 API via ``boto3``, so it works against AWS S3, MinIO, Cloudflare
R2, Backblaze B2, and any other S3-compatible endpoint — point it at one
with ``endpoint_url``.

Object layout::

    {prefix}{thread_id}/{checkpoint_id}.json      # AgentState payload
    {prefix}{thread_id}/{checkpoint_id}.meta.json # per-checkpoint metadata,
                                                  # iteration and size
    {prefix}{thread_id}/_latest                   # pointer to newest id

Every checkpoint is its own object, so a thread's history is complete and
its storage grows with the square of the conversation's length;
``max_checkpoints_per_thread`` bounds it (a bucket lifecycle rule is the
server-side alternative). ``list`` takes one listing plus one GET of the
small ``.meta.json`` per checkpoint described.

S3 keeps an object's modification time to the second, and a turn writes
several checkpoints in that, so the listing alone cannot order them; the
write time each ``.meta.json`` records (to the microsecond) orders the
checkpoints that share a second, and only those are read for it
(:func:`~locus.memory.time_travel.order_by_write_time`).

The ``_latest`` pointer lets ``load(thread_id)`` do a single GET instead
of a list + sort every turn.

``boto3`` is synchronous; every call is dispatched to a threadpool via
:func:`asyncio.to_thread` so the backend composes with the async agent
loop.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.expiry import EXPIRES_AT_KEY, is_expired
from locus.memory.time_travel import (
    CheckpointInfo,
    checkpoint_iteration,
    checkpoint_size,
    info_from,
    link_parents,
    order_by_write_time,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState
    from locus.memory.encryption import EncryptedSerializer
    from locus.memory.scope import CheckpointScope


_LATEST_POINTER = "_latest"


class S3Backend(BaseCheckpointer):
    """S3-compatible object-storage-backed checkpointer.

    Example (AWS S3, ambient credentials)::

        checkpointer = S3Backend(bucket="my-checkpoints")
        agent = Agent(config=cfg, checkpointer=checkpointer)

    Example (MinIO / local)::

        checkpointer = S3Backend(
            bucket="checkpoints",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )

    Args:
        bucket: Bucket name.
        prefix: Key prefix for all objects. Default ``locus/checkpoints/``.
        endpoint_url: Custom S3 endpoint (set for MinIO / R2 / B2).
        region_name: AWS region (also used when auto-creating the bucket).
        aws_access_key_id: Explicit access key. When ``None``, boto3's
            default credential chain is used.
        aws_secret_access_key: Explicit secret key (pairs with
            ``aws_access_key_id``).
        create_bucket: Create the bucket on first use if it doesn't exist.
            Defaults to True (convenient for dev / MinIO / tests).
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints, deleting older ones after each save. ``None`` (the
            default) keeps them all. The tool-effect ledger's sidecar thread
            is never trimmed.
        ttl: How long a checkpoint stays loadable after its save (seconds or
            a ``timedelta``). Recorded as ``expires_at`` in the payload and
            the ``.meta.json``; a load never returns an expired checkpoint,
            an instance with a ``ttl`` never lists one, and :meth:`expire`
            deletes them (a bucket lifecycle rule is the server-side
            alternative; object expiry is not set on the objects).
        encryption: Seals the state inside the payload object
            (:class:`~locus.memory.encryption.EncryptedSerializer`); the
            ``.meta.json`` stays plaintext.
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`): the scope's
            prefix becomes part of the object key, ahead of the thread id.
        _client: Injection seam for tests — a pre-built boto3 S3 client
            bypasses the lazy import and credential resolution.

    Capabilities: ``metadata_query``, ``vacuum``, ``branching``,
    ``list_threads``, ``list_with_metadata``, ``persistent_checkpoint_ids``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "locus/checkpoints/",
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        create_bucket: bool = True,
        max_checkpoints_per_thread: int | None = None,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
        _client: Any = None,
    ) -> None:
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self.bucket = bucket
        self.prefix = prefix
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._configure(ttl=ttl, encryption=encryption, scope=scope)
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._create_bucket = create_bucket
        self._client_override = _client
        self._client: Any = None
        self._initialized = False

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        return CheckpointerCapabilities(
            metadata_query=True,
            vacuum=True,
            branching=True,
            list_threads=True,
            list_with_metadata=True,
            persistent_checkpoint_ids=True,
        )

    # ------------------------------------------------------------------
    # Client + bucket bootstrap
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        try:
            import boto3  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                'boto3 is not installed. Install with: pip install "locus-sdk[s3]"'
            ) from e
        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )
        return self._client

    async def _ensure_bucket(self) -> None:
        if self._initialized:
            return

        def _ensure() -> None:
            client = self._get_client()
            from botocore.exceptions import ClientError  # noqa: PLC0415

            try:
                client.head_bucket(Bucket=self.bucket)
            except ClientError:
                if not self._create_bucket:
                    raise
                kwargs: dict[str, Any] = {"Bucket": self.bucket}
                # us-east-1 rejects an explicit LocationConstraint.
                if self._region_name and self._region_name != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region_name}
                client.create_bucket(**kwargs)

        await asyncio.to_thread(_ensure)
        self._initialized = True

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _thread_prefix(self, thread_key: str) -> str:
        """The key prefix of a thread, from the id it is stored under (its scope included)."""
        return f"{self.prefix}{thread_key}/"

    def _checkpoint_key(self, thread_key: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_key)}{checkpoint_id}.json"

    def _meta_key(self, thread_key: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_key)}{checkpoint_id}.meta.json"

    def _latest_key(self, thread_key: str) -> str:
        return f"{self._thread_prefix(thread_key)}{_LATEST_POINTER}"

    # ------------------------------------------------------------------
    # Raw object helpers
    # ------------------------------------------------------------------

    async def _put_bytes(self, key: str, body: bytes, content_type: str) -> None:
        client = self._get_client()

        def _put() -> None:
            client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

        await asyncio.to_thread(_put)

    async def _put_json(self, key: str, payload: dict[str, Any]) -> None:
        await self._put_bytes(key, json.dumps(payload).encode("utf-8"), "application/json")

    async def _get_bytes(self, key: str) -> bytes | None:
        client = self._get_client()

        def _get() -> bytes | None:
            from botocore.exceptions import ClientError  # noqa: PLC0415

            try:
                response = client.get_object(Bucket=self.bucket, Key=key)
                data: bytes = response["Body"].read()
                return data
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404", "NoSuchBucket"):
                    return None
                raise

        return await asyncio.to_thread(_get)

    async def _get_json(self, key: str) -> dict[str, Any] | None:
        body = await self._get_bytes(key)
        if body is None:
            return None
        result: dict[str, Any] = json.loads(body.decode("utf-8"))
        return result

    async def _delete_object(self, key: str) -> None:
        client = self._get_client()
        await asyncio.to_thread(lambda: client.delete_object(Bucket=self.bucket, Key=key))

    async def _list_keys(self, prefix: str, delimiter: str | None = None) -> dict[str, Any]:
        client = self._get_client()

        def _list() -> dict[str, Any]:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if delimiter is not None:
                kwargs["Delimiter"] = delimiter
            contents: list[dict[str, Any]] = []
            common_prefixes: list[str] = []
            token: str | None = None
            from botocore.exceptions import ClientError  # noqa: PLC0415

            while True:
                if token:
                    kwargs["ContinuationToken"] = token
                try:
                    resp = client.list_objects_v2(**kwargs)
                except ClientError as e:
                    # A bucket not made yet holds nothing; a read never creates it.
                    if e.response.get("Error", {}).get("Code", "") == "NoSuchBucket":
                        break
                    raise
                contents.extend(resp.get("Contents", []))
                common_prefixes.extend(p["Prefix"] for p in resp.get("CommonPrefixes", []))
                if resp.get("IsTruncated"):
                    token = resp.get("NextContinuationToken")
                else:
                    break
            return {"contents": contents, "prefixes": common_prefixes}

        return await asyncio.to_thread(_list)

    # ------------------------------------------------------------------
    # BaseCheckpointer API
    # ------------------------------------------------------------------

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        await self._ensure_bucket()
        checkpoint_id = checkpoint_id or uuid4().hex
        now = datetime.now(UTC)
        key = self._scoped(thread_id)
        expires_at = self._expires_at(now)

        state_data = self._encode_state(state, key, checkpoint_id)
        payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": key,
            "created_at": now.isoformat(),
            "state": state_data,
        }
        meta_payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": key,
            "created_at": now.isoformat(),
            "metadata": metadata or {},
            "iteration": checkpoint_iteration(state_data),
            "size": checkpoint_size(state_data),
        }
        if expires_at is not None:
            payload[EXPIRES_AT_KEY] = expires_at.isoformat()
            meta_payload[EXPIRES_AT_KEY] = expires_at.isoformat()
        await self._put_json(self._checkpoint_key(key, checkpoint_id), payload)
        await self._put_json(self._meta_key(key, checkpoint_id), meta_payload)
        await self._put_bytes(self._latest_key(key), checkpoint_id.encode("utf-8"), "text/plain")
        if self.max_checkpoints_per_thread is not None:
            await self._trim_history(thread_id, self.max_checkpoints_per_thread)
        return checkpoint_id

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        key = self._scoped(thread_id)
        if checkpoint_id is None:
            return await self._load_latest(key)
        return await self._load_checkpoint(key, checkpoint_id)

    async def _load_checkpoint(self, key: str, checkpoint_id: str) -> AgentState | None:
        data = await self._get_json(self._checkpoint_key(key, checkpoint_id))
        if data is None or is_expired(data.get(EXPIRES_AT_KEY)):
            return None
        return self._decode_state(data["state"], key, checkpoint_id)

    async def _load_latest(self, key: str) -> AgentState | None:
        """The thread's newest live checkpoint.

        ``_latest`` names the last save. Written with a shorter ``ttl`` than
        an earlier checkpoint, it expires first; the listing then finds the
        newest checkpoint still live, as it does when the pointer is gone.
        """
        pointer = await self._get_bytes(self._latest_key(key))
        if pointer is not None:
            state = await self._load_checkpoint(key, pointer.decode("utf-8").strip())
            if state is not None:
                return state
        for checkpoint_id, _, _ in await self._checkpoint_objects(key):
            state = await self._load_checkpoint(key, checkpoint_id)
            if state is not None:
                return state
        return None

    async def _checkpoint_objects(
        self, thread_key: str
    ) -> list[tuple[str, Any, dict[str, Any] | None]]:
        """``(checkpoint_id, last_modified, meta)`` of the thread's checkpoints, newest first.

        Ordered by the store's modification time, and within one second of
        it by the write time in each ``.meta.json``; ``meta`` is that object
        where it had to be read, ``None`` otherwise. On an instance with a
        ``ttl`` every ``.meta.json`` is read and the expired checkpoints are
        left out.
        """
        listing = await self._list_keys(self._thread_prefix(thread_key))
        items: list[tuple[str, Any]] = []
        for obj in listing["contents"]:
            key = obj["Key"]
            if key.endswith((".meta.json", _LATEST_POINTER)):
                continue
            if not key.endswith(".json"):
                continue
            items.append((key.rsplit("/", 1)[-1][: -len(".json")], obj.get("LastModified")))
        return await _live_in_order(
            items,
            lambda cid: self._get_json(self._meta_key(thread_key, cid)),
            "created_at",
            read_all=self.ttl is not None,
        )

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        return [
            cid for cid, _, _ in (await self._checkpoint_objects(self._scoped(thread_id)))[:limit]
        ]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        key = self._scoped(thread_id)
        if checkpoint_id is None:
            listing = await self._list_keys(self._thread_prefix(key))
            keys = [obj["Key"] for obj in listing["contents"]]
            if not keys:
                return False
            for object_key in keys:
                await self._delete_object(object_key)
            return True

        ckpt_key = self._checkpoint_key(key, checkpoint_id)
        if await self._get_bytes(ckpt_key) is None:
            return False
        await self._delete_object(ckpt_key)
        await self._delete_object(self._meta_key(key, checkpoint_id))
        await self._repoint_latest(key, checkpoint_id)
        return True

    async def _repoint_latest(self, thread_key: str, deleted: str) -> None:
        """Repoint ``_latest`` when it named a deleted checkpoint: the next newest, or none.

        The pointer is right for any other checkpoint, and finding the next
        newest lists the thread, so this runs only when it has to.
        """
        pointer = await self._get_bytes(self._latest_key(thread_key))
        if pointer is None or pointer.decode("utf-8").strip() != deleted:
            return
        remaining = await self._checkpoint_objects(thread_key)
        if remaining:
            await self._put_bytes(
                self._latest_key(thread_key), remaining[0][0].encode("utf-8"), "text/plain"
            )
        else:
            await self._delete_object(self._latest_key(thread_key))

    async def expire(self) -> int:
        """Delete the checkpoints whose ``expires_at`` has passed, in every thread and scope.

        One listing of the prefix, one GET per ``.meta.json``; each expired
        checkpoint's payload and metadata objects are deleted and a thread's
        ``_latest`` pointer is moved when it named one. Returns the
        checkpoints deleted. A bucket lifecycle rule does the same
        server-side, without reading anything.
        """
        await self._ensure_bucket()
        now = datetime.now(UTC)
        listing = await self._list_keys(self.prefix)
        deleted = 0
        repoint: dict[str, list[str]] = {}
        for obj in listing["contents"]:
            object_key = obj["Key"]
            if not object_key.endswith(".meta.json"):
                continue
            meta = await self._get_json(object_key)
            if meta is None or not is_expired(meta.get(EXPIRES_AT_KEY), now):
                continue
            thread_prefix, name = object_key.rsplit("/", 1)
            checkpoint_id = name[: -len(".meta.json")]
            await self._delete_object(f"{thread_prefix}/{checkpoint_id}.json")
            await self._delete_object(object_key)
            deleted += 1
            repoint.setdefault(thread_prefix[len(self.prefix) :], []).append(checkpoint_id)
        for thread_key, gone in repoint.items():
            pointer = await self._get_bytes(self._latest_key(thread_key))
            if pointer is not None and pointer.decode("utf-8").strip() in gone:
                await self._repoint_latest(thread_key, pointer.decode("utf-8").strip())
        return deleted

    async def get_metadata(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        self._require_capability("metadata_query")
        key = self._scoped(thread_id)
        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(key))
            if pointer is None:
                return None
            checkpoint_id = pointer.decode("utf-8").strip()
        meta = await self._get_json(self._meta_key(key, checkpoint_id))
        if meta is None or is_expired(meta.get(EXPIRES_AT_KEY)):
            return None
        result: dict[str, Any] = meta.get("metadata", {})
        return result

    async def list_threads(
        self, limit: int = 100, pattern: str = "*", *, scope: CheckpointScope | None = None
    ) -> list[str]:
        """The threads of one scope: the common prefixes one level under the scope's prefix."""
        self._require_capability("list_threads")
        scoped = self._listing_scope(scope)
        base = f"{self.prefix}{scoped.prefix}"
        listing = await self._list_keys(base, delimiter="/")
        threads: list[str] = []
        for p in listing["prefixes"]:
            name = p[len(self.prefix) :].rstrip("/")
            thread_id = scoped.unkey(name)
            if thread_id:
                threads.append(thread_id)
        return threads[:limit]

    async def list_with_metadata(self, limit: int = 100) -> list[dict[str, Any]]:
        self._require_capability("list_with_metadata")
        listing = await self._list_keys(self.prefix)
        results: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for obj in listing["contents"]:
            key = obj["Key"]
            if not key.endswith(".meta.json"):
                continue
            meta = await self._get_json(key)
            if meta is not None and not is_expired(meta.get(EXPIRES_AT_KEY), now):
                results.append(meta)
            if len(results) >= limit:
                break
        return results

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        """Copy every object of a thread under another thread's prefix.

        An encrypted checkpoint is bound to its thread and cannot be copied
        as bytes: each one is unsealed and sealed again under the new
        thread, its ``.meta.json`` copied beside it.
        """
        self._require_capability("branching")
        client = self._get_client()
        src_key, dst_key = self._scoped(source_thread_id), self._scoped(dest_thread_id)
        src_prefix, dst_prefix = self._thread_prefix(src_key), self._thread_prefix(dst_key)
        listing = await self._list_keys(src_prefix)
        if not listing["contents"]:
            return False

        def _copy(src: str, dst: str) -> None:
            client.copy_object(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": src},
                Key=dst,
            )

        for obj in listing["contents"]:
            object_key = obj["Key"]
            name = object_key[len(src_prefix) :]
            target = dst_prefix + name
            is_payload = name.endswith(".json") and not name.endswith(".meta.json")
            payload = await self._get_json(object_key) if is_payload and self.encryption else None
            if payload is not None:
                checkpoint_id = name[: -len(".json")]
                state = self._decode_state(payload["state"], src_key, checkpoint_id)
                payload["state"] = self._encode_state(state, dst_key, checkpoint_id)
                payload["thread_id"] = dst_key
                await self._put_json(target, payload)
                continue
            await asyncio.to_thread(_copy, object_key, target)
        return True

    async def vacuum(self, older_than_days: int = 30) -> int:
        self._require_capability("vacuum")
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        listing = await self._list_keys(self.prefix)
        deleted = 0
        for obj in listing["contents"]:
            lm = obj.get("LastModified")
            if lm is None:
                continue
            if lm.tzinfo is None:
                lm = lm.replace(tzinfo=UTC)
            if lm < cutoff:
                await self._delete_object(obj["Key"])
                deleted += 1
        return deleted

    async def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                await asyncio.to_thread(close)
            self._client = None

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first: one listing, then each one's ``.meta.json``."""
        await self._ensure_bucket()
        key = self._scoped(thread_id)
        ordered = await self._checkpoint_objects(key)
        # The checkpoint just past the window is the parent of the oldest in it.
        older = ordered[limit][0] if len(ordered) > limit else None
        objects = ordered[:limit]
        metas = await asyncio.gather(
            *(
                self._get_json(self._meta_key(key, cid)) if meta is None else _ready(meta)
                for cid, _, meta in objects
            )
        )
        infos: list[CheckpointInfo] = []
        now = datetime.now(UTC)
        for (checkpoint_id, modified, _), raw_meta in zip(objects, metas, strict=True):
            meta = raw_meta or {}
            if is_expired(meta.get(EXPIRES_AT_KEY), now):
                continue
            iteration, size = meta.get("iteration"), meta.get("size")
            if iteration is None or size is None:
                data = await self._get_json(self._checkpoint_key(key, checkpoint_id))
                state_data = (data or {}).get("state")
                if isinstance(state_data, dict):
                    iteration, size = state_data.get("iteration"), checkpoint_size(state_data)
            infos.append(
                info_from(
                    checkpoint_id,
                    thread_id,
                    created_at=meta.get("created_at") or modified,
                    metadata=meta.get("metadata") or {},
                    iteration=iteration,
                    size=size,
                )
            )
        return link_parents(infos, older)


async def _ready(value: dict[str, Any]) -> dict[str, Any]:
    """``value`` as an awaitable, beside the GETs of a gather."""
    return value


async def _live_in_order(
    items: list[tuple[str, Any]],
    read_meta: Any,
    time_key: str,
    *,
    read_all: bool,
) -> list[tuple[str, Any, dict[str, Any] | None]]:
    """:func:`order_by_write_time`, dropping the expired checkpoints when every meta is read.

    With ``read_all`` (an instance with a ``ttl``) each ``.meta.json`` is
    fetched up front, the checkpoints whose ``expires_at`` has passed are
    left out, and the ordering reads nothing more; without it only the
    same-second ties are read, as before.
    """
    if not read_all:
        ordered: list[tuple[str, Any, dict[str, Any] | None]] = await order_by_write_time(
            items, read_meta, time_key
        )
        return ordered
    metas = dict(
        zip(
            [cid for cid, _ in items],
            await asyncio.gather(*(read_meta(cid) for cid, _ in items)),
            strict=True,
        )
    )
    now = datetime.now(UTC)
    live = [
        (cid, modified)
        for cid, modified in items
        if not is_expired((metas.get(cid) or {}).get(EXPIRES_AT_KEY), now)
    ]

    async def cached(cid: str) -> dict[str, Any] | None:
        return metas.get(cid)

    return [
        (cid, modified, meta if meta is not None else metas.get(cid))
        for cid, modified, meta in await order_by_write_time(live, cached, time_key)
    ]
