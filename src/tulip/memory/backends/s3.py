# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""S3-compatible object-storage checkpointer.

``S3Backend`` implements :class:`BaseCheckpointer` directly, so it can be
passed to ``Agent(checkpointer=...)`` without any adapter glue. It speaks
the S3 API via ``boto3``, so it works against AWS S3, MinIO, Cloudflare
R2, Backblaze B2, and any other S3-compatible endpoint — point it at one
with ``endpoint_url``.

Object layout::

    {prefix}{thread_id}/{checkpoint_id}.json      # AgentState payload
    {prefix}{thread_id}/{checkpoint_id}.meta.json # per-checkpoint metadata
    {prefix}{thread_id}/_latest                   # pointer to newest id

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

from tulip.core.protocols import CheckpointerCapabilities
from tulip.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from tulip.core.state import AgentState


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
        prefix: Key prefix for all objects. Default ``tulip/checkpoints/``.
        endpoint_url: Custom S3 endpoint (set for MinIO / R2 / B2).
        region_name: AWS region (also used when auto-creating the bucket).
        aws_access_key_id: Explicit access key. When ``None``, boto3's
            default credential chain is used.
        aws_secret_access_key: Explicit secret key (pairs with
            ``aws_access_key_id``).
        create_bucket: Create the bucket on first use if it doesn't exist.
            Defaults to True (convenient for dev / MinIO / tests).
        _client: Injection seam for tests — a pre-built boto3 S3 client
            bypasses the lazy import and credential resolution.

    Capabilities: ``metadata_query``, ``vacuum``, ``branching``,
    ``list_threads``, ``list_with_metadata``, ``persistent_checkpoint_ids``.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "tulip/checkpoints/",
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        create_bucket: bool = True,
        _client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
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
                'boto3 is not installed. Install with: pip install "tulip-agents[s3]"'
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

    def _thread_prefix(self, thread_id: str) -> str:
        return f"{self.prefix}{thread_id}/"

    def _checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{checkpoint_id}.json"

    def _meta_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{checkpoint_id}.meta.json"

    def _latest_key(self, thread_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{_LATEST_POINTER}"

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
            while True:
                if token:
                    kwargs["ContinuationToken"] = token
                resp = client.list_objects_v2(**kwargs)
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

        payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "created_at": now.isoformat(),
            "state": state.to_checkpoint(),
        }
        meta_payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "created_at": now.isoformat(),
            "metadata": metadata or {},
        }
        await self._put_json(self._checkpoint_key(thread_id, checkpoint_id), payload)
        await self._put_json(self._meta_key(thread_id, checkpoint_id), meta_payload)
        await self._put_bytes(
            self._latest_key(thread_id), checkpoint_id.encode("utf-8"), "text/plain"
        )
        return checkpoint_id

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        from tulip.core.state import AgentState  # noqa: PLC0415

        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(thread_id))
            if pointer is None:
                return None
            checkpoint_id = pointer.decode("utf-8").strip()

        data = await self._get_json(self._checkpoint_key(thread_id, checkpoint_id))
        if data is None:
            return None
        return AgentState.from_checkpoint(data["state"])

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        listing = await self._list_keys(self._thread_prefix(thread_id))
        items: list[tuple[str, str]] = []  # (last_modified_iso, checkpoint_id)
        for obj in listing["contents"]:
            key = obj["Key"]
            if key.endswith((".meta.json", _LATEST_POINTER)):
                continue
            if not key.endswith(".json"):
                continue
            checkpoint_id = key.rsplit("/", 1)[-1][: -len(".json")]
            lm = obj.get("LastModified")
            items.append((lm.isoformat() if lm else "", checkpoint_id))
        items.sort(key=lambda t: t[0], reverse=True)
        return [cid for _, cid in items[:limit]]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if checkpoint_id is None:
            listing = await self._list_keys(self._thread_prefix(thread_id))
            keys = [obj["Key"] for obj in listing["contents"]]
            if not keys:
                return False
            for key in keys:
                await self._delete_object(key)
            return True

        ckpt_key = self._checkpoint_key(thread_id, checkpoint_id)
        if await self._get_bytes(ckpt_key) is None:
            return False
        await self._delete_object(ckpt_key)
        await self._delete_object(self._meta_key(thread_id, checkpoint_id))
        # Repoint _latest if we just deleted the newest checkpoint.
        remaining = await self.list_checkpoints(thread_id, limit=1)
        if remaining:
            await self._put_bytes(
                self._latest_key(thread_id), remaining[0].encode("utf-8"), "text/plain"
            )
        else:
            await self._delete_object(self._latest_key(thread_id))
        return True

    async def get_metadata(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        self._require_capability("metadata_query")
        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(thread_id))
            if pointer is None:
                return None
            checkpoint_id = pointer.decode("utf-8").strip()
        meta = await self._get_json(self._meta_key(thread_id, checkpoint_id))
        if meta is None:
            return None
        result: dict[str, Any] = meta.get("metadata", {})
        return result

    async def list_threads(self, limit: int = 100, pattern: str = "*") -> list[str]:
        self._require_capability("list_threads")
        listing = await self._list_keys(self.prefix, delimiter="/")
        threads: list[str] = []
        for p in listing["prefixes"]:
            thread_id = p[len(self.prefix) :].rstrip("/")
            if thread_id:
                threads.append(thread_id)
        return threads[:limit]

    async def list_with_metadata(self, limit: int = 100) -> list[dict[str, Any]]:
        self._require_capability("list_with_metadata")
        listing = await self._list_keys(self.prefix)
        results: list[dict[str, Any]] = []
        for obj in listing["contents"]:
            key = obj["Key"]
            if not key.endswith(".meta.json"):
                continue
            meta = await self._get_json(key)
            if meta is not None:
                results.append(meta)
            if len(results) >= limit:
                break
        return results

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        self._require_capability("branching")
        client = self._get_client()
        src_prefix = self._thread_prefix(source_thread_id)
        dst_prefix = self._thread_prefix(dest_thread_id)
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
            src_key = obj["Key"]
            dst_key = dst_prefix + src_key[len(src_prefix) :]
            await asyncio.to_thread(_copy, src_key, dst_key)
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
