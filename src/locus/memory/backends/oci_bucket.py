# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI Object Storage checkpointer.

``OCIBucketBackend`` implements :class:`BaseCheckpointer` directly, so it
can be passed to ``Agent(checkpointer=...)`` without any adapter glue.

Object layout::

    {prefix}{thread_id}/{checkpoint_id}.json      # AgentState payload
    {prefix}{thread_id}/{checkpoint_id}.meta.json # per-checkpoint metadata
    {prefix}{thread_id}/_latest                   # text pointer to the
                                                  # newest checkpoint id

The ``_latest`` pointer lets ``load(thread_id)`` do a single GET instead of
a list + sort every turn — matters when the bucket is hot.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from oci.object_storage import ObjectStorageClient

    from locus.core.state import AgentState


_LATEST_POINTER = "_latest"


class OCIBucketConfig(BaseModel):
    """Configuration for OCI Object Storage checkpointer."""

    bucket_name: str
    namespace: str
    prefix: str = "locus/checkpoints/"
    compartment_id: str | None = None
    profile_name: str = "DEFAULT"
    config_file: str = "~/.oci/config"
    auth_type: str = "api_key"  # api_key | security_token | instance_principal | resource_principal
    region: str | None = None


class OCIBucketBackend(BaseCheckpointer):
    """OCI Object Storage-backed checkpointer.

    Durable, per-checkpoint storage with lifecycle-policy support. Pass the
    instance directly to :class:`~locus.agent.Agent` — no adapter needed.

    Example::

        checkpointer = OCIBucketBackend(
            bucket_name="my-checkpoints",
            namespace="yzhbfkqxqsx9",
            profile_name="API_KEY_AUTH",
        )
        agent = Agent(config=cfg, checkpointer=checkpointer)

    With an OCI compute instance principal::

        checkpointer = OCIBucketBackend(
            bucket_name="my-checkpoints",
            namespace="yzhbfkqxqsx9",
            auth_type="instance_principal",
        )

    Capabilities:
        - ``list_threads`` — yes (via object prefix delimiter listing)
        - ``list_with_metadata`` — yes
        - ``metadata_query`` — yes (via ``get_metadata``)
        - ``branching`` — yes (via ``copy_thread``)
        - ``vacuum`` — yes (prefer bucket lifecycle policies for prod)
        - ``persistent_checkpoint_ids`` — yes
    """

    def __init__(
        self,
        bucket_name: str,
        namespace: str,
        prefix: str = "locus/checkpoints/",
        profile_name: str = "DEFAULT",
        auth_type: str = "api_key",
        region: str | None = None,
        retry_strategy: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = OCIBucketConfig(
            bucket_name=bucket_name,
            namespace=namespace,
            prefix=prefix,
            profile_name=profile_name,
            auth_type=auth_type,
            region=region,
            **kwargs,
        )
        self._client: ObjectStorageClient | None = None
        self._initialized = False
        # Override the default retry strategy by passing one explicitly.
        # Default (None) resolves to ``oci.retry.DEFAULT_RETRY_STRATEGY`` at
        # first call — exponential backoff with jitter on 429 / 5xx /
        # transport errors. Pass ``oci.retry.NoneRetryStrategy()`` to
        # disable retries (e.g. for tests).
        self._retry_strategy: Any = retry_strategy

    def _get_retry_strategy(self) -> Any:
        """Return the OCI retry strategy used for every transient-prone call."""
        if self._retry_strategy is not None:
            return self._retry_strategy
        try:
            import oci
        except ImportError:  # pragma: no cover
            return None
        return oci.retry.DEFAULT_RETRY_STRATEGY

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

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

    def _get_client(self) -> ObjectStorageClient:
        if self._client is not None:
            return self._client

        try:
            import oci
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "OCIBucketBackend requires the 'oci' package. Install with: pip install locus[oci]"
            ) from e

        from pathlib import Path

        cfg = self.config
        config_file = Path(cfg.config_file).expanduser()

        if cfg.auth_type == "instance_principal":
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self._client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
        elif cfg.auth_type == "resource_principal":
            signer = oci.auth.signers.get_resource_principals_signer()
            self._client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
        elif cfg.auth_type == "security_token":
            oci_config = oci.config.from_file(str(config_file), cfg.profile_name)
            token_file = oci_config.get("security_token_file")
            if not token_file:
                raise ValueError("security_token_file not found in config")
            with open(Path(token_file).expanduser()) as f:
                token = f.read().strip()
            private_key = oci.signer.load_private_key_from_file(oci_config.get("key_file"))
            signer = oci.auth.signers.SecurityTokenSigner(token=token, private_key=private_key)
            self._client = oci.object_storage.ObjectStorageClient(config=oci_config, signer=signer)
        else:
            oci_config = oci.config.from_file(str(config_file), cfg.profile_name)
            if cfg.region:
                oci_config["region"] = cfg.region
            self._client = oci.object_storage.ObjectStorageClient(oci_config)

        return self._client

    async def _ensure_bucket(self) -> None:
        if self._initialized:
            return

        def check_bucket() -> None:
            client = self._get_client()
            retry = self._get_retry_strategy()
            try:
                client.get_bucket(
                    namespace_name=self.config.namespace,
                    bucket_name=self.config.bucket_name,
                    retry_strategy=retry,
                )
            except Exception as e:
                if "BucketNotFound" in str(e) and self.config.compartment_id:
                    from oci.object_storage.models import CreateBucketDetails

                    client.create_bucket(
                        namespace_name=self.config.namespace,
                        create_bucket_details=CreateBucketDetails(
                            name=self.config.bucket_name,
                            compartment_id=self.config.compartment_id,
                            storage_tier="Standard",
                            public_access_type="NoPublicAccess",
                        ),
                        retry_strategy=retry,
                    )
                else:
                    raise

        await asyncio.to_thread(check_bucket)
        self._initialized = True

    # ------------------------------------------------------------------
    # Object path helpers
    # ------------------------------------------------------------------

    def _thread_prefix(self, thread_id: str) -> str:
        return f"{self.config.prefix}{thread_id}/"

    def _checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{checkpoint_id}.json"

    def _meta_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{checkpoint_id}.meta.json"

    def _latest_key(self, thread_id: str) -> str:
        return f"{self._thread_prefix(thread_id)}{_LATEST_POINTER}"

    # ------------------------------------------------------------------
    # Raw object-level helpers (private)
    # ------------------------------------------------------------------

    async def _put_json(self, object_name: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        await self._put_bytes(object_name, body, "application/json")

    async def _put_bytes(self, object_name: str, body: bytes, content_type: str) -> None:
        client = self._get_client()
        retry = self._get_retry_strategy()

        def _put() -> None:
            client.put_object(
                namespace_name=self.config.namespace,
                bucket_name=self.config.bucket_name,
                object_name=object_name,
                put_object_body=body,
                content_type=content_type,
                retry_strategy=retry,
            )

        await asyncio.to_thread(_put)

    async def _get_json(self, object_name: str) -> dict[str, Any] | None:
        body = await self._get_bytes(object_name)
        if body is None:
            return None
        data: dict[str, Any] = json.loads(body.decode("utf-8"))
        return data

    async def _get_bytes(self, object_name: str) -> bytes | None:
        client = self._get_client()
        retry = self._get_retry_strategy()

        def _get() -> bytes | None:
            try:
                response = client.get_object(
                    namespace_name=self.config.namespace,
                    bucket_name=self.config.bucket_name,
                    object_name=object_name,
                    retry_strategy=retry,
                )
                content: bytes = response.data.content
                return content
            except Exception as e:
                if "ObjectNotFound" in str(e) or "404" in str(e):
                    return None
                raise

        return await asyncio.to_thread(_get)

    async def _delete_object(self, object_name: str) -> bool:
        client = self._get_client()
        retry = self._get_retry_strategy()

        def _delete() -> bool:
            try:
                client.delete_object(
                    namespace_name=self.config.namespace,
                    bucket_name=self.config.bucket_name,
                    object_name=object_name,
                    retry_strategy=retry,
                )
                return True
            except Exception as e:
                if "ObjectNotFound" in str(e) or "404" in str(e):
                    return False
                raise

        return await asyncio.to_thread(_delete)

    async def _list_objects(
        self,
        prefix: str,
        limit: int = 1000,
        delimiter: str | None = None,
    ) -> tuple[list[Any], list[str]]:
        """Return (objects, prefixes) from a ListObjects call."""
        client = self._get_client()
        retry = self._get_retry_strategy()

        def _list() -> tuple[list[Any], list[str]]:
            kwargs: dict[str, Any] = {
                "namespace_name": self.config.namespace,
                "bucket_name": self.config.bucket_name,
                "prefix": prefix,
                "limit": min(limit, 1000),
                "fields": "name,timeModified,size",
                "retry_strategy": retry,
            }
            if delimiter is not None:
                kwargs["delimiter"] = delimiter
            response = client.list_objects(**kwargs)
            objects = list(response.data.objects or [])
            prefixes = list(response.data.prefixes or [])
            return objects, prefixes

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
            "updated_at": now.isoformat(),
            "metadata": metadata or {},
        }

        # Write checkpoint + metadata in parallel.
        await asyncio.gather(
            self._put_json(self._checkpoint_key(thread_id, checkpoint_id), payload),
            self._put_json(self._meta_key(thread_id, checkpoint_id), meta_payload),
        )
        # Update the "latest" pointer only after the payload is durable.
        await self._put_bytes(
            self._latest_key(thread_id),
            checkpoint_id.encode("utf-8"),
            "text/plain",
        )
        return checkpoint_id

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        from locus.core.state import AgentState

        await self._ensure_bucket()

        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(thread_id))
            if pointer is None:
                # No pointer — fall back to listing (old bucket, restored
                # from lifecycle, or pointer deleted mid-flight).
                ids = await self.list_checkpoints(thread_id, limit=1)
                if not ids:
                    return None
                checkpoint_id = ids[0]
            else:
                checkpoint_id = pointer.decode("utf-8").strip()

        data = await self._get_json(self._checkpoint_key(thread_id, checkpoint_id))
        if data is None:
            return None
        return AgentState.from_checkpoint(data["state"])

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        await self._ensure_bucket()

        objects, _ = await self._list_objects(self._thread_prefix(thread_id), limit=1000)

        thread_prefix_len = len(self._thread_prefix(thread_id))
        entries: list[tuple[str, Any]] = []
        for obj in objects:
            name = obj.name
            if name.endswith((".meta.json", _LATEST_POINTER)):
                continue
            if not name.endswith(".json"):
                continue
            checkpoint_id = name[thread_prefix_len:-5]
            entries.append((checkpoint_id, getattr(obj, "time_modified", None)))

        # Newest first. time_modified comes through as datetime; fall back to
        # id string if absent (shouldn't happen in practice).
        entries.sort(key=lambda x: x[1] or "", reverse=True)
        return [cp_id for cp_id, _ in entries[:limit]]

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        await self._ensure_bucket()

        if checkpoint_id is not None:
            results = await asyncio.gather(
                self._delete_object(self._checkpoint_key(thread_id, checkpoint_id)),
                self._delete_object(self._meta_key(thread_id, checkpoint_id)),
            )
            # If we just deleted the latest, clear the pointer too.
            pointer = await self._get_bytes(self._latest_key(thread_id))
            if pointer is not None and pointer.decode("utf-8").strip() == checkpoint_id:
                await self._delete_object(self._latest_key(thread_id))
            return any(results)

        # Delete every object under the thread prefix.
        objects, _ = await self._list_objects(self._thread_prefix(thread_id), limit=1000)
        if not objects:
            return False
        await asyncio.gather(*(self._delete_object(o.name) for o in objects))
        return True

    async def exists(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        await self._ensure_bucket()

        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(thread_id))
            return pointer is not None
        return (await self._get_bytes(self._checkpoint_key(thread_id, checkpoint_id))) is not None

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    async def get_metadata(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        await self._ensure_bucket()

        if checkpoint_id is None:
            pointer = await self._get_bytes(self._latest_key(thread_id))
            if pointer is None:
                return None
            checkpoint_id = pointer.decode("utf-8").strip()

        return await self._get_json(self._meta_key(thread_id, checkpoint_id))

    async def list_threads(
        self,
        limit: int = 100,
        pattern: str = "*",
    ) -> list[str]:
        await self._ensure_bucket()

        # delimiter="/" returns synthetic "sub-folders" under our prefix; each
        # entry is exactly `{prefix}{thread_id}/`, which maps 1:1 to threads.
        _, prefixes = await self._list_objects(
            self.config.prefix,
            limit=limit * 4 if pattern != "*" else limit,
            delimiter="/",
        )

        prefix_len = len(self.config.prefix)
        threads: list[str] = []
        for p in prefixes:
            if not p.endswith("/"):
                continue
            threads.append(p[prefix_len:-1])

        if pattern != "*":
            import fnmatch

            threads = [t for t in threads if fnmatch.fnmatch(t, pattern)]

        return threads[:limit]

    async def list_with_metadata(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        await self._ensure_bucket()

        threads = await self.list_threads(limit=limit)

        async def _fetch(thread_id: str) -> dict[str, Any] | None:
            meta = await self.get_metadata(thread_id)
            if meta is None:
                return None
            return {
                "thread_id": thread_id,
                "checkpoint_id": meta.get("checkpoint_id"),
                "updated_at": meta.get("updated_at"),
                "metadata": meta.get("metadata", {}),
            }

        results = await asyncio.gather(*(_fetch(t) for t in threads))
        return [r for r in results if r is not None]

    async def vacuum(self, older_than_days: int = 30) -> int:
        """Delete threads whose latest checkpoint is older than the cutoff.

        For production, prefer an OCI Object Storage lifecycle rule — it runs
        server-side and costs nothing in client CPU.
        """
        await self._ensure_bucket()

        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        threads = await self.list_threads(limit=1000)

        async def _maybe_delete(thread_id: str) -> bool:
            meta = await self.get_metadata(thread_id)
            if not meta:
                return False
            updated = meta.get("updated_at")
            if not updated:
                return False
            try:
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return False
            if updated_dt >= cutoff:
                return False
            await self.delete(thread_id)
            return True

        results = await asyncio.gather(*(_maybe_delete(t) for t in threads))
        return sum(1 for r in results if r)

    async def copy_thread(
        self,
        source_thread_id: str,
        dest_thread_id: str,
    ) -> bool:
        """Copy every checkpoint under source to dest (for branching)."""
        await self._ensure_bucket()

        checkpoints = await self.list_checkpoints(source_thread_id, limit=1000)
        if not checkpoints:
            return False

        for cp_id in checkpoints:
            payload, meta = await asyncio.gather(
                self._get_json(self._checkpoint_key(source_thread_id, cp_id)),
                self._get_json(self._meta_key(source_thread_id, cp_id)),
            )
            if payload is None:
                continue
            payload["thread_id"] = dest_thread_id
            if meta is not None:
                meta["thread_id"] = dest_thread_id
                await asyncio.gather(
                    self._put_json(self._checkpoint_key(dest_thread_id, cp_id), payload),
                    self._put_json(self._meta_key(dest_thread_id, cp_id), meta),
                )
            else:
                await self._put_json(self._checkpoint_key(dest_thread_id, cp_id), payload)

        # Point dest's latest at the most-recent source checkpoint.
        await self._put_bytes(
            self._latest_key(dest_thread_id),
            checkpoints[0].encode("utf-8"),
            "text/plain",
        )
        return True

    def __repr__(self) -> str:
        return (
            f"OCIBucketBackend(bucket='{self.config.bucket_name}', "
            f"namespace='{self.config.namespace}', prefix='{self.config.prefix}')"
        )
