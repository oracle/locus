# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.backends.s3.S3Backend``.

Exercises the error / pagination / edge-case branches that the existing
``test_memory_backends_s3.py`` (happy path via ``moto``) does not reach.
A few low-level paths (non-404 GET errors, list pagination, vacuum on
objects with missing / naive ``LastModified``) are driven through a tiny
hand-rolled S3 client double injected via the ``_client=`` seam, because
``moto`` cannot easily produce them.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

import pytest


pytest.importorskip("moto")
pytest.importorskip("boto3")

from botocore.exceptions import ClientError  # noqa: E402
from moto import mock_aws  # noqa: E402

from tulip.core.state import AgentState  # noqa: E402
from tulip.memory.backends.s3 import S3Backend  # noqa: E402


def _backend(**kwargs: Any) -> S3Backend:
    kwargs.setdefault("region_name", "us-east-1")
    return S3Backend(bucket="tulip-test", **kwargs)


# ---------------------------------------------------------------------------
# Injected client double for branches moto cannot reach
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """Minimal S3 client double exercising list / get / delete branches."""

    def __init__(
        self,
        *,
        pages: list[dict[str, Any]] | None = None,
        get_error: BaseException | None = None,
    ) -> None:
        self._pages = pages if pages is not None else [{"Contents": [], "IsTruncated": False}]
        self._idx = 0
        self._get_error = get_error
        self.deleted: list[str] = []
        self.closed = False

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        page = self._pages[min(self._idx, len(self._pages) - 1)]
        self._idx += 1
        return page

    def delete_object(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs["Key"])

    def get_object(self, **kwargs: Any) -> Any:
        if self._get_error is not None:
            raise self._get_error
        raise AssertionError("unexpected get_object call")

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Client bootstrap
# ---------------------------------------------------------------------------


def test_get_client_returns_override() -> None:
    fake = _FakeS3Client()
    backend = _backend(_client=fake)
    assert backend._get_client() is fake


def test_get_client_raises_when_boto3_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # No override, no cached client → lazy import of boto3 fails.
    monkeypatch.setitem(sys.modules, "boto3", None)
    backend = _backend()
    with pytest.raises(ImportError, match="boto3 is not installed"):
        backend._get_client()


async def test_ensure_bucket_raises_when_create_disabled() -> None:
    with mock_aws():
        backend = _backend(create_bucket=False)
        with pytest.raises(ClientError):
            await backend.save(AgentState(), "thread1")


async def test_create_bucket_with_non_us_east_region() -> None:
    with mock_aws():
        backend = S3Backend(bucket="regional-bucket", region_name="us-west-2")
        cid = await backend.save(AgentState(), "thread1")
        assert cid
        assert await backend.load("thread1") is not None


# ---------------------------------------------------------------------------
# Raw object helpers
# ---------------------------------------------------------------------------


async def test_get_bytes_reraises_non_404_client_error() -> None:
    err = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    backend = _backend(_client=_FakeS3Client(get_error=err))
    with pytest.raises(ClientError):
        await backend._get_bytes("some/key")


async def test_load_explicit_missing_checkpoint_returns_none() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")
        # Explicit, non-existent checkpoint id → _get_json returns None.
        assert await backend.load("t", "does-not-exist") is None


async def test_list_keys_paginates_with_continuation_token() -> None:
    pages = [
        {"Contents": [{"Key": "a"}], "IsTruncated": True, "NextContinuationToken": "tok"},
        {"Contents": [{"Key": "b"}], "IsTruncated": False},
    ]
    backend = _backend(_client=_FakeS3Client(pages=pages))
    result = await backend._list_keys("prefix/")
    assert [c["Key"] for c in result["contents"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# list_checkpoints / delete
# ---------------------------------------------------------------------------


async def test_list_checkpoints_skips_non_json_objects() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")
        # Drop a stray non-json object under the thread prefix.
        await backend._put_bytes(f"{backend._thread_prefix('t')}stray.txt", b"junk", "text/plain")
        ids = await backend.list_checkpoints("t")
        assert ids == ["cp1"]


async def test_delete_entire_thread() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")
        await backend.save(AgentState(), "t", checkpoint_id="cp2")

        assert await backend.delete("t") is True
        assert await backend.list_checkpoints("t") == []


async def test_delete_empty_thread_returns_false() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "other")  # create the bucket
        assert await backend.delete("nonexistent") is False


async def test_delete_only_checkpoint_removes_latest_pointer() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")

        assert await backend.delete("t", "cp1") is True
        # _latest pointer should be gone, so a latest-load returns None.
        assert await backend.load("t") is None


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


async def test_get_metadata_latest_pointer_path() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1", metadata={"tag": "v1"})
        # No checkpoint_id → resolve via the _latest pointer.
        assert await backend.get_metadata("t") == {"tag": "v1"}


async def test_get_metadata_missing_thread_returns_none() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "other")  # create the bucket
        assert await backend.get_metadata("missing") is None


async def test_get_metadata_missing_checkpoint_returns_none() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")
        assert await backend.get_metadata("t", "no-such-cp") is None


# ---------------------------------------------------------------------------
# list_with_metadata / copy_thread / vacuum / close
# ---------------------------------------------------------------------------


async def test_list_with_metadata_respects_limit() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1", metadata={"n": 1})
        await backend.save(AgentState(), "t", checkpoint_id="cp2", metadata={"n": 2})

        results = await backend.list_with_metadata(limit=1)
        assert len(results) == 1
        assert "metadata" in results[0]


async def test_copy_thread_missing_source_returns_false() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "other")  # create the bucket
        assert await backend.copy_thread("missing", "dest") is False


async def test_vacuum_handles_missing_and_naive_timestamps() -> None:
    # Object with no LastModified is skipped; a naive, old timestamp is
    # normalised to UTC and deleted.
    pages = [
        {
            "Contents": [
                {"Key": "no-timestamp"},
                {"Key": "old-naive", "LastModified": datetime(2000, 1, 1)},  # noqa: DTZ001
            ],
            "IsTruncated": False,
        }
    ]
    fake = _FakeS3Client(pages=pages)
    backend = _backend(_client=fake)
    deleted = await backend.vacuum(older_than_days=30)
    assert deleted == 1
    assert fake.deleted == ["old-naive"]


async def test_close_releases_client() -> None:
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t")  # forces a real boto3 client
        assert backend._client is not None
        await backend.close()
        assert backend._client is None
