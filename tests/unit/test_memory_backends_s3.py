# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the S3 checkpointer.

Runs against an in-process mock S3 via ``moto`` — no AWS account, no
network, free. Skips if ``moto`` / ``boto3`` are not installed.

``moto``'s ``mock_aws`` is used as a context manager inside each (async)
test, since the decorator form does not compose with ``async def``.
"""

import pytest


pytest.importorskip("moto")
pytest.importorskip("boto3")

from moto import mock_aws  # noqa: E402

from tulip.core.state import AgentState  # noqa: E402
from tulip.memory.backends.s3 import S3Backend  # noqa: E402


def _backend() -> S3Backend:
    # region us-east-1 so create_bucket needs no LocationConstraint.
    return S3Backend(bucket="tulip-test", region_name="us-east-1")


async def test_save_and_load_latest():
    with mock_aws():
        backend = _backend()
        cid = await backend.save(AgentState(), "thread1")
        assert isinstance(cid, str)
        assert cid
        assert await backend.load("thread1") is not None


async def test_load_missing_thread_returns_none():
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "other")  # creates the bucket
        assert await backend.load("nonexistent") is None


async def test_list_checkpoints_and_metadata():
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1", metadata={"tag": "first"})
        await backend.save(AgentState(), "t", checkpoint_id="cp2", metadata={"tag": "second"})

        ids = await backend.list_checkpoints("t")
        assert set(ids) == {"cp1", "cp2"}
        assert await backend.get_metadata("t", "cp1") == {"tag": "first"}


async def test_delete_specific_and_repoint_latest():
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "t", checkpoint_id="cp1")
        await backend.save(AgentState(), "t", checkpoint_id="cp2")

        assert await backend.delete("t", "cp2") is True
        assert await backend.delete("t", "cp2") is False
        assert await backend.list_checkpoints("t") == ["cp1"]


async def test_list_threads_and_branching():
    with mock_aws():
        backend = _backend()
        await backend.save(AgentState(), "alpha", checkpoint_id="c1")
        await backend.save(AgentState(), "beta", checkpoint_id="c1")

        assert set(await backend.list_threads()) == {"alpha", "beta"}
        assert await backend.copy_thread("alpha", "alpha-branch") is True
        assert await backend.load("alpha-branch", "c1") is not None


def test_capabilities():
    backend = _backend()
    caps = backend.capabilities
    assert caps.branching
    assert caps.list_threads
    assert caps.metadata_query
