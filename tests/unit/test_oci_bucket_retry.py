"""Unit tests for OCIBucketBackend retry strategy wiring.

These verify that the retry strategy passed at construction time
(or the default `oci.retry.DEFAULT_RETRY_STRATEGY`) is actually
threaded through to every OCI Object Storage call.

We do not need a real bucket — the OCI client is mocked so we can
inspect the kwargs passed to each method.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


oci = pytest.importorskip("oci")  # skip the whole module if oci SDK isn't installed

from locus.memory.backends.oci_bucket import OCIBucketBackend


@pytest.fixture
def backend_with_explicit_retry() -> tuple[OCIBucketBackend, MagicMock, object]:
    """Build a backend whose `_client` is mocked so we can spy on kwargs.

    Returns the backend, the mock client, and the retry strategy instance
    we passed in (so the test can assert identity).
    """
    sentinel_retry = oci.retry.NoneRetryStrategy()
    backend = OCIBucketBackend(
        bucket_name="test-bucket",
        namespace="test-ns",
        retry_strategy=sentinel_retry,
    )
    mock_client = MagicMock()
    backend._client = mock_client  # bypass real client construction
    return backend, mock_client, sentinel_retry


def test_default_retry_strategy_is_oci_default():
    """Without an explicit retry, the backend uses oci.retry.DEFAULT_RETRY_STRATEGY."""
    backend = OCIBucketBackend(bucket_name="b", namespace="n")
    resolved = backend._get_retry_strategy()
    assert resolved is oci.retry.DEFAULT_RETRY_STRATEGY


def test_explicit_retry_strategy_preserved(backend_with_explicit_retry):
    backend, _, sentinel = backend_with_explicit_retry
    assert backend._get_retry_strategy() is sentinel


def test_put_object_passes_retry_strategy(backend_with_explicit_retry):
    """Saving a checkpoint must thread retry_strategy through to put_object."""
    import asyncio

    backend, mock_client, sentinel = backend_with_explicit_retry

    asyncio.run(backend._put_bytes("some/key.json", b"{}", "application/json"))

    mock_client.put_object.assert_called_once()
    kwargs = mock_client.put_object.call_args.kwargs
    assert kwargs["retry_strategy"] is sentinel


def test_get_object_passes_retry_strategy(backend_with_explicit_retry):
    import asyncio

    backend, mock_client, sentinel = backend_with_explicit_retry
    mock_client.get_object.return_value.data.content = b"{}"

    asyncio.run(backend._get_bytes("some/key.json"))

    mock_client.get_object.assert_called_once()
    kwargs = mock_client.get_object.call_args.kwargs
    assert kwargs["retry_strategy"] is sentinel


def test_delete_object_passes_retry_strategy(backend_with_explicit_retry):
    import asyncio

    backend, mock_client, sentinel = backend_with_explicit_retry

    asyncio.run(backend._delete_object("some/key.json"))

    mock_client.delete_object.assert_called_once()
    kwargs = mock_client.delete_object.call_args.kwargs
    assert kwargs["retry_strategy"] is sentinel


def test_list_objects_passes_retry_strategy(backend_with_explicit_retry):
    import asyncio

    backend, mock_client, sentinel = backend_with_explicit_retry
    mock_client.list_objects.return_value.data.objects = []
    mock_client.list_objects.return_value.data.prefixes = []

    asyncio.run(backend._list_objects("some/prefix/"))

    mock_client.list_objects.assert_called_once()
    kwargs = mock_client.list_objects.call_args.kwargs
    assert kwargs["retry_strategy"] is sentinel
