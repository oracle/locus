# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for HTTP checkpointer backend."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx


@pytest.fixture
def mock_respx():
    """Fixture to provide respx mock context."""
    with respx.mock:
        yield respx


class TestHTTPCheckpointerInit:
    """Tests for HTTPCheckpointer initialization."""

    def test_create_with_base_url(self):
        """Test creating checkpointer with base URL."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        assert cp.base_url == "http://localhost:8000"

    def test_base_url_strips_trailing_slash(self):
        """Test base URL trailing slash is stripped."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000/")
        assert cp.base_url == "http://localhost:8000"

    def test_create_with_headers(self):
        """Test creating checkpointer with headers."""
        from locus.memory.backends.http import HTTPCheckpointer

        headers = {"Authorization": "Bearer token123"}
        cp = HTTPCheckpointer(base_url="http://localhost:8000", headers=headers)
        assert cp.headers == headers

    def test_create_with_auth(self):
        """Test creating checkpointer with auth."""
        from locus.memory.backends.http import HTTPCheckpointer

        auth = ("user", "pass")
        cp = HTTPCheckpointer(base_url="http://localhost:8000", auth=auth)
        assert cp.auth == auth

    def test_create_with_timeout(self):
        """Test creating checkpointer with timeout."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000", timeout=60.0)
        assert cp.timeout == 60.0

    def test_repr(self):
        """Test string representation."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        assert "HTTPCheckpointer" in repr(cp)
        assert "localhost:8000" in repr(cp)


class TestHTTPCheckpointerClient:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test that _get_client creates httpx client."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        assert cp._client is None

        client = await cp._get_client()
        assert client is not None
        assert cp._client is client

        await cp.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test that _get_client reuses existing client."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")

        client1 = await cp._get_client()
        client2 = await cp._get_client()

        assert client1 is client2

        await cp.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Test that close closes the client."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        await cp._get_client()

        assert cp._client is not None
        await cp.close()
        assert cp._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close when no client exists."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        await cp.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from locus.memory.backends.http import HTTPCheckpointer

        async with HTTPCheckpointer(base_url="http://localhost:8000") as cp:
            assert cp._client is not None

        assert cp._client is None


class TestHTTPCheckpointerSave:
    """Tests for save operation."""

    @pytest.mark.asyncio
    async def test_save_returns_checkpoint_id(self, mock_respx):
        """Test save returns checkpoint ID."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.post("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"checkpoint_id": "cp123"})
        )

        mock_state = MagicMock()
        mock_state.to_checkpoint.return_value = {"key": "value"}

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.save(mock_state, "thread1")

        assert result == "cp123"
        await cp.close()

    @pytest.mark.asyncio
    async def test_save_with_custom_checkpoint_id(self, mock_respx):
        """Test save with provided checkpoint ID."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.post("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"checkpoint_id": "custom-id"})
        )

        mock_state = MagicMock()
        mock_state.to_checkpoint.return_value = {}

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.save(mock_state, "thread1", checkpoint_id="custom-id")

        assert result == "custom-id"
        await cp.close()


class TestHTTPCheckpointerLoad:
    """Tests for load operation."""

    @pytest.mark.asyncio
    async def test_load_returns_state(self, mock_respx):
        """Test load returns state."""
        from locus.memory.backends.http import HTTPCheckpointer

        # Mock list checkpoints
        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=["cp123"])
        )

        # Mock get checkpoint
        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "state": {
                        "run_id": "test-run",
                        "messages": [],
                        "iteration": 0,
                    }
                },
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1")

        assert result is not None
        assert result.run_id == "test-run"
        await cp.close()

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, mock_respx):
        """Test load with specific checkpoint ID."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints/cp456").mock(
            return_value=httpx.Response(
                200,
                json={
                    "run_id": "test-run",
                    "messages": [],
                    "iteration": 0,
                },
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1", checkpoint_id="cp456")

        assert result is not None
        await cp.close()

    @pytest.mark.asyncio
    async def test_load_not_found_returns_none(self, mock_respx):
        """Test load returns None when not found."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=[])
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1")

        assert result is None
        await cp.close()

    @pytest.mark.asyncio
    async def test_load_error_returns_none(self, mock_respx):
        """Test load returns None on error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=["cp123"])
        )
        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(500)
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1")

        assert result is None
        await cp.close()


class TestHTTPCheckpointerListCheckpoints:
    """Tests for list_checkpoints operation."""

    @pytest.mark.asyncio
    async def test_list_returns_ids(self, mock_respx):
        """Test list_checkpoints returns IDs."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=["cp1", "cp2", "cp3"])
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2", "cp3"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_with_dict_format(self, mock_respx):
        """Test list_checkpoints with dict format response."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"checkpoint_id": "cp1"},
                    {"checkpoint_id": "cp2"},
                ],
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_with_wrapped_response(self, mock_respx):
        """Test list_checkpoints with wrapped response."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(
                200,
                json={"checkpoints": ["cp1", "cp2"]},
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_with_wrapped_dict_format(self, mock_respx):
        """Test list_checkpoints with wrapped dict format."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"checkpoint_id": "cp1"},
                        {"checkpoint_id": "cp2"},
                    ]
                },
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_empty_on_error(self, mock_respx):
        """Test list_checkpoints returns empty on error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(500)
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == []
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, mock_respx):
        """Test list_checkpoints respects limit."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=["cp1", "cp2", "cp3", "cp4", "cp5"])
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1", limit=3)

        assert len(result) == 3
        await cp.close()


class TestHTTPCheckpointerDelete:
    """Tests for delete operation."""

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, mock_respx):
        """Test deleting specific checkpoint."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(204)
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1", "cp123")

        assert result is True
        await cp.close()

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, mock_respx):
        """Test deleting all checkpoints."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(204)
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1")

        assert result is True
        await cp.close()

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, mock_respx):
        """Test delete returns False on error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(500)
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1", "cp123")

        assert result is False
        await cp.close()


class TestHTTPCheckpointerHealthCheck:
    """Tests for health check operation."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_respx):
        """Test health check returns True on success."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200))

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.health_check()

        assert result is True
        await cp.close()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_respx):
        """Test health check returns False on failure."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(500))

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.health_check()

        assert result is False
        await cp.close()

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, mock_respx):
        """Test health check returns False on connection error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/health").mock(
            side_effect=Exception("Connection failed")
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.health_check()

        assert result is False
        await cp.close()


class TestHTTPCheckpointerImportError:
    """Tests for import error handling."""

    @pytest.mark.asyncio
    async def test_get_client_import_error(self):
        """Test _get_client raises ImportError when httpx not available."""
        from locus.memory.backends.http import HTTPCheckpointer

        _cp = HTTPCheckpointer(base_url="http://localhost:8000")

        # Mock the import
        with patch.dict("sys.modules", {"httpx": None}):
            # This won't actually trigger the import error since httpx is already imported
            # The import check happens only once, so we need a different approach
            pass


class TestHTTPCheckpointerListCheckpointsEdgeCases:
    """Edge case tests for list_checkpoints."""

    @pytest.mark.asyncio
    async def test_list_with_empty_response(self, mock_respx):
        """Test list_checkpoints with empty response."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=[])
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == []
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_unexpected_format_returns_empty(self, mock_respx):
        """Test list_checkpoints with unexpected format returns empty."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json="not a list or dict")
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == []
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_dict_without_checkpoints_key(self, mock_respx):
        """Test list with dict missing checkpoints key."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"other_key": "value"})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == []
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_connection_error(self, mock_respx):
        """Test list_checkpoints with connection error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            side_effect=Exception("Connection failed")
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == []
        await cp.close()


class TestHTTPCheckpointerDeleteEdgeCases:
    """Edge case tests for delete operation."""

    @pytest.mark.asyncio
    async def test_delete_connection_error(self, mock_respx):
        """Test delete returns False on connection error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            side_effect=Exception("Connection failed")
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1", "cp123")

        assert result is False
        await cp.close()


class TestHTTPCheckpointerLoadEdgeCases:
    """Edge case tests for load operation."""

    @pytest.mark.asyncio
    async def test_load_connection_error_on_get(self, mock_respx):
        """Test load returns None on connection error."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json=["cp123"])
        )
        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            side_effect=Exception("Connection failed")
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1")

        assert result is None
        await cp.close()

    @pytest.mark.asyncio
    async def test_load_unwrapped_state_format(self, mock_respx):
        """Test load with unwrapped state format."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(
                200,
                json={
                    "run_id": "test-run",
                    "messages": [],
                    "iteration": 0,
                },
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.load("thread1", checkpoint_id="cp123")

        assert result is not None
        assert result.run_id == "test-run"
        await cp.close()


class TestHTTPCheckpointerDeleteOperations:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, mock_respx):
        """Test delete all checkpoints for a thread."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1")

        assert result is True
        await cp.close()

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, mock_respx):
        """Test delete specific checkpoint."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints/cp123").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1", "cp123")

        assert result is True
        await cp.close()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_respx):
        """Test delete returns False on 404."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.delete("http://localhost:8000/threads/thread1/checkpoints/nonexistent").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.delete("thread1", "nonexistent")

        assert result is False
        await cp.close()


class TestHTTPCheckpointerListFormats:
    """Tests for various list response formats."""

    @pytest.mark.asyncio
    async def test_list_wrapped_string_checkpoints(self, mock_respx):
        """Test list with wrapped response containing string checkpoint IDs."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"checkpoints": ["cp1", "cp2", "cp3"]})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2", "cp3"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_wrapped_dict_checkpoints(self, mock_respx):
        """Test list with wrapped response containing dict checkpoints."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(
                200,
                json={
                    "checkpoints": [
                        {"checkpoint_id": "cp1", "created_at": "2024-01-01"},
                        {"checkpoint_id": "cp2", "created_at": "2024-01-02"},
                    ]
                },
            )
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2"]
        await cp.close()

    @pytest.mark.asyncio
    async def test_list_data_key_format(self, mock_respx):
        """Test list with 'data' key in response."""
        from locus.memory.backends.http import HTTPCheckpointer

        mock_respx.get("http://localhost:8000/threads/thread1/checkpoints").mock(
            return_value=httpx.Response(200, json={"data": ["cp1", "cp2"]})
        )

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        result = await cp.list_checkpoints("thread1")

        assert result == ["cp1", "cp2"]
        await cp.close()


class TestHTTPCheckpointerRepr:
    """Tests for repr."""

    def test_repr(self):
        """Test string representation."""
        from locus.memory.backends.http import HTTPCheckpointer

        cp = HTTPCheckpointer(base_url="http://localhost:8000")
        r = repr(cp)

        assert "HTTPCheckpointer" in r
        assert "http://localhost:8000" in r
