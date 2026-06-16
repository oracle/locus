# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""HTTP API checkpoint backend for Locus.

This backend stores checkpoints via HTTP API, enabling:
- Centralized storage for distributed agents
- Integration with external persistence services
- Cloud-based checkpoint storage

Requires httpx for async HTTP requests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import quote
from uuid import uuid4

from locus.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from locus.core.state import AgentState


def _encode_path_segment(value: str) -> str:
    """Percent-encode a value so it is safe to embed as a single URL path segment.

    Prevents path traversal and query/fragment injection via caller-supplied
    thread_id / checkpoint_id values. `safe=""` ensures `/`, `?`, `#`, and `..`
    characters are all encoded.
    """
    return quote(str(value), safe="")


class HTTPCheckpointer(BaseCheckpointer):
    """
    HTTP API-based checkpointer for remote storage.

    Stores checkpoints via HTTP API calls, suitable for distributed
    systems or cloud-based storage backends.

    The API is expected to implement the following endpoints:
    - POST /threads/{thread_id}/checkpoints - Create checkpoint
    - GET /threads/{thread_id}/checkpoints/{checkpoint_id} - Get checkpoint
    - GET /threads/{thread_id}/checkpoints - List checkpoints
    - DELETE /threads/{thread_id}/checkpoints/{checkpoint_id} - Delete checkpoint

    Args:
        base_url: Base URL of the checkpoint API
        headers: Additional headers to include in requests
        auth: Authentication tuple (username, password) for basic auth
        timeout: Request timeout in seconds

    Example:
        ```python
        checkpointer = HTTPCheckpointer(
            base_url="https://api.example.com/v1",
            headers={"Authorization": "Bearer token"},
        )

        # Save state
        checkpoint_id = await checkpointer.save(state, "thread-1")

        # Load state
        restored = await checkpointer.load("thread-1")
        ```
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create the HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for HTTPCheckpointer. Install it with: pip install httpx"
                ) from e

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save agent state via HTTP POST.

        Args:
            state: Current agent state
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint ID
            metadata: Optional metadata for querying/filtering checkpoints

        Returns:
            Checkpoint ID for the saved state

        Raises:
            HTTPError: If the request fails
        """
        checkpoint_id = checkpoint_id or uuid4().hex

        client = await self._get_client()

        payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "created_at": datetime.now(UTC).isoformat(),
            "state": state.to_checkpoint(),
            "metadata": metadata or {},
        }

        response = await client.post(
            f"/threads/{_encode_path_segment(thread_id)}/checkpoints",
            json=payload,
        )
        response.raise_for_status()

        # Extract checkpoint_id from response if provided
        result: dict[str, Any] = response.json()
        returned_id: str = result.get("checkpoint_id", checkpoint_id)
        return returned_id

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """
        Load agent state via HTTP GET.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Restored AgentState or None if not found
        """
        from locus.core.state import AgentState

        client = await self._get_client()

        if checkpoint_id is None:
            # Get latest checkpoint
            checkpoints = await self.list_checkpoints(thread_id, limit=1)
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[0]

        try:
            response = await client.get(
                f"/threads/{_encode_path_segment(thread_id)}"
                f"/checkpoints/{_encode_path_segment(checkpoint_id)}",
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001 — missing/unreachable == absent by design
            return None

        data = response.json()

        # Handle both wrapped and unwrapped state formats
        state_data = data.get("state", data)
        return AgentState.from_checkpoint(state_data)

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """
        List available checkpoints via HTTP GET.

        Args:
            thread_id: Thread identifier
            limit: Maximum number to return

        Returns:
            List of checkpoint IDs, newest first
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"/threads/{_encode_path_segment(thread_id)}/checkpoints",
                params={"limit": limit},
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001 — unreachable == empty by design
            return []

        data = response.json()

        # Handle various response formats
        if isinstance(data, list):
            # Direct list of checkpoints
            if data and isinstance(data[0], str):
                return data[:limit]
            if data and isinstance(data[0], dict):
                return [cp["checkpoint_id"] for cp in data[:limit]]
        elif isinstance(data, dict):
            # Wrapped response
            checkpoints = data.get("checkpoints", data.get("data", []))
            if checkpoints and isinstance(checkpoints[0], str):
                truncated: list[str] = checkpoints[:limit]
                return truncated
            if checkpoints and isinstance(checkpoints[0], dict):
                return [cp["checkpoint_id"] for cp in checkpoints[:limit]]

        return []

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """
        Delete checkpoint(s) via HTTP DELETE.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint to delete (all if None)

        Returns:
            True if deletion was successful
        """
        client = await self._get_client()

        try:
            if checkpoint_id is None:
                # Delete all checkpoints for thread
                response = await client.delete(
                    f"/threads/{_encode_path_segment(thread_id)}/checkpoints",
                )
            else:
                response = await client.delete(
                    f"/threads/{_encode_path_segment(thread_id)}"
                    f"/checkpoints/{_encode_path_segment(checkpoint_id)}",
                )
            response.raise_for_status()
            return True
        except Exception:  # noqa: BLE001 — delete is idempotent; report boolean result
            return False

    async def health_check(self) -> bool:
        """
        Check if the API is reachable.

        Returns:
            True if the API responds successfully
        """
        client = await self._get_client()

        try:
            response = await client.get("/health")
            ok: bool = response.status_code < 400
            return ok
        except Exception:  # noqa: BLE001 — health check is a boolean probe
            return False

    def __repr__(self) -> str:
        return f"HTTPCheckpointer(base_url={self.base_url!r})"

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()
