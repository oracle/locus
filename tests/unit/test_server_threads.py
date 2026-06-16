# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``GET /threads/{tid}`` and ``DELETE /threads/{tid}``.

The README and ``docs/concepts/server.md`` documented these endpoints from
day one but the routes weren't actually registered. Now that they are,
guard:

- 404 when no checkpointer is configured (helpful error, not a 500).
- 404 when the thread isn't found.
- 200 + payload shape on a successful load.
- 200 + ``{"deleted": true}`` on a successful delete; ``{"deleted": false}``
  if the thread didn't exist (idempotent delete).
- Principal scoping: a per-key prefix is applied to ``thread_id`` so two
  API keys can't read each other's threads even when sharing the same
  AgentServer instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("fastapi", reason="fastapi not installed"),
    reason="fastapi not installed",
)


def _build_state(thread_id: str) -> Any:
    """Construct a real ``AgentState`` so ``model_dump(mode='json')`` works.

    A MagicMock would slip through Pydantic's serializer with garbage values;
    we want the round-trip to look like a real production response.
    """
    from locus.core.messages import Message
    from locus.core.state import AgentState, ToolExecution

    state = AgentState()
    state = state.with_message(Message.system("You are helpful."))
    state = state.with_message(Message.user("Hi"))
    state = state.with_message(Message.assistant("Hello!"))
    state = state.with_tool_execution(
        ToolExecution(
            tool_name="echo",
            tool_call_id="tc-1",
            arguments={"msg": "hi"},
            result="hi",
        )
    )
    return state.model_copy(update={"iteration": 3})


def _make_agent(checkpointer: Any | None) -> MagicMock:
    """Mock agent whose ``config.checkpointer`` is the provided fake."""
    agent = MagicMock()
    agent.config.checkpointer = checkpointer
    return agent


# =============================================================================
# 404s when no checkpointer is wired
# =============================================================================


class TestNoCheckpointer:
    def test_get_thread_returns_404(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        server = AgentServer(agent=_make_agent(None))
        client = TestClient(server.app)
        r = client.get("/threads/foo")
        assert r.status_code == 404
        assert "checkpointer" in r.json()["detail"].lower()

    def test_delete_thread_returns_404(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        server = AgentServer(agent=_make_agent(None))
        client = TestClient(server.app)
        r = client.delete("/threads/foo")
        assert r.status_code == 404
        assert "checkpointer" in r.json()["detail"].lower()


# =============================================================================
# Happy path
# =============================================================================


class TestGetThread:
    def test_returns_state_payload(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        state = _build_state("user-c42")
        ckp = MagicMock()
        ckp.load = AsyncMock(return_value=state)

        server = AgentServer(agent=_make_agent(ckp))
        client = TestClient(server.app)
        r = client.get("/threads/user-c42")

        assert r.status_code == 200
        data = r.json()
        assert data["thread_id"] == "user-c42"
        assert data["iteration"] == 3
        assert isinstance(data["messages"], list)
        assert any(m.get("role") == "assistant" for m in data["messages"])
        assert isinstance(data["tool_executions"], list)
        assert data["tool_executions"][0]["tool_name"] == "echo"
        # The principal prefix should be applied at the storage layer; the
        # client-facing response keeps the unprefixed id.
        assert data["thread_id"] == "user-c42"

    def test_404_when_thread_missing(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.load = AsyncMock(return_value=None)
        server = AgentServer(agent=_make_agent(ckp))
        client = TestClient(server.app)
        r = client.get("/threads/missing")
        assert r.status_code == 404
        assert "missing" in r.json()["detail"]


class TestDeleteThread:
    def test_deletes_existing_thread(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.delete = AsyncMock(return_value=True)
        server = AgentServer(agent=_make_agent(ckp))
        client = TestClient(server.app)
        r = client.delete("/threads/user-c42")
        assert r.status_code == 200
        assert r.json() == {"thread_id": "user-c42", "deleted": True}

    def test_idempotent_when_missing(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.delete = AsyncMock(return_value=False)
        server = AgentServer(agent=_make_agent(ckp))
        client = TestClient(server.app)
        r = client.delete("/threads/never-existed")
        assert r.status_code == 200
        assert r.json() == {"thread_id": "never-existed", "deleted": False}


# =============================================================================
# Principal scoping
# =============================================================================


class TestPrincipalScoping:
    def test_thread_id_is_prefixed_with_principal_hash(self):
        """When an API key is set, the checkpointer.load argument must be
        scoped — different keys must hit different namespaces.
        """
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        state = _build_state("t1")
        ckp = MagicMock()
        ckp.load = AsyncMock(return_value=state)

        server = AgentServer(agent=_make_agent(ckp), api_key="alice-key")
        client = TestClient(server.app)
        r = client.get(
            "/threads/t1",
            headers={"Authorization": "Bearer alice-key"},
        )
        assert r.status_code == 200

        # The principal-scoped id sent to the checkpointer must NOT be the
        # raw client-supplied id — that would let two clients sharing one
        # AgentServer collide.
        scoped_id = ckp.load.call_args.args[0]
        assert scoped_id != "t1"
        assert "t1" in scoped_id  # the unprefixed id is still in the scoped form

    def test_two_keys_get_different_namespaces(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.load = AsyncMock(return_value=None)
        server = AgentServer(agent=_make_agent(ckp), api_key="alice-key")
        client = TestClient(server.app)

        client.get("/threads/shared", headers={"Authorization": "Bearer alice-key"})
        first_call = ckp.load.call_args.args[0]

        # New server instance with a different key — same client thread name,
        # different scoped id at the storage layer.
        server2 = AgentServer(agent=_make_agent(ckp), api_key="bob-key")
        client2 = TestClient(server2.app)
        client2.get("/threads/shared", headers={"Authorization": "Bearer bob-key"})
        second_call = ckp.load.call_args.args[0]

        assert first_call != second_call


class TestAuthRequired:
    def test_get_threads_rejected_without_bearer(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.load = AsyncMock(return_value=None)
        server = AgentServer(agent=_make_agent(ckp), api_key="secret")
        client = TestClient(server.app)
        r = client.get("/threads/any")
        assert r.status_code in (401, 403)

    def test_delete_threads_rejected_without_bearer(self):
        from fastapi.testclient import TestClient

        from locus.server import AgentServer

        ckp = MagicMock()
        ckp.delete = AsyncMock(return_value=True)
        server = AgentServer(agent=_make_agent(ckp), api_key="secret")
        client = TestClient(server.app)
        r = client.delete("/threads/any")
        assert r.status_code in (401, 403)
