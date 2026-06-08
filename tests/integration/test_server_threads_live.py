# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""End-to-end: AgentServer's ``/threads/{id}`` round-trips real agent state.

Stands up an in-process FastAPI server backed by a real Agent +
``MemoryCheckpointer``, calls ``/invoke`` to populate a thread, then
verifies ``GET /threads/{tid}`` returns the persisted state and
``DELETE`` removes it.

Runs against two providers:

* OpenAI direct (gpt-4o-mini) — needs ``OPENAI_API_KEY``
* Anthropic direct (claude-haiku-4.5) — needs ``ANTHROPIC_API_KEY``

Each provider that's unavailable is silently skipped.
"""

from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.integration]


_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))


def _build_openai_native():
    if not _OPENAI:
        return None
    pytest.importorskip("openai")
    from tulip.models.native.openai import OpenAIModel

    return OpenAIModel(model=os.environ.get("TULIP_OPENAI_TEST_MODEL", "gpt-4o-mini"))


def _build_anthropic_native():
    if not _ANTHROPIC:
        return None
    pytest.importorskip("anthropic")
    from tulip.models.native.anthropic import AnthropicModel

    return AnthropicModel(
        model=os.environ.get("TULIP_ANTHROPIC_TEST_MODEL", "claude-haiku-4-5-20251001")
    )


_PROVIDER_FACTORIES = [
    pytest.param(_build_openai_native, id="openai-native-gpt-4o-mini"),
    pytest.param(_build_anthropic_native, id="anthropic-native-claude-haiku"),
]


def _build_server(model):
    from tulip.agent import Agent
    from tulip.memory.backends.memory import MemoryCheckpointer
    from tulip.server import AgentServer

    agent = Agent(
        model=model,
        tools=[],
        system_prompt="Reply briefly.",
        checkpointer=MemoryCheckpointer(),
        max_iterations=2,
    )
    return AgentServer(agent=agent)


@pytest.mark.parametrize("factory", _PROVIDER_FACTORIES)
def test_threads_round_trip(factory):
    """Full ``/invoke`` → ``GET /threads`` → ``DELETE /threads`` cycle.

    Confirms that the persisted thread is reachable via the new endpoints
    on real provider runs (not just unit-level mock state).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    model = factory()
    if model is None:
        pytest.skip("provider credentials missing")

    server = _build_server(model)
    client = TestClient(server.app)

    # 1) Drive the agent through /invoke; a thread gets persisted.
    invoke_resp = client.post(
        "/invoke",
        json={"prompt": "What is the capital of France?", "thread_id": "round-trip"},
    )
    assert invoke_resp.status_code == 200, invoke_resp.text

    # 2) GET /threads/round-trip returns the persisted state.
    get_resp = client.get("/threads/round-trip")
    assert get_resp.status_code == 200, get_resp.text
    thread = get_resp.json()
    assert thread["thread_id"] == "round-trip"
    assert thread["iteration"] >= 1
    # Should at least carry the user prompt + an assistant reply.
    roles = [m.get("role") for m in thread["messages"]]
    assert "user" in roles
    assert "assistant" in roles

    # 3) GET on an unknown thread is a clean 404.
    miss_resp = client.get("/threads/never-existed")
    assert miss_resp.status_code == 404

    # 4) DELETE removes it; subsequent GET is 404.
    del_resp = client.delete("/threads/round-trip")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True
    after = client.get("/threads/round-trip")
    assert after.status_code == 404
