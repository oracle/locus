# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for :class:`OracleAgentMemoryManager`.

The adapter is exercised against a hand-rolled stub of the
``oracleagentmemory.core.OracleAgentMemory`` surface (we patch the
import inside ``locus.memory.managers.oracle_agent_memory`` so we
don't pay the dependency cost or need a live ADB). Each test asserts
on the side-effects we care about: which scopes get passed to the
client, that the context card lands as a system message, and that
``add_messages_async`` is fed the right payload at session end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from locus.core.messages import Message, Role
from locus.core.state import AgentState


# ---------------------------------------------------------------------------
# Stubs that stand in for the real oracleagentmemory surface.
# ---------------------------------------------------------------------------


class _StubThread:
    """Records ``add_messages_async`` / ``get_context_card_async`` calls."""

    def __init__(self, thread_id: str, *, card_text: str | None = None) -> None:
        self.thread_id = thread_id
        self._card_text = card_text
        self.add_messages_async = AsyncMock(return_value=["msg-id-1"])
        self.get_context_card_async = AsyncMock(
            return_value=SimpleNamespace(text=card_text) if card_text else None
        )


class _StubOracleAgentMemory:
    """Stands in for ``OracleAgentMemory`` — enough surface for the adapter."""

    def __init__(self, *, card_text: str | None = "stub context card") -> None:
        # Record ctor kwargs so tests can assert on what the adapter passed.
        self.init_kwargs: dict[str, Any] = {}
        self.threads: dict[str, _StubThread] = {}
        self._card_text = card_text
        self.search_async = AsyncMock(return_value=[])
        self.add_memory_async = AsyncMock(return_value="mem-id-1")

    # Mimic OracleAgentMemory's sync methods. The adapter calls these
    # synchronously (they live outside the async hook bodies).
    def get_thread(self, thread_id: str) -> _StubThread:
        if thread_id not in self.threads:
            raise LookupError(thread_id)
        return self.threads[thread_id]

    def create_thread(self, *, thread_id: str, **_: Any) -> _StubThread:
        thread = _StubThread(thread_id, card_text=self._card_text)
        self.threads[thread_id] = thread
        return thread


# ---------------------------------------------------------------------------
# Fixture wiring — patch the lazy import so the real package never loads.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_client_factory(monkeypatch):
    """Yield a factory that monkeypatches the adapter's import.

    Returns the latest _StubOracleAgentMemory instance so the test can
    assert on init kwargs + recorded calls.
    """
    holder: dict[str, _StubOracleAgentMemory] = {}

    def build(*, card_text: str | None = "stub context card") -> _StubOracleAgentMemory:
        stub = _StubOracleAgentMemory(card_text=card_text)

        # Patch the symbols the adapter's lazy import would resolve.
        # The adapter does three imports inside __init__:
        #   from oracleagentmemory.core import OracleAgentMemory
        #   from oracleagentmemory.core.embedders import Embedder
        #   from oracleagentmemory.core.llms import Llm
        # We inject fake modules for all three so neither the real
        # oracleagentmemory nor its dependency stack (litellm + httpx)
        # is exercised during the unit-test path.
        sys = __import__("sys")

        class _FakeEmbedder:
            def __init__(self, model: str, **_: Any) -> None:
                self.model = model

        class _FakeLlm:
            def __init__(self, model: str, **_: Any) -> None:
                self.model = model

        fake_core = SimpleNamespace(OracleAgentMemory=lambda **kw: _record(stub, kw))
        fake_embedders = SimpleNamespace(Embedder=_FakeEmbedder)
        fake_llms = SimpleNamespace(Llm=_FakeLlm)
        fake_root = SimpleNamespace(core=fake_core)

        monkeypatch.setitem(sys.modules, "oracleagentmemory", fake_root)
        monkeypatch.setitem(sys.modules, "oracleagentmemory.core", fake_core)
        monkeypatch.setitem(sys.modules, "oracleagentmemory.core.embedders", fake_embedders)
        monkeypatch.setitem(sys.modules, "oracleagentmemory.core.llms", fake_llms)
        holder["stub"] = stub
        return stub

    def _record(stub: _StubOracleAgentMemory, kwargs: dict[str, Any]) -> _StubOracleAgentMemory:
        stub.init_kwargs = kwargs
        return stub

    return build


def _state(
    *, metadata: dict[str, Any] | None = None, messages: tuple[Message, ...] = ()
) -> AgentState:
    return AgentState(
        agent_id="test-agent",
        metadata=metadata or {},
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_construct_requires_connection_or_store(stub_client_factory):
    """Either ``connection`` or ``store`` must be provided."""
    stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    with pytest.raises(ValueError, match=r"connection.*store"):
        OracleAgentMemoryManager()


def test_construct_rejects_both_connection_and_store(stub_client_factory):
    stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    with pytest.raises(ValueError, match="OR"):
        OracleAgentMemoryManager(connection="x", store="y")


def test_construct_forwards_kwargs_to_client(stub_client_factory):
    stub = stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    OracleAgentMemoryManager(
        connection="fake-conn",
        embedder="oci/cohere.embed-english-v3.0",
        llm="oci/openai.gpt-4.1",
        extract_memories=True,
        table_name_prefix="test_oam_",
    )

    # String embedder/llm specs are auto-wrapped into Embedder/Llm
    # instances by the adapter (workaround: upstream auto-wraps the
    # embedder but not the llm). Check the wrapped instances carry the
    # model spec, not that they're string-equal.
    assert stub.init_kwargs["connection"] == "fake-conn"
    assert stub.init_kwargs["embedder"].model == "oci/cohere.embed-english-v3.0"
    assert stub.init_kwargs["llm"].model == "oci/openai.gpt-4.1"
    assert stub.init_kwargs["extract_memories"] is True
    assert stub.init_kwargs["table_name_prefix"] == "test_oam_"


async def test_on_session_start_injects_context_card(stub_client_factory):
    stub = stub_client_factory(card_text="Alice prefers us-chicago-1.")
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    state = _state(
        metadata={"thread_id": "t-1", "user_id": "alice"},
        messages=(Message(role=Role.SYSTEM, content="You are helpful."),),
    )

    out = await manager.on_session_start(state)

    # The thread was created with the resolved scope ids.
    assert "t-1" in stub.threads
    # Context card landed as a system message between original system + future user.
    system_messages = [m for m in out.messages if m.role == Role.SYSTEM]
    assert len(system_messages) == 2
    assert "Alice prefers us-chicago-1." in system_messages[1].content


async def test_on_session_start_returns_state_when_card_is_empty(stub_client_factory):
    stub_client_factory(card_text=None)
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    state = _state(messages=(Message(role=Role.SYSTEM, content="You are helpful."),))

    out = await manager.on_session_start(state)

    assert out is state or len(out.messages) == 1


async def test_on_session_end_ships_messages_to_thread(stub_client_factory):
    stub = stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    state = _state(
        metadata={"thread_id": "t-2", "user_id": "bob"},
        messages=(
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="hello"),
        ),
    )

    await manager.on_session_end(state)

    assert "t-2" in stub.threads
    thread = stub.threads["t-2"]
    thread.add_messages_async.assert_awaited_once()
    payload = thread.add_messages_async.await_args.args[0]
    assert {p["role"] for p in payload} == {"system", "user", "assistant"}


async def test_on_session_end_silent_when_no_messages(stub_client_factory):
    stub = stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    state = _state()

    await manager.on_session_end(state)

    # No thread created, no messages sent.
    assert not stub.threads


async def test_thread_id_defaults_to_run_id(stub_client_factory):
    """When metadata doesn't include thread_id, fall back to state.run_id."""
    stub = stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    state = _state(messages=(Message(role=Role.USER, content="hi"),))

    await manager.on_session_end(state)

    assert next(iter(stub.threads)) == state.run_id


async def test_custom_resolvers(stub_client_factory):
    stub = stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(
        connection="conn",
        thread_id_resolver=lambda s: "fixed-thread",
        user_id_resolver=lambda s: "fixed-user",
        agent_id_resolver=lambda s: "fixed-agent",
    )
    await manager.on_session_end(
        _state(messages=(Message(role=Role.USER, content="hi"),)),
    )

    assert "fixed-thread" in stub.threads


async def test_extract_is_noop(stub_client_factory):
    """The lifecycle path uses add_messages_async, so extract returns []."""
    stub_client_factory()
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    result = await manager.extract([Message(role=Role.USER, content="hi")])
    assert result == []


async def test_save_calls_add_memory_async_per_memory(stub_client_factory):
    stub = stub_client_factory()
    from locus.memory.manager import Memory, MemoryType
    from locus.memory.managers import OracleAgentMemoryManager

    manager = OracleAgentMemoryManager(connection="conn")
    memories = [
        Memory(
            type=MemoryType.USER,
            key="pref-1",
            content="alice prefers us-chicago-1",
            metadata={"user_id": "alice"},
        ),
        Memory(
            type=MemoryType.PROJECT,
            key="proj-1",
            content="team is shipping Atlas in Q3",
            metadata={},
        ),
    ]

    await manager.save(memories)

    assert stub.add_memory_async.await_count == 2
    first_call = stub.add_memory_async.await_args_list[0]
    assert first_call.kwargs["user_id"] == "alice"
    assert "content" in first_call.kwargs
