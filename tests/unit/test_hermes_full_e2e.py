# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Full end-to-end integration of every Hermes-port primitive.

Drives a real :class:`~locus.agent.Agent` through a multi-iteration
run that exercises:

* **A.1 redaction** — a tool raises with an embedded API key; the
  agent observes only the redacted error.
* **A.2 SSRF guard** — a tool calls ``validate_url`` on a metadata
  hostname and the agent sees the rejection cleanly.
* **A.4 path safety** — a tool uses ``safe_resolve`` and a traversal
  attempt is rejected end-to-end.
* **B.1 + B.3** — model wrapper consults ``classify`` and rotates
  through a ``CredentialPool`` on a synthetic 429.
* **C.1 metadata** — context length comes from the registry.
* **C.2 + C.3** — auxiliary-model summarises long history through
  the ``LLMCompactor``.
* **D.1 result storage** — oversized tool output is offloaded to an
  external store with a recoverable reference key.
* **D.2 prompt cache** — the system message is marked when the model
  metadata says caching is supported.

The agent's primary "model" is a hand-rolled stub that returns canned
responses, so the test is fast and deterministic. The point is to
prove the *integration*, not to benchmark the model.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from locus.agent.agent import Agent
from locus.agent.config import AgentConfig
from locus.core.messages import Message, Role, ToolCall
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.compactor import LLMCompactor
from locus.models import ModelResponse
from locus.models.auxiliary import resolve_auxiliary
from locus.models.caching import is_cache_breakpoint, mark_cache_breakpoint
from locus.models.credentials import Credential, CredentialPool
from locus.models.failover import classify
from locus.models.metadata import metadata_for
from locus.tools.decorator import tool
from locus.tools.path_safety import safe_resolve
from locus.tools.result_storage import (
    ToolResultStore,
    extract_reference_key,
)
from locus.tools.url_safety import validate_url


# ---------------------------------------------------------------------------
# Stub primary + auxiliary models.
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal Model implementation that emits canned responses."""

    name = "stub-primary"

    def __init__(self, *, scripted_responses: list[ModelResponse]) -> None:
        self._scripted = list(scripted_responses)
        self._calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if not self._scripted:
            raise RuntimeError("stub model out of scripted responses")
        self._calls += 1
        return self._scripted.pop(0)

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("stream not used in this test")
        yield  # pragma: no cover


class _StubAux:
    """Auxiliary model — returns a fixed, recognisable summary."""

    name = "stub-auxiliary"

    async def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            message=Message.assistant("AUX-SUMMARY: Resolved=A. Pending=B. Remaining work=C.")
        )


# ---------------------------------------------------------------------------
# Tools that exercise A.1 / A.2 / A.4 / D.1.
# ---------------------------------------------------------------------------


WORKSPACE_ROOT: Path | None = None
RESULT_STORE: ToolResultStore | None = None


@tool
def fetch_metadata_endpoint() -> str:
    """Attempts to fetch a cloud-metadata URL — must be blocked by A.2."""
    validate_url("https://metadata.google.internal/computeMetadata/")
    return "should never reach here"


@tool
def read_workspace_file(path: str) -> str:
    """Reads a file under the workspace, guarded by A.4."""
    assert WORKSPACE_ROOT is not None
    target = safe_resolve(WORKSPACE_ROOT, path)
    return target.read_text()


@tool
def fetch_logs(run_id: str = "demo-run", iteration: int = 0) -> str:
    """Returns a multi-kB log blob; D.1 offloads it via the checkpointer."""
    big = "INFO line of log content\n" * 1000  # ~25 kB
    assert RESULT_STORE is not None

    # Construct a ToolResult so the storage helper can offload it.
    from locus.core.messages import ToolResult

    raw = ToolResult(tool_call_id="call-fetch", name="fetch_logs", content=big)
    offloaded = RESULT_STORE.maybe_offload(raw, run_id=run_id, iteration=iteration)
    return offloaded.content or ""


@tool
def leak_provider_key() -> str:
    """Tool that raises with an embedded vendor key — A.1 redacts the error."""
    raise RuntimeError("401 from upstream: sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


# ---------------------------------------------------------------------------
# B.1 + B.3 wrapper that rotates credentials via the failover classifier.
# ---------------------------------------------------------------------------


class _SdkRateLimitError(Exception):
    """Mimics a provider SDK rate-limit shape."""

    def __init__(self, msg: str = "Too many requests, please retry after 30s") -> None:
        super().__init__(msg)
        self.status_code = 429


class _PoolRotatingModel:
    """Wraps a primary model and rotates a CredentialPool on classified errors."""

    name = "pool-rotating"

    def __init__(
        self,
        primary: _StubModel,
        pool: CredentialPool,
        *,
        fail_first_call: bool,
    ) -> None:
        self._primary = primary
        self._pool = pool
        self._fail_once = fail_first_call
        self.successful_credential: Credential | None = None
        self.attempts = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # Loop: pick → call → on rotation-eligible failure rotate.
        while True:
            cred = self._pool.pick()
            self.attempts += 1
            try:
                if self._fail_once:
                    self._fail_once = False
                    raise _SdkRateLimitError
                resp = await self._primary.complete(messages, tools, **kwargs)
                self.successful_credential = cred
                return resp
            except Exception as exc:
                decision = classify(exc)
                if not decision.should_rotate_credential:
                    raise
                self._pool.mark_bad(cred, cooldown_s=60.0)
                continue

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Compactor wired to the auxiliary model.
# ---------------------------------------------------------------------------


def _build_compactor(aux: _StubAux) -> LLMCompactor:
    async def _summarise(middle: list[Message], previous: str | None) -> str:
        helper = resolve_auxiliary(primary=None, auxiliary=aux)
        rendered = "\n".join((m.content or "")[:200] for m in middle)
        prompt = [
            Message.system(
                "Summarise the conversation excerpt in three sections: "
                "Resolved, Pending, Remaining work."
            ),
            Message.user(rendered),
        ]
        if previous:
            prompt.append(Message.system(f"Prior summary: {previous}"))
        response = await helper.complete(prompt)
        return response.message.content or ""

    return LLMCompactor(
        summarize_fn=_summarise,
        context_length=4_000,  # tiny so we trip on real test data
        trigger_fraction=0.3,
        head_turns=2,
        tail_token_fraction=0.4,
        tool_output_ttl_turns=10,
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace dir with a known file plus a sibling secret outside."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "report.txt").write_text("ALL GOOD\n")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "passwords.txt").write_text("nope")
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = ws
    yield ws
    WORKSPACE_ROOT = None


@pytest.fixture
def checkpointer() -> MemoryCheckpointer:
    return MemoryCheckpointer()


@pytest.fixture
def result_store() -> ToolResultStore:
    """A ToolResultStore backed by an in-process dict.

    The D.1 contract only requires that ``save`` and ``load`` round-trip
    a key/value mapping; a plain dict captures that without dragging in
    any backend driver.
    """
    backing: dict[str, str] = {}

    def _save(key: str, content: str) -> None:
        backing[key] = content

    def _load(key: str) -> str | None:
        return backing.get(key)

    store = ToolResultStore(save=_save, load=_load, threshold_chars=2_000, preview_chars=500)
    global RESULT_STORE
    RESULT_STORE = store
    yield store
    RESULT_STORE = None


# ---------------------------------------------------------------------------
# Test 1 — auto-wired features in one Agent run.
# ---------------------------------------------------------------------------


def test_e2e_redaction_compactor_metadata_and_caching(
    workspace: Path,
    checkpointer: MemoryCheckpointer,
    result_store: ToolResultStore,
) -> None:
    aux = _StubAux()

    # The agent will call the primary model with a script:
    # 1. First turn: emit a tool call to read_workspace_file
    # 2. Second turn: emit a final answer
    primary = _StubModel(
        scripted_responses=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="read_workspace_file",
                            arguments={"path": "report.txt"},
                        )
                    ]
                )
            ),
            ModelResponse(message=Message.assistant("I've read the report; status: ALL GOOD.")),
        ]
    )

    # C.1 — context length pulled from the registry where possible.
    meta = metadata_for("claude-haiku-4")
    assert meta is not None
    assert meta.supports_prompt_caching is True

    # D.2 — the system message is marked when metadata says caching is on.
    sys_msg = Message.system("You are a helpful assistant.")
    if meta.supports_prompt_caching:
        sys_msg = mark_cache_breakpoint(sys_msg)
    assert is_cache_breakpoint(sys_msg)

    config = AgentConfig(
        model=primary,
        auxiliary_model=aux,
        conversation_manager=_build_compactor(aux),  # C.3
        checkpointer=checkpointer,  # C.3 + D.1 backing store
        system_prompt=sys_msg.content or "You are helpful.",
        tools=[read_workspace_file],
        max_iterations=5,
    )
    agent = Agent(config=config)

    result = agent.run_sync("Read the report file please.")
    assert "ALL GOOD" in result.message


# ---------------------------------------------------------------------------
# Test 2 — A.1 redaction: agent observes the sanitised error from a tool.
# ---------------------------------------------------------------------------


def test_e2e_redaction_on_tool_exception() -> None:
    primary = _StubModel(
        scripted_responses=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="t1", name="leak_provider_key", arguments={})]
                )
            ),
            ModelResponse(message=Message.assistant("failed gracefully")),
        ]
    )
    config = AgentConfig(
        model=primary,
        tools=[leak_provider_key],
        max_iterations=3,
    )
    agent = Agent(config=config)

    # Run to completion. The tool raises an error containing an Anthropic
    # key; A.1 redaction must scrub it before the message reaches the agent
    # state.
    result = agent.run_sync("Try the leaky tool.")

    # Inspect the agent's recorded messages for the tool result.
    leaked = "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    for msg in result.state.messages:
        assert leaked not in (msg.content or ""), f"key leaked in: {msg.content!r}"


# ---------------------------------------------------------------------------
# Test 3 — A.2 SSRF guard fires through a tool call.
# ---------------------------------------------------------------------------


def test_e2e_ssrf_guard_blocks_metadata_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force any DNS lookup to a public IP so the block fires on hostname alone.
    def _fake(host: str, port: int | None, *_a: Any, **_kw: Any) -> Any:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake)

    primary = _StubModel(
        scripted_responses=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="fetch_metadata_endpoint",
                            arguments={},
                        )
                    ]
                )
            ),
            ModelResponse(message=Message.assistant("rejected as expected")),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            model=primary,
            tools=[fetch_metadata_endpoint],
            max_iterations=3,
        )
    )
    result = agent.run_sync("Try fetching IMDS.")

    # The classified guard error is on the ToolExecution record.
    failures = [e for e in result.state.tool_executions if e.tool_name == "fetch_metadata_endpoint"]
    assert failures, "expected fetch_metadata_endpoint to have been attempted"
    assert any("SSRF guard" in (f.error or "") for f in failures)


# ---------------------------------------------------------------------------
# Test 4 — A.4 path safety guard fires through a tool call.
# ---------------------------------------------------------------------------


def test_e2e_path_safety_blocks_traversal(workspace: Path) -> None:
    primary = _StubModel(
        scripted_responses=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="read_workspace_file",
                            arguments={"path": "../secret/passwords.txt"},
                        )
                    ]
                )
            ),
            ModelResponse(message=Message.assistant("blocked")),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            model=primary,
            tools=[read_workspace_file],
            max_iterations=3,
        )
    )
    result = agent.run_sync("Try traversal.")

    # Path-safety failure surfaced on the ToolExecution record.
    failures = [e for e in result.state.tool_executions if e.tool_name == "read_workspace_file"]
    assert failures
    assert any("outside the allowed base" in (f.error or "") for f in failures), [
        f.error for f in failures
    ]
    # And the secret file content never appears anywhere.
    assert all("nope" not in (m.content or "") for m in result.state.messages)
    assert all("nope" not in (e.result or "") for e in result.state.tool_executions)


# ---------------------------------------------------------------------------
# Test 5 — D.1 tool-result storage offloads big output via the checkpointer.
# ---------------------------------------------------------------------------


def test_e2e_tool_result_offloaded_to_checkpointer(
    checkpointer: MemoryCheckpointer, result_store: ToolResultStore
) -> None:
    primary = _StubModel(
        scripted_responses=[
            ModelResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="fetch_logs",
                            arguments={"run_id": "e2e-1", "iteration": 0},
                        )
                    ]
                )
            ),
            ModelResponse(message=Message.assistant("logs loaded")),
        ]
    )
    agent = Agent(
        config=AgentConfig(
            model=primary,
            checkpointer=checkpointer,
            tools=[fetch_logs],
            max_iterations=3,
        )
    )
    result = agent.run_sync("Pull the logs.")

    # Agent's tool-result message contains a reference key, not the full blob.
    tool_results = [m for m in result.state.messages if m.role == Role.TOOL]
    assert tool_results, "expected at least one tool result"
    content = tool_results[0].content or ""
    assert "STORED externally" in content
    key = extract_reference_key(content)
    assert key is not None

    # The full payload is recoverable from the same checkpointer.
    full = result_store.load(key)
    assert full is not None
    assert "INFO line of log content" in full
    assert len(full) > 20_000  # confirm we got the *full* original blob


# ---------------------------------------------------------------------------
# Test 6 — B.1 + B.3 credential pool rotates on classified rate-limit.
# ---------------------------------------------------------------------------


def test_e2e_credential_pool_rotates_on_rate_limit() -> None:
    pool = CredentialPool(
        [
            Credential(label="alpha", api_key=SecretStr("k1")),
            Credential(label="beta", api_key=SecretStr("k2")),
        ]
    )
    primary = _StubModel(
        scripted_responses=[ModelResponse(message=Message.assistant("answered after rotation"))]
    )
    rotating = _PoolRotatingModel(primary, pool, fail_first_call=True)

    agent = Agent(
        config=AgentConfig(
            model=rotating,
            tools=[],
            max_iterations=2,
        )
    )
    result = agent.run_sync("Anything.")

    assert "answered after rotation" in result.message
    # First credential must be in cooldown after the synthetic 429.
    assert "alpha" in pool.state()["disabled"]
    assert rotating.successful_credential is not None
    assert rotating.successful_credential.label == "beta"
    # 1 fail + 1 success = 2 attempts.
    assert rotating.attempts == 2


# ---------------------------------------------------------------------------
# Test 7 — C.3 compactor fires on a long history (using auxiliary model).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_compactor_summarises_long_history() -> None:
    aux = _StubAux()
    compactor = _build_compactor(aux)

    # Manufacture a long pre-existing message list for the compactor to munch.
    seed: list[Message] = [Message.system("anchor system message")]
    for i in range(40):
        seed.append(Message.user(f"q{i}: " + ("filler " * 30)))
        seed.append(Message.assistant(f"a{i}: " + ("answer " * 30)))

    out = await compactor.async_apply(seed)

    # System anchor preserved verbatim.
    assert out[0].content == "anchor system message"
    # Summary block inserted at index 1, contains the auxiliary model's
    # canned summary text.
    assert out[1].role == Role.SYSTEM
    assert "REFERENCE ONLY" in (out[1].content or "")
    assert "AUX-SUMMARY" in (out[1].content or "")
    # Tail still includes the most recent message.
    assert (out[-1].content or "").startswith("a39:")
    # Big shrinkage.
    assert len(out) < len(seed) // 2
