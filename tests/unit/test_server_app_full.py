# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Full coverage tests for ``locus.server.app`` (AgentServer).

Existing ``test_server_threads.py`` covers the thread routes. These
tests exercise the rest of ``server/app.py``:

- ``_principal_key`` (anon + hashed)
- ``_is_loopback``
- bearer-token auth on every route
- ``/invoke`` happy path with terminate + tool events
- ``/stream`` SSE shapes for think / tool_start / tool_complete /
  terminate / unknown event types, and the error-sanitisation path
- ``run()`` loopback gate (refuses non-loopback bind without auth)
- ``run()`` raises clear ImportError when uvicorn missing
- the docs gate when settings raise
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from locus.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.server.app import (
    AgentServer,
    _is_loopback,
    _principal_key,
)


# ---------------------------------------------------------------------------
# Stub agent
# ---------------------------------------------------------------------------


@dataclass
class _StubConfig:
    checkpointer: Any = None


@dataclass
class _StubAgent:
    """Minimal agent — yields a configurable list of events from ``run``."""

    events: list[Any] = field(default_factory=list)
    seen_kwargs: list[dict[str, Any]] = field(default_factory=list)
    config: _StubConfig = field(default_factory=_StubConfig)
    raise_on_run: bool = False

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        self.seen_kwargs.append({"prompt": prompt, **kwargs})
        if self.raise_on_run:
            raise RuntimeError("DSN=postgres://leak/secret")
        for event in self.events:
            yield event


def _think(text: str) -> ThinkEvent:
    return ThinkEvent(iteration=0, reasoning=text, tool_calls=[])


def _tool_start(name: str = "search") -> ToolStartEvent:
    return ToolStartEvent(tool_name=name, tool_call_id="t1", arguments={"q": "x"})


def _tool_complete(name: str = "search") -> ToolCompleteEvent:
    return ToolCompleteEvent(
        tool_name=name,
        tool_call_id="t1",
        result="ok",
        error=None,
        duration_ms=1.0,
    )


def _terminate(*, message: str = "done", reason: str = "complete") -> TerminateEvent:
    return TerminateEvent(
        reason=reason,
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=message,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestPrincipalKey:
    def test_anon_when_no_api_key(self) -> None:
        assert _principal_key(None) == "anon"

    def test_anon_when_empty_api_key(self) -> None:
        assert _principal_key("") == "anon"

    def test_returns_12_hex_chars(self) -> None:
        out = _principal_key("secret-key")
        assert len(out) == 12
        assert all(c in "0123456789abcdef" for c in out)

    def test_deterministic_for_same_key(self) -> None:
        assert _principal_key("k") == _principal_key("k")

    def test_different_for_different_keys(self) -> None:
        assert _principal_key("a") != _principal_key("b")


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
    def test_recognises_loopback(self, host: str) -> None:
        assert _is_loopback(host) is True

    @pytest.mark.parametrize("host", ["8.8.8.8", "192.0.2.1", "example.com"])
    def test_rejects_non_loopback(self, host: str) -> None:
        assert _is_loopback(host) is False


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_unauthenticated_ok(self) -> None:
        agent = _StubAgent()
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestBearerAuth:
    def test_missing_bearer_returns_401(self) -> None:
        server = AgentServer(agent=_StubAgent(), api_key="secret")  # noqa: S106
        client = TestClient(server.app)
        # /invoke is auth-gated (health is not).
        r = client.post("/invoke", json={"prompt": "hi"})
        assert r.status_code == 401
        assert "Missing bearer token" in r.text

    def test_wrong_bearer_returns_401(self) -> None:
        server = AgentServer(agent=_StubAgent(), api_key="secret")  # noqa: S106
        client = TestClient(server.app)
        r = client.post(
            "/invoke",
            headers={"Authorization": "Bearer wrong"},
            json={"prompt": "hi"},
        )
        assert r.status_code == 401
        assert "Invalid bearer token" in r.text

    def test_non_bearer_scheme_returns_401(self) -> None:
        server = AgentServer(agent=_StubAgent(), api_key="secret")  # noqa: S106
        client = TestClient(server.app)
        r = client.post(
            "/invoke",
            headers={"Authorization": "Basic abc"},
            json={"prompt": "hi"},
        )
        assert r.status_code == 401

    def test_valid_bearer_passes(self) -> None:
        agent = _StubAgent(events=[_terminate(message="answer")])
        server = AgentServer(agent=agent, api_key="secret")  # noqa: S106
        client = TestClient(server.app)
        r = client.post(
            "/invoke",
            headers={"Authorization": "Bearer secret"},
            json={"prompt": "hi"},
        )
        assert r.status_code == 200
        assert r.json()["message"] == "answer"

    def test_principal_scoped_thread_id(self) -> None:
        agent = _StubAgent(events=[_terminate(message="done")])
        server = AgentServer(agent=agent, api_key="secret")  # noqa: S106
        client = TestClient(server.app)
        r = client.post(
            "/invoke",
            headers={"Authorization": "Bearer secret"},
            json={"prompt": "hi", "thread_id": "alpha"},
        )
        assert r.status_code == 200
        # Server prefixed thread_id with hashed principal.
        seen = agent.seen_kwargs[0]["thread_id"]
        assert seen != "alpha"
        assert seen.endswith(":alpha")


# ---------------------------------------------------------------------------
# /invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_invoke_terminate_only(self) -> None:
        agent = _StubAgent(events=[_terminate(message="answer", reason="complete")])
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        r = client.post("/invoke", json={"prompt": "hi"})
        body = r.json()
        assert body["message"] == "answer"
        assert body["stop_reason"] == "complete"
        assert body["iterations"] == 1

    def test_invoke_counts_tool_complete_events(self) -> None:
        agent = _StubAgent(
            events=[
                _think("thinking"),
                _tool_start(),
                _tool_complete(),
                _terminate(message="done"),
            ]
        )
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        body = client.post("/invoke", json={"prompt": "hi"}).json()
        assert body["tool_calls"] == 1
        # ``iterations`` counts every event yielded.
        assert body["iterations"] == 4

    def test_invoke_passes_metadata_to_agent(self) -> None:
        agent = _StubAgent(events=[_terminate()])
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        client.post("/invoke", json={"prompt": "hi", "metadata": {"trace": "abc"}})
        assert agent.seen_kwargs[0]["metadata"] == {"trace": "abc"}

    # ------ duration_ms (B3) ------------------------------------------------

    def test_invoke_reports_real_duration_ms(self) -> None:
        """``duration_ms`` must reflect the wall time the run took.

        Regression: it used to be hardcoded to 0.0, which made every
        client-side latency metric useless. Even a zero-event /
        immediate-terminate run takes a non-zero number of microseconds
        to schedule the async generator, so ``duration_ms > 0`` is a
        meaningful assertion against the old behaviour.
        """
        agent = _StubAgent(events=[_terminate(message="x")])
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        body = client.post("/invoke", json={"prompt": "hi"}).json()
        assert body["duration_ms"] > 0.0

    # ------ success derivation (B4) -----------------------------------------

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("complete", True),
            ("confidence_met", True),
            ("terminal_tool", True),
            ("max_iterations", False),
            ("tool_loop", False),
            ("error", False),
        ],
    )
    def test_invoke_success_derived_from_stop_reason(self, reason: str, expected: bool) -> None:
        """``success`` must follow the terminal stop_reason.

        Previously hardcoded to ``True`` regardless of how the run
        ended, which made the field uninformative for callers that
        wanted to branch on outcome without parsing ``stop_reason``.
        """
        agent = _StubAgent(events=[_terminate(message="x", reason=reason)])
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        body = client.post("/invoke", json={"prompt": "hi"}).json()
        assert body["stop_reason"] == reason
        assert body["success"] is expected

    def test_invoke_success_helper_directly(self) -> None:
        """The ``_invoke_success`` mapping is the single source of truth."""
        from locus.server.app import _INVOKE_SUCCESS_REASONS, _invoke_success

        # Documented success reasons.
        assert frozenset({"complete", "confidence_met", "terminal_tool"}) == _INVOKE_SUCCESS_REASONS
        for r in _INVOKE_SUCCESS_REASONS:
            assert _invoke_success(r) is True
        # Anything else is a failure — including unrecognised strings.
        for r in ("error", "max_iterations", "tool_loop", "", "unknown"):
            assert _invoke_success(r) is False


# ---------------------------------------------------------------------------
# /stream
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """Decode ``data: …\\n\\n`` lines as JSON, skipping the [DONE] sentinel."""
    out = []
    for line in body.split("\n\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            try:
                out.append(json.loads(line.removeprefix("data: ")))
            except json.JSONDecodeError:
                continue
    return out


class TestStream:
    def test_emits_think_tool_terminate_events(self) -> None:
        agent = _StubAgent(
            events=[
                _think("thinking..."),
                _tool_start("search"),
                _tool_complete("search"),
                _terminate(message="final"),
            ]
        )
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        with client.stream("POST", "/stream", json={"prompt": "hi"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        types = [e["type"] for e in events]
        assert "think" in types
        assert "tool_start" in types
        assert "tool_complete" in types
        assert "done" in types
        assert "[DONE]" in body

    def test_unknown_event_type_falls_through_to_event_type_field(self) -> None:
        # Provide a duck-typed event the server doesn't recognise — it
        # should still be serialised as ``{"type": event.event_type, ...}``.
        from locus.core.events import ModelChunkEvent

        agent = _StubAgent(
            events=[
                ModelChunkEvent(content="streaming"),
                _terminate(message="ok"),
            ]
        )
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        with client.stream("POST", "/stream", json={"prompt": "hi"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        assert any(e["type"] == "model_chunk" for e in events)

    def test_sanitises_agent_errors(self) -> None:
        agent = _StubAgent(raise_on_run=True)
        server = AgentServer(agent=agent, allow_unauthenticated=True)
        client = TestClient(server.app)
        with client.stream("POST", "/stream", json={"prompt": "hi"}) as r:
            body = "".join(r.iter_text())
        # The DSN fragment must NOT leak; the sanitised payload has
        # ``"internal error"`` + a correlation_id.
        assert "DSN=postgres" not in body
        assert "internal error" in body
        assert "correlation_id" in body


# ---------------------------------------------------------------------------
# run() — bind gate + uvicorn import
# ---------------------------------------------------------------------------


class TestRunBindGate:
    def test_unauthenticated_non_loopback_refused(self) -> None:
        server = AgentServer(agent=_StubAgent())
        with pytest.raises(RuntimeError, match="Refusing to bind"):
            server.run(host="8.8.8.8", port=9999)

    def test_loopback_path_invokes_uvicorn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        captured: dict[str, Any] = {}

        fake_uvicorn = types.ModuleType("uvicorn")

        def fake_run(app: Any, **kwargs: Any) -> None:
            captured["app"] = app
            captured["kwargs"] = kwargs

        fake_uvicorn.run = fake_run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

        server = AgentServer(agent=_StubAgent())
        server.run(host="127.0.0.1", port=12345)
        assert captured["kwargs"] == {"host": "127.0.0.1", "port": 12345}

    def test_uvicorn_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "uvicorn", None)
        server = AgentServer(agent=_StubAgent())
        with pytest.raises(ImportError, match="uvicorn is required"):
            server.run(host="127.0.0.1")

    def test_authed_server_can_bind_anywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        fake = types.ModuleType("uvicorn")
        fake.run = lambda *a, **k: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "uvicorn", fake)

        server = AgentServer(
            agent=_StubAgent(),
            api_key="secret",  # noqa: S106
        )
        # No exception even on a public host — auth is configured.
        server.run(host="0.0.0.0", port=9999)  # noqa: S104


# ---------------------------------------------------------------------------
# Docs gate
# ---------------------------------------------------------------------------


class TestDocsGate:
    def test_docs_disabled_when_settings_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force ``get_settings`` to raise — server should not enable docs.
        def boom() -> None:
            raise RuntimeError("no settings")

        monkeypatch.setattr("locus.core.config.get_settings", boom)
        server = AgentServer(agent=_StubAgent(), allow_unauthenticated=True)
        assert server._resolve_docs_enabled() is False
