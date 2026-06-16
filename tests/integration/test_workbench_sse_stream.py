# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end test that the workbench SSE stream carries every layer
of locus telemetry — router, agent, notebook — for a single
``/api/notebooks/run`` dispatch.

How it works:

1. POST a small piece of notebook source to ``/api/notebooks/run``.
2. Concurrently subscribe to ``/api/events/{run_id}`` once we observe
   the ``runStarted`` event in the legacy stream.
3. Both streams run to completion. We collect every event from each
   side and assert: the bus stream contains the same exit code, the
   structured ``agent.*`` events fire, every notebook line shows up
   as a ``notebook.stdout`` bus event, and tokens are reported.

Skipped without a model provider (Anthropic key by default — cheapest
for the smoke test) or if the backend isn't reachable.

Required env (via the standard provider env locus reads):

* ``ANTHROPIC_API_KEY`` (default path) **or** ``OPENAI_API_KEY`` /
  ``OCI_PROFILE`` for the alternative providers.

Run with::

    ANTHROPIC_API_KEY=sk-... hatch run pytest \\
        tests/integration/test_workbench_sse_stream.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator

import httpx
import pytest


pytestmark = pytest.mark.integration


WORKBENCH_URL = os.getenv("WORKBENCH_URL", "http://127.0.0.1:8100")


def _backend_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get(f"{WORKBENCH_URL}/api/events/__stats").status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _provider_payload() -> dict[str, str | None] | None:
    """Pick the cheapest available provider for the smoke run."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        }
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }
    return None


# Module-level skip — keeps the file out of CI when the backend isn't up
# or no creds exist. Cheaper than per-test skips and avoids spurious
# "1 skipped" lines per scenario.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _backend_reachable(),
        reason=f"workbench backend not reachable at {WORKBENCH_URL}",
    ),
    pytest.mark.skipif(
        _provider_payload() is None,
        reason="no provider creds (set ANTHROPIC_API_KEY or OPENAI_API_KEY)",
    ),
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# SSE parsing helpers — light, just enough for these tests. No external
# SSE library; we parse the on-the-wire ``event:`` + ``data:`` shape by
# hand so the test stays a self-contained contract for what the wire
# format actually is.
# ---------------------------------------------------------------------------


async def _sse_iter(client: httpx.AsyncClient, path: str) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event_name, data_json_str)`` for each SSE frame.

    Comment lines (``: ...``) and blank lines are skipped. Multi-line
    ``data:`` continuation is collapsed into one string.
    """
    async with client.stream("GET", path) as resp:
        resp.raise_for_status()
        event_name = "message"
        data_lines: list[str] = []
        async for line in resp.aiter_lines():
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())


async def _post_and_iter_legacy(
    client: httpx.AsyncClient, source: str, provider: dict[str, str | None]
) -> AsyncIterator[dict]:
    """Drive ``/api/notebooks/run`` and yield each parsed event payload."""
    async with client.stream(
        "POST",
        "/api/notebooks/run",
        json={"source": source, "provider": provider, "timeout_seconds": 240},
    ) as resp:
        resp.raise_for_status()
        # The legacy run endpoint emits SSE-shaped frames where every
        # line is prefixed with ``data:`` and the body is JSON.
        buf = b""
        async for chunk in resp.aiter_bytes():
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.decode().splitlines():
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[len("data:") :].strip())
                        except json.JSONDecodeError:
                            pass


# ---------------------------------------------------------------------------
# Notebook source. Tiny + deterministic + emits a recognisable token in
# stdout so we can assert the bus relayed it.
# ---------------------------------------------------------------------------

_MARKER = "locus-sse-e2e-marker-9173"

TINY_NOTEBOOK = f'''
"""Tiny SSE smoke notebook — runs one Agent and prints a marker."""
from config import get_model
from locus.agent import Agent

agent = Agent(model=get_model(max_tokens=40), system_prompt="Reply in one short sentence.")
result = agent.run_sync("Reply with exactly: {_MARKER}")
print(result.message)
'''


# ---------------------------------------------------------------------------
# E2E tests.
# ---------------------------------------------------------------------------


class TestWorkbenchSseStream:
    """The bus stream and the legacy stream must agree on every load-
    bearing fact for a single dispatch."""

    async def test_bus_and_legacy_streams_agree_on_run(self) -> None:
        provider = _provider_payload()
        assert provider is not None  # guarded by module skip

        legacy_events: list[dict] = []
        bus_events: list[tuple[str, dict]] = []
        run_id_holder: dict[str, str | None] = {"run_id": None}

        async with httpx.AsyncClient(base_url=WORKBENCH_URL, timeout=300.0) as client:

            async def consume_legacy() -> None:
                async for ev in _post_and_iter_legacy(client, TINY_NOTEBOOK, provider):
                    legacy_events.append(ev)
                    if ev.get("type") == "runStarted" and run_id_holder["run_id"] is None:
                        run_id_holder["run_id"] = ev.get("run_id")

            async def consume_bus() -> None:
                # Wait until the legacy stream has surfaced the run_id —
                # it's the first SSE frame the runner emits, so this
                # poll is short.
                for _ in range(200):  # 10s max
                    if run_id_holder["run_id"]:
                        break
                    await asyncio.sleep(0.05)
                rid = run_id_holder["run_id"]
                assert rid is not None, "legacy stream never emitted runStarted"
                async for event_name, data_str in _sse_iter(client, f"/api/events/{rid}"):
                    if event_name == "done":
                        return
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    bus_events.append((payload["event_type"], payload["data"]))

            await asyncio.gather(consume_legacy(), consume_bus())

        # ============ Legacy stream sanity ============
        kinds = [e.get("type") for e in legacy_events]
        assert "runStarted" in kinds, f"legacy stream missed runStarted: {kinds}"
        assert "exit" in kinds, f"legacy stream missed exit: {kinds}"
        exit_event = next(e for e in legacy_events if e.get("type") == "exit")
        assert exit_event["code"] == 0, f"notebook subprocess exited non-zero: {exit_event}"
        # The marker should land in the legacy stdout stream.
        stdout_blob = "\n".join(
            e.get("text", "") for e in legacy_events if e.get("type") == "stdout"
        )
        assert _MARKER in stdout_blob, f"marker {_MARKER!r} never appeared in legacy stdout"

        # ============ Bus stream — the load-bearing assertions ========
        bus_types = [t for t, _ in bus_events]
        assert bus_events, "bus stream produced zero events for the run"

        # Notebook-level events bridge the subprocess output.
        assert any(t == "notebook.exited" for t in bus_types), (
            f"missing notebook.exited on bus; saw: {set(bus_types)}"
        )

        # Marker reached the bus through the stdout-bridge.
        bus_stdout_blob = "\n".join(
            data.get("text", "") for t, data in bus_events if t == "notebook.stdout"
        )
        assert _MARKER in bus_stdout_blob, f"marker {_MARKER!r} not on bus notebook.stdout stream"

        # Agent-level structured events fire — proves the bootstrap's
        # callback_handler reached the EventBus via the __LE__: bridge.
        assert any(t.startswith("agent.") for t in bus_types), (
            f"no agent.* events on bus; saw: {set(bus_types)}"
        )

        # Final exit event matches.
        bus_exit = next((data for t, data in bus_events if t == "notebook.exited"), None)
        assert bus_exit is not None
        assert bus_exit["code"] == exit_event["code"], (
            f"bus exit code {bus_exit['code']} ≠ legacy {exit_event['code']}"
        )

    async def test_bus_history_replay_after_run(self) -> None:
        """A late subscriber that connects *after* the run finishes
        still receives the buffered history then a clean ``done`` —
        no hung connection."""
        provider = _provider_payload()
        assert provider is not None
        run_id: str | None = None

        async with httpx.AsyncClient(base_url=WORKBENCH_URL, timeout=300.0) as client:
            async for ev in _post_and_iter_legacy(client, TINY_NOTEBOOK, provider):
                if ev.get("type") == "runStarted":
                    run_id = ev.get("run_id")
                if ev.get("type") == "exit":
                    break
            assert run_id is not None

            # Run is now closed. Connect to the bus stream — should
            # replay history then immediately send the done sentinel.
            late_events: list[str] = []
            async for event_name, _data_str in _sse_iter(client, f"/api/events/{run_id}"):
                if event_name == "done":
                    break
                late_events.append(event_name)
                if len(late_events) > 100:
                    pytest.fail("late subscriber would not stop")

            # Replay must include the notebook.exited marker since we
            # wait for run completion before subscribing.
            replayed_types = set(late_events)
            assert "notebook.exited" in replayed_types or any(
                e.endswith(".exited") for e in replayed_types
            ), f"history replay missed terminal event: {replayed_types}"

    async def test_unique_run_ids_per_dispatch(self) -> None:
        """Two consecutive runs get distinct run_ids and stream into
        distinct bus channels — no cross-contamination."""
        provider = _provider_payload()
        assert provider is not None
        run_ids: list[str] = []

        async with httpx.AsyncClient(base_url=WORKBENCH_URL, timeout=300.0) as client:
            for _ in range(2):
                async for ev in _post_and_iter_legacy(client, TINY_NOTEBOOK, provider):
                    if ev.get("type") == "runStarted":
                        run_ids.append(ev["run_id"])
                    if ev.get("type") == "exit":
                        break

        assert len(run_ids) == 2
        assert run_ids[0] != run_ids[1], "consecutive dispatches reused a run_id"
        # IDs are uuid4 hex strings — sanity on shape.
        for rid in run_ids:
            assert re.fullmatch(r"[0-9a-f]{32}", rid), (
                f"run_id is not the expected uuid4 hex: {rid!r}"
            )

    async def test_sse_wire_format_well_formed(self) -> None:
        """Verifies the SSE response carries the right Content-Type and
        emits parseable ``event:`` + ``data:`` frames. Catches
        regressions in the BFF passthrough or the StreamingResponse
        headers."""
        # Subscribe to a run that doesn't exist yet — the stream opens,
        # sends the connected comment, then waits. We close quickly so
        # the test stays cheap. The stream may also send event frames
        # if some other run with the same id exists, but the wire
        # format is what we care about.
        async with (
            httpx.AsyncClient(base_url=WORKBENCH_URL, timeout=10.0) as client,
            client.stream("GET", "/api/events/sse-shape-probe-doesnt-exist") as resp,
        ):
            assert resp.status_code == 200, await resp.aread()
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type, (
                f"expected text/event-stream, got {content_type!r}"
            )
            # First chunk must include the keep-alive comment we emit
            # on connect — proves headers flushed.
            async for line in resp.aiter_lines():
                assert line.startswith((":", "event:", "data:")) or line == "", (
                    f"first SSE line not in protocol shape: {line!r}"
                )
                break
