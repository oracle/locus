"""The box runner survives an OpenShell box's connection cuts.

OpenShell cuts a box's connections whenever its policy generation moves on (the first
settings poll after start does so). The runner's gateway client asks again what is safe
to repeat, and its model reads each turn whole and asks again when a turn is cut.
"""

from __future__ import annotations

import httpx
import pytest

from tulip.runner import client as client_module
from tulip.runner.client import GatewayClient, GatewayUnavailable, RunnerConfig, _safe_to_repeat


URL = "http://gateway.test:8421"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "GATEWAY_RETRY_DELAY", 0.0)


def _flaky(cut: type[httpx.TransportError], cuts: int) -> tuple[httpx.MockTransport, list[str]]:
    """A gateway whose first ``cuts`` calls are cut with ``cut``; then it answers."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if len(seen) <= cuts:
            raise cut("cut by the box's network", request=request)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handle), seen


def _client(transport: httpx.MockTransport) -> GatewayClient:
    config = RunnerConfig(url=URL, run_id="r1", token="t")  # noqa: S106 — a test token
    return GatewayClient(config, transport=transport)


async def test_a_read_cut_once_is_asked_again() -> None:
    transport, seen = _flaky(httpx.RemoteProtocolError, 1)
    assert await _client(transport).request("GET", "/internal/v1/runner/next") == {"ok": True}
    assert len(seen) == 2


async def test_an_event_batch_cut_once_is_sent_again() -> None:
    transport, seen = _flaky(httpx.ReadError, 1)
    path = "/internal/v1/runs/r1/events"
    assert await _client(transport).request("POST", path, json={"seq_from": 0, "events": []}) == {
        "ok": True
    }
    assert seen == [f"POST {path}", f"POST {path}"]


async def test_an_admission_cut_after_connecting_is_never_sent_twice() -> None:
    """The first may have reached the gateway: a second would be a second decision."""
    transport, seen = _flaky(httpx.RemoteProtocolError, 1)
    with pytest.raises(GatewayUnavailable):
        await _client(transport).request("POST", "/v1/admit", json={"call_id": "c1"})
    assert len(seen) == 1


async def test_an_admission_that_never_connected_is_sent_again() -> None:
    transport, seen = _flaky(httpx.ConnectError, 1)
    assert await _client(transport).request("POST", "/v1/admit", json={"call_id": "c1"}) == {
        "ok": True
    }
    assert len(seen) == 2


async def test_retries_are_bounded() -> None:
    transport, seen = _flaky(httpx.ConnectError, 99)
    with pytest.raises(GatewayUnavailable):
        await _client(transport).request("GET", "/internal/v1/runner/next")
    assert len(seen) == client_module.GATEWAY_RETRIES + 1


def test_safe_to_repeat() -> None:
    request = httpx.Request("POST", URL)
    cut = httpx.RemoteProtocolError("cut", request=request)
    assert _safe_to_repeat("GET", "/internal/v1/runner/manifest", cut)
    assert _safe_to_repeat("PUT", "/internal/v1/runs/r1/checkpoint", cut)
    assert _safe_to_repeat("POST", "/internal/v1/runs/r1/events", cut)
    assert not _safe_to_repeat("POST", "/v1/admit", cut)
    assert not _safe_to_repeat("POST", "/internal/v1/runs/r1/result", cut)
    assert not _safe_to_repeat("POST", "/internal/v1/runs/r1/tools/x", cut)
    assert _safe_to_repeat("POST", "/v1/admit", httpx.ConnectError("refused", request=request))


def test_the_runner_model_reads_turns_whole_and_reconnects() -> None:
    from tulip.runner import build

    assert build.MODEL_STREAM_RECONNECTS >= 1
