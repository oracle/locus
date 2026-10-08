# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A fake Tulip gateway for the runner protocol tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from tulip.runner.client import GatewayClient, RunnerConfig


TOKEN = "wl-secret-token-value"  # noqa: S105 — a test value
RUN_ID = "run-1"

Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class FakeGateway:
    """Routes by ``(method, path)``; records every request it is sent."""

    routes: dict[tuple[str, str], Handler | Any] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    down: bool = False

    def on(self, method: str, path: str, answer: Handler | Any) -> None:
        self.routes[(method, path)] = answer

    def bodies(self, method: str, path: str) -> list[Any]:
        return [
            json.loads(r.content) if r.content else None
            for r in self.requests
            if r.method == method and r.url.path == path
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("gateway unreachable", request=request)
        answer = self.routes.get((request.method, request.url.path))
        if answer is None:
            return httpx.Response(404, json={"detail": "no such route"})
        if callable(answer):
            result: httpx.Response = answer(request)
            return result
        return httpx.Response(200, json=answer)


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def client(gateway: FakeGateway) -> GatewayClient:
    config = RunnerConfig(url="http://gateway.test", run_id=RUN_ID, token=TOKEN)
    return GatewayClient(config, transport=httpx.MockTransport(gateway))
