# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The engine moved out of the gateway without changing a byte of what it does.

``fixtures/golden.json`` is the full trace of every scripted scenario in
``golden_scenarios.py`` as tulip-gateway's own engine produced it (``make_golden.py``):
the events, the audit records, every call's answer, the final graph, the model's prose,
the ``when`` corpus. The same scripts run here on :mod:`tulip.playbooks.v2`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.unit.playbooks_v2.golden_scenarios import FIXTURES, SCENARIOS, Engine
from tulip.playbooks.v2 import engine, results, when


GOLDEN: dict[str, Any] = json.loads((FIXTURES / "golden.json").read_text("utf-8"))


def sdk_engine() -> Engine:
    def make(pb: Any, *, audit: list[dict[str, Any]], **kwargs: Any) -> Any:
        def record(event: str, fields: dict[str, Any]) -> None:
            audit.append({"event": event, **fields})

        return engine.PlaybookRuntime(pb, record=record, **kwargs)

    def mode(pb: Any, deployment: str) -> str:
        return engine.enforcement_mode(pb, deployment)

    return Engine(engine=engine, results=results, when=when, make=make, mode=mode)


def test_the_golden_covers_every_scenario() -> None:
    assert GOLDEN["source"].startswith("tulip-gateway ")
    assert set(GOLDEN["scenarios"]) == set(SCENARIOS)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_sdk_engine_reproduces_the_gateway_trace(name: str) -> None:
    assert SCENARIOS[name](sdk_engine()) == GOLDEN["scenarios"][name]
