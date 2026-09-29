# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A test that reconfigures global settings must not leak into the next one.

The two tests run in file order in one process. On a tree without the
suite-level settings fixture (``tests/unit/conftest.py``) the second test
sees ``debug=True`` left behind by the first — the same leak that made the
SSE ``on_error`` tests fail only under ``pytest -n auto``.
"""

from __future__ import annotations

import json

from tulip.core.config import configure, get_settings
from tulip.streaming import SSEHandler


def test_a_reconfigures_debug_and_does_not_clean_up() -> None:
    configure({"debug": True})
    assert get_settings().debug is True


async def test_b_next_test_sees_default_settings() -> None:
    assert get_settings().debug is False

    handler = SSEHandler()
    await handler.on_error(ValueError("secret dsn=postgres://u:p@h/db"))
    data = json.loads(handler.get_messages()[0].data)
    assert data["error"] == "internal error"
