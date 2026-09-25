# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Suite-wide isolation for process-global state.

``tulip.core.config`` caches one :class:`TulipSettings` per process. A test
that calls ``configure({... "debug": True})`` and leaves it behind changes
the behaviour of every later test in the same process — e.g. the SSE
``on_error`` payload switches from the sanitized ``"internal error"`` to the
raw exception text. Run serially, file order happened to hide the leak (a
later test in the same class reset the global); under ``pytest -n auto`` a
worker can receive the leaking test without its successor and then fail an
unrelated SSE test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import tulip.core.config as _config


@pytest.fixture(autouse=True)
def _restore_global_settings() -> Iterator[None]:
    """Give every test back the settings object that was current before it."""
    saved = _config._settings
    yield
    _config._settings = saved
