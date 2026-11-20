# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The per-test budget from ``conftest.py`` is on every integration item.

Needs no service, so it runs on every integration job and fails the moment
the hook stops marking items.
"""

from __future__ import annotations

import os

import pytest


def test_integration_items_carry_the_default_budget(request) -> None:
    config = request.config
    if not config.pluginmanager.hasplugin("timeout"):
        pytest.fail("pytest-timeout is not installed; install the dev extras")
    marker = request.node.get_closest_marker("timeout")
    run_sets_a_timeout = (
        config.getoption("timeout") is not None
        or bool(config.getini("timeout"))
        or bool(os.environ.get("PYTEST_TIMEOUT"))
    )
    if run_sets_a_timeout:
        # The run's own budget replaces the default, so no marker is added.
        assert marker is None
    else:
        assert marker is not None
        assert marker.args == (300,)  # INTEGRATION_TIMEOUT_SECONDS in conftest.py
