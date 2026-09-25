# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The dependency gate for the MCP loopback-server test modules.

Call :func:`require_server_deps` at the top of a module that drives the real
server in ``_mcp_loopback_server.py``. Locally a missing ``mcp``/``uvicorn``
skips the module; with ``TULIP_REQUIRE_MCP_SERVER_TESTS`` set (CI's full-deps
jobs) it fails collection instead. The gate exists because these modules once
skipped silently on every CI job after the resolver moved to mcp 2.x, which
has no ``mcp.server.fastmcp``.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest


REQUIRE_ENV = "TULIP_REQUIRE_MCP_SERVER_TESTS"


def require_server_deps() -> ModuleType:
    """Import ``mcp`` and ``uvicorn``; return ``uvicorn``."""
    if os.environ.get(REQUIRE_ENV):
        importlib.import_module("mcp")
        return importlib.import_module("uvicorn")
    pytest.importorskip("mcp")
    return pytest.importorskip("uvicorn")
