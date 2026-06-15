# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — SIEM search graduated into the SDK.

This adapter is now first-class in :mod:`tulip.security.siem`. Importing
from here still works; prefer::

    from tulip.security import query_siem, siem_query_tool
"""

from __future__ import annotations

from tulip.security.siem import query_siem, siem_query_tool


__all__ = ["query_siem", "siem_query_tool"]
