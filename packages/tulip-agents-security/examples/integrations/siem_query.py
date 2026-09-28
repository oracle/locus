# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — SIEM search graduated into the SDK.

This adapter is now first-class in :mod:`tulip_security.siem`. Importing
from here still works; prefer::

    from tulip_security import query_siem, siem_query_tool
"""

from __future__ import annotations

from tulip_security.siem import query_siem, siem_query_tool


__all__ = ["query_siem", "siem_query_tool"]
