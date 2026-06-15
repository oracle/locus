# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — threat-intel IOC enrichment graduated into the SDK.

This adapter is now first-class in :mod:`tulip.security.intel`. Importing
from here still works; prefer::

    from tulip.security import (
        enrich_indicator,
        enrich_indicator_tool,
        enrich_to_finding,
    )
"""

from __future__ import annotations

from tulip.security.intel import (
    classify_indicator,
    enrich_indicator,
    enrich_indicator_tool,
    enrich_to_finding,
)


__all__ = [
    "classify_indicator",
    "enrich_indicator",
    "enrich_indicator_tool",
    "enrich_to_finding",
]
