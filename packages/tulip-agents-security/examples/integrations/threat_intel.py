# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — threat-intel IOC enrichment graduated into the SDK.

This adapter is now first-class in :mod:`tulip_security.intel`. Importing
from here still works; prefer::

    from tulip_security import (
        enrich_indicator,
        enrich_indicator_tool,
        enrich_to_finding,
    )
"""

from __future__ import annotations

from tulip_security.intel import (
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
