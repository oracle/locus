# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Back-compat re-export — the adapter toolkit now lives in :mod:`tulip.security.adapter`.

The shared helper conventions were promoted to the public contract module
``tulip.security.adapter`` (the langchain-core-style boundary). This module
re-exports them so the bundled adapters' ``from tulip.security._adapters import …``
imports keep working; new code should import from ``tulip.security`` or
``tulip.security.adapter``.
"""

from __future__ import annotations

from tulip.security.adapter import (
    as_json,
    env,
    indicator_type,
    inference_claim,
    tool_match,
)


__all__ = [
    "as_json",
    "env",
    "indicator_type",
    "inference_claim",
    "tool_match",
]
