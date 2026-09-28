# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Back-compat re-export — the adapter toolkit now lives in :mod:`tulip_security.adapter`.

The shared helper conventions were promoted to the public contract module
``tulip_security.adapter`` (the langchain-core-style boundary). This module
re-exports them so the bundled adapters' ``from tulip_security._adapters import …``
imports keep working; new code should import from ``tulip_security`` or
``tulip_security.adapter``.
"""

from __future__ import annotations

from tulip_security.adapter import (
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
