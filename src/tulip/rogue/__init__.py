# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The rogue-agent challenge — try to make a governed agent fire a real action.

Ships inside the package so it runs straight from an install, with no clone
and no API key::

    pip install tulip-agents
    python -m tulip.rogue

The agent holds live-looking production tools and the offline model arrives
already compromised. What holds is :func:`tulip.control.admit`, not the
model's judgement.
"""

from tulip.rogue.challenge import build_agent, main, pick_mode


__all__ = ["build_agent", "main", "pick_mode"]
