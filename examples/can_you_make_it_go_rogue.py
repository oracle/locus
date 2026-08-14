#!/usr/bin/env python
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""🔓 Can you make the agent go rogue?  (see ``tulip.rogue`` for the source)

The challenge now ships inside the package, so it runs from an install with
no clone:

    pip install tulip-agents
    python -m tulip.rogue

This file stays as the examples-directory entry point and is equivalent::

    python examples/can_you_make_it_go_rogue.py
"""

import asyncio

from tulip.rogue.challenge import main


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
