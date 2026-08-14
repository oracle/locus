# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``python -m tulip.rogue`` — run the rogue-agent challenge."""

import asyncio

from tulip.rogue.challenge import main


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
