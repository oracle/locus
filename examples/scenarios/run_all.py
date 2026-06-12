#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Run every threat→defense scenario gist offline and report pass/fail.

Each gist is a standalone script that demonstrates one AI-security threat
and the Tulip defense for it, deterministically and with no credentials.
This runner executes them all under the mock model and exits non-zero if
any fail — the gate that keeps the scenario catalog honest.

    python examples/scenarios/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    gists = sorted(p for p in here.glob("*.py") if p.name not in {"run_all.py", "__init__.py"})

    env = {**os.environ, "TULIP_MODEL_PROVIDER": "mock"}
    # Make the sibling examples/integrations/ package importable by gists
    # that use a real probe (e.g. model_extraction.py).
    env["PYTHONPATH"] = os.pathsep.join([str(here.parent), env.get("PYTHONPATH", "")])

    results: dict[str, bool] = {}
    for gist in gists:
        proc = subprocess.run(  # noqa: S603 — trusted: our own interpreter + a local gist path
            [sys.executable, str(gist)], env=env, capture_output=True, text=True, check=False
        )
        results[gist.name] = proc.returncode == 0
        if proc.returncode != 0:
            print(f"--- {gist.name} FAILED ---\n{proc.stdout}\n{proc.stderr}")

    passed = sum(1 for ok in results.values() if ok)
    print(f"\n{'=' * 50}")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
