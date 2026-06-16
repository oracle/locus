#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-file coverage ratchet.

Runs after ``pytest --cov --cov-report=json``. Reads ``coverage.json`` and
compares each tracked file's line coverage against the values stored in
``scripts/coverage_baseline.json``. The job fails if any file dropped by
more than ``--tolerance`` percentage points; new files are accepted as
long as they meet ``--min-new`` (default 90%).

Why a baseline file instead of just ``fail_under``: a global threshold lets
heavily-tested files mask regressions in lightly-tested ones. The ratchet
gives every file its own floor and only allows downward movement when the
baseline is explicitly refreshed (``--update``), which forces a code review.

Usage:
    # Check current coverage against baseline (CI mode).
    python scripts/coverage_ratchet.py --check

    # Refresh the baseline after an intentional change. Commit the result.
    python scripts/coverage_ratchet.py --update

The baseline file is JSON: ``{ "<relative-path>": <percentage>, ... }``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_JSON = REPO_ROOT / "coverage.json"
BASELINE_JSON = REPO_ROOT / "scripts" / "coverage_baseline.json"


def _load_current_coverage() -> dict[str, float]:
    """Return ``{filename: line-coverage-percent}`` for every tracked file.

    Reads the JSON output produced by ``coverage json``. Filenames are
    stored relative to the repo root so they survive CI vs. local runs.
    """
    if not COVERAGE_JSON.exists():
        sys.exit(
            f"error: {COVERAGE_JSON} not found. Run "
            f"`pytest --cov=src/locus --cov-report=json tests/unit` first."
        )
    data: dict[str, Any] = json.loads(COVERAGE_JSON.read_text())
    files: dict[str, dict[str, Any]] = data.get("files", {})
    out: dict[str, float] = {}
    for raw_path, file_data in files.items():
        # Coverage.py emits absolute paths under some configurations and
        # relative ones under others. Normalise to repo-relative.
        path = Path(raw_path)
        if path.is_absolute():
            try:
                path = path.relative_to(REPO_ROOT)
            except ValueError:
                # Not under the repo (e.g. site-packages); skip it.
                continue
        rel = path.as_posix()
        # ``percent_covered`` is the line-coverage figure coverage.py
        # already calculated (line + branch when branch=True).
        summary = file_data.get("summary", {})
        pct = summary.get("percent_covered")
        if pct is None:
            continue
        out[rel] = round(float(pct), 2)
    return out


def _load_baseline() -> dict[str, float]:
    if not BASELINE_JSON.exists():
        return {}
    raw: dict[str, float] = json.loads(BASELINE_JSON.read_text())
    return {k: float(v) for k, v in raw.items()}


def _save_baseline(values: dict[str, float]) -> None:
    BASELINE_JSON.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
    )


def _check(tolerance: float, min_new: float) -> int:
    current = _load_current_coverage()
    baseline = _load_baseline()
    regressions: list[tuple[str, float, float]] = []
    new_below_floor: list[tuple[str, float]] = []

    for path, pct in sorted(current.items()):
        prior = baseline.get(path)
        if prior is None:
            if pct + 1e-9 < min_new:
                new_below_floor.append((path, pct))
            continue
        # Allow ``tolerance`` percentage points of drift to absorb branch
        # coverage noise on small files (a single branch flip can move a
        # 30-statement file by ~1.5 points).
        if pct + tolerance < prior:
            regressions.append((path, prior, pct))

    if regressions:
        print("Coverage regressions vs baseline:")
        for path, prior, pct in regressions:
            print(f"  {path}: {prior:.2f}% -> {pct:.2f}% (drop {prior - pct:.2f}pp)")
    if new_below_floor:
        print(f"New files below {min_new:.0f}% floor:")
        for path, pct in new_below_floor:
            print(f"  {path}: {pct:.2f}%")

    if regressions or new_below_floor:
        print(
            "\nIf the change is intentional, refresh the baseline with: "
            "python scripts/coverage_ratchet.py --update"
        )
        return 1

    print(f"Coverage ratchet: {len(current)} files checked, no regressions.")
    return 0


def _update() -> int:
    current = _load_current_coverage()
    _save_baseline(current)
    print(f"Baseline updated: {len(current)} files written to {BASELINE_JSON}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Compare against baseline (CI).")
    group.add_argument("--update", action="store_true", help="Refresh baseline.")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Allowed downward drift in percentage points (default: 0.5).",
    )
    p.add_argument(
        "--min-new",
        type=float,
        default=90.0,
        help="Minimum coverage for files not yet in the baseline (default: 90).",
    )
    args = p.parse_args()
    if args.update:
        return _update()
    return _check(tolerance=args.tolerance, min_new=args.min_new)


if __name__ == "__main__":
    raise SystemExit(main())
