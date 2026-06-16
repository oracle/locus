#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
# Workbench helper script — relax lint rules that don't apply here.
# ruff: noqa: BLE001, PLW2901, T201

"""Run every model-only notebook through the workbench harness.

Standalone integration sweep — not part of the playwright e2e (e2e is
fast UI smoke; this is the slow real-GenAI sweep). Exits non-zero if
any attempted notebook finished with a non-zero exit.

Usage:
    # With runner :8100 + BFF :3101 already up + valid OCI session.
    python workbench/scripts/test-all-notebooks.py

    # Scope to specific notebooks:
    python workbench/scripts/test-all-notebooks.py --include 1,2,11

    # Bigger budget per notebook (default 180s):
    python workbench/scripts/test-all-notebooks.py --timeout 240
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any

import requests


# Notebooks that pause for human input via locus.core.interrupt(). The
# harness skips them — they'd hang until timeout in subprocess mode.
RED = {9, 45, 46, 47, 48}

DEFAULT_BFF = os.environ.get("LOCUS_BFF", "http://127.0.0.1:3101")
DEFAULT_PROFILE = os.environ.get("OCI_PROFILE", "MY_PROFILE")
DEFAULT_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
DEFAULT_COMPARTMENT = os.environ.get(
    "OCI_COMPARTMENT",
    "ocid1.compartment.oc1..aaaaaaaa00000000000000000000000000000000000000000000000000000000",
)
DEFAULT_MODEL = os.environ.get("OCI_MODEL", "openai.gpt-5.5-2026-04-23")


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def run_one(bff: str, tut: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST the unedited source to /api/notebooks/run; collect outcome."""
    payload = {
        "source": tut["source"],
        "provider": {
            "provider": "oci-session",
            "model": DEFAULT_MODEL,
            "profile": DEFAULT_PROFILE,
            "region": DEFAULT_REGION,
            "compartment_id": DEFAULT_COMPARTMENT,
            "oci_transport": "auto",
        },
        "timeout_seconds": timeout,
    }
    started = time.monotonic()
    stderr_tail: deque[str] = deque(maxlen=30)
    stdout_lines = 0
    exit_code: int | None = None
    err_msg: str | None = None
    try:
        with requests.post(
            f"{bff}/api/notebooks/run",
            json=payload,
            stream=True,
            timeout=timeout + 30,
        ) as resp:
            if resp.status_code != 200:
                err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode(errors="replace")
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type", "")
                    if t == "stdout":
                        stdout_lines += 1
                    elif t == "stderr":
                        stderr_tail.append(ev.get("text", ""))
                    elif t == "error":
                        stderr_tail.append(f"[error] {ev.get('text', '')}")
                    elif t == "exit":
                        exit_code = int(ev.get("code", -1))
                        break
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started
    status = "PASS" if exit_code == 0 else ("FAIL" if exit_code is not None else "ERR")
    return {
        "id": tut["id"],
        "number": tut["number"],
        "title": tut["title"],
        "status": status,
        "exit_code": exit_code,
        "duration": elapsed,
        "stdout_lines": stdout_lines,
        "stderr_tail": list(stderr_tail),
        "err_msg": err_msg,
    }


def parse_include(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(int(tok))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bff", default=DEFAULT_BFF, help="BFF base URL")
    p.add_argument("--timeout", type=int, default=180, help="Per-notebook timeout (s)")
    p.add_argument("--include", default=None, help="Comma list of notebook numbers")
    args = p.parse_args()

    include = parse_include(args.include)

    print(f"# Workbench notebook sweep — {DEFAULT_MODEL}")
    print(f"# bff={args.bff}  timeout={args.timeout}s")
    print()

    try:
        catalog = requests.get(f"{args.bff}/api/notebooks", timeout=10).json()
    except Exception as exc:
        print(f"failed to reach BFF /api/notebooks: {exc}", file=sys.stderr)
        return 2

    skipped = []
    runnable = []
    for entry in catalog:
        if include is not None and entry["number"] not in include:
            continue
        if entry["number"] in RED or entry.get("needs_stdin"):
            skipped.append(entry)
            continue
        runnable.append(entry)

    print(
        f"Catalog: {len(catalog)} notebooks, "
        f"{len(skipped)} skipped (needs stdin), "
        f"{len(runnable)} attempting"
    )
    print()

    print("| #  | id                                   | status | exit | duration  | stdout |")
    print("|---:|--------------------------------------|--------|-----:|----------:|-------:|")

    results: list[dict[str, Any]] = []
    for entry in runnable:
        # Re-fetch detail to get source.
        try:
            tut = requests.get(f"{args.bff}/api/notebooks/{entry['id']}", timeout=10).json()
        except Exception as exc:
            print(
                f"| {entry['number']:>2} | {entry['id']:<36} | ERR    |    - |     -     |   -   |"
                f"  ← fetch failed: {exc}"
            )
            continue
        r = run_one(args.bff, tut, args.timeout)
        results.append(r)
        status_icon = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "ERR": "⚠️  ERR "}[r["status"]]
        ex = "—" if r["exit_code"] is None else str(r["exit_code"])
        print(
            f"| {r['number']:>2} | {r['id']:<36} | {status_icon:<6} | {ex:>4} | "
            f"{fmt_duration(r['duration']):>9} | {r['stdout_lines']:>6} |"
        )
        sys.stdout.flush()

    failures = [r for r in results if r["status"] != "PASS"]
    print()
    print(f"## Summary: {len(results) - len(failures)}/{len(results)} passed")
    if skipped:
        ids = ", ".join(str(e["number"]) for e in skipped)
        print(f"## Skipped (needs stdin): {ids}")
    if failures:
        print("\n## Failures\n")
        for r in failures:
            print(f"### {r['id']} (exit {r['exit_code']}, {fmt_duration(r['duration'])})")
            if r["err_msg"]:
                print(f"  {r['err_msg']}")
            for line in r["stderr_tail"]:
                print(f"  {line}")
            print()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
