#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""What Tulip costs you, with the model held still.

An agent framework should be a thin runtime. The only way to say that
honestly is to measure the part that is ours and publish it, so this file
measures Tulip with the model replaced by :class:`tulip.testing.ScriptedModel`
— a double that returns a canned turn immediately.

That is the whole trick. Against a real provider every number here would be
swamped by network and inference time, and a framework could hide almost any
amount of overhead inside a 900 ms model call. With the model pinned at
roughly zero, what remains *is* the framework: message assembly, the loop,
tool dispatch, hook fan-out, event construction.

    python benchmarks/framework_overhead.py            # all of it
    python benchmarks/framework_overhead.py --json     # machine-readable

Numbers are per-operation medians over many iterations, with the interpreter
warmed first, reported alongside the interquartile range so a reader can see
the spread rather than trusting a single figure. Import cost is measured in a
subprocess, since it can only be paid once per process.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------


def _measure(fn: Callable[[], Any], *, iterations: int, warmup: int) -> dict[str, float]:
    """Median and IQR of ``fn`` over ``iterations``, in milliseconds.

    The median rather than the mean: one GC pause in the middle of a run
    should not become the headline number.
    """
    for _ in range(warmup):
        fn()

    gc.collect()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    q1 = statistics.median(samples[: len(samples) // 2])
    q3 = statistics.median(samples[(len(samples) + 1) // 2 :])
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 4),
        "iqr_ms": round(q3 - q1, 4),
        "iterations": iterations,
    }


def _measure_import() -> dict[str, float]:
    """Cost of ``import tulip`` in a cold interpreter.

    Measured in a subprocess because an import happens once per process and
    is cached thereafter — timing it in-process measures a dict lookup.
    """
    code = "import time; t=time.perf_counter(); import tulip; print((time.perf_counter()-t)*1000)"
    samples = []
    for _ in range(5):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        samples.append(float(out.stdout.strip()))
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "min_ms": round(samples[0], 2),
        "iterations": len(samples),
    }


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------


def build_cases() -> list[tuple[str, str, Callable[[], Any], int]]:
    """Each case is (key, human label, thunk, iterations)."""
    from tulip import Agent, tool
    from tulip.testing import ScriptedModel, text, tool_call

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def _scripted(turns: int = 1) -> ScriptedModel:
        if turns == 1:
            return ScriptedModel([text("done")])
        return ScriptedModel([tool_call("add", a=2, b=2), text("done")])

    def construct() -> None:
        Agent(model=_scripted(), system_prompt="Be brief.")

    def construct_with_tools() -> None:
        Agent(model=_scripted(), tools=[add], system_prompt="Be brief.")

    def one_turn() -> None:
        agent = Agent(model=_scripted(), system_prompt="Be brief.")
        agent.run_sync("hello")

    def one_turn_with_tool() -> None:
        agent = Agent(model=_scripted(2), tools=[add], system_prompt="Be brief.")
        agent.run_sync("add 2 and 2")

    async def _stream_first_event() -> None:
        agent = Agent(model=_scripted(), system_prompt="Be brief.")
        async for _event in agent.run("hello"):
            break  # time to first event is the number that matters

    def stream_first_event() -> None:
        asyncio.run(_stream_first_event())

    return [
        ("agent_construction", "Agent(...) — no tools", construct, 2000),
        ("agent_construction_tools", "Agent(...) — 1 tool", construct_with_tools, 2000),
        ("run_sync_1_turn", "run_sync — 1 model turn", one_turn, 1000),
        ("run_sync_tool_call", "run_sync — tool call + turn", one_turn_with_tool, 1000),
        ("stream_first_event", "run() — time to first event", stream_first_event, 500),
    ]


def measure_memory() -> dict[str, float]:
    """Peak Python heap for one agent doing one turn."""
    from tulip import Agent, tool
    from tulip.testing import ScriptedModel, text

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    agent = Agent(model=ScriptedModel([text("done")]), tools=[add], system_prompt="Be brief.")
    agent.run_sync("hello")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "peak_kib": round((peak - base) / 1024, 1),
        "retained_kib": round((current - base) / 1024, 1),
    }


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--quick", action="store_true", help="fewer iterations, for smoke runs")
    args = ap.parse_args()

    import tulip

    results: dict[str, Any] = {
        "tulip_version": tulip.__version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "model": "tulip.testing.ScriptedModel (no network, no inference)",
    }

    results["import"] = _measure_import()
    for key, label, fn, default_iterations in build_cases():
        iterations = max(20, default_iterations // 50) if args.quick else default_iterations
        stats = _measure(fn, iterations=iterations, warmup=max(5, iterations // 20))
        stats["label"] = label
        results[key] = stats
    results["memory"] = measure_memory()

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(
        f"\ntulip {results['tulip_version']} · python {results['python']} · {results['platform']}"
    )
    print("model replaced by ScriptedModel — every number below is framework time\n")
    print(f"  {'operation':<32} {'median':>10} {'p95':>10} {'iqr':>9}")
    print(f"  {'-' * 32} {'-' * 10} {'-' * 10} {'-' * 9}")
    imp = results["import"]
    print(f"  {'import tulip (cold process)':<32} {imp['median_ms']:>9.2f}ms {'—':>10} {'—':>9}")
    for key, _label, _fn, _i in build_cases():
        r = results[key]
        print(
            f"  {r['label']:<32} {r['median_ms']:>9.3f}ms "
            f"{r['p95_ms']:>9.3f}ms {r['iqr_ms']:>8.3f}ms"
        )
    mem = results["memory"]
    print(f"\n  peak heap, one agent + one turn : {mem['peak_kib']:.1f} KiB")
    print(f"  retained after the turn         : {mem['retained_kib']:.1f} KiB\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
