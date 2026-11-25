#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Check the router's training split and export it as training rows.

The training split is ``benchmarks/router/train.json``; the routing corpus,
``benchmarks/router/cases.json``, is the held-out set a classifier is gated
on. This script holds the split to the corpus's rules, refuses a prompt that
overlaps a held-out one, checks that every row's profile compiles to the
protocol its label names, and prints the cases per task type and difficulty.
No model is called.

    python scripts/router_training_set.py
    python scripts/router_training_set.py --jsonl router-train.jsonl
    python scripts/router_training_set.py --measured difficulty.json --jsonl router-train.jsonl

``--jsonl`` writes one row a line: the prompt, the profile to emit for it
(task type, difficulty, risk, needs) and the protocol that profile compiles
to. ``--measured`` names a file written by ``scripts/router_difficulty.py
--corpus benchmarks/router/train.json --out``; each row then also carries the
level that measurement found. The exit code is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from locus.router.benchmark import benchmark_compiler, load_router_corpus  # noqa: E402
from locus.router.training import (  # noqa: E402
    cell_table,
    load_router_training_set,
    training_examples,
    write_jsonl,
)


TRAIN = ROOT / "benchmarks" / "router" / "train.json"
HELD_OUT = ROOT / "benchmarks" / "router" / "cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default=str(TRAIN), help="the training split")
    parser.add_argument("--held-out", default=str(HELD_OUT), help="the corpus it must not overlap")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="word overlap at or above which two prompts count as the same (default 0.5)",
    )
    parser.add_argument("--measured", help="a measurement from router_difficulty.py --out")
    parser.add_argument("--jsonl", help="write the training rows to this file")
    args = parser.parse_args()

    held_out = load_router_corpus(args.held_out)
    try:
        cases = load_router_training_set(args.train, held_out=held_out, threshold=args.threshold)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    measured = None
    if args.measured:
        payload = json.loads(Path(args.measured).read_text())
        measured = {row["case"]: row["measured"] for row in payload["cases"]}
    examples = training_examples(cases, measured=measured)

    compiler = benchmark_compiler(None)
    wrong = [
        e.name
        for e in examples
        if compiler.explain(e.profile.to_frame()).selected_protocol != e.protocol
    ]
    if wrong:
        print(f"FAIL: {len(wrong)} profile(s) do not compile to their label: {', '.join(wrong)}")
        return 1

    print(f"{len(examples)} training rows, none overlapping the {len(held_out)} held-out cases\n")
    print(cell_table(cases))
    needs = Counter(need for e in examples for need in e.profile.needs)
    print("\nneeds: " + ", ".join(f"{need} {count}" for need, count in needs.most_common()))
    shapes = Counter(e.protocol for e in examples)
    print("protocols: " + ", ".join(f"{shape} {count}" for shape, count in shapes.most_common()))
    if measured is not None:
        levels = Counter(e.measured_difficulty or "not measured" for e in examples)
        print("measured: " + ", ".join(f"{level} {count}" for level, count in levels.items()))
    if args.jsonl:
        write_jsonl(examples, args.jsonl)
        print(f"\nrows written to {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
