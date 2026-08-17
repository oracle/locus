# Benchmarks

An agent framework should be a thin runtime. That is a claim, and claims about
performance are worth exactly as much as the numbers behind them, so this
directory measures Tulip and publishes what it finds — including the number we
would rather not have.

## Method

Every measurement replaces the model with
[`tulip.testing.ScriptedModel`](../src/tulip/testing.py), a double that returns
a canned turn immediately. Against a real provider a single call costs hundreds
of milliseconds and a framework can hide almost any amount of overhead inside
it. With the model pinned at roughly zero, what remains **is** the framework:
message assembly, the loop, tool dispatch, hook fan-out, event construction.

Figures are per-operation medians over 500–2000 iterations with the interpreter
warmed first, reported with p95 and interquartile range so the spread is
visible rather than implied. Import cost is measured in a fresh subprocess,
because an import is paid once per process and timing it in-process measures a
dictionary lookup.

```bash
python benchmarks/framework_overhead.py           # table
python benchmarks/framework_overhead.py --json    # machine-readable
python benchmarks/framework_overhead.py --quick   # smoke run
```

## Results

`tulip 2.10.0` · CPython 3.12.3 · Linux x86-64 · 2026-08-16

| operation | median | p95 | IQR |
| --- | ---: | ---: | ---: |
| `Agent(...)` — no tools | **0.263 ms** | 1.018 ms | 0.078 ms |
| `Agent(...)` — 1 tool | **0.257 ms** | 0.992 ms | 0.072 ms |
| `run_sync` — 1 model turn | **0.802 ms** | 1.782 ms | 0.335 ms |
| `run_sync` — tool call + turn | **3.357 ms** | 5.535 ms | 1.323 ms |
| `run()` — time to first event | **0.790 ms** | 1.657 ms | 0.392 ms |
| `import tulip` — cold process | **255.9 ms** | — | — |

Peak Python heap for one agent doing one turn: **138 KiB**, of which 128 KiB is
still retained afterwards.

## Reading these

**The runtime is thin.** Sub-millisecond agent construction and a
sub-millisecond turn mean that against any real model — where a fast provider
answers in 300 ms and a reasoning model takes several seconds — Tulip is well
under 1% of wall-clock. A full tool round trip at 3.4 ms is the most expensive
in-loop operation and is still noise beside inference.

**Import is not thin, and that is the honest weak spot.** 256 ms to
`import tulip` is slow enough to be felt in a CLI, a serverless cold start, or
a test suite that spawns processes. Profiling with `python -X importtime`
attributes it roughly as:

```
270 ms  tulip
252 ms    tulip.core.config
169 ms      tulip.core.command
 63 ms        pydantic          (54 ms pydantic_core, 33 ms asyncio)
```

About 63 ms is pydantic and effectively fixed — it is a core dependency and
every framework built on it pays the same. The remainder sits under
`tulip.core.command`, and that part is ours. Deferring work there behind lazy
imports is the obvious lever, and this table exists partly so that improvement
is visible when it lands rather than asserted.

## What is deliberately not measured

**Throughput and concurrency.** Both are dominated by the provider and by how
you deploy, so a number here would say more about the harness than about
Tulip.

**Comparisons against other frameworks.** A fair cross-framework benchmark
needs equivalent configurations, equivalent middleware, and an author with no
stake in the outcome. Any of those numbers published here would deserve to be
distrusted, so they are not published here. The methodology above is
deliberately simple enough to reproduce against anything else.
