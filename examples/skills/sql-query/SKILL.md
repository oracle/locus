---
name: sql-query
description: Use when the user asks the agent to write, review, or explain a SQL query — enforces correctness, safety, and read-only-by-default behavior.
allowed-tools: kb_search
license: UPL-1.0
metadata:
  author: locus
  domain: data
---

# SQL query authoring

When asked to produce or review a SQL query, walk these four steps in order. Each step's output is required in the final response — no shortcuts.

## 1. Restate the question

In one sentence, restate what data the user is asking for. If the question has more than one interpretation, list them and ask the user to pick *before* writing SQL.

## 2. Read-only by default

Default to `SELECT`. Only emit `INSERT` / `UPDATE` / `DELETE` / `TRUNCATE` / `DROP` / `ALTER` when the user has explicitly used a writing verb in their request *and* explicitly confirmed the table they want changed.

If the request is ambiguous about read vs. write, treat it as read.

## 3. Write the query

Constraints, in order of importance:

1. **Always include a `LIMIT`** on `SELECT *` against tables that may be large. Default to `LIMIT 100` if you don't know the row count.
2. **Always qualify columns** when joining (`u.email`, not `email`).
3. **Use parameterised placeholders** (`:name` or `?`) for any value that came from the user — never inline literals from user input.
4. **Prefer explicit `JOIN ... ON`** over comma-joins.
5. **Avoid `SELECT *`** in production-bound queries — list the columns the consumer actually needs.

## 4. Annotate

After the query, list:

- **Indexes used** (or "unknown — schema not provided")
- **Estimated row scan** (small / medium / large / unbounded)
- **Side effects** ("none — read-only" or a precise description)
- **Rollback plan** for any non-`SELECT` query

## Anti-patterns

- ❌ Don't write `DELETE` without a `WHERE` clause unless the user explicitly says "delete every row in <table>."
- ❌ Don't combine multiple statements without explaining the order and side effects.
- ❌ Don't skip the annotation block "because it's a simple query."
