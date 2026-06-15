---
name: siem-query
description: Use when the user asks the agent to write, review, or explain a SIEM / log search (Splunk SPL, Elastic/KQL, or SQL over a log table) — enforces correctness, scoping, and read-only-by-default behavior.
allowed-tools: kb_search
license: Apache-2.0
metadata:
  author: tulip
  domain: security
---

# SIEM / log-query authoring

When asked to produce or review a detection or hunt query, walk these
four steps in order. Each step's output is required in the final
response — no shortcuts.

## 1. Restate the hunt

In one sentence, restate what activity the user is looking for (e.g.
"failed logons followed by a success from the same source IP"). If the
question has more than one interpretation, list them and ask the user to
pick *before* writing the query.

## 2. Read-only by default

Detection and hunting are read paths. Emit **search/read** queries only.
Never emit anything that mutates the SIEM or the underlying store —
no index deletion, no `| delete`, no `DROP`/`ALTER`, no retention
changes — even if asked in passing; surface that as a separate,
explicitly-confirmed action.

## 3. Write the query

Constraints, in order of importance:

1. **Always bound the time window** (`earliest=-24h`, `@timestamp >= now-1d`).
   An unbounded hunt over all history is the query equivalent of a full
   table scan — default to the last 24h if the user doesn't say.
2. **Scope to the relevant index / sourcetype / data model** rather than
   searching everything.
3. **Filter early, enrich late** — narrow on indexed fields before
   `stats`/`eval`/joins.
4. **Parameterise indicators** (host, user, IP) instead of inlining
   values pulled from an alert, so the query is reusable.
5. **Map to a detection framework** where one applies — name the MITRE
   ATT&CK technique the query is meant to surface.

## 4. Annotate

After the query, list:

- **Fields / indexes touched** (or "unknown — schema not provided")
- **Expected volume** (small / medium / large / unbounded)
- **False-positive sources** (benign activity that would also match)
- **Tuning notes** — what to add to cut noise without losing the signal

## Anti-patterns

- ❌ Don't run an unbounded all-time search "to be thorough" — it buries
  the analyst and hammers the cluster.
- ❌ Don't alert on a single raw event when a threshold or sequence is
  what actually indicates the behavior.
- ❌ Don't skip the false-positive + tuning block "because the rule is
  obvious."
