---
name: incident-triage
description: Use when diagnosing a live production incident — guides the agent through alert triage, metric correlation, and impact assessment.
allowed-tools: get_metric list_alerts kb_search
license: UPL-1.0
metadata:
  author: locus
  domain: observability
---

# Incident triage runbook

When the user reports a live incident, work the following four steps in order. Skip steps that are clearly not applicable, but say *why* you skipped them.

## 1. Establish the symptom envelope

State, in one line each:

- **Surface**: which service, endpoint, or screen is affected
- **Onset**: when did the symptom first appear (if known)
- **Scope**: percentage of traffic, regions, customer tiers

If any of these are missing, ask the user *once* before moving on. Do not loop.

## 2. Pull active alerts in the affected window

Call `list_alerts` with a `window_minutes` value that comfortably brackets the symptom onset (default 30 minutes if unknown). Quote the alert IDs you receive — never invent IDs.

## 3. Pull the headline metric for each alert

For every alert, call `get_metric` with the metric name referenced by the alert (e.g. `latency_p99`, `cpu`, `errors_5xx`). Report the raw value verbatim before interpreting it.

## 4. Correlate and report

Produce a one-paragraph summary in this exact shape:

> **Likely cause**: <one sentence>
>
> **Evidence**: <list of (alert_id, metric_name, value) tuples that support the conclusion>
>
> **Recommended next action**: <one of: "investigate further", "page on-call", "rollback recent change", "no action — within budget">

If evidence is insufficient for any conclusion, say so explicitly and recommend "investigate further" with a list of the *specific* missing data points.

## Anti-patterns

- ❌ Don't invent alert IDs or metric values that didn't come back from a tool call.
- ❌ Don't recommend `rollback` without naming a specific change.
- ❌ Don't skip step 2 even if step 3's output looks "obvious."
