---
name: stripe-subscriptions-read
description: >
  Establish a subscription's state before anything is changed — its status,
  current period, schedule and items. Use before any cancellation, proration or
  refund on recurring revenue.
allowed-tools: [stripe_api_read]
min-tool-calls: 2
max-tool-calls: 12
required_probes:
  - name: subscription_read
    match: "/v1/subscriptions"
    description: Status and current period — a refund's meaning depends on both.
  - name: schedule_read
    match: "/v1/subscription_schedules"
    description: Pending changes — the shape it is about to become.
    description: The schedule or items behind it, where the future changes live.
---

# Establish the subscription

`status` and `current_period_end` decide whether a refund is a refund or a
credit. Read the schedule too: a subscription with a pending schedule change is
about to become something else, and acting on today's shape misses it.
