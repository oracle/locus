---
name: stripe-webhooks-read
description: >
  Read webhook endpoints and portal configuration to establish whether the
  integration was told about an event. Use when the question is "why did our
  system not know?"
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 6
required_probes:
  - name: portal_config
    match: "/v1/billing_portal"
    description: What the customer can change themselves, before you change it for them.
  - name: endpoint_read
    match: "/v1/webhook_endpoints"
    description: Which endpoints exist and which events they are subscribed to.
---

# Establish what was notified

A missing internal record is often a missing subscription, not a missing event.
Read the endpoints and their `enabled_events` before concluding that Stripe did
not send something.
