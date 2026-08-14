---
name: stripe-checkout-read
description: >
  Read a Checkout Session and what was actually in the basket. Use when the
  dispute or question is about what the customer believed they were buying.
allowed-tools: [stripe_api_read]
min-tool-calls: 2
max-tool-calls: 8
required_probes:
  - name: session_read
    match: "/v1/checkout/sessions"
    description: The session, for its mode, status and totals.
  - name: line_items_read
    match: "line_items"
    description: What was in it. "They ordered the wrong thing" is settled here.
---

# Read what was actually bought

The session totals answer "how much"; the line items answer "for what". A
dispute reasoned from the total alone cannot tell a wrong-item claim from a
wrong-price one.
