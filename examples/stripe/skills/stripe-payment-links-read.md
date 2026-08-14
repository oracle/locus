---
name: stripe-payment-links-read
description: >
  Read a payment link and its line items to establish what a customer was sent.
  Use when the charge originated from a link rather than a checkout you control.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 6
required_probes:
  - name: link_read
    match: "/v1/payment_links"
    description: The link as it was configured when it was sent.
---

# Establish what the link offered

A payment link is a standing offer. Read it and its line items before accepting
a claim about what it said — links are edited, and the current shape is not
necessarily the one the customer saw.
