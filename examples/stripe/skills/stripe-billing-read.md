---
name: stripe-billing-read
description: >
  Establish the billing context around a payment — invoices, invoice items,
  subscriptions and schedules. Use when the question is about what the customer
  was billed for rather than about a single charge.
allowed-tools:
  - stripe_api_read
min-tool-calls: 2
max-tool-calls: 14
required_probes:
  - name: invoice_read
    match: "/v1/invoices"
    description: The invoice behind the charge.
  - name: invoice_items
    match: "/v1/invoiceitems"
    description: The line items, where a wrong-amount claim is actually settled.
  - name: subscription_context
    match: "/v1/subscriptions"
    description: Whether this is recurring, which changes what a refund means.
---

# Establish what was billed

A charge is the money; the invoice is the reason. Read `/v1/invoices` and its
line items, then `/v1/subscriptions` — a refund on a recurring charge without
touching the subscription refunds one month and bills the next.
