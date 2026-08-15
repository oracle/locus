---
name: stripe-invoices-write
description: >
  Change an invoice — create, update, finalize, void, or mark uncollectible.
  Use only when a person has decided which of those is correct; they are not
  interchangeable and two of them are irreversible.
allowed-tools: [stripe_api_write]
min-tool-calls: 1
max-tool-calls: 3
required_probes:
  - name: invoice_targeted
    match: "/v1/invoices"
    description: The write names an invoice path, not some other resource.
---

# Change an invoice, deliberately

`void` and `mark_uncollectible` both end an invoice and they mean opposite
things to the accounts: void says it should never have existed, uncollectible
says it existed and will not be paid. Pick the one the person approved and say
which you are using.

This skill carries the generic writer because Stripe has no specific tool for
these. That is a reason for more care, not less: `stripe_api_write` can reach
the entire API, and only the path you pass keeps it on an invoice.
