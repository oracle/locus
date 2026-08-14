---
name: stripe-payments-read
description: >
  Establish the facts of a payment: the charge, its payment intent, the
  customer, and any refunds already issued. Use before any decision that turns
  on what was actually paid.
allowed-tools:
  - stripe_api_read
min-tool-calls: 2
max-tool-calls: 12
required_probes:
  - name: charge_read
    match: "/v1/charges"
    description: The charge itself was read, not inferred from the request.
  - name: refunds_checked
    match: "/v1/refunds"
    description: Existing refunds were checked — a second refund is the classic failure.
---

# Establish what was paid

Work from `/v1/charges` and `/v1/payment_intents`, never from the description in
the ticket. Read `/v1/refunds` for the charge before recommending anything: a
charge that is already refunded is the single most expensive thing to miss.
