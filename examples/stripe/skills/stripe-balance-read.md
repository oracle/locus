---
name: stripe-balance-read
description: >
  Establish whether the account can actually absorb a refund or payout —
  balance, balance transactions and pending payouts. Use before approving money
  movement of material size.
allowed-tools:
  - stripe_api_read
min-tool-calls: 1
max-tool-calls: 8
required_probes:
  - name: balance_read
    match: "/v1/balance"
    description: The available balance, before promising money out of it.
---

# Establish the balance

A refund that overdraws the balance becomes a negative-balance event, not a
refund. Read `/v1/balance` and, when the amount is material, `/v1/payouts` for
what is already committed.
