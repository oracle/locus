---
name: stripe-treasury-read
description: >
  Read the v2 money-management surface — financial accounts, transactions,
  received credits and debits, outbound payments and transfers. Use for
  questions about money that has moved between accounts rather than a charge.
allowed-tools: [stripe_api_read, get_balance_summary]
min-tool-calls: 1
max-tool-calls: 12
required_probes:
  - name: account_read
    match: "/v2/core/accounts"
    description: Which account the money belongs to — on a platform, rarely yours.
  - name: money_movement_read
    match: "/v2/money_management"
    description: The v2 surface, which is where transfers and credits live.
---

# Establish where the money went

`/v1/balance` is what you hold; `/v2/money_management` is what moved. A question
about a missing payout is answered in the second, and looking only at the first
reports a balance without explaining it.
