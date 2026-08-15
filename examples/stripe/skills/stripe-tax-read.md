---
name: stripe-tax-read
description: >
  Read tax settings, codes and registrations to establish why tax was or was
  not charged. Use before refunding tax, which is rarely the same decision as
  refunding the goods.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 8
required_probes:
  - name: tax_read
    match: "/v1/tax"
    description: Settings or registrations — the reason a rate applied.
---

# Establish the tax position

Refunding a charge does not automatically make its tax recoverable, and the
registration is what decides. Read it before recommending a full refund on a
taxed charge in a jurisdiction where you are registered.
