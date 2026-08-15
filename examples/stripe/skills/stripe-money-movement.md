---
name: stripe-money-movement
description: >
  Move money that a person has approved — a refund for the exact charge and
  amount they saw. Use only as the final act of a procedure, never to explore.
allowed-tools:
  - create_refund
min-tool-calls: 1
max-tool-calls: 1
required_probes:
  - name: bound_to_charge
    match: "ch_"
    description: The refund names the charge that was established earlier.
---

# Move the money, once

One call. `create_refund` rather than `stripe_api_write`: the specific tool
names the consequence, and a procedure that reaches for the generic writer to
issue a refund has given itself the whole write surface to do one thing.

If the amount or the charge has changed since a person approved, stop and say
so. A refund nobody authorised is worse than a refund not issued.
