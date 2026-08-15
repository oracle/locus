---
name: stripe-disputes-read
description: >
  Read a dispute and its history: the reason code, the evidence deadline, and
  whether this customer has disputed before. Use whenever a dispute is the
  subject rather than a plain refund request.
allowed-tools:
  - stripe_api_read
min-tool-calls: 2
max-tool-calls: 10
required_probes:
  - name: dispute_read
    match: "/v1/disputes"
    description: The dispute itself, for its reason code and deadline.
  - name: dispute_history
    match: "limit"
    description: More than one dispute was looked at — a pattern, not an incident.
---

# Read the dispute

The reason code decides which evidence is worth submitting; the deadline decides
whether any of it matters. Read `/v1/disputes` for both, then list the
customer's prior disputes.

Never write here. Gathering and deciding are separate on purpose.
