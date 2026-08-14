---
name: stripe-customers-read
description: >
  Establish who the customer is and what they already have — their record,
  their payment methods and their standing. Use when a request names a person
  or company rather than a specific charge.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 10
required_probes:
  - name: customer_read
    match: "/v1/customers"
    description: The customer record itself, not the name in the ticket.
---

# Establish the customer

A name in a support ticket is not an identity. Resolve to `cus_...` via
`/v1/customers` with a search or an email filter, and say so when more than one
matches rather than picking the first.
