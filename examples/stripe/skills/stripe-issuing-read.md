---
name: stripe-issuing-read
description: >
  Read Issuing authorizations, cards, cardholders, transactions and disputes.
  Use when the subject is a card you issued rather than a payment you received.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 10
required_probes:
  - name: issuing_read
    match: "/v1/issuing"
    description: The Issuing surface — a different object graph entirely.
---

# Establish the card's side of it

An Issuing authorization is not a charge and its dispute is not a `/v1/disputes`
dispute. Read the authorization and the transaction before reasoning about a
decline: most "the card was refused" questions are answered by the
authorization's own reason.
