---
name: stripe-catalog-read
description: >
  Read products and prices to establish what something costs and in which
  currency. Use whenever an amount has to be explained rather than merely
  reported.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 10
required_probes:
  - name: product_read
    match: "/v1/products"
    description: What the thing IS, before what it costs.
  - name: price_read
    match: "/v1/prices"
    description: The price object — currency and interval, not just a number.
---

# Establish the catalogue

An amount without its price object cannot be explained: the same 1200 is £12.00
or ¥1200 depending on `currency`, and monthly or annual depending on
`recurring.interval`.
