---
name: stripe-promotions-read
description: >
  Read coupons and promotion codes to establish whether a discount explains a
  disputed amount. Use when the customer says they were charged more than they
  expected.
allowed-tools: [stripe_api_read]
min-tool-calls: 1
max-tool-calls: 8
required_probes:
  - name: discount_read
    match: "/v1/coupons"
    description: The coupon, or its absence, which is equally an answer.
---

# Establish the discount

"I had a code" is checkable. Read `/v1/coupons` and `/v1/promotion_codes` and
report which applied — including when none did, because that is usually the
whole dispute.
