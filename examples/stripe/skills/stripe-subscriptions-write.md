---
name: stripe-subscriptions-write
description: >
  Create, update or cancel a subscription once a person has approved it. Use
  only after the current state has been read — recurring revenue changed on an
  assumption is the expensive kind of mistake.
allowed-tools: [stripe_api_write]
min-tool-calls: 1
max-tool-calls: 2
required_probes:
  - name: subscription_targeted
    match: "/v1/subscriptions"
    description: The write names a subscription path.
---

# Change the subscription a person approved

Cancelling at period end and cancelling immediately are different products to
the customer. Use the one that was approved, name which it was, and never
retry a cancellation whose outcome you have not read.
