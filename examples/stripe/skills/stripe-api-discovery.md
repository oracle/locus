---
name: stripe-api-discovery
description: >
  Find the right Stripe API method before calling it. Use at the start of any
  task whose method is not already known — Stripe's surface is large and
  guessing a path wastes a call and can hit the wrong resource entirely.
allowed-tools:
  - stripe_api_search
  - stripe_api_details
  - search_stripe_documentation
min-tool-calls: 1
max-tool-calls: 6
required_probes:
  - name: method_located
    match: "stripe_api_search"
    description: The method was looked up rather than assumed from memory.
---

# Find the method before you call it

`stripe_api_read` and `stripe_api_write` will accept any path you give them.
That is precisely why the path must be established, not recalled: a plausible
wrong path returns a plausible wrong answer.

1. `stripe_api_search` for the resource in the request's own words.
2. `stripe_api_details` on the chosen method, for its required parameters.

Never move to a write without having read the details of the method you intend
to write with.
