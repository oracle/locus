---
name: cert-expiry-check
description: Find certificates close to expiry.
allowed-tools: query_metrics
required-probes:
  - name: expiry
    match: "tls_cert_not_after_seconds"
---

# Certificate expiry

Query certificate expiry per endpoint and compare it with the renewal window.
