# Provenance

This file records where the code in this repository comes from. It
complements `NOTICE` and the `LICENSES/` directory.

## Licensing history

- The project started at Oracle in 2025 under the Universal Permissive License
  v1.0 (UPL-1.0): the `0.x` line, published to PyPI as `locus-sdk`, with
  copyright held by Oracle and/or its affiliates.
- From 2026-06-08 to 2026-10-10 the project continued outside Oracle as
  `tulip-agents` (github.com/tuliplabs-ai/tulip-agents). Those releases —
  `1.0.0` through `2.28.0` — were published under the Apache License,
  Version 2.0, by the same maintainer. `tulip-agents` is retired and
  receives no further releases.
- With `3.0.0` (2026-10-22) the development line returned to
  github.com/oracle/locus with its full commit history. Everything in this
  repository from `3.0.0` on is licensed under the UPL-1.0 with Oracle
  copyright; the relicensing was made by the copyright holder of the
  intermediate releases. The Oracle Cloud Infrastructure and Oracle Database
  integrations removed in June 2026 were restored in the same release.
- Third-party code adapted into the tree keeps its own notice: the
  holographic-memory encoding and the memory-scrubbing approach in
  `src/locus/memory/` are adapted from NousResearch/hermes-agent (MIT) and
  say so in their headers; `THIRD_PARTY_LICENSES.txt` lists the rest.

## Research artifacts

- The research pages on the documentation site publish their method and
  scoring code, which are runnable against your own endpoint.
- The full held-out corpus for the policy-blindness study and the evaluated
  model (Clusiana-Admit-4B) are not redistributed.
- The evaluations, and the training of Clusiana-Admit-4B, ran on the authors'
  own accounts, API keys and rented GPUs.

## Security tooling

Offensive-capable tooling (red-team agents, model and hardware
fingerprinting) ships only in the separate, opt-in `locus-sdk-security`
package under `packages/locus-sdk-security/`, not in the core
`locus-sdk` distribution.
