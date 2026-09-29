# Provenance

This file records where the code in this repository comes from. It
complements `NOTICE` and the `LICENSES/` directory.

## Licensing history

- The project's first commit is dated 2026-06-08.
- Releases before 1.0.0 (the 0.x line) were published by The Tulip Authors
  under the Universal Permissive License v1.0 (UPL-1.0).
- The project relicensed itself to the Apache License, Version 2.0 at 1.0.0
  (2026-06-10). Code as it stood in the 0.x releases remains available under
  the UPL-1.0; all changes and additions since 1.0.0 are Apache-2.0.
- `LICENSES/UPL-1.0-FILES.txt` lists the files that date from the UPL-1.0
  releases.

## Research artifacts

- The research pages on the documentation site publish their method and
  scoring code, which are runnable against your own endpoint.
- The full held-out corpus for the policy-blindness study and the evaluated
  model (Clusiana-Admit-4B) are not redistributed.
- The evaluations, and the training of Clusiana-Admit-4B, ran on the authors'
  own accounts, API keys and rented GPUs.

## Security tooling

Offensive-capable tooling (red-team agents, model and hardware
fingerprinting) ships only in the separate, opt-in `tulip-agents-security`
package under `packages/tulip-agents-security/`, not in the core
`tulip-agents` distribution.
