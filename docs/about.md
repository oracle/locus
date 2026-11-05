---
title: About the SDK — maintainers, history, and project health
description: Who maintains the Oracle Generative AI – Multi-Agent Locus SDK, where it came from, and how releases, security reports, and contributions are managed.
---

# About the Oracle Generative AI – Multi-Agent Locus SDK

The SDK is an agentic harness: one `Agent` class, eight
multi-agent shapes, and the runtime around the model — tools, memory, RAG,
observability, guardrails and an admission gate with a signed audit trail —
built first-class for **OCI Generative AI** (including Dedicated AI Clusters)
and **Oracle Database 26ai**, and open to every other provider. It is
built inside Oracle, used in production, and open source under the
[Universal Permissive License](legal.md). Maintainers and contributors are
visible in the repository's
[Git history](https://github.com/oracle/locus/graphs/contributors).

## Project history

The project began at Oracle as `locus-sdk` (the `0.x` line, UPL-1.0). From June to
October 2026 development continued outside Oracle as `tulip-agents` under the
tuliplabs-ai GitHub organisation (`1.0.0` – `2.28.0`, Apache-2.0), where the
admission gate, the control layer, playbooks, the coding harness and most of
today's runtime took shape. With `3.0.0` (October 2026) the project returned
to [oracle/locus](https://github.com/oracle/locus) with its full history, the
Oracle Cloud and Oracle Database integrations restored, and `tulip-agents`
retired. See [Legal notices](legal.md#provenance) for the licensing detail.

## Assessing maintenance

| Signal | Where to check |
|---|---|
| Releases and changes | [Changelog](https://github.com/oracle/locus/blob/main/CHANGELOG.md) and [PyPI](https://pypi.org/project/locus-sdk/) |
| Supported Python and dependency ranges | [Compatibility policy](compatibility.md) and package metadata |
| Open work and decisions | [Issues](https://github.com/oracle/locus/issues) |
| Contribution process | [Contributing guide](https://github.com/oracle/locus/blob/main/CONTRIBUTING.md) — contributions need the [Oracle Contributor Agreement](https://oca.opensource.oracle.com) |
| Security reporting | [Security policy](https://github.com/oracle/locus/blob/main/SECURITY.md) |
| License and provenance | [Legal notices](legal.md) |

The documentation uses status and execution-mode labels where an example could
otherwise be mistaken for a live or generally available integration.
