# Legal notices

## About &amp; disclaimer

**Locus is built and maintained by Oracle.** The project is developed in the
open under the Universal Permissive License and does not claim endorsement by
any third-party model provider or integration vendor. Its licensing history is
described under [Provenance](#provenance).

### Disclaimer of warranty and liability

The software is provided **"AS IS", without warranty of any kind**, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. **The authors and copyright holders accept no liability** for any claim, damages, or other consequence arising from the software or its use. **Use is entirely at your own risk.** You are **solely responsible** for ensuring that your use is lawful, authorized, and compliant with all applicable obligations — including security, privacy, export-control, and any policies that govern the systems and data you touch.

### Responsible use

The core `locus-sdk` package does not include offensive-capable tooling. The separate, opt-in `locus-sdk-security` package does — red-team agents, model and hardware fingerprinting, and similar reconnaissance capabilities. **Use it only against systems you own or are explicitly authorized to test.** Running these capabilities against systems you do not own or have not been authorized to assess may be unlawful.

*This page is not legal advice. Consult your own counsel.*

## License

Locus is released under the **Universal Permissive License (UPL), Version 1.0**.

Copyright (c) 2025, 2026 Oracle and/or its affiliates.

Permission is hereby granted, free of charge and under any and all copyright rights in the Software, and any and all patent rights owned or freely licensable by each licensor, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and either this complete permission notice or at a minimum a reference to the UPL must be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

Full text: [LICENSE](https://github.com/oracle/locus/blob/main/LICENSE)

---

### Provenance

Locus started at Oracle under the **Universal Permissive License v1.0 (UPL-1.0)** (the `0.x` line, `locus-sdk` on PyPI). From June to October 2026 the project continued outside Oracle as `tulip-agents` (releases `1.0.0` – `2.28.0`, Apache License 2.0). With `3.0.0` the project returned to [oracle/locus](https://github.com/oracle/locus), under Oracle copyright and the UPL-1.0. The intermediate Apache-2.0 releases remain available under that licence from their tags; everything in this repository from `3.0.0` on is UPL-1.0. [`PROVENANCE.md`](https://github.com/oracle/locus/blob/main/PROVENANCE.md) records the detail.

Full text: [LICENSE](https://github.com/oracle/locus/blob/main/LICENSE) · [THIRD_PARTY_LICENSES.txt](https://github.com/oracle/locus/blob/main/THIRD_PARTY_LICENSES.txt)

---

## Third-party components

locus uses the following open-source libraries. Their licenses are reproduced below and in full in [`THIRD_PARTY_LICENSES.txt`](https://github.com/oracle/locus/blob/main/THIRD_PARTY_LICENSES.txt).

### Core dependencies

| Package | License | Copyright |
|---|---|---|
| `httpx` | BSD-3-Clause | © 2019 Encode OSS Ltd. |
| `pydantic` | MIT | © Pydantic Services Inc. and contributors |
| `pydantic-settings` | MIT | © 2022 Samuel Colvin and contributors |
| `typing-extensions` | PSF | © Python Software Foundation |

### Model providers (optional)

| Package | License | Copyright |
|---|---|---|
| `openai` | MIT | © OpenAI |
| `anthropic` | MIT | © Anthropic |
| `cohere` | MIT | © Cohere |
| `oci` | UPL-1.0 OR Apache-2.0 | © Oracle and/or its affiliates |

### Observability (optional)

| Package | License | Copyright |
|---|---|---|
| `opentelemetry-api` | Apache-2.0 | © The OpenTelemetry Authors |
| `opentelemetry-sdk` | Apache-2.0 | © The OpenTelemetry Authors |
| `opentelemetry-exporter-otlp` | Apache-2.0 | © The OpenTelemetry Authors |

### MCP integration (optional)

| Package | License | Copyright |
|---|---|---|
| `mcp` | MIT | © Anthropic, PBC |

### Checkpoint / memory backends (optional)

| Package | License | Copyright |
|---|---|---|
| `redis` | MIT | © Redis Inc. |
| `asyncpg` | Apache-2.0 | © MagicStack Inc. |
| `opensearch-py` | Apache-2.0 | © OpenSearch Contributors |
| `oracledb` | UPL-1.0 OR Apache-2.0 | © Oracle and/or its affiliates |
| `oracleagentmemory` | UPL-1.0 | © Oracle and/or its affiliates |
