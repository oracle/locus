# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Tool abuse: SSRF and path traversal via model-supplied arguments.

Threat: an agent with a fetch-or-read tool is steered to reach the cloud
metadata service (http://169.254.169.254/ → instance credentials) or to
escape its workspace (../../etc/passwd). The model is the confused deputy;
the tool is the weapon.

Defense (built-in SDK primitives): tulip.tools.url_safety.is_safe_url
rejects loopback/link-local/private/metadata targets; tulip.tools.
path_safety.safe_resolve confines a path to an allowed root.

Taxonomy: OWASP ASI02 (Tool Misuse) · OWASP LLM06 (Excessive Agency).
"""

from __future__ import annotations

from pathlib import Path

from tulip.core.errors import ValidationError
from tulip.tools.path_safety import safe_resolve
from tulip.tools.url_safety import is_safe_url


def main() -> None:
    print("Scenario: tool abuse — SSRF + path traversal  [ASI02 · LLM06]\n")

    print("SSRF guard (url_safety.is_safe_url):")
    urls = [
        "https://docs.example.com/runbook.md",  # benign, public
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",  # AWS IMDS
        "http://localhost:6379/",  # internal service
        "http://metadata.google.internal/computeMetadata/v1/",  # GCP metadata
    ]
    for url in urls:
        verdict = "ALLOW" if is_safe_url(url) else "BLOCK"
        print(f"  [{verdict}] {url}")

    print("\nPath-traversal guard (path_safety.safe_resolve):")
    workspace = Path("/srv/workspace")
    for rel in ("cases/IR-2026-001.md", "../../etc/passwd", "/etc/shadow"):
        try:
            resolved = safe_resolve(workspace, rel)
            print(f"  [ALLOW] {rel} -> {resolved}")
        except ValidationError as exc:
            print(f"  [BLOCK] {rel} -> {exc}")

    print("\nThe agent can ask for anything; the guard decides what the tool may reach.")


if __name__ == "__main__":
    main()
