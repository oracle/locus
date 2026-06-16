# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Locus Agent — Quick Start.

Uses ``OCIChatCompletionsModel`` against OCI GenAI's OpenAI-compatible
``/openai/v1`` endpoint — real SSE streaming, day-0 model support, no
GenAI Project OCID required. The compartment is auto-derived from the
profile's tenancy.

Set ``OCI_PROFILE`` to the OCI config profile you want to use (defaults
to ``DEFAULT``). For Cohere R-series, use ``OCIModel`` instead — see
``docs/how-to/oci-models.md``.
"""

import os

from locus.agent import Agent
from locus.models import OCIChatCompletionsModel


def main():
    model = OCIChatCompletionsModel(
        model="openai.gpt-5",
        profile=os.environ.get("OCI_PROFILE", "DEFAULT"),
    )

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Be concise.",
    )

    result = agent.run_sync("What is the capital of France?")
    print(f"Response: {result.message}")
    print(f"Iterations: {result.metrics.iterations}")


if __name__ == "__main__":
    main()
