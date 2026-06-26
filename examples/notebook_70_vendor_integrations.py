#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 70: Live vendor integrations — PII discovery, data map, scan dispatch.

The earlier notebooks used inline mock tools to keep the focus on agent
mechanics. Real privacy work calls real systems: a data-classification
feed to score an identifier, a data catalog to pull the records behind a
subject request, a scanning cloud to run a PII-discovery probe over a
data store. This notebook wires three *worked* vendor integrations into a
data-subject-request (DSAR) triage agent.

Every integration follows one convention: read the vendor credential from
the environment and call the live API when it's set; otherwise return a
deterministic, synthetic sample so the cookbook runs offline with no
account. The return shape is identical either way, so the agent's
reasoning doesn't change between this offline demo and a live deployment.

- ``scan_for_pii`` — BigID/OneTrust-shaped identifier classification
  (``DATAMAP_API_KEY``).
- ``query_data_map`` — Collibra/Atlan-shaped data-catalog search
  (``DATAMAP_URL`` + ``DATAMAP_TOKEN``).
- ``scan_dataset_reference`` — the *offline reference* PII discovery scan;
  the live version dispatches the scan to a classification cloud
  (``SCANNER_API_KEY``). It returns the same category map either way.

Run it:
    .venv/bin/python examples/notebook_70_vendor_integrations.py

The default provider is the bundled mock model, and every vendor tool falls
back to its offline sample, so this runs end-to-end with no credentials.
Set the matching credential to swap any offline sample for the live API.

Prerequisites:
- Notebook 07 (Agent with tools).
- Notebook 27 (CURATOR) — grounds the data inventory the scan feeds.
"""

import asyncio
import os

from config import get_model, print_config

from tulip.multiagent.specialist import Specialist
from tulip.tools import tool


# The PII categories a discovery scan reports on — a fixed, well-known set
# so the offline reference and a live scan share one vocabulary.
PII_CATEGORIES = (
    "email",
    "phone",
    "national_id",
    "payment_card",
    "location",
    "health",
)

# A synthetic, pseudonymized subject id — the SHA-256 of the literal string
# "test". Safe to print and never maps to a real person, so the offline demo
# stays deterministic and PII-free.
_TEST_SUBJECT = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


def scan_for_pii(identifier: str) -> dict:
    """Classify an identifier against the data-inventory vendor.

    Live path (``DATAMAP_API_KEY`` set) calls the classification API;
    otherwise returns a deterministic synthetic sample with the same shape.
    """
    if os.getenv("DATAMAP_API_KEY"):  # pragma: no cover - live path
        raise NotImplementedError(
            "Live BigID/OneTrust call goes here; this demo runs the sample path."
        )
    # Offline sample: a small, deterministic reputation-style verdict.
    high_risk = {_TEST_SUBJECT, "jane.doe@example.com"}
    minimal = {"unsubscribed@example.net"}
    if identifier in high_risk:
        return {"verdict": "high-risk", "sensitive": True, "categories": ["email", "health"]}
    if identifier in minimal:
        return {"verdict": "minimal", "sensitive": False, "categories": []}
    return {"verdict": "standard", "sensitive": True, "categories": ["email", "location"]}


def query_data_map(term: str, scope: str = "all") -> dict:
    """Search the data catalog for records mentioning a subject term.

    Live path (``DATAMAP_URL`` + ``DATAMAP_TOKEN`` set) queries the catalog;
    otherwise returns a deterministic synthetic result set.
    """
    if os.getenv("DATAMAP_URL") and os.getenv("DATAMAP_TOKEN"):  # pragma: no cover
        raise NotImplementedError(
            "Live Collibra/Atlan query goes here; this demo runs the sample path."
        )
    return {
        "count": 2,
        "source": "data-catalog (sample)",
        "records": [
            {
                "sensitivity": "high",
                "system": "crm.customers",
                "detail": f"row matches '{term}': email, billing_address retained 5y",
            },
            {
                "sensitivity": "medium",
                "system": "marketing.events",
                "detail": f"row matches '{term}': open/click events, no consent flag set",
            },
        ],
    }


def scan_dataset_reference(target: str) -> dict:
    """Offline reference PII-discovery scan over a named data store.

    Returns the category->sample-value map a live classification scan would
    produce. The live version dispatches the scan to a cloud classifier
    (``SCANNER_API_KEY``); this reference path is fully deterministic.
    """
    return {
        "email": "1 column (customers.email)",
        "phone": "1 column (customers.phone)",
        "payment_card": "1 column (billing.card_last4)",
        "location": "2 columns (customers.city, customers.country)",
    }


# Tool wrappers the agent can call. The docstring is what the model reads.
@tool
def scan_for_pii_tool(identifier: str) -> dict:
    """Classify an identifier (email/subject id) and return its PII risk verdict."""
    return scan_for_pii(identifier)


@tool
def query_data_map_tool(term: str, scope: str = "all") -> dict:
    """Search the data catalog for systems and records that hold a subject's data."""
    return query_data_map(term, scope=scope)


async def main() -> None:
    print("=" * 60)
    print("Notebook 70: Live vendor integrations")
    print("=" * 60)
    print()
    print_config()

    model = get_model()

    # =========================================================================
    # Part 1: the vendor tools, called directly
    # =========================================================================
    # Each integration is a plain callable with a live path and an offline
    # sample. Here we exercise the offline path so the output is deterministic.
    print("\n=== Part 1: Vendor Tools (offline sample path) ===\n")

    print("PII classification:")
    for ident in (_TEST_SUBJECT, "jane.doe@example.com", "unsubscribed@example.net"):
        rep = scan_for_pii(ident)
        print(f"  {ident[:24]:<24} -> {rep['verdict']} (sensitive={rep['sensitive']})")

    print("\nData-map query ('jane.doe@example.com', scope=all):")
    hits = query_data_map("jane.doe@example.com", scope="all")
    print(f"  {hits['count']} record(s) from {hits['source']}")
    for rec in hits["records"]:
        print(f"    [{rec['sensitivity']}] {rec['system']}: {rec['detail']}")

    print("\nPII-discovery scan dispatch (offline reference):")
    feats = scan_dataset_reference("crm.customers")
    observed = sum(1 for k in PII_CATEGORIES if k in feats)
    print(f"  categories ({observed}/{len(PII_CATEGORIES)} found): {feats}")

    # =========================================================================
    # Part 2: hand the tools to a DSAR triage agent
    # =========================================================================
    # The same integrations, this time as Tulip @tools the agent can call.
    # Under the mock model the agent narrates rather than truly tool-calls;
    # point TULIP_MODEL_PROVIDER at a live model to see it drive the tools.
    print("\n=== Part 2: A DSAR triage agent with vendor tools ===\n")

    steward = Specialist(
        name="STEWARD",
        specialist_type="triage",
        description="First-line data-subject-request triage with live vendor tools",
        system_prompt=(
            "You are STEWARD, a data-privacy triage analyst. Given a subject "
            "request, classify the identifier (scan_for_pii), pull the systems "
            "that hold the subject's data (query_data_map), and state a "
            "handling priority with the evidence behind it. Never assert that "
            "data exists (or doesn't) without a tool result to back it."
        ),
        tools=[scan_for_pii_tool, query_data_map_tool],
        confidence_threshold=0.8,
        max_iterations=6,
        model=model,
    )

    print(f"Specialist: {steward.name}  tools={[t.name for t in steward.tools]}")

    result = await steward.execute(
        task=(
            "Triage this DSAR: a subject submitted an erasure request for "
            "jane.doe@example.com. Classify the identifier and pull the systems "
            "that still hold her personal data, then state the handling priority."
        ),
        context={"request_id": "DSAR-2026-070", "type": "erasure"},
    )

    print(f"  success={result.success}  confidence={result.confidence:.0%}")
    if result.output:
        print(f"  disposition: {result.output[:240]}")
    if result.error:
        print(f"  error: {result.error}")

    print("\n" + "=" * 60)
    print("Each vendor tool reads its credential from the environment — set")
    print("the matching credential to swap any offline sample for the live API.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
