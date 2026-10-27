# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 70: Drive the entire OCI control plane from natural language — `use_oci` + `describe_oci`.

Locus ships two built-in tools that, together, expose every OCI Python
SDK operation to an agent without per-service plumbing:

- ``describe_oci(service?, client?, operation?)`` — runtime
  introspection. Progressively zooms into the SDK: list services
  (~190), list clients in a service, list operations on a client,
  return the parameter schema for one operation (parsed from the
  OCI SDK docstring).
- ``use_oci(service, client, operation, parameters, ...)`` — execute
  any introspected call. Builds the right ``oci.<service>.<Client>``
  with the requested auth, invokes the method, and serialises the
  response (snake_case Python attrs are rendered with their wire
  camelCase names via ``attribute_map``).

The agent's loop is: hand the model the two tools, the model picks
``service`` + ``client`` + ``operation`` from natural-language intent,
optionally calls ``describe_oci`` to verify the parameter shape, then
calls ``use_oci`` to run it. One open-spec primitive covers the
entire OCI surface — Identity, Compute, Database, Object Storage,
Networking, Vault, Functions, Bastion, GenAI, the lot.

Safety: ``use_oci`` is read-only by default. Operations whose names
don't start with a read-only prefix (``list_``, ``get_``, ``head_``,
``summarize_``, ``describe_``, ``search_``, ``fetch_``, ``compute_``,
``preview_``, ``validate_``, ``test_``) are refused unless either
``allow_mutations=True`` is passed explicitly or the env var
``LOCUS_USE_OCI_ALLOW_MUTATIONS=1`` is set.

Key concepts:

- A *service* is an OCI Python SDK submodule under ``oci.*``
  (snake_case: ``identity``, ``core``, ``database``, ``object_storage``).
- A *client* is a class inside the service module (CamelCase, ends in
  ``Client``: ``IdentityClient``, ``ComputeClient``, ``DatabaseClient``).
- An *operation* is a method on the client (snake_case, mirrors the
  REST operation id: ``list_compartments``, ``get_autonomous_database``).
- Auth flows through ``~/.oci/config`` profiles. ``auth_type`` accepts
  ``api_key`` (default), ``security_token`` (session-token profiles
  from ``oci session authenticate``), ``instance_principal`` (running
  on an OCI compute instance with a dynamic group), or
  ``resource_principal`` (running as an OCI Function).

Run it::

    # 1. Pick a tenancy-side profile for the OCI control-plane calls
    export OCI_USE_PROFILE=API_FREE_TIER         # or any other profile in ~/.oci/config
    export OCI_USE_REGION=ca-toronto-1           # whichever region you have resources in
    export OCI_USE_TENANCY=ocid1.tenancy.oc1...  # your tenancy OCID (for compartment-scoped reads)

    # 2. Pick a GenAI-side profile for the model driving the agent
    export OCI_GENAI_PROFILE=LUIGI_FRA_API       # any profile with OCI GenAI access
    export OCI_GENAI_REGION=us-chicago-1

    python examples/notebook_89_oci_tools.py
    python examples/notebook_89_oci_tools.py --part introspection   # just describe_oci
    python examples/notebook_89_oci_tools.py --part execute         # just use_oci
    python examples/notebook_89_oci_tools.py --part agent           # just the agent loop

Difficulty: Intermediate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any


def _env(name: str, default: str | None = None, *, fallbacks: tuple[str, ...] = ()) -> str:
    """Read env var ``name``; fall back to any of ``fallbacks`` if unset.

    Supports the ``OCI_USE_*`` aliases documented in this notebook AND
    the standard ``OCI_PROFILE`` / ``OCI_REGION`` / ``OCI_COMPARTMENT``
    envelope, so users with stock OCI environment variables don't have
    to re-export anything just to run this notebook.
    """
    val = os.environ.get(name)
    if not val:
        for fb in fallbacks:
            val = os.environ.get(fb)
            if val:
                break
    if not val:
        val = default
    if not val:
        tried = [name, *fallbacks]
        sys.stderr.write(
            f"skipped: missing env var (tried {tried}) — see the prerequisites in the notebook "
            "docstring\n"
        )
        # 77 = skipped: scripts/run_notebooks.py reports SKIP, not FAIL.
        sys.exit(77)
    return val


# =============================================================================
# Part 1 — describe_oci: walk the SDK at four levels
# =============================================================================


async def part1_introspection() -> None:
    """Show every level of API discovery, no network calls required."""
    from locus.tools import describe_oci

    print("=== describe_oci — runtime introspection ===\n")

    # Level 1: every service module under oci.* (~190 of them)
    services = json.loads(await describe_oci.execute())
    print(f"Level 1 — services: {services['count']} modules")
    print(f"  sample: {services['services'][:6]} ...\n")

    # Level 2: clients inside a service
    core = json.loads(await describe_oci.execute(service="core"))
    print(f"Level 2 — clients in 'core': {core['clients']}\n")

    # Level 3: operations on one client, partitioned by read/write
    ops = json.loads(
        await describe_oci.execute(service="object_storage", client="ObjectStorageClient")
    )
    print(f"Level 3 — operations on ObjectStorageClient: {ops['count']} total")
    print(f"  read-only (first 5): {ops['read_only_operations'][:5]}")
    print(f"  mutating  (first 5): {ops['mutating_operations'][:5]}\n")

    # Level 4: parameter schema for one operation, parsed from docstring
    schema = json.loads(
        await describe_oci.execute(
            service="identity",
            client="IdentityClient",
            operation="list_compartments",
        )
    )
    print(f"Level 4 — schema for {schema['signature']}")
    print(f"  read_only:        {schema['read_only']}")
    print(f"  required params:  {schema['required_parameters']}")
    print("  all params (truncated):")
    for p in schema["parameters"][:6]:
        flag = "required" if p["required"] else "optional"
        print(f"    - {p['name']:32s} {p['type']:10s} {flag:8s} {p['description'][:60]}")
    print()


# =============================================================================
# Part 2 — use_oci: dispatch any operation, with safety gating
# =============================================================================


async def part2_execute() -> None:
    """Call real OCI services directly through use_oci."""
    from locus.tools import use_oci

    profile = _env("OCI_USE_PROFILE", fallbacks=("OCI_PROFILE",))
    region = _env("OCI_USE_REGION", fallbacks=("OCI_REGION", "OCI_GENAI_REGION"))
    tenancy = _env("OCI_USE_TENANCY", fallbacks=("OCI_COMPARTMENT", "OCI_TENANCY"))

    print(f"=== use_oci — direct dispatch (profile={profile}, region={region}) ===\n")

    # Identity: list every compartment in the tenancy subtree
    result = json.loads(
        await use_oci.execute(
            service="identity",
            client="IdentityClient",
            operation="list_compartments",
            parameters={"compartment_id_in_subtree": True},
            compartment_id=tenancy,
            profile=profile,
            region=region,
            label="list every compartment in the tenancy",
        )
    )
    print(
        f"identity.list_compartments → http {result.get('http_status')} "
        f"opc-request-id={result.get('opc_request_id', '')[:32]}..."
    )
    print(f"  found {len(result.get('data', []))} compartment(s)")
    for c in result.get("data", [])[:3]:
        print(f"    - {c.get('name'):30s} {c.get('lifecycleState')}")
    print()

    # Object Storage: get the tenancy's namespace (an unauth'd-looking call
    # that still uses your signer).
    result = json.loads(
        await use_oci.execute(
            service="object_storage",
            client="ObjectStorageClient",
            operation="get_namespace",
            parameters={},
            profile=profile,
            region=region,
        )
    )
    print(
        f"object_storage.get_namespace → http {result.get('http_status')} "
        f"namespace='{result.get('data')}'\n"
    )

    # Safety gate: a mutating call is refused without the opt-in.
    result = json.loads(
        await use_oci.execute(
            service="object_storage",
            client="ObjectStorageClient",
            operation="delete_bucket",
            parameters={"namespace_name": "nope", "bucket_name": "nope"},
            profile=profile,
            region=region,
        )
    )
    print("Safety: delete_bucket without allow_mutations →")
    print(f"  status='{result['status']}'  error='{result['error'][:90]}...'\n")


# =============================================================================
# Part 3 — Agent loop: NL → describe_oci → use_oci → English answer
# =============================================================================


def _system_prompt(tenancy: str, profile: str) -> str:
    return (
        f"You are an OCI cloud assistant. The user's profile is "
        f"'{profile}' (api_key auth). Their tenancy OCID is:\n"
        f"{tenancy}\n\n"
        f"You have two tools:\n"
        f"1. `describe_oci(service?, client?, operation?)` — introspect the OCI "
        f"Python SDK. Call this first whenever you're unsure which "
        f"service/client/operation/parameters to use. It progressively narrows:\n"
        f"   - no args → list every service module under oci.*\n"
        f"   - service     → list every *Client class in that service\n"
        f"   - service + client → list every operation on that client\n"
        f"   - service + client + operation → return parameter schema\n"
        f"2. `use_oci(service, client, operation, parameters, ...)` — execute "
        f"the call. Always pass profile='{profile}'. Use compartment_id=<tenancy "
        f"OCID> for tenancy-scoped reads.\n\n"
        f"OCI Python SDK conventions:\n"
        f"- service = snake_case module name ('identity', 'core', 'database')\n"
        f"- client = CamelCase class ('IdentityClient', 'ComputeClient')\n"
        f"- operation = snake_case method ('list_compartments', 'get_instance')\n\n"
        f"Discover before executing. Answer in plain English."
    )


async def part3_agent() -> None:
    """Watch the agent self-discover the SDK and answer free-form questions."""
    from locus.agent import Agent
    from locus.models import get_model
    from locus.tools import describe_oci, use_oci

    use_profile = _env("OCI_USE_PROFILE", fallbacks=("OCI_PROFILE",))
    use_region = _env("OCI_USE_REGION", fallbacks=("OCI_REGION", "OCI_GENAI_REGION"))
    tenancy = _env("OCI_USE_TENANCY", fallbacks=("OCI_COMPARTMENT", "OCI_TENANCY"))
    genai_profile = _env("OCI_GENAI_PROFILE", fallbacks=("OCI_PROFILE",))
    genai_region = _env("OCI_GENAI_REGION", "us-chicago-1", fallbacks=("OCI_REGION",))

    print(
        f"=== Agent loop (model via {genai_profile}@{genai_region}, "
        f"OCI calls via {use_profile}@{use_region}) ===\n"
    )

    model = get_model("oci:openai.gpt-4o", profile=genai_profile, region=genai_region)
    agent = Agent(
        model=model,
        tools=[describe_oci, use_oci],
        system_prompt=_system_prompt(tenancy, use_profile),
        max_iterations=12,
    )

    prompts = [
        "What regions am I subscribed to?",
        "Are there any compute instances running anywhere in my tenancy?",
    ]

    for prompt in prompts:
        print(f"USER: {prompt}")
        result = agent.run_sync(prompt)
        print(f"AGENT: {result.message.strip()}")
        # Show the call sequence the model picked — proof it self-discovered.
        calls: list[str] = []
        for msg in result.state.messages:
            for tc in msg.tool_calls or []:
                args: Any = tc.arguments if isinstance(tc.arguments, dict) else {}
                if tc.name == "use_oci":
                    calls.append(
                        f"  -> use_oci({args.get('service')}.{args.get('client')}."
                        f"{args.get('operation')}, region={args.get('region')})"
                    )
                elif tc.name == "describe_oci":
                    bits = [
                        f"{k}={args.get(k)}"
                        for k in ("service", "client", "operation")
                        if args.get(k)
                    ]
                    calls.append(f"  -> describe_oci({', '.join(bits)})")
        if calls:
            print("Tool-call sequence:")
            print("\n".join(calls))
        print()


# =============================================================================
# Why this is a great primitive
# =============================================================================
#
# Adding a new OCI resource type to your agent normally means:
#
#   1. Read the OCI Python SDK docs.
#   2. Write a thin @tool wrapper around the right ``oci.<service>.<Client>``
#      method, with the right ``compartment_id`` plumbing, the right
#      paginator, the right serialiser.
#   3. Register that tool with the agent.
#   4. Repeat for every operation you want to expose (~190 services x N ops).
#
# With ``use_oci`` + ``describe_oci``:
#
#   1. Hand the model the two tools. Done.
#
# The model uses ``describe_oci`` like a man page — discovers what
# operations exist, what parameters they need — and then calls
# ``use_oci`` to execute. There is exactly one execution path, with one
# auth code-path, one serialiser, one mutation-safety gate. Adding
# coverage for a new service does not require writing any new code.
# =============================================================================


async def main(part: str) -> None:
    if part in {"all", "introspection"}:
        await part1_introspection()
    if part in {"all", "execute"}:
        await part2_execute()
    if part in {"all", "agent"}:
        await part3_agent()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        choices=["all", "introspection", "execute", "agent"],
        default="all",
    )
    asyncio.run(main(parser.parse_args().part))
