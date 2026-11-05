# OCI tools

The SDK ships two built-in tools that together expose every operation of
the Oracle Cloud Infrastructure (OCI) Python SDK to an agent without a
wrapper per service. `describe_oci(service?, client?, operation?)` is
the reference: it narrows progressively through the installed SDK,
listing the service modules, the clients in one service, the operations
on one client and the parameter schema of one operation, parsed from the
SDK docstring. `use_oci(service, client, operation, parameters, ...)` is
the single execution path: it builds the requested
`oci.<service>.<Client>` with the requested authentication, invokes the
method and serialises the response, rendering snake_case Python
attributes under their camelCase wire names through the SDK's
`attribute_map`.

Every call names a *service* (a snake_case submodule under `oci.*`, such
as `identity`, `core` or `object_storage`), a *client* (a class in that
module whose name ends in `Client`, such as `IdentityClient` or
`ComputeClient`) and an *operation* (a snake_case method that mirrors
the REST operation id, such as `list_compartments` or
`get_autonomous_database`). The agent loop hands the model both tools:
from natural-language intent the model picks the three names, calls
`describe_oci` when it is unsure of the parameter shape, then calls
`use_oci` to run it. One generic tool pair reaches Identity, Compute,
Database, Object Storage, Networking, Vault, Functions, Bastion,
Generative AI and every other service in the installed SDK.

`use_oci` is read-only by default. An operation whose name does not
start with a read-only prefix (`list_`, `get_`, `head_`, `summarize_`,
`describe_`, `search_`, `fetch_`, `compute_`, `preview_`, `validate_`,
`test_`) is refused before any client is built, unless the call passes
`allow_mutations=True` or the environment variable
`LOCUS_USE_OCI_ALLOW_MUTATIONS=1` is set.

What this notebook covers:

- `describe_oci` walking the installed SDK at four levels without a
  network call
- `use_oci` running two read-only calls, on Identity and Object Storage,
  through a `~/.oci/config` profile
- the mutation gate refusing `delete_bucket` when `allow_mutations` is
  not set
- an `Agent` holding both tools that discovers the right call for a
  free-form question, and the tool-call sequence it chose

## Prerequisites

- The `oci` SDK extra (`pip install 'locus-sdk[oci]'`, or
  `pip install -e ".[oci]"` from a checkout). Part 1 introspects the
  installed SDK, so even the offline part needs it.
- For Parts 2 and 3, a profile in `~/.oci/config` with permission to
  read the tenancy, and the tenancy's Oracle Cloud Identifier (OCID).
- For Part 3, a profile with OCI Generative AI access and a region where
  the service is offered.

```bash
# Tenancy-side profile for the OCI control-plane calls
export OCI_USE_PROFILE=<your-profile>          # any profile in ~/.oci/config
export OCI_USE_REGION=us-ashburn-1             # whichever region holds your resources
export OCI_USE_TENANCY=ocid1.tenancy.oc1...    # tenancy OCID for compartment-scoped reads

# GenAI-side profile for the model driving the agent
export OCI_GENAI_PROFILE=<your-genai-profile>  # any profile with OCI Generative AI access
export OCI_GENAI_REGION=us-chicago-1
```

Each variable falls back to the OCI CLI's standard names:
`OCI_USE_PROFILE` to `OCI_PROFILE`; `OCI_USE_REGION` to `OCI_REGION`,
then `OCI_GENAI_REGION`; `OCI_USE_TENANCY` to `OCI_COMPARTMENT`, then
`OCI_TENANCY`; `OCI_GENAI_PROFILE` to `OCI_PROFILE`; and
`OCI_GENAI_REGION` to `OCI_REGION`, then to `us-chicago-1`. An
environment with only `OCI_PROFILE`, `OCI_REGION` and `OCI_COMPARTMENT`
exported therefore runs every part against a single profile and region.
The compartment used for the Generative AI calls is `OCI_COMPARTMENT`
(or `OCI_COMPARTMENT_ID`) when set, otherwise the tenancy of the
Generative AI profile.

Part 2 calls `use_oci` with `auth_type` at its default, `api_key`, and
the system prompt in Part 3 tells the model the same, so the
control-plane profile needs an API key. The model built from the
Generative AI profile signs with that profile's API key or, when the
profile carries a `security_token_file`, with its session token.
`use_oci` itself also accepts `security_token` (a session profile
created by `oci session authenticate`), `instance_principal` (a compute
instance in a dynamic group) and `resource_principal` (an OCI Function);
the last two need no configuration file.

## Key lines

`describe_oci` walks the installed SDK at four levels — the service
modules under `oci.*`, the clients in one service, the operations on one
client partitioned into read-only and mutating, and the parameter schema
of one operation — without a network call; every level is `importlib`,
`pkgutil` and `inspect` over the installed package:

{{ locus_source examples/notebook_36_oci_tools.py::part1_introspection }}

`use_oci` dispatches two read-only calls through the profile from
Prerequisites. For `identity.IdentityClient.list_compartments` the
tenancy OCID travels as `compartment_id`, which `use_oci` merges into
`parameters` because they do not already carry one;
`object_storage.ObjectStorageClient.get_namespace` needs none.
The third call asks for `delete_bucket` without `allow_mutations`, and
the gate returns `status='error'` before any client is built:

{{ locus_source examples/notebook_36_oci_tools.py::part2_execute }}

The agent is `get_model` on `oci:openai.gpt-4o` under the Generative AI
profile, both tools and a system prompt that tells the model to discover
before executing, capped at twelve iterations; `run_sync` answers two
free-form questions, and the tool-call sequence printed afterwards is
read from `result.state.messages`, showing which `describe_oci` and
`use_oci` calls the model chose:

{{ locus_source examples/notebook_36_oci_tools.py::part3_agent }}

## Run it

```bash
python examples/notebook_36_oci_tools.py
```

The script builds its model with `locus.models.get_model` on the
`oci:openai.gpt-4o` id and does not use the shared `examples/config.py`
harness, so there is no mock fallback and `LOCUS_MODEL_PROVIDER` is not
read. Part 1 only introspects the installed OCI Python SDK and runs
offline. Parts 2 and 3 read the variables from Prerequisites; the first
one missing, after its fallbacks, makes the script write a skip notice
to stderr and exit with status 77, which `scripts/run_notebooks.py`
reports as a skip rather than a failure, so an offline run completes
Part 1 and stops before the first network call.

The `--part` flag runs one part on its own:

```bash
python examples/notebook_36_oci_tools.py --part introspection   # describe_oci only
python examples/notebook_36_oci_tools.py --part execute         # use_oci only
python examples/notebook_36_oci_tools.py --part agent           # the agent loop only
```

## See also

- [Notebook 81 — grounded cloud-posture agent](notebook_81_cloud_posture_agent.md) — a read-only auditing agent built on the same two tools
- [Concepts: Cloud posture](../concepts/cloud-posture.md) — the read-only gate and the identity backstop behind it
- [API: built-in OCI tools](../api/tools.md#oci-tools) — the `use_oci` and `describe_oci` signatures
- [OCI Python SDK — Oracle docs](https://docs.oracle.com/en-us/iaas/tools/python/latest/) — the services, clients and operations the tools introspect

## Source

??? note "Full source — `examples/notebook_36_oci_tools.py`"

    ````python
    --8<-- "examples/notebook_36_oci_tools.py"
    ````
