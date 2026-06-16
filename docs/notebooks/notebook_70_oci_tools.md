# Notebook 70: Drive the entire OCI control plane from natural language — `use_oci` + `describe_oci`

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

    python examples/notebook_70_oci_tools.py
    python examples/notebook_70_oci_tools.py --part introspection   # just describe_oci
    python examples/notebook_70_oci_tools.py --part execute         # just use_oci
    python examples/notebook_70_oci_tools.py --part agent           # just the agent loop

Difficulty: Intermediate

## Source

```python
--8<-- "examples/notebook_70_oci_tools.py"
```
