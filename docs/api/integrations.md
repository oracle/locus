# Integrations

Adapters that bridge Locus to external frameworks, clouds, and vendor APIs.
Cloud posture is one worked example domain — the same adapter pattern
applies to any vendor API.

## FastMCP

Expose a Locus agent (or any of its tools) as a Model Context Protocol
server, and consume MCP tools from any compliant client as native
Locus `Tool`s.

::: locus.integrations.fastmcp.LocusMCPServer
::: locus.integrations.fastmcp.create_mcp_server
::: locus.integrations.fastmcp.mcp_tool_to_locus

## OCI cloud-posture (read-only)

Two generic, spec-driven tools driven by the OCI Python SDK — the agent
discovers the shape of the control plane with `describe_oci` and runs
read-only operations with `use_oci`, whose responses become grounded-finding
evidence. Read-only by default: `use_oci` refuses a mutating operation before
a call is made. Both are core tools, documented under
[Tools](tools.md#oci-tools); the
[`create_soc_analyst`](../concepts/cloud-posture.md) security-operations (SOC)
analyst factory in `locus-sdk-security` composes them into a grounded posture
agent. See the [cloud-posture agent](../concepts/cloud-posture.md) for the
full workflow.

Install the extra: `pip install 'locus-sdk[oci]'`.

## Vendor integration examples

The notebooks ship *worked* vendor integrations in
[`examples/integrations/`][int]. Each is an ordinary Locus `@tool` following
one convention — **bring your own credentials**: read the vendor key from the
environment and call the live API when it's set, otherwise return a
deterministic offline sample so the example runs with no account. The return
shape is identical either way, so the agent's reasoning doesn't change between
the offline demo and a live deployment.

| Tool | Vendor shape | Credential |
|------|--------------|------------|
| [`measure_endpoint_timing`][rt] | Streaming time-to-first-token / cadence probe | none (uses any reachable endpoint) |
| [`dispatch_timing_probe`][gpu] | Offline reference GPU timing-probe dispatch (returns the sample feature vector; a live RunPod / Lambda probe is an integration you write) | none |
| [`enrich_indicator`][ti] | VirusTotal / GreyNoise indicator-of-compromise (IOC) reputation | `VT_API_KEY` |
| [`query_siem`][siem] | Splunk / Elastic log/alert search (a SIEM) | `SIEM_URL`, `SIEM_TOKEN` |

Hand these to a triage agent end-to-end in
[live vendor integrations](../notebooks/notebook_80_vendor_integrations.md);
the GPU probe grounds into a fingerprint finding in
[specialist agents](../notebooks/notebook_23_specialist_agents.md). The
bring-your-own-credentials contract is documented in
[`examples/integrations/README.md`][int].

[int]: https://github.com/oracle/locus/tree/main/packages/locus-sdk-security/examples/integrations
[ti]: https://github.com/oracle/locus/blob/main/packages/locus-sdk-security/examples/integrations/threat_intel.py
[siem]: https://github.com/oracle/locus/blob/main/packages/locus-sdk-security/examples/integrations/siem_query.py
[gpu]: https://github.com/oracle/locus/blob/main/packages/locus-sdk-security/examples/integrations/gpu_probe_dispatch.py
[rt]: https://github.com/oracle/locus/blob/main/packages/locus-sdk-security/examples/integrations/remote_timing.py
