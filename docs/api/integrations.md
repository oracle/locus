# Integrations

Adapters that bridge the SDK to external frameworks, clouds, and vendor APIs.

## FastMCP

Expose an agent (or any of its tools) as a Model Context Protocol
server, and consume MCP tools from any compliant client as native
`Tool`s.

::: locus.integrations.fastmcp.LocusMCPServer
::: locus.integrations.fastmcp.create_mcp_server
::: locus.integrations.fastmcp.mcp_tool_to_locus

## OCI cloud-posture (read-only)

Two generic, spec-driven tools driven by the OCI Python SDK — the agent
discovers the shape of the control plane with `describe_oci` and runs
read-only operations with `use_oci`, whose responses become grounded-finding
evidence. Read-only by default: `use_oci` refuses a mutating operation before
a call is made. Both are core tools, documented under
[Tools](tools.md#oci-tools); [notebook 36](../notebooks/notebook_36_oci_tools.md)
runs them inside an agent.

Install the extra: `pip install 'locus-sdk[oci]'`.
