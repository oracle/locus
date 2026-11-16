# Integrations

Adapters that bridge the SDK to external frameworks, clouds, and vendor APIs.

## FastMCP

Expose an agent (or any of its tools) as a Model Context Protocol
server, and consume MCP tools from any compliant client as native
`Tool`s.

::: locus.integrations.fastmcp.LocusMCPServer
::: locus.integrations.fastmcp.create_mcp_server
::: locus.integrations.fastmcp.create_tool_server
::: locus.integrations.fastmcp.mcp_tool_to_locus

### Signing in to a remote server

`MCPClient(oauth=MCPOAuth(...))` runs the OAuth 2.1 authorization-code
flow (PKCE, discovery, refresh) against a Streamable HTTP server that asks
for it; the token store is pluggable. See
[MCP — signing in](../concepts/mcp.md#signing-in-to-a-remote-server-oauth-21).

::: locus.integrations.mcp_auth.MCPOAuth
::: locus.integrations.mcp_auth.MCPTokenStore
::: locus.integrations.mcp_auth.InMemoryTokenStore
::: locus.integrations.mcp_auth.FileTokenStore
::: locus.integrations.mcp_auth.KeyringTokenStore

### A server's questions

::: locus.integrations.fastmcp.MCPElicitation
::: locus.integrations.fastmcp.ElicitationAnswer

### Trace context in `_meta`

::: locus.integrations.mcp_tracing.trace_meta
::: locus.integrations.mcp_tracing.trace_scope

## OCI cloud-posture (read-only)

Two generic, spec-driven tools driven by the OCI Python SDK — the agent
discovers the shape of the control plane with `describe_oci` and runs
read-only operations with `use_oci`, whose responses become grounded-finding
evidence. Read-only by default: `use_oci` refuses a mutating operation before
a call is made. Both are core tools, documented under
[Tools](tools.md#oci-tools); [notebook 36](../notebooks/notebook_36_oci_tools.md)
runs them inside an agent.

Install the extra: `pip install 'locus-sdk[oci]'`.
