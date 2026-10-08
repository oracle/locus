# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The run manifest: everything a box runner needs to run one agent, pinned.

The gateway compiles a :class:`RunManifest` from the agent's pinned definition
and hands it to the runner in the box. The runner builds its agent from the
manifest and nothing else, and the manifest's :meth:`~RunManifest.digest` is
recorded on the run, so what ran can be checked against what was approved.

What the manifest deliberately does **not** carry:

- **Labels the runner could use to judge a call.** Each tool's labels are there
  so the runner can describe its tools truthfully, but admission never trusts
  them: :class:`~tulip.runner.gate.RemoteGate` sends only a call's name and
  arguments, and the gateway weighs the call against its own copy of the
  surface.
- **Secrets.** The model's key is named only by the environment variable that
  holds its OpenShell placeholder (:attr:`ModelRoute.api_key_env`); the
  sandbox's proxy swaps the placeholder for the key on the model's host and
  nowhere else.

The digest is ``sha256:`` over canonical JSON: keys sorted, no whitespace,
UTF-8. Two manifests with the same content have the same digest whatever
order their fields were written in.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


__all__ = [
    "DEFAULT_PARK_TTL_S",
    "MANIFEST_VERSION",
    "ApiConnector",
    "ApiOperation",
    "Budgets",
    "McpMount",
    "ModelRoute",
    "RunManifest",
    "ToolEntry",
    "canonical_digest",
]

#: The manifest format this SDK reads and writes.
MANIFEST_VERSION: Literal[1] = 1

#: Where a tool runs: in the box, on the gateway, on a mounted MCP server, or
#: as one operation of an API connector.
_RUNS = re.compile(r"^(box|gateway|(mcp|api):[A-Za-z0-9][A-Za-z0-9._-]{0,127})$")

#: Seconds a runner waits on a held call before it parks the run.
DEFAULT_PARK_TTL_S = 600.0

#: A ``{name}`` placeholder in an operation's path.
_PATH_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def canonical_digest(value: Any) -> str:
    """``sha256:<hex>`` of ``value`` as canonical JSON (sorted keys, no spaces)."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiOperation(_Strict):
    """How one ``api:<id>`` tool's arguments become one HTTP request.

    The mapping is fixed, so the gateway can compute the exact request a call
    will make when it admits it (and bind the call's decision token to it) and
    the box guard can check the request on the wire against that token.
    :func:`tulip.runner.connectors.api_request` is the one implementation both
    sides use.

    - ``path`` may name arguments as ``{name}``; each is URL-quoted into place.
    - ``query`` names the arguments sent as query parameters.
    - Every other argument goes in the JSON body when ``body`` is ``"json"``;
      with ``"none"`` an argument that is neither in the path nor the query is
      refused.
    """

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    path: str = Field(min_length=1, description="Relative to the connector's base URL.")
    query: tuple[str, ...] = ()
    body: Literal["json", "none"] = "none"

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"an operation's path must start with '/', not {value!r}")
        return value

    @property
    def path_params(self) -> tuple[str, ...]:
        """The argument names the path takes, in order."""
        return tuple(_PATH_PARAM.findall(self.path))


class ApiConnector(_Strict):
    """An HTTP API the agent's ``api:<id>`` tools call, from the box.

    The credential, if any, is named only by the env var holding its OpenShell
    placeholder; the sandbox's proxy swaps it for the real value on the
    connector's host and nowhere else.
    """

    id: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=1, description="Scheme and host, optionally a base path.")
    credential_env: str | None = Field(
        default=None, description="The env var holding the credential's placeholder."
    )
    credential_header: str = Field(
        default="authorization", description="The header the credential is sent in."
    )
    credential_prefix: str = Field(
        default="Bearer ", description="Written before the placeholder ('' for a bare key)."
    )


class ToolEntry(_Strict):
    """One tool the agent is offered, and where its body runs."""

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="The tool's JSON Schema, as the model sees it.",
    )
    labels: tuple[str, ...] = Field(
        default=(), description="The gateway's labels for this tool. Descriptive only."
    )
    runs: str = Field(
        default="box",
        description="'box' (in this sandbox), 'gateway' (performed server-side), "
        "'mcp:<mount id>' (on that mounted MCP server), or 'api:<connector id>' "
        "(one operation of that API connector).",
    )
    operation: ApiOperation | None = Field(
        default=None, description="For an 'api:<id>' tool: the request it makes."
    )

    @field_validator("runs")
    @classmethod
    def _runs(cls, value: str) -> str:
        if not _RUNS.match(value):
            raise ValueError(
                f"runs must be 'box', 'gateway', 'mcp:<id>' or 'api:<id>', not {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _operation(self) -> ToolEntry:
        if self.connector is not None and self.operation is None:
            raise ValueError(f"tool {self.name!r} runs on {self.runs} but names no operation")
        if self.connector is None and self.operation is not None:
            raise ValueError(f"tool {self.name!r} has an operation but runs on {self.runs}")
        return self

    @property
    def mount(self) -> str | None:
        """The MCP mount id when ``runs`` is ``mcp:<id>``, else ``None``."""
        return self.runs[4:] if self.runs.startswith("mcp:") else None

    @property
    def connector(self) -> str | None:
        """The API connector id when ``runs`` is ``api:<id>``, else ``None``."""
        return self.runs[4:] if self.runs.startswith("api:") else None


class McpMount(_Strict):
    """An MCP server the agent's ``mcp:<id>`` tools are called on."""

    id: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1)
    transport: Literal["streamable_http", "sse"] = "streamable_http"
    tools: tuple[str, ...] = Field(default=(), description="The tools of it mounted; () = all.")
    credential_env: str | None = Field(
        default=None, description="The env var holding this server's credential placeholder."
    )


class Budgets(_Strict):
    """Limits on one run. ``None`` means the gateway set none."""

    max_iterations: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    timeout_s: float | None = Field(default=None, gt=0)
    park_ttl_s: float = Field(
        default=DEFAULT_PARK_TTL_S,
        ge=0,
        description="How long the runner waits on a held call before it parks the run.",
    )


class ModelRoute(_Strict):
    """How the runner reaches its model."""

    model: str = Field(min_length=1, description="The model name to request.")
    base_url: str = Field(min_length=1, description="An OpenAI-compatible endpoint.")
    api_key_env: str = Field(
        min_length=1,
        description="The env var that holds the key's OpenShell placeholder, never the key.",
    )
    provider: str = Field(default="openai", description="The SDK provider that speaks to it.")


class RunManifest(_Strict):
    """What a box runner runs: one agent, pinned, for one run.

    Built by the gateway; read by the runner. Validation is strict (unknown
    fields are refused) so a manifest from a newer gateway fails loudly
    instead of being half-understood.
    """

    version: Literal[1] = MANIFEST_VERSION
    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent: str = Field(min_length=1, description="The agent's name in the registry.")
    agent_version: str | None = None
    definition: dict[str, Any] = Field(description="The agent definition, as pinned.")
    definition_digest: str | None = Field(
        default=None, description="The registry's digest of ``definition``."
    )
    instructions: str = ""
    input: str = Field(default="", description="The run's input: what the agent is asked.")
    tools: tuple[ToolEntry, ...] = ()
    mcp: tuple[McpMount, ...] = ()
    api: tuple[ApiConnector, ...] = ()
    harness: dict[str, Any] | None = Field(default=None, description="The harness spec, if any.")
    playbook: dict[str, Any] | None = Field(default=None, description="A v2 playbook, if any.")
    budgets: Budgets = Field(default_factory=Budgets)
    model: ModelRoute

    @model_validator(mode="after")
    def _consistent(self) -> RunManifest:
        names = [tool.name for tool in self.tools]
        duplicate = next((name for name in names if names.count(name) > 1), None)
        if duplicate is not None:
            raise ValueError(f"tool {duplicate!r} is listed twice")
        mounts = {mount.id for mount in self.mcp}
        connectors = {connector.id for connector in self.api}
        for tool in self.tools:
            if tool.mount is not None and tool.mount not in mounts:
                raise ValueError(
                    f"tool {tool.name!r} runs on mcp:{tool.mount}, which is not mounted"
                )
            if tool.connector is not None and tool.connector not in connectors:
                raise ValueError(
                    f"tool {tool.name!r} runs on api:{tool.connector}, which is not connected"
                )
        return self

    def connector(self, connector_id: str) -> ApiConnector | None:
        """The API connector called ``connector_id``, or ``None``."""
        return next((c for c in self.api if c.id == connector_id), None)

    def mount(self, mount_id: str) -> McpMount | None:
        """The MCP mount called ``mount_id``, or ``None``."""
        return next((m for m in self.mcp if m.id == mount_id), None)

    def digest(self) -> str:
        """``sha256:`` over this manifest's canonical JSON."""
        return canonical_digest(self.model_dump(mode="json"))

    def tool(self, name: str) -> ToolEntry | None:
        """The tool called ``name``, or ``None``."""
        return next((tool for tool in self.tools if tool.name == name), None)

    @classmethod
    def from_json(cls, data: str | bytes) -> RunManifest:
        """Parse and validate a manifest."""
        return cls.model_validate_json(data)
