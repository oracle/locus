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
    "MANIFEST_VERSION",
    "Budgets",
    "McpMount",
    "ModelRoute",
    "RunManifest",
    "ToolEntry",
    "canonical_digest",
]

#: The manifest format this SDK reads and writes.
MANIFEST_VERSION: Literal[1] = 1

#: Where a tool runs: in the box, on the gateway, or on a mounted MCP server.
_RUNS = re.compile(r"^(box|gateway|mcp:[A-Za-z0-9][A-Za-z0-9._-]{0,127})$")


def canonical_digest(value: Any) -> str:
    """``sha256:<hex>`` of ``value`` as canonical JSON (sorted keys, no spaces)."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
        "or 'mcp:<mount id>' (on that mounted MCP server).",
    )

    @field_validator("runs")
    @classmethod
    def _runs(cls, value: str) -> str:
        if not _RUNS.match(value):
            raise ValueError(f"runs must be 'box', 'gateway' or 'mcp:<id>', not {value!r}")
        return value

    @property
    def mount(self) -> str | None:
        """The MCP mount id when ``runs`` is ``mcp:<id>``, else ``None``."""
        return self.runs[4:] if self.runs.startswith("mcp:") else None


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
    tools: tuple[ToolEntry, ...] = ()
    mcp: tuple[McpMount, ...] = ()
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
        for tool in self.tools:
            if tool.mount is not None and tool.mount not in mounts:
                raise ValueError(
                    f"tool {tool.name!r} runs on mcp:{tool.mount}, which is not mounted"
                )
        return self

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
