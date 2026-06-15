# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The public contract a security integration implements.

This is the langchain-core-style boundary: `tulip` (core) defines the
**contract** and the helper toolkit; vendor integrations live in a separate
distribution (`tulip-integrations`), depend on this core, and implement
:class:`SecurityAdapter`. Core never imports the integrations package — the
dependency is one-way.

A :class:`SecurityAdapter` is a named, vendored bundle of agent-ready
:class:`~tulip.tools.decorator.Tool` objects. Every tool an integration ships
should follow the toolkit conventions used by the bundled reference adapters:

- read bring-your-own credentials from the environment (:func:`env`);
- fall back to a deterministic, benign **offline sample** when no credentials
  are present (so it runs in CI with no network);
- return a JSON string (:func:`as_json`);
- where it asserts something about an asset, build a GSAR partition (with
  :func:`tool_match` / :func:`inference_claim`) and route it through
  :func:`tulip.security.ground_finding`, so an ungrounded result abstains.

Conformance against this contract is checked by :mod:`tulip.security.testing`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tulip.reasoning.gsar import Claim, EvidenceType
from tulip.security.taxonomy import IndicatorType
from tulip.tools.decorator import Tool


# --------------------------------------------------------------------------- #
# Helper toolkit (the public conventions integrations reuse)
# --------------------------------------------------------------------------- #


def env(*names: str) -> str | None:
    """Return the first non-empty value among ``names`` in the environment.

    Adapters use this to detect bring-your-own credentials; when it returns
    ``None`` the adapter takes its offline-sample path.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def as_json(obj: Any) -> str:
    """Serialise an adapter result for a tool return (``default=str``)."""
    return json.dumps(obj, default=str)


def tool_match(text: str, *evidence_refs: str) -> Claim:
    """A grounded, tool-backed :class:`~tulip.reasoning.gsar.Claim`.

    The strongest evidence tier (:attr:`EvidenceType.TOOL_MATCH`) — use it for a
    statement read directly off a scanner / API response.
    """
    return Claim(text=text, type=EvidenceType.TOOL_MATCH, evidence_refs=list(evidence_refs))


def inference_claim(text: str, *evidence_refs: str) -> Claim:
    """A weak, model-internal :class:`~tulip.reasoning.gsar.Claim`.

    Lands in the ungrounded bucket so a finding built only from inference
    abstains.
    """
    return Claim(text=text, type=EvidenceType.INFERENCE, evidence_refs=list(evidence_refs))


def indicator_type(kind: str, value: str) -> IndicatorType:
    """Map a coarse adapter ``kind`` ("hash"/"ip"/"domain"/…) to a typed enum."""
    if kind == "ip":
        return IndicatorType.IP
    if kind == "domain":
        return IndicatorType.DOMAIN
    if kind == "url":
        return IndicatorType.URL
    if kind == "hash":
        return IndicatorType.MD5 if len(value) == 32 else IndicatorType.SHA256
    return IndicatorType.HOST


# --------------------------------------------------------------------------- #
# The adapter contract
# --------------------------------------------------------------------------- #


@runtime_checkable
class SecurityAdapter(Protocol):
    """A named, vendored bundle of agent-ready security tools.

    The contract an integration implements. `name` is a stable id (e.g.
    ``"splunk"``); `vendor` is a human label (e.g. ``"Splunk / Elastic SIEM"``);
    `tools()` returns the :class:`~tulip.tools.decorator.Tool` objects to hand an
    agent. Pass them to :func:`tulip.security.security_toolset` via ``extra=`` or
    straight to ``Agent(tools=...)``.
    """

    name: str
    vendor: str

    def tools(self) -> list[Tool]: ...


@dataclass(frozen=True)
class ToolAdapter:
    """The simplest concrete :class:`SecurityAdapter` — wrap a list of tools.

    Integrations that don't need their own class can just construct one::

        splunk = ToolAdapter(
            name="splunk", vendor="Splunk SIEM", _tools=[splunk_siem_tool]
        )
        agent = Agent(tools=splunk.tools())
    """

    name: str
    vendor: str
    _tools: list[Tool] = field(default_factory=list)

    def tools(self) -> list[Tool]:
        return list(self._tools)


__all__ = [
    "SecurityAdapter",
    "ToolAdapter",
    "as_json",
    "env",
    "indicator_type",
    "inference_claim",
    "tool_match",
]
