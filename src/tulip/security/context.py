# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""`SecurityContext` — investigate by domain, not by vendor.

You don't think *"I'm querying Splunk, then CrowdStrike, then Okta."* You think
*"I'm investigating an incident."* `SecurityContext` is that mental model in
code: one handle exposing security **domains** —

    ctx = SecurityContext()
    await ctx.logs.search("failed login spike")
    await ctx.endpoint.get_host("WS-0142")
    await ctx.identity.get_user("jsmith@example.com")
    await ctx.threat_intel.enrich("198.51.100.23")
    ctx.actions.request_approval(action, finding=f, verdict=v)

Each domain is a small Protocol (a *port*); the default provider is the bundled
offline reference adapter, so `SecurityContext()` works with **zero config**.
Swap in a real vendor by injecting a provider — `SecurityContext(logs=SplunkLogs())`
— which lives in the one-way-dependent ``tulip-integrations`` package. Core never
imports a vendor; you wire them explicitly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tulip.security.aws import describe_aws, use_aws
from tulip.security.edr import fetch_host_timeline, isolate_host, list_detections
from tulip.security.findings import Finding
from tulip.security.intel import enrich_indicator
from tulip.security.policy import Action, ApprovalDecision, SecurityPolicy, approve
from tulip.security.siem import query_siem
from tulip.security.verify import Verdict


# --------------------------------------------------------------------------- #
# Domain ports — the contract each provider implements.
# --------------------------------------------------------------------------- #


@runtime_checkable
class LogSource(Protocol):
    """Log / SIEM search."""

    async def search(self, query: str, *, window: str = "24h") -> dict[str, Any]: ...


@runtime_checkable
class EndpointSource(Protocol):
    """EDR host forensics + containment."""

    async def get_host(self, host: str, *, window: str = "24h") -> dict[str, Any]: ...
    async def detections(self, host: str | None = None) -> dict[str, Any]: ...
    async def isolate(self, host_id: str) -> dict[str, Any]: ...


@runtime_checkable
class IdentitySource(Protocol):
    """Identity provider — the surface most attacks touch."""

    async def get_user(self, user: str) -> dict[str, Any]: ...
    async def risk(self, user: str) -> dict[str, Any]: ...
    async def signins(self, user: str) -> dict[str, Any]: ...
    async def disable(self, user: str) -> dict[str, Any]: ...


@runtime_checkable
class CloudSource(Protocol):
    """Cloud control-plane evidence (read-only)."""

    async def describe(
        self, service: str | None = None, operation: str | None = None
    ) -> dict[str, Any]: ...
    async def events(
        self, service: str, operation: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


@runtime_checkable
class ThreatIntelSource(Protocol):
    """IOC reputation / enrichment."""

    async def enrich(self, indicator: str) -> dict[str, Any]: ...


@runtime_checkable
class ActionsPort(Protocol):
    """Gate a response action against evidence + verification + policy."""

    def request_approval(
        self,
        action: Action,
        *,
        finding: Finding | None = None,
        verdict: Verdict | None = None,
    ) -> ApprovalDecision: ...

    async def execute(
        self,
        action: Action,
        perform: Callable[[], Awaitable[Any]],
        *,
        finding: Finding | None = None,
        verdict: Verdict | None = None,
    ) -> Any:
        """Admission control: run ``perform`` only if the action is admitted."""
        ...


# --------------------------------------------------------------------------- #
# Default reference providers (offline) — wrap the bundled adapters.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RefLogs:
    async def search(self, query: str, *, window: str = "24h") -> dict[str, Any]:
        return query_siem(query, window=window)


@dataclass(frozen=True)
class _RefEndpoint:
    async def get_host(self, host: str, *, window: str = "24h") -> dict[str, Any]:
        return fetch_host_timeline(host, window=window)

    async def detections(self, host: str | None = None) -> dict[str, Any]:
        return list_detections(host)

    async def isolate(self, host_id: str) -> dict[str, Any]:
        return isolate_host(host_id)


# Deterministic, benign identity samples (a low-risk and a high-risk user).
_IDENTITY_SAMPLE: dict[str, dict[str, Any]] = {
    "jsmith@example.com": {
        "user": "jsmith@example.com",
        "risk": "low",
        "mfa": True,
        "signins": [{"ip": "198.51.100.10", "city": "Toronto", "result": "success"}],
    },
    "mallory@example.com": {
        "user": "mallory@example.com",
        "risk": "high",
        "mfa": False,
        "impossible_travel": True,
        "signins": [
            {"ip": "203.0.113.7", "city": "Toronto", "result": "success"},
            {"ip": "192.0.2.55", "city": "Minsk", "result": "success"},
        ],
    },
}


@dataclass(frozen=True)
class _RefIdentity:
    """Okta/Entra-shaped offline reference. Live providers live in integrations."""

    def _record(self, user: str) -> dict[str, Any]:
        return _IDENTITY_SAMPLE.get(
            user, {"user": user, "risk": "unknown", "mfa": None, "signins": []}
        )

    async def get_user(self, user: str) -> dict[str, Any]:
        return {"source": "offline-sample", **self._record(user)}

    async def risk(self, user: str) -> dict[str, Any]:
        rec = self._record(user)
        return {
            "user": user,
            "risk": rec["risk"],
            "impossible_travel": rec.get("impossible_travel", False),
        }

    async def signins(self, user: str) -> dict[str, Any]:
        return {"user": user, "signins": self._record(user).get("signins", [])}

    async def disable(self, user: str) -> dict[str, Any]:
        # Offline: a simulated receipt. A write — gate it via ctx.actions first.
        return {"user": user, "disabled": True, "source": "offline-sample"}


@dataclass(frozen=True)
class _RefCloud:
    async def describe(
        self, service: str | None = None, operation: str | None = None
    ) -> dict[str, Any]:
        try:
            return describe_aws(service=service, operation=operation)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully without boto3
            return {"available": False, "reason": f"{exc}", "hint": "pip install tulip-agents[aws]"}

    async def events(
        self, service: str, operation: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            return use_aws(service, operation, parameters=parameters)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully without boto3
            return {"available": False, "reason": f"{exc}", "hint": "pip install tulip-agents[aws]"}


@dataclass(frozen=True)
class _RefThreatIntel:
    async def enrich(self, indicator: str) -> dict[str, Any]:
        return enrich_indicator(indicator)


@dataclass(frozen=True)
class _RefActions:
    policy: SecurityPolicy = field(default_factory=SecurityPolicy)

    def request_approval(
        self,
        action: Action,
        *,
        finding: Finding | None = None,
        verdict: Verdict | None = None,
    ) -> ApprovalDecision:
        return approve(action, policy=self.policy, finding=finding, verdict=verdict)

    async def execute(
        self,
        action: Action,
        perform: Callable[[], Awaitable[Any]],
        *,
        finding: Finding | None = None,
        verdict: Verdict | None = None,
    ) -> Any:
        from tulip.security.admit import admit

        return await admit(action, perform, policy=self.policy, finding=finding, verdict=verdict)


# --------------------------------------------------------------------------- #
# The facade.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SecurityContext:
    """One handle over the security domains. Defaults run offline, zero-config.

    Inject a vendor provider per domain to go live::

        from tulip_integrations.siem.splunk import SplunkLogs

        ctx = SecurityContext(logs=SplunkLogs())
    """

    logs: LogSource = field(default_factory=_RefLogs)
    endpoint: EndpointSource = field(default_factory=_RefEndpoint)
    identity: IdentitySource = field(default_factory=_RefIdentity)
    cloud: CloudSource = field(default_factory=_RefCloud)
    threat_intel: ThreatIntelSource = field(default_factory=_RefThreatIntel)
    actions: ActionsPort = field(default_factory=_RefActions)

    def toolset(self, **flags: Any) -> list[Any]:
        """The agent-ready tool bundle (delegates to :func:`security_toolset`).

        The domain handles above are the *programmatic* facade; this is the
        *agent* facade — hand it to ``Agent(tools=...)``.
        """
        from tulip.security import security_toolset

        return security_toolset(**flags)


__all__ = [
    "ActionsPort",
    "CloudSource",
    "EndpointSource",
    "IdentitySource",
    "LogSource",
    "SecurityContext",
    "ThreatIntelSource",
]
