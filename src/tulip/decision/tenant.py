# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-tenant decision heads, each decision on its own tenant's audit chain.

The enterprise shape of the same seam. Every tenant has its own trained heads —
its own served model, or its own LoRA adapter on a shared base (vLLM serves an
adapter under its own model name, so a tenant's provider is a
:class:`~tulip.decision.LogprobDecider` naming that adapter). The router:

- serves a tenant **only** from that tenant's provider. An unknown tenant is
  refused, never answered by another tenant's head. A ``public`` provider may
  be named for tenants with no head of their own; it must be a base model
  trained on no tenant's data, and the decision records that it was used.
- appends every decision to **that tenant's** audit chain — field names,
  labels, probabilities, model and latency. Never the input text unless
  ``record_text=True``: the input may be a customer's words, or a child's.
- keeps nothing between calls but what the resolvers return. No cache, no
  shared state keyed without the tenant.

If the audit write fails, the decision is not returned: a decision the record
cannot show was never made.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tulip.control.audit import AuditTrail
from tulip.decision.fields import Decision, DecisionError, Field
from tulip.decision.provider import DecisionProvider, _run_coroutine_sync


__all__ = ["TenantDecisionRouter", "UnknownTenantError", "per_tenant_trails"]

#: The audit event type a decision is recorded under.
DECISION_EVENT = "decision"

_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class UnknownTenantError(DecisionError):
    """No head is configured for this tenant, and no public head may stand in."""


def _check_tenant(tenant: str) -> str:
    if not isinstance(tenant, str) or not _TENANT_ID.match(tenant) or ".." in tenant:
        raise UnknownTenantError(f"not a tenant id: {tenant!r}")
    return tenant


def per_tenant_trails(directory: str | Path) -> Callable[[str], AuditTrail]:
    """A zero-infra ``audit_for``: one hash-chained JSONL file per tenant.

    ``<directory>/<tenant>/decisions.jsonl``. Tenant ids are checked against a
    strict pattern before they become a path. The enterprise form backs the
    same callable with the tenant's chain in a row-level-secured database.
    """
    root = Path(directory)
    trails: dict[str, AuditTrail] = {}
    lock = threading.Lock()

    def audit_for(tenant: str) -> AuditTrail:
        _check_tenant(tenant)
        with lock:
            trail = trails.get(tenant)
            if trail is None:
                trail = AuditTrail(path=root / tenant / "decisions.jsonl")
                trails[tenant] = trail
            return trail

    return audit_for


class TenantDecisionRouter:
    """Routes each decision to the tenant's own head and records it on the tenant's chain.

    Args:
        providers: ``tenant -> provider``, as a mapping or a resolver returning
            ``None`` for an unknown tenant.
        audit_for: ``tenant -> AuditTrail``. Called on every decision; it must
            return that tenant's chain and no one else's.
        public: A provider for tenants without a head of their own. Off by
            default: an unknown tenant is refused.
        record_text: Also record the input text on the chain. Off by default.
    """

    provider_name = "tenant"

    def __init__(
        self,
        providers: Mapping[str, DecisionProvider] | Callable[[str], DecisionProvider | None],
        *,
        audit_for: Callable[[str], AuditTrail] | None = None,
        public: DecisionProvider | None = None,
        record_text: bool = False,
    ) -> None:
        if isinstance(providers, Mapping):
            table = dict(providers)
            for tenant in table:
                _check_tenant(tenant)
            self._resolve: Callable[[str], DecisionProvider | None] = table.get
        else:
            self._resolve = providers
        self._audit_for = audit_for
        self._public = public
        self.record_text = record_text

    def _provider(self, tenant: str) -> tuple[DecisionProvider, str]:
        provider = self._resolve(_check_tenant(tenant))
        if provider is not None:
            return provider, "tenant"
        if self._public is not None:
            return self._public, "public"
        raise UnknownTenantError(f"no decision head is configured for tenant {tenant!r}")

    def _recorded(self, tenant: str, served_by: str, text: str, decision: Decision) -> Decision:
        decision = Decision(
            answers=decision.answers,
            model=decision.model,
            provider=decision.provider,
            latency_ms=decision.latency_ms,
            meta={**decision.meta, "tenant": tenant, "served_by": served_by},
        )
        if self._audit_for is not None:
            payload: dict[str, Any] = {
                "tenant": tenant,
                "served_by": served_by,
                **decision.as_record(),
            }
            if self.record_text:
                payload["text"] = text
            try:
                self._audit_for(tenant).record(DECISION_EVENT, payload)
            except Exception as exc:
                raise DecisionError(
                    f"the decision for tenant {tenant!r} could not be recorded"
                ) from exc
        return decision

    async def decide(self, text: str, fields: Sequence[Field], *, tenant: str) -> Decision:
        """Answer ``fields`` about ``text`` with ``tenant``'s head, and record it."""
        provider, served_by = self._provider(tenant)
        decision = await provider.decide(text, fields)
        return self._recorded(tenant, served_by, text, decision)

    def decide_sync(self, text: str, fields: Sequence[Field], *, tenant: str) -> Decision:
        """The same, from synchronous code."""
        provider, served_by = self._provider(tenant)
        own = getattr(provider, "decide_sync", None)
        if callable(own):
            decision = own(text, fields)
        else:
            decision = _run_coroutine_sync(lambda: provider.decide(text, fields))
        if not isinstance(decision, Decision):
            raise DecisionError(f"{type(provider).__name__} returned no Decision")
        return self._recorded(tenant, served_by, text, decision)

    def for_tenant(self, tenant: str) -> DecisionProvider:
        """A :class:`DecisionProvider` bound to one tenant, e.g. for a ``DecisionAdvisor``."""
        return _BoundTenant(self, _check_tenant(tenant))


class _BoundTenant:
    """The router with the tenant fixed, so it fits the plain provider protocol."""

    def __init__(self, router: TenantDecisionRouter, tenant: str) -> None:
        self._router = router
        self.tenant = tenant

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        return await self._router.decide(text, fields, tenant=self.tenant)

    def decide_sync(self, text: str, fields: Sequence[Field]) -> Decision:
        return self._router.decide_sync(text, fields, tenant=self.tenant)
