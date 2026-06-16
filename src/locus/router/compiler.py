# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Cognitive compiler — turn a typed :class:`GoalFrame` into a runnable graph.

The compiler is the deterministic core: every step after the LLM
produces the :class:`GoalFrame` is rule-driven (protocol selection,
capability binding, policy gate, builder dispatch). No primitive in
``locus`` is modified — the compiler only composes existing pieces.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from locus.observability.router_events import (
    emit_picker_fallback,
    emit_policy_verdict,
    emit_protocol_no_match,
    emit_protocol_selected,
    emit_runnable_compiled,
)
from locus.router.capability import CapabilityIndex
from locus.router.goal_frame import GoalFrame
from locus.router.picker import LLMProtocolPicker
from locus.router.policy import PolicyDeniedError, PolicyGate, PolicyVerdict
from locus.router.protocol import (
    BuilderContext,
    NoMatchingProtocolError,
    Protocol,
    ProtocolRegistry,
    _rank_key,
)
from locus.router.runnable import Runnable, RunnableResult
from locus.router.skill_index import SkillIndex


ApprovalCallback = Callable[[GoalFrame, PolicyVerdict], Awaitable[bool]]
"""Async callback used when a verdict requires approval.

Returning ``True`` lets the compiled runnable execute; returning
``False`` raises :class:`PolicyDeniedError`. Defaults deny.
"""


async def _default_deny(_frame: GoalFrame, verdict: PolicyVerdict) -> bool:
    return False


class _ApprovalRunnable(BaseModel):
    """Wraps a Runnable with an approval check that fires before execution.

    Used when the policy gate verdict is ``require_approval=True``. The
    follow-up ``approval_gated_execution`` protocol replaces this with a
    StateGraph + ``interrupt()`` node so the workbench's interrupt UI
    drives the approval; for now the callback is the simplest contract
    that works for the three v1 protocols (none of which are graphs).
    """

    inner: Any
    frame: Any
    verdict: Any
    callback: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        approved = await self.callback(self.frame, self.verdict)
        if not approved:
            raise PolicyDeniedError(
                f"approval denied for protocol={self.inner.protocol_id!r}: {self.verdict.reason}",
            )
        result: RunnableResult = await self.inner.execute(task)
        return result


class CognitiveCompiler:
    """Glue between protocols, capabilities, policy, and the model.

    Parameters
    ----------
    protocols:
        :class:`ProtocolRegistry` populated with built-in or custom
        protocols.
    capabilities:
        :class:`CapabilityIndex` over the surrounding ``ToolRegistry``.
        The index resolves capability ids to real tools at compile time.
    policy:
        :class:`PolicyGate` that runs between selection and build.
    model:
        A locus model instance (or model string) injected into every
        builder. Builders pass it to :class:`~locus.Agent` / specialist
        constructors.
    skills:
        Optional :class:`SkillIndex`. When provided, every emitted
        :class:`Agent` is configured with a
        :class:`~locus.skills.SkillsPlugin` containing the skills tagged
        for ``frame.domain`` (plus any globally-tagged skills). The
        agent loop's L1 / L2 / L3 progressive disclosure surfaces them
        at runtime.
    on_approval:
        Optional async callback fired when the verdict requires
        approval. Defaults to denying — wire your workbench / CLI
        approval flow here.
    protocol_picker:
        Optional :class:`LLMProtocolPicker`. When present, the compiler
        delegates the *last-mile* protocol pick to the model whenever
        the filter leaves more than one candidate. When absent (the
        default), selection is the deterministic
        :func:`_rank_key`-based ranker. The picker is the **only** part
        of the routing pipeline that uses the model; everything else
        — filtering, policy gating, capability binding, builder dispatch
        — remains rule-based.
    """

    def __init__(
        self,
        *,
        protocols: ProtocolRegistry,
        capabilities: CapabilityIndex,
        policy: PolicyGate,
        model: Any,
        skills: SkillIndex | None = None,
        a2a_endpoint: str | None = None,
        on_approval: ApprovalCallback | None = None,
        protocol_picker: LLMProtocolPicker | None = None,
    ) -> None:
        self.protocols = protocols
        self.capabilities = capabilities
        self.policy = policy
        self.model = model
        self.skills = skills
        self.a2a_endpoint = a2a_endpoint
        self._on_approval: ApprovalCallback = on_approval or _default_deny
        self.protocol_picker = protocol_picker

    def _build_context(self) -> BuilderContext:
        return BuilderContext(
            model=self.model,
            capabilities=self.capabilities,
            skills=self.skills,
            a2a_endpoint=self.a2a_endpoint,
        )

    async def _pick_protocol(
        self,
        frame: GoalFrame,
        candidates: list[Protocol],
        *,
        run_id: str | None,
    ) -> tuple[Protocol, str, str | None]:
        """Resolve the final protocol pick from a filtered candidate set.

        Returns ``(protocol, method, rationale)`` so the caller can
        emit a consistent ``router.protocol.selected`` event.

        Decision ladder:

        1. No picker configured → rule-based ranker via ``_rank_key``.
        2. Exactly one candidate → short-circuit (no LLM call,
           no token cost).
        3. Multiple candidates + picker → call the picker. On any
           failure (``PickerError`` or unexpected exception) emit
           ``router.protocol.picker_fallback`` and fall back to the
           rule-based ranker. Emergent mode never reduces availability.
        """
        if self.protocol_picker is None:
            picked = min(candidates, key=lambda p: _rank_key(p, frame))
            return picked, "rule_based", None

        if len(candidates) == 1:
            return candidates[0], "single_candidate", None

        try:
            picked, rationale = await self.protocol_picker.pick(frame, candidates)
            return picked, "llm_picked", rationale
        except Exception as exc:  # noqa: BLE001 — picker may surface arbitrary errors
            if run_id:
                await emit_picker_fallback(run_id, frame, f"{type(exc).__name__}: {exc}")
            picked = min(candidates, key=lambda p: _rank_key(p, frame))
            return picked, "rule_based_fallback", None

    async def compile(self, frame: GoalFrame, run_id: str | None = None) -> Runnable:
        """Pick a protocol, run the gate, build the runnable.

        ``run_id`` (when provided) scopes every emitted
        :class:`StreamEvent` so the workbench's SSE consumer can
        correlate selection / verdict / compile events with one
        cognitive dispatch.
        """
        available = {c.id for c in self.capabilities.all()}
        candidates = self.protocols.filter_candidates(frame, available_capabilities=available)
        if not candidates:
            err = NoMatchingProtocolError(
                f"No protocol handles primary_goal={frame.primary_goal!r} "
                f"at risk={frame.risk!r} with capabilities={sorted(available)}.",
            )
            if run_id:
                await emit_protocol_no_match(run_id, frame, str(err))
            raise err

        protocol, method, rationale = await self._pick_protocol(frame, candidates, run_id=run_id)

        if run_id:
            await emit_protocol_selected(
                run_id, frame, protocol, method=method, rationale=rationale
            )

        verdict = self.policy.check(frame, protocol)
        if run_id:
            await emit_policy_verdict(run_id, frame, protocol, verdict)
        if not verdict.allow:
            raise PolicyDeniedError(verdict.reason)

        # Intersect with the actual registry — the LLM extractor sometimes
        # hallucinates capability ids that don't exist. Strict lookup
        # would crash the whole dispatch on a single bad id; lenient
        # filtering is the right call at this boundary because the
        # protocol registry's selection step has *already* checked that
        # every protocol-required capability is available, so the only
        # things being dropped here are extractor-suggested extras.
        requested = list(frame.required_capabilities)
        valid = [cid for cid in requested if cid in available]
        caps = self.capabilities.lookup(valid) if valid else []
        runnable = protocol.builder(frame, caps, self._build_context())

        if verdict.require_approval:
            runnable = _ApprovalRunnable(
                inner=runnable,
                frame=frame,
                verdict=verdict,
                callback=self._on_approval,
            )

        if run_id:
            await emit_runnable_compiled(
                run_id,
                protocol_id=protocol.id,
                runnable_type=type(runnable).__name__,
                capability_count=len(caps),
            )
        return runnable
