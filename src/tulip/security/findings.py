# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Typed security findings — the artifacts a Tulip security agent emits.

A :class:`Evidence` is the unit a SOC consumes: a titled, severity-banded
statement about an asset, tagged with the threat taxonomy, and — the part
that matters — carrying the grounding score and evidence references that
admitted it. There is no public path that builds a ``Evidence`` without a
grounding score; the sanctioned factory is
:func:`tulip.security.ground_finding`, which only returns one when its
evidence clears the GSAR threshold. An ungrounded finding is therefore a
false positive *by construction* and never reaches the queue.

:class:`FingerprintFinding` specialises this for the AI-security flagship:
identifying the model / inference engine / hardware behind an endpoint
from timing side-channels, with the timing feature vector as evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tulip.security.taxonomy import IndicatorType, Severity, TaxonomyTag


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A probability / confidence in ``[0, 1]`` — mirrors the GSAR shape."""


class Indicator(BaseModel):
    """A typed indicator of compromise — an observable (type + value).

    Benign by design: examples use documentation-range addresses
    (RFC 5737), ``*.example`` domains, and well-known test artifacts.
    """

    type: IndicatorType = Field(description="The kind of observable.")
    value: str = Field(description="The observable value (e.g. an IP or domain).")

    model_config = {"frozen": True}


class Evidence(BaseModel):
    """A grounded security finding.

    The ``gsar_score`` and ``evidence_refs`` fields are required: a
    ``Evidence`` always knows how strongly it is grounded and what it is
    grounded in. Build findings via :func:`tulip.security.ground_finding`
    rather than constructing them directly — that is the path that
    enforces the grounding threshold.
    """

    title: str = Field(description="One-line summary of the finding.")
    description: str = Field(description="What was observed and why it matters.")
    severity: Severity = Field(description="Severity band.")
    asset: str = Field(description="The affected asset / host / service / endpoint.")
    remediation: str = Field(description="Recommended remediation.")
    gsar_score: Confidence = Field(
        description="GSAR grounding score that admitted this finding.",
    )
    confidence: Confidence = Field(
        default=1.0,
        description="Analyst-facing confidence, distinct from the grounding score.",
    )
    indicators: list[Indicator] = Field(
        default_factory=list,
        description="Indicators of compromise associated with the finding.",
    )
    taxonomy: list[TaxonomyTag] = Field(
        default_factory=list,
        description="MITRE ATLAS / OWASP LLM / OWASP ASI tags.",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Opaque references to the grounding evidence — flattened from "
            "the GSAR partition's claims (e.g. "
            "``tool:scan_endpoint:row=3:tls_expiry``)."
        ),
    )

    model_config = {"frozen": True}


class FingerprintVerdict(BaseModel):
    """A timing side-channel inference-fingerprinting verdict.

    Identifies what is serving an endpoint from observable timing
    features. ``feature_coverage`` is the fraction of the expected
    feature schema actually observed; low coverage should drive an
    abstention rather than an asserted fingerprint.
    """

    model: str = Field(description="Identified model (or family).")
    engine: str = Field(description="Identified inference engine.")
    hardware: str = Field(description="Identified accelerator / hardware class.")
    confidence: Confidence = Field(description="Classifier confidence.")
    feature_coverage: Confidence = Field(
        description="Fraction of the expected feature schema observed.",
    )

    model_config = {"frozen": True}


@runtime_checkable
class FingerprintClassifier(Protocol):
    """A callable mapping a timing feature vector to a verdict.

    Examples ship a deterministic mock; real deployments plug a
    fingerprinting service in behind this same signature.
    """

    def __call__(self, features: Mapping[str, float]) -> FingerprintVerdict: ...


class FingerprintFinding(Evidence):
    """A :class:`Evidence` specialised for inference fingerprinting.

    Carries the :class:`FingerprintVerdict` whose feature vector is the
    finding's evidence.
    """

    verdict: FingerprintVerdict = Field(description="The fingerprint verdict.")

    model_config = {"frozen": True}


__all__ = [
    "Confidence",
    "Evidence",
    "FingerprintClassifier",
    "FingerprintFinding",
    "FingerprintVerdict",
    "Indicator",
]
