# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Security severity, indicator, and threat-taxonomy reference enums.

These are the vocabulary a :class:`~tulip.security.findings.Evidence` tags
itself with. The threat enums mirror the published catalogues so a
finding is portable into a SIEM, a compliance report, or another tool
without a translation layer:

- :class:`AtlasTechnique` — MITRE ATLAS (``AML.Txxxx``), the adversarial
  technique catalogue for AI systems.
- :class:`OwaspLLM` — OWASP Top 10 for LLM Applications, 2025
  (``LLM01``–``LLM10``).
- :class:`OwaspASI` — OWASP Top 10 for Agentic Applications, 2026
  (``ASI01``–``ASI10``), from the Agentic Security Initiative.

Each enum carries a representative subset (ATLAS) or the full list
(OWASP); values are the canonical IDs so ``Severity.HIGH == "high"`` and
``OwaspLLM.PROMPT_INJECTION == "LLM01"`` both hold.
"""

from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    """Ordered severity band. ``StrEnum`` so it serialises as the bare string.

    Not directly comparable with ``<`` (string ordering would be wrong);
    use :func:`severity_at_least` or :data:`SEVERITY_ORDER` for ranking.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Ordinal rank for each band, lowest to highest. Used by
# :func:`severity_at_least` and by callers that gate on a threshold.
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def severity_at_least(value: Severity, floor: Severity) -> bool:
    """Whether ``value`` ranks at or above ``floor`` (e.g. ``>= HIGH``)."""
    return SEVERITY_ORDER[value] >= SEVERITY_ORDER[floor]


class IndicatorType(StrEnum):
    """The kind of an :class:`~tulip.security.findings.Indicator`."""

    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    SHA256 = "sha256"
    MD5 = "md5"
    EMAIL = "email"
    FILE_PATH = "file_path"
    USER = "user"
    HOST = "host"
    ENDPOINT = "endpoint"
    """A model / inference endpoint — the subject of a fingerprint finding."""


class AtlasTechnique(StrEnum):
    """MITRE ATLAS techniques (``AML.Txxxx``) — representative subset.

    A curated set covering the AI-security surfaces Tulip's examples
    exercise; the full matrix lives at <https://atlas.mitre.org/>. Values
    are the canonical ATLAS IDs.
    """

    CRAFT_ADVERSARIAL_DATA = "AML.T0043"
    """Craft Adversarial Data."""

    PROMPT_INJECTION = "AML.T0051"
    """LLM Prompt Injection (direct or indirect via tool output / RAG)."""

    JAILBREAK = "AML.T0054"
    """LLM Jailbreak — bypassing model controls."""

    POISON_TRAINING_DATA = "AML.T0020"
    """Poison Training Data."""

    BACKDOOR_ML_MODEL = "AML.T0018"
    """Backdoor ML Model."""

    INFERENCE_API_ACCESS = "AML.T0040"
    """AI Model Inference API Access."""

    EXFILTRATION_VIA_INFERENCE_API = "AML.T0024"
    """Exfiltration via AI Inference API (e.g. model extraction probing)."""

    EXFILTRATION_VIA_AGENT_TOOL = "AML.T0086"
    """Exfiltration via AI Agent Tool Invocation."""

    AGENT_TOOL_POISONING = "AML.T0110"
    """AI Agent Tool Poisoning."""

    EXTERNAL_HARMS = "AML.T0048"
    """External Harms — financial, reputational, or physical harm."""


class OwaspLLM(StrEnum):
    """OWASP Top 10 for LLM Applications, 2025 (``LLM01``–``LLM10``).

    See <https://genai.owasp.org/llm-top-10/>.
    """

    PROMPT_INJECTION = "LLM01"
    """Prompt Injection."""

    SENSITIVE_INFORMATION_DISCLOSURE = "LLM02"
    """Sensitive Information Disclosure."""

    SUPPLY_CHAIN = "LLM03"
    """Supply Chain."""

    DATA_AND_MODEL_POISONING = "LLM04"
    """Data and Model Poisoning."""

    IMPROPER_OUTPUT_HANDLING = "LLM05"
    """Improper Output Handling."""

    EXCESSIVE_AGENCY = "LLM06"
    """Excessive Agency."""

    SYSTEM_PROMPT_LEAKAGE = "LLM07"
    """System Prompt Leakage."""

    VECTOR_AND_EMBEDDING_WEAKNESSES = "LLM08"
    """Vector and Embedding Weaknesses."""

    MISINFORMATION = "LLM09"
    """Misinformation."""

    UNBOUNDED_CONSUMPTION = "LLM10"
    """Unbounded Consumption."""


class OwaspASI(StrEnum):
    """OWASP Top 10 for Agentic Applications, 2026 (``ASI01``–``ASI10``).

    From the OWASP Agentic Security Initiative. See
    <https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/>.
    """

    AGENT_GOAL_HIJACK = "ASI01"
    """Agent Goal Hijack."""

    TOOL_MISUSE = "ASI02"
    """Tool Misuse."""

    IDENTITY_AND_PRIVILEGE_ABUSE = "ASI03"
    """Identity & Privilege Abuse."""

    AGENTIC_SUPPLY_CHAIN = "ASI04"
    """Agentic Supply Chain Vulnerabilities."""

    UNEXPECTED_CODE_EXECUTION = "ASI05"
    """Unexpected Code Execution."""

    MEMORY_AND_CONTEXT_POISONING = "ASI06"
    """Memory & Context Poisoning."""

    INSECURE_INTER_AGENT_COMMUNICATION = "ASI07"
    """Insecure Inter-Agent Communication."""

    CASCADING_FAILURES = "ASI08"
    """Cascading Failures."""

    HUMAN_AGENT_TRUST_EXPLOITATION = "ASI09"
    """Human-Agent Trust Exploitation."""

    ROGUE_AGENTS = "ASI10"
    """Rogue Agents."""


# A finding may tag itself with techniques drawn from any of the three
# catalogues. Kept as a union of StrEnums so the canonical ID round-trips
# through ``model_dump`` while the type stays precise.
TaxonomyTag = AtlasTechnique | OwaspLLM | OwaspASI


__all__ = [
    "SEVERITY_ORDER",
    "AtlasTechnique",
    "IndicatorType",
    "OwaspASI",
    "OwaspLLM",
    "Severity",
    "TaxonomyTag",
    "severity_at_least",
]
