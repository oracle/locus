# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Typed decisions: pick from a fixed list, with a probability, in one forward pass.

Much of what an agent must decide is not writing. *Is this message for me?*
*Which of these thirteen commands is it?* *Does this reply ask for a home
address?* A model that writes answers those slowly and gives no measure of how
sure it is. A **decision model** answers them as a classifier does: every listed
answer gets a probability, read from the logits of a single forward pass.

    from locus.decision import Choice, LogprobDecider, YesNo

    decider = LogprobDecider("http://127.0.0.1:8000/v1", model="my-head")
    decision = await decider.decide(
        "can you build me a castle",
        [
            Choice("intent", "What does the speaker want?", ("come", "build", "chat")),
            YesNo("personal", "Does it ask for personal details?"),
        ],
    )
    decision["intent"].label           # "build"
    decision["personal"].p_yes         # 0.004

- **Fields** — :class:`Choice`, :class:`YesNo`, :class:`Score`.
- **Answers** — :class:`Answer` (a distribution over the listed labels, its
  argmax, margin and coverage) gathered in a :class:`Decision`.
- **Providers** — the :class:`DecisionProvider` protocol; the zero-infra
  default :class:`LogprobDecider` over any OpenAI-compatible server; the
  Decisions API on OpenAI (:class:`OpenAIDecider`) or on OCI Generative AI
  (:class:`OCIDecider`), also by name with :func:`get_decider`
  (``"openai:<model>"``, ``"oci:<model>"``); the per-tenant
  :class:`TenantDecisionRouter`.
- **The gate** — :class:`DecisionAdvisor` plugs an admit head into
  ``approve(advisor=)``; :func:`verification_from_decision` feeds safety heads
  into ``ControlPolicy`` verification.

The wire format (:data:`SYSTEM_PROMPT`, :func:`render`) is a contract: train a
head on it and any server that returns logprobs serves it.
"""

from __future__ import annotations

from locus.decision.api import MAX_QUESTIONS, MAX_SCORE_LEVELS, OCIDecider, OpenAIDecider
from locus.decision.control import (
    ADMIT_FIELD,
    ADMIT_QUESTION,
    DecisionAdvisor,
    describe_action,
    verification_from_decision,
)
from locus.decision.fields import (
    LETTERS,
    SYSTEM_PROMPT,
    YES_NO,
    Answer,
    Choice,
    Decision,
    DecisionError,
    Field,
    Score,
    YesNo,
    answer_from_logprobs,
    render,
)
from locus.decision.provider import DecisionProvider, LogprobDecider, decide_sync
from locus.decision.registry import get_decider, list_deciders, register_decider
from locus.decision.tenant import TenantDecisionRouter, UnknownTenantError, per_tenant_trails


__all__ = [
    "ADMIT_FIELD",
    "ADMIT_QUESTION",
    "LETTERS",
    "MAX_QUESTIONS",
    "MAX_SCORE_LEVELS",
    "SYSTEM_PROMPT",
    "YES_NO",
    "Answer",
    "Choice",
    "Decision",
    "DecisionAdvisor",
    "DecisionError",
    "DecisionProvider",
    "Field",
    "LogprobDecider",
    "OCIDecider",
    "OpenAIDecider",
    "Score",
    "TenantDecisionRouter",
    "UnknownTenantError",
    "YesNo",
    "answer_from_logprobs",
    "decide_sync",
    "describe_action",
    "get_decider",
    "list_deciders",
    "per_tenant_trails",
    "register_decider",
    "render",
    "verification_from_decision",
]
