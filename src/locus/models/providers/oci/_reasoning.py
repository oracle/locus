"""The chat endpoint's refusal of function tools under reasoning, handled once for both transports.

OCI Generative AI's OpenAI-compatible chat endpoint answers a request that
carries function tools for a model of the gpt-5.6 or gpt-6 families with
``400 Function tools with reasoning_effort are not supported for <model> in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'``. Both chat transports (:class:`OCIModel` on the
SDK, :class:`OCIChatCompletionsModel` on the OpenAI-compatible API) treat that
answer the way the message asks: the request is sent again with the effort set
to ``none``, the model instance keeps sending ``none`` with tools from then on,
and a warning is logged once. It is logged rather than raised through
:mod:`warnings` because it is operational news about a request the library
already repaired, not a misuse the caller has to fix; under a test runner that
turns warnings into errors it would otherwise fail the very call it describes.
A caller who wants reasoning together with tools uses :class:`OCIResponsesModel`.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


#: The effort value the endpoint asks for.
NO_REASONING = "none"


def refuses_tools_under_reasoning(message: str | None) -> bool:
    """Whether ``message`` is the endpoint's refusal of function tools under reasoning."""
    text = (message or "").lower()
    return "function tools" in text and "reasoning_effort" in text and "'none'" in text


def is_no_reasoning(effort: object) -> bool:
    """Whether ``effort`` already asks for no reasoning (``None`` sends nothing, which is not that)."""
    return isinstance(effort, str) and effort.lower() == NO_REASONING


def warn_reasoning_disabled(model: str, transport: str) -> None:
    """Log once that ``transport`` now sends ``reasoning_effort='none'`` with tools for ``model``."""
    logger.warning(
        "OCI chat: the endpoint refused function tools for %r while reasoning was on, "
        "so %s sent this request again with reasoning_effort='none' and keeps sending "
        "'none' with tools for the rest of this instance's lifetime. Pass "
        "reasoning_effort='none' at construction to skip the first attempt, or use "
        "OCIResponsesModel to keep reasoning together with tools.",
        model,
        transport,
    )
