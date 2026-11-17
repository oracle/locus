# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests: the guardrail and the response cache of the sample gateway.

Deployment validation for ``examples/litellm-gateway/``: the in-process PII
guardrail both config files carry must rewrite a prompt before the upstream
call, and the Redis cache ``config.platform.yaml`` turns on must answer an
identical request from the cache. Same env gate as the other gateway tests,
plus one for the cache::

    export LITELLM_GATEWAY_URL="http://localhost:4000"
    export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"      # or a virtual key
    export LITELLM_GATEWAY_MODEL="gpt-4o-mini"            # alias from config.yaml
    export LITELLM_GATEWAY_CACHE=1   # only when the gateway runs config.platform.yaml
    pytest tests/integration/test_litellm_gateway_platform.py -v

Auto-skips without the gateway env vars; the cache test also skips without
``LITELLM_GATEWAY_CACHE``.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest


_GATEWAY_URL = os.environ.get("LITELLM_GATEWAY_URL", "").rstrip("/")
_GATEWAY_KEY = os.environ.get("LITELLM_GATEWAY_KEY", "")
_GATEWAY_MODEL = os.environ.get("LITELLM_GATEWAY_MODEL", "gpt-4o-mini")
_CACHE_ENABLED = os.environ.get("LITELLM_GATEWAY_CACHE", "") not in ("", "0", "false")

_SYNTHETIC_SSN = "123-45-6789"
_SYNTHETIC_CARD = "4111 1111 1111 1111"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATEWAY_URL and _GATEWAY_KEY),
        reason=(
            "LITELLM_GATEWAY_URL / LITELLM_GATEWAY_KEY not set — bring up the "
            "sample gateway under examples/litellm-gateway/ and export the URL "
            "+ a key. See docs/how-to/litellm-gateway.md."
        ),
    ),
]


def _complete(content: str, **extra: object) -> httpx.Response:
    return httpx.post(
        f"{_GATEWAY_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {_GATEWAY_KEY}"},
        json={
            "model": _GATEWAY_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 60,
            "temperature": 0,
            **extra,
        },
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# Guardrail — the masked prompt is what leaves the gateway
# ---------------------------------------------------------------------------


def test_pii_guardrail_masks_ssn_and_card_before_the_upstream_call():
    """Ask the model to echo a line holding a synthetic SSN and card number.
    The guardrail rewrites the prompt in ``pre_call``, so the echo can only
    contain the masks: the provider never saw the digits."""
    resp = _complete(
        "Return the text inside the square brackets exactly as written, nothing else: "
        f"[reference {_SYNTHETIC_SSN} / {_SYNTHETIC_CARD}]"
    )
    resp.raise_for_status()
    assert "pii-mask" in resp.headers.get("x-litellm-applied-guardrails", ""), dict(resp.headers)
    content = resp.json()["choices"][0]["message"]["content"]
    assert _SYNTHETIC_SSN not in content, content
    assert _SYNTHETIC_CARD not in content, content
    assert "[US_SSN_REDACTED]" in content, content
    assert "[CREDIT_CARD_REDACTED]" in content, content


@pytest.mark.asyncio
async def test_pii_guardrail_applies_to_locus_requests_too():
    """The same through Locus's ``OpenAIModel`` — the client the docs
    instruct users to build; the gateway masks regardless of caller."""
    from locus.core.messages import Message
    from locus.models.native.openai import OpenAIModel

    model = OpenAIModel(
        model=_GATEWAY_MODEL, api_key=_GATEWAY_KEY, base_url=_GATEWAY_URL, max_tokens=60
    )
    resp = await model.complete(
        messages=[
            Message.user(
                "Return the text inside the square brackets exactly as written, nothing else: "
                f"[case {_SYNTHETIC_SSN}]"
            )
        ],
    )
    assert resp.message.content
    assert _SYNTHETIC_SSN not in resp.message.content
    assert "[US_SSN_REDACTED]" in resp.message.content


# ---------------------------------------------------------------------------
# Cache — an identical request is answered from Redis the second time
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _CACHE_ENABLED,
    reason="LITELLM_GATEWAY_CACHE not set — the gateway must run config.platform.yaml with Redis",
)
def test_identical_request_is_served_from_the_cache_the_second_time():
    """A nonce keeps earlier runs from pre-filling the entry. The first call
    goes upstream; the second returns the stored response (same ``id``),
    names its cache entry in ``x-litellm-cache-key`` and costs nothing."""
    prompt = f"Reply with the single word OK. Request {uuid.uuid4().hex}."

    first = _complete(prompt)
    first.raise_for_status()
    assert "x-litellm-cache-key" not in first.headers, "a fresh prompt must not be a cache hit"

    second = _complete(prompt)
    second.raise_for_status()
    assert second.headers.get("x-litellm-cache-key"), dict(second.headers)
    assert second.json()["id"] == first.json()["id"]
    assert float(second.headers["x-litellm-response-cost"]) == 0.0

    # A caller can still skip the cache for one request.
    fresh = _complete(prompt, cache={"no-cache": True})
    fresh.raise_for_status()
    assert "x-litellm-cache-key" not in fresh.headers
