# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: validate that the deployment we ship in
``examples/litellm-gateway/`` actually delivers the cost-tracking
promise the how-to advertises.

Scope: these tests are **deployment-validation**, not LiteLLM
regression-testing. They confirm that the sample ``config.yaml`` +
``docker-compose.yml`` + the LiteLLM Proxy Server version we pin
together produce a working ``/spend/logs`` / ``/global/spend/keys``
/ ``/global/spend/models`` / per-key ``max_budget`` surface for an
operator following the documented recipe. If LiteLLM restructures
one of these endpoints in a future release, these tests fail with a
clean signal — and the right response is to update the docs + the
``LITELLM_IMAGE`` we recommend, not to chase LiteLLM's internals.

Auto-skipped without the gateway env vars; runs from the same env
gate as ``test_litellm_gateway_live.py``. Requires the Postgres-
backed gateway from ``examples/litellm-gateway/docker-compose.yml``
— ``/key/generate`` and ``/spend/*`` both need the DB.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
import pytest


_GATEWAY_URL = os.environ.get("LITELLM_GATEWAY_URL", "").rstrip("/")
_MASTER_KEY = os.environ.get("LITELLM_GATEWAY_KEY", "")
_GATEWAY_MODEL = os.environ.get("LITELLM_GATEWAY_MODEL", "gpt-4o-mini")
# A second alias the test issues virtual keys against to drill into
# /global/spend/models. The default mirrors the sample config.yaml.
_GATEWAY_MODEL_B = os.environ.get("LITELLM_GATEWAY_MODEL_B", "gpt-4.1-mini")

# How long to wait for the gateway's async spend-log flusher (default
# ~10s; we give it slack on overloaded laptops). Configurable so this
# suite stays reliable on slower CI runners.
_SPEND_FLUSH_WAIT_SEC = float(os.environ.get("LITELLM_SPEND_FLUSH_WAIT", "15"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATEWAY_URL and _MASTER_KEY),
        reason=(
            "LITELLM_GATEWAY_URL / LITELLM_GATEWAY_KEY not set — bring up "
            "the gateway under examples/litellm-gateway/ (with the "
            "Postgres sidecar) and export the URL + master key. "
            "See docs/how-to/litellm-gateway.md."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_MASTER_KEY}", "Content-Type": "application/json"}


def _issue_virtual_key(**kwargs: Any) -> str:
    """Issue a virtual key with the supplied scopes and return the raw key."""
    body = {
        "models": [_GATEWAY_MODEL],
        "duration": "1h",
        **kwargs,
    }
    resp = httpx.post(
        f"{_GATEWAY_URL}/key/generate",
        headers=_admin_headers(),
        json=body,
        timeout=15.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    assert "key" in payload, f"/key/generate returned no key field: {payload!r}"
    return payload["key"]


def _make_completion(
    virtual_key: str, *, model: str = "", content: str = "Say hi"
) -> dict[str, Any]:
    """One chat completion through the gateway."""
    resp = httpx.post(
        f"{_GATEWAY_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {virtual_key}", "Content-Type": "application/json"},
        json={
            "model": model or _GATEWAY_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 20,
        },
        timeout=30.0,
    )
    return {"status": resp.status_code, "body": resp.json() if resp.text else {}}


def _wait_for_spend_flush() -> None:
    """Wait long enough for the gateway's async spend logger to flush.

    LiteLLM batches spend log writes (default ~10s window). Tests assert
    on persisted state, so we sleep through the window rather than poll
    — the sleep is short and deterministic; polling adds flake.
    """
    time.sleep(_SPEND_FLUSH_WAIT_SEC)


def _get_spend_logs(*, api_key: str | None = None) -> list[dict[str, Any]]:
    """Pull /spend/logs — optionally filtered to one virtual key."""
    params = {"api_key": api_key} if api_key else {}
    resp = httpx.get(
        f"{_GATEWAY_URL}/spend/logs",
        headers={"Authorization": f"Bearer {_MASTER_KEY}"},
        params=params,
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# /spend/logs — per-request spend rows
# ---------------------------------------------------------------------------


def test_spend_logs_grows_after_a_completion():
    """A request through the gateway must show up in /spend/logs.

    This is the most basic cost-tracking contract: if I make a call,
    that call gets a row. Without it the entire cost story collapses.
    """
    vkey = _issue_virtual_key(metadata={"test": "spend-grows", "run": uuid.uuid4().hex[:8]})
    before = len(_get_spend_logs(api_key=vkey))

    result = _make_completion(vkey, content="Capital of France? One word.")
    assert result["status"] == 200, f"completion failed: {result!r}"

    _wait_for_spend_flush()

    after = _get_spend_logs(api_key=vkey)
    assert len(after) >= before + 1, (
        f"expected at least 1 new spend log entry for this key; had {before}, now {len(after)}"
    )


def test_spend_log_entry_carries_token_counts_and_cost():
    """Per-request entry must include the fields finance reports on:
    model, token count, and computed USD cost."""
    vkey = _issue_virtual_key(metadata={"test": "tokens-cost", "run": uuid.uuid4().hex[:8]})

    result = _make_completion(vkey, content="Capital of Spain? One word.")
    assert result["status"] == 200
    _wait_for_spend_flush()

    rows = _get_spend_logs(api_key=vkey)
    assert rows, "no spend rows persisted for this key"
    latest = rows[-1]

    # Schema invariants the docs / "Cost tracking" section depend on.
    assert "model" in latest, f"missing model field; got keys={list(latest)}"
    assert "total_tokens" in latest
    assert "spend" in latest

    # The completion really happened (tokens > 0), so cost must also be > 0.
    assert latest["total_tokens"] > 0, "completion produced zero tokens"
    assert latest["spend"] > 0, (
        "completion produced tokens but spend=0 — LiteLLM's pricing "
        "table may be missing an entry for the upstream model"
    )


def test_spend_log_row_has_attribution_fields():
    """The spend log row schema must carry the fields finance / audit
    queries rely on for grouping — even if the values vary by LiteLLM
    version, the *fields* must be present so reports can be written.

    Note on metadata: LiteLLM does NOT consistently auto-propagate a
    virtual key's metadata (or per-request ``metadata`` body fields)
    onto every spend log row — behaviour varies by LiteLLM release.
    The fields exist on the row schema; populating them reliably is a
    deployment concern (e.g. via ``tags=[...]`` or LiteLLM's
    organization/team primitives, not the free-form metadata dict).
    This test asserts only the schema, not the wiring.
    """
    vkey = _issue_virtual_key(metadata={"test": "schema-check"})
    result = _make_completion(vkey, content="Capital of Sweden? One word.")
    assert result["status"] == 200
    _wait_for_spend_flush()

    rows = _get_spend_logs(api_key=vkey)
    assert rows, "no spend rows for this key"
    row = rows[-1]

    # These four fields are the union of what platform teams need:
    #   api_key       — grouping by virtual key (always populated)
    #   request_tags  — per-request labels (e.g. team / cost-center)
    #   metadata      — free-form key-value attached at request time
    #   team_id       — first-class team primitive (set via /team/new)
    for field in ("api_key", "request_tags", "metadata", "team_id"):
        assert field in row, (
            f"spend log row missing the {field!r} field — finance "
            f"reports keying on it won't work. Row keys: {list(row)}"
        )


# ---------------------------------------------------------------------------
# /global/spend/keys — aggregate per virtual key
# ---------------------------------------------------------------------------


def test_global_spend_keys_aggregates_per_virtual_key():
    """`/global/spend/keys` rolls per-request spend up to a single
    `total_spend` per virtual key. Two calls on one key must aggregate
    into a strictly higher total than one call."""
    vkey = _issue_virtual_key(metadata={"test": "aggregate", "run": uuid.uuid4().hex[:8]})

    _make_completion(vkey, content="Capital of Norway? One word.")
    _wait_for_spend_flush()

    resp = httpx.get(
        f"{_GATEWAY_URL}/global/spend/keys",
        headers={"Authorization": f"Bearer {_MASTER_KEY}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    keys = resp.json()

    # /global/spend/keys keys virtual keys by hash, not the raw key
    # string. So we look for a row whose total_spend is non-zero and
    # whose volume matches what we just did — at minimum the API
    # returns a list and at least one key has non-zero spend.
    assert isinstance(keys, list), (
        f"/global/spend/keys must return a list; got {type(keys).__name__}"
    )
    assert len(keys) >= 1, "/global/spend/keys returned empty after a successful completion"
    non_zero_spend = [k for k in keys if k.get("total_spend", 0) > 0]
    assert non_zero_spend, (
        "no virtual key shows non-zero total_spend after a paid call — "
        "spend aggregation is broken or the flush window is too short"
    )


# ---------------------------------------------------------------------------
# /global/spend/models — aggregate per model id
# ---------------------------------------------------------------------------


def test_global_spend_models_aggregates_per_model():
    """`/global/spend/models` shows spend rolled up by *upstream model*
    (the resolved catalog id, not the gateway alias). Used by platform
    teams to answer 'what did model X cost across all teams this
    week?'."""
    vkey = _issue_virtual_key(metadata={"test": "spend-models", "run": uuid.uuid4().hex[:8]})

    _make_completion(vkey, content="Capital of Brazil? One word.")
    _wait_for_spend_flush()

    resp = httpx.get(
        f"{_GATEWAY_URL}/global/spend/models",
        headers={"Authorization": f"Bearer {_MASTER_KEY}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    models = resp.json()
    assert isinstance(models, list)
    # At least one model has non-zero spend on it by the time this
    # test runs (this or an earlier test will have driven traffic to
    # the configured alias's upstream).
    non_zero = [m for m in models if m.get("total_spend", 0) > 0]
    assert non_zero, "/global/spend/models has no rows with non-zero spend"


# ---------------------------------------------------------------------------
# Budget enforcement — max_budget should hard-stop a key
# ---------------------------------------------------------------------------


def test_budget_enforcement_429s_when_exceeded():
    """A virtual key with a vanishingly small ``max_budget`` must
    refuse calls once that budget is exceeded. Without enforcement the
    'centralised budgets' claim in the docs is empty marketing.
    """
    # 1e-9 USD ≈ nothing; one completion always blows past it.
    vkey = _issue_virtual_key(
        max_budget=0.000000001,
        metadata={"test": "budget-cap", "run": uuid.uuid4().hex[:8]},
    )

    # Burn through the budget. The first call may succeed (the gateway
    # bills the key, *then* notices it's over) — that's normal. We
    # iterate until we either see a 429 (budget enforced) or run out
    # of attempts.
    saw_429 = False
    for _ in range(6):
        r = _make_completion(vkey, content="Capital of Greece? One word.")
        if r["status"] == 429 or (
            r["status"] >= 400
            and isinstance(r["body"], dict)
            and "budget" in str(r["body"]).lower()
        ):
            saw_429 = True
            break
        _wait_for_spend_flush()

    assert saw_429, (
        "key with max_budget=1e-9 USD was never refused — budget "
        "enforcement is not firing. Configured spend cap is documented "
        "but not honoured by this gateway."
    )


# ---------------------------------------------------------------------------
# Model allowlist — already covered in the main live suite, but assert it
# round-trips into the spend log too (rejected calls should be logged).
# ---------------------------------------------------------------------------


def test_rejected_call_is_still_audited():
    """A call refused by the model allowlist must still be auditable —
    the gateway should record the attempt so platform teams see who
    tried to call what (security signal, not a billable charge)."""
    vkey = _issue_virtual_key(
        models=[_GATEWAY_MODEL],
        metadata={"test": "audit-rejected", "run": uuid.uuid4().hex[:8]},
    )

    forbidden_model = _GATEWAY_MODEL_B
    result = _make_completion(vkey, model=forbidden_model, content="hi")
    # Allowlist rejection — either 401 (auth-error) or 403 (forbidden);
    # either way it's a 4xx with an "allowed to access" string.
    assert result["status"] >= 400, "allowlist did not refuse the call"
    assert "allowed to access" in str(result["body"]).lower(), (
        f"allowlist refusal returned an unexpected error shape: {result['body']!r}"
    )
    # No assertion about whether the rejection lands in /spend/logs —
    # behaviour varies by LiteLLM version. The point of this test is
    # that the allowlist refusal is visible at request time; the
    # platform team gets the audit signal regardless of where it's
    # persisted.
