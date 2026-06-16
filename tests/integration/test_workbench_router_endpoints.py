# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""BFF ↔ SDK contract tests for the workbench router endpoints.

These tests prove the workbench surface stays consistent with the SDK
source of truth. They hit the locally-running BFF (default
``http://127.0.0.1:3101``) and compare its responses against:

* :func:`locus.router.builtin_protocols` for ``/api/protocols``
* :class:`locus.skills.Skill.from_directory` for ``/api/skills``

If the BFF isn't reachable, the entire module is skipped — these are
end-to-end tests that depend on the workbench being up. The intent is
that any drift between the BFF's catalogue endpoints and the SDK
surface fails CI loudly the moment somebody changes one without the
other.

Run with: ``pytest tests/integration/test_workbench_router_endpoints.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from locus.router import builtin_protocols


pytestmark = pytest.mark.integration


BFF_URL = os.getenv("BFF_URL", "http://127.0.0.1:3101")
SKILLS_DIR = (Path(__file__).resolve().parents[2] / "examples" / "skills").resolve()


def _bff_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{BFF_URL}/api/protocols")
            return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not _bff_reachable(),
        reason=f"workbench BFF not reachable at {BFF_URL} — start it with `npm run dev`",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http() -> httpx.Client:
    with httpx.Client(timeout=10.0, base_url=BFF_URL) as client:
        yield client


@pytest.fixture(scope="module")
def sdk_protocols() -> dict[str, object]:
    """Map of protocol_id → Protocol instance, straight from the SDK."""
    return {p.id: p for p in builtin_protocols()}


# ---------------------------------------------------------------------------
# /api/protocols — list endpoint.
# ---------------------------------------------------------------------------


class TestProtocolsListEndpoint:
    """The catalogue the BFF serves must mirror the SDK's
    :func:`builtin_protocols` exactly. Drift here = the workbench
    sidebar is showing protocols that don't exist (or hiding ones that
    do)."""

    def test_status_and_count(self, http: httpx.Client, sdk_protocols: dict[str, object]) -> None:
        r = http.get("/api/protocols")
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == len(sdk_protocols), (
            f"BFF returned {len(data)} protocols; SDK has {len(sdk_protocols)}. "
            f"Did somebody add/remove a builtin without updating the other side?"
        )

    def test_ids_match_sdk(self, http: httpx.Client, sdk_protocols: dict[str, object]) -> None:
        api_ids = {p["id"] for p in http.get("/api/protocols").json()}
        assert api_ids == set(sdk_protocols.keys()), (
            f"BFF protocol ids {sorted(api_ids)} do not match "
            f"SDK builtin_protocols() ids {sorted(sdk_protocols.keys())}"
        )

    def test_each_entry_has_required_fields(self, http: httpx.Client) -> None:
        required = {
            "id",
            "name",
            "description",
            "handles",
            "primary_for",
            "requires_capabilities",
            "risk_max",
            "cost",
            "latency",
            "supports_streaming",
            "supports_repair",
        }
        for p in http.get("/api/protocols").json():
            missing = required - p.keys()
            assert not missing, f"protocol {p.get('id')!r} on the BFF is missing fields: {missing}"


# ---------------------------------------------------------------------------
# /api/protocols/{pid} — detail endpoint.
# ---------------------------------------------------------------------------


class TestProtocolsDetailEndpoint:
    """Per-protocol metadata round-trip against the SDK."""

    def test_unknown_id_returns_404(self, http: httpx.Client) -> None:
        r = http.get("/api/protocols/does_not_exist")
        assert r.status_code == 404

    @pytest.mark.parametrize(
        "pid",
        [p.id for p in builtin_protocols()],
        ids=[p.id for p in builtin_protocols()],
    )
    def test_metadata_round_trips(
        self,
        http: httpx.Client,
        sdk_protocols: dict[str, object],
        pid: str,
    ) -> None:
        sdk = sdk_protocols[pid]
        api = http.get(f"/api/protocols/{pid}").json()

        # String / list / bool fields should be exact equality.
        assert api["id"] == sdk.id
        assert api["description"] == sdk.description
        assert api["risk_max"] == sdk.risk_max.value
        assert api["cost"] == sdk.cost
        assert api["latency"] == sdk.latency
        assert api["supports_streaming"] is sdk.supports_streaming
        assert api["supports_repair"] is sdk.supports_repair
        # Order-sensitive: ``handles`` and ``primary_for`` are lists in
        # the SDK and the BFF preserves the registration order.
        assert api["handles"] == [t.value for t in sdk.handles]
        assert api["primary_for"] == [t.value for t in sdk.primary_for]
        assert api["requires_capabilities"] == list(sdk.requires_capabilities)

    @pytest.mark.parametrize(
        "pid",
        [p.id for p in builtin_protocols()],
        ids=[p.id for p in builtin_protocols()],
    )
    def test_runtime_shape_present(self, http: httpx.Client, pid: str) -> None:
        # The runtime_shape string is what the workbench shows in the
        # detail callout. Every protocol must have one — no silent
        # fallbacks. (The structural-audit suite at
        # tests/unit/test_router_compiled_shape.py pins these strings
        # to the actual compiled object graph.)
        api = http.get(f"/api/protocols/{pid}").json()
        assert "runtime_shape" in api, f"{pid!r} missing runtime_shape"
        assert api["runtime_shape"], f"{pid!r} has empty runtime_shape"
        assert api["runtime_shape"] != "(no shape recorded)", (
            f"{pid!r} fell back to the default shape — add an entry to "
            f"_RUNTIME_SHAPES in workbench/backend/runner.py"
        )


# ---------------------------------------------------------------------------
# /api/skills — list + detail endpoints.
# ---------------------------------------------------------------------------


def _filesystem_skills() -> list[str]:
    """The skill ids the loader would see, sorted, from the on-disk
    examples/skills/ directory. Used as the source of truth in the
    contract tests below."""
    if not SKILLS_DIR.is_dir():
        return []
    out: list[str] = []
    for child in sorted(SKILLS_DIR.iterdir()):
        if child.is_dir() and (child / "SKILL.md").exists():
            out.append(child.name)
    return out


class TestSkillsEndpoints:
    def test_list_matches_filesystem(self, http: httpx.Client) -> None:
        api_ids = sorted(s["id"] for s in http.get("/api/skills").json())
        fs_ids = _filesystem_skills()
        assert api_ids == fs_ids, (
            f"BFF skills {api_ids} drifted from on-disk {fs_ids} — "
            f"either a SKILL.md package was added/removed without "
            f"a backend restart or the loader silently rejected one"
        )

    def test_unknown_id_returns_404(self, http: httpx.Client) -> None:
        assert http.get("/api/skills/does_not_exist").status_code == 404

    @pytest.mark.parametrize(
        "sid",
        _filesystem_skills(),
        ids=_filesystem_skills() or ["no-skills-installed"],
    )
    def test_detail_includes_instructions_and_resources(self, http: httpx.Client, sid: str) -> None:
        if not _filesystem_skills():
            pytest.skip("no SKILL.md packages installed")
        d = http.get(f"/api/skills/{sid}").json()
        assert d["id"] == sid
        # Every well-formed SKILL.md has a non-empty description and
        # body. The loader rejects packages missing either, so the BFF
        # should never serve an empty one.
        assert d["description"], f"skill {sid!r} has empty description"
        assert d["instructions"], f"skill {sid!r} has empty instructions body"
        # ``resources`` is a list (possibly empty) of relative file paths.
        assert isinstance(d["resources"], list)


## End of live-BFF tests. ##
