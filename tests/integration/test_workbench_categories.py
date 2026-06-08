# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Tests for the workbench /categories endpoints.

The workbench backend serves three catalogues — notebooks, skills,
protocols — each with a parallel ``/categories`` endpoint that feeds
the sidebar's topic-progression section headers. The contract:

* ``GET /api/{noun}/categories`` returns
  ``[{"id", "name", "description"}, ...]`` in declaration order.
* Every list item's ``category`` field is the id of one of those
  categories OR a fallback sentinel (``"misc"`` for notebooks,
  ``"other"`` for skills/protocols).
* The user-facing nav rests on a curated set of category ids
  (``fundamentals``, ``graphs``, ..., ``observability``) — that set
  must stay in sync with the workbench docs.

These tests run via FastAPI's ``TestClient`` straight against the
runner module, so they execute in CI without needing the live
workbench process up.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


WORKBENCH_BACKEND = (Path(__file__).resolve().parents[2] / "workbench" / "backend").resolve()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Spin up a TestClient against the runner module — no need for the
    live BFF / Vite / FastAPI processes."""
    sys.path.insert(0, str(WORKBENCH_BACKEND))
    try:
        import runner  # type: ignore[import-not-found]
    finally:
        # Don't pollute sys.path beyond this fixture's scope.
        if str(WORKBENCH_BACKEND) in sys.path:
            sys.path.remove(str(WORKBENCH_BACKEND))
    return TestClient(runner.app)


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------


class TestNotebookCategories:
    def test_endpoint_returns_curated_categories(self, client: TestClient) -> None:
        r = client.get("/api/notebooks/categories")
        assert r.status_code == 200, r.text
        cats = r.json()
        assert cats, "expected at least one category"
        ids = {c["id"] for c in cats}
        # Must include the cardinal sections — these power the user-
        # facing learning path. Drift here = the README / nav docs are
        # describing categories that no longer exist.
        #
        # ``router-observability`` is a single combined category — the
        # cognitive router and the EventBus observability surface ship
        # together as one learning track, and the workbench reflects that
        # in its NOTEBOOK_CATEGORIES list.
        for required in (
            "fundamentals",
            "graphs",
            "multi-agent",
            "router-observability",
        ):
            assert required in ids, f"missing notebook category: {required}"
        for c in cats:
            assert c["name"], f"category {c['id']} has empty name"
            assert c["description"], f"category {c['id']} has empty description"

    def test_every_notebook_has_known_category(self, client: TestClient) -> None:
        cats = {c["id"] for c in client.get("/api/notebooks/categories").json()}
        cats.add("misc")
        for t in client.get("/api/notebooks").json():
            assert t.get("category") in cats, (
                f"notebook {t['id']} has unknown category {t.get('category')!r}"
            )

    def test_router_observability_groups_router_plus_eventbus(self, client: TestClient) -> None:
        """The combined ``router-observability`` track must surface the
        cognitive router (notebook 58) and the EventBus / observability
        notebooks (59, 60, 61) as a single sidebar group. Drift here
        means the curated learning path lost a notebook to ``misc``."""
        track_numbers = sorted(
            t["number"]
            for t in client.get("/api/notebooks").json()
            if t.get("category") == "router-observability"
        )
        for n in (58, 59, 60, 61):
            assert n in track_numbers, (
                f"notebook {n} missing from 'router-observability' (got {track_numbers})"
            )

    def test_notebooks_sorted_by_category_then_order(self, client: TestClient) -> None:
        """The catalogue is pre-sorted by (category position,
        category_order, number). Two consecutive notebooks in the same
        category must have non-decreasing ``category_order``."""
        ts = client.get("/api/notebooks").json()
        for prev, curr in pairwise(ts):
            if prev["category"] == curr["category"]:
                assert prev.get("category_order", 0) <= curr.get("category_order", 0), (
                    f"{prev['id']} and {curr['id']} are out of order in {curr['category']}"
                )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class TestSkillCategories:
    def test_endpoint_returns_categories(self, client: TestClient) -> None:
        r = client.get("/api/skills/categories")
        assert r.status_code == 200, r.text
        cats = r.json()
        ids = {c["id"] for c in cats}
        # Three curated buckets: engineering / operations / data.
        for required in ("engineering", "operations", "data"):
            assert required in ids, f"missing skill category: {required}"

    def test_every_skill_has_known_category(self, client: TestClient) -> None:
        cats = {c["id"] for c in client.get("/api/skills/categories").json()}
        cats.add("other")
        for s in client.get("/api/skills").json():
            assert s.get("category") in cats, (
                f"skill {s['id']} has unknown category {s.get('category')!r}"
            )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TestProtocolCategories:
    def test_endpoint_returns_categories(self, client: TestClient) -> None:
        r = client.get("/api/protocols/categories")
        assert r.status_code == 200, r.text
        cats = r.json()
        ids = {c["id"] for c in cats}
        for required in ("single", "linear", "parallel", "delegation", "gated"):
            assert required in ids, f"missing protocol category: {required}"

    def test_no_protocol_falls_through_to_other(self, client: TestClient) -> None:
        """Every built-in protocol is curated by hand — none should land
        on the ``other`` sentinel."""
        for p in client.get("/api/protocols").json():
            assert p.get("category") != "other", (
                f"protocol {p['id']} fell through — add it to PROTOCOL_CATEGORIES"
            )

    def test_protocols_sorted_by_category(self, client: TestClient) -> None:
        ps = client.get("/api/protocols").json()
        for prev, curr in pairwise(ps):
            if prev["category"] == curr["category"]:
                assert prev.get("category_order", 0) <= curr.get("category_order", 0), (
                    f"{prev['id']} and {curr['id']} are out of order in {curr['category']}"
                )
