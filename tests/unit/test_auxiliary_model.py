# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for auxiliary-model plumbing: AgentConfig slot + resolver."""

from __future__ import annotations

import pytest

from locus.agent.config import AgentConfig
from locus.models.auxiliary import resolve_auxiliary


class TestConfigSlot:
    def test_defaults_to_none(self) -> None:
        cfg = AgentConfig(model="openai:gpt-4o")
        assert cfg.auxiliary_model is None

    def test_accepts_string(self) -> None:
        cfg = AgentConfig(
            model="openai:gpt-4o",
            auxiliary_model="openai:gpt-4o-mini",
        )
        assert cfg.auxiliary_model == "openai:gpt-4o-mini"

    def test_accepts_arbitrary_model_instance(self) -> None:
        # ``arbitrary_types_allowed=True`` lets users pass a
        # ModelProtocol instance. Use a stand-in to avoid importing
        # a concrete provider in unit tests.
        class _StubModel:
            name = "stub"

        stub = _StubModel()
        cfg = AgentConfig(model="openai:gpt-4o", auxiliary_model=stub)
        assert cfg.auxiliary_model is stub


class TestResolveAuxiliary:
    def test_auxiliary_wins_when_set(self) -> None:
        assert (
            resolve_auxiliary(
                primary="openai:gpt-4o",
                auxiliary="openai:gpt-4o-mini",
            )
            == "openai:gpt-4o-mini"
        )

    def test_falls_back_to_primary_when_none(self) -> None:
        assert resolve_auxiliary(primary="openai:gpt-4o", auxiliary=None) == "openai:gpt-4o"

    def test_model_instance_works_both_slots(self) -> None:
        class _Primary:
            name = "primary"

        class _Aux:
            name = "aux"

        p = _Primary()
        a = _Aux()
        assert resolve_auxiliary(primary=p, auxiliary=a) is a
        assert resolve_auxiliary(primary=p, auxiliary=None) is p

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="no auxiliary or primary"):
            resolve_auxiliary(primary=None, auxiliary=None)
