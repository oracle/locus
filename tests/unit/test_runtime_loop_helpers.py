# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for the *helpers* in ``locus.agent.runtime_loop``.

The runtime loop methods themselves require a fully-initialised ``Agent``
to drive — those are exercised by the agent end-to-end tests. This file
hits the small, self-contained pieces:

- ``_normalize_stop_reason`` — maps free-form reasons to the typed Literal
"""

from __future__ import annotations

import pytest

from locus.agent.runtime_loop import _normalize_stop_reason


class TestNormalizeStopReason:
    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_falsy_input_defaults_to_complete(self, raw: str | None) -> None:
        # The implementation only treats falsy values (None / "") as default.
        # A non-empty whitespace string is *not* falsy and falls through to
        # the substring search, which then falls through to "complete" too.
        assert _normalize_stop_reason(raw) == "complete"

    @pytest.mark.parametrize(
        "reason",
        [
            "complete",
            "terminal_tool",
            "confidence_met",
            "max_iterations",
            "tool_loop",
            "no_tools",
            "grounding_failed",
            "token_budget",
            "time_budget",
            "interrupted",
            "error",
            "cancelled",
        ],
    )
    def test_exact_match_returned(self, reason: str) -> None:
        assert _normalize_stop_reason(reason) == reason

    def test_tool_called_prefix_routes_to_terminal_tool(self) -> None:
        assert _normalize_stop_reason("tool_called:terminate_session") == "terminal_tool"

    def test_text_mention_prefix_routes_to_complete(self) -> None:
        assert _normalize_stop_reason("text_mention:DONE") == "complete"

    def test_substring_known_reason(self) -> None:
        # Free-form reason that contains a known token should be normalised
        # to the matching value.
        assert _normalize_stop_reason("hit max_iterations limit") == "max_iterations"
        assert _normalize_stop_reason("error talking to model") == "error"

    def test_unknown_reason_falls_back_to_complete(self) -> None:
        assert _normalize_stop_reason("nonsense reason") == "complete"
