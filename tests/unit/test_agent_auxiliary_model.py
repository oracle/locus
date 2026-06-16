# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``AgentConfig.auxiliary_model`` wiring.

The field existed for months and was tested in isolation
(``test_auxiliary_model.py`` covers ``resolve_auxiliary``) but the agent
loop never read it. These tests guard the wiring:

- The agent resolves the auxiliary model on init when set.
- Grounding eval uses the auxiliary model when ``grounding.model`` is None.
- Grounding eval still prefers ``grounding.model`` when explicitly set.
- The max-iterations summary call routes through the auxiliary model.
"""

from __future__ import annotations

from typing import Any

from locus.agent import Agent
from locus.agent.config import GroundingConfig
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelResponse


class _LabelledModel:
    """Test double that tags every response with its own ``label`` so callers
    can inspect which model was hit.

    This is the cheapest way to assert "the auxiliary model handled this
    call, not the primary": both models are scripted, the response carries
    the model identity, and we read it back off the message content.
    """

    def __init__(self, label: str, responses: list[ModelResponse] | None = None):
        self.label = label
        self.calls = 0
        self._responses = list(responses) if responses else []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        # Pop scripted responses one-by-one. After they're exhausted,
        # emit a labelled assistant text so the test can detect which
        # model was hit.
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(
            message=Message.assistant(content=f"[{self.label}] reply #{self.calls}"),
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _assist_with_tool(tool_name: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=None,
            tool_calls=[ToolCall(name=tool_name, arguments={})],
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


# =============================================================================
# Resolution at init time
# =============================================================================


class TestAuxiliaryResolution:
    def test_unset_falls_back_to_primary(self):
        primary = _LabelledModel("primary")
        agent = Agent(model=primary)
        assert agent._auxiliary_model is primary

    def test_explicit_instance_kept(self):
        primary = _LabelledModel("primary")
        aux = _LabelledModel("aux")
        agent = Agent(model=primary, auxiliary_model=aux)
        assert agent._auxiliary_model is aux

    def test_grounding_uses_auxiliary_when_grounding_model_unset(self):
        primary = _LabelledModel("primary")
        aux = _LabelledModel("aux")
        agent = Agent(
            model=primary,
            auxiliary_model=aux,
            grounding=GroundingConfig(),  # no `model=` -> use auxiliary
        )
        assert agent._grounding_model is aux

    def test_grounding_model_overrides_auxiliary(self):
        # Given grounding.model=string, the agent calls get_model on it.
        # We can't easily inject a string-resolvable model from a unit test
        # without monkeypatching, so guard the precedence by direct
        # observation: even when auxiliary_model is set, grounding.model
        # wins. We use a callable model identity check.
        from unittest.mock import patch

        primary = _LabelledModel("primary")
        aux = _LabelledModel("aux")
        explicit = _LabelledModel("grounding-explicit")

        with patch("locus.agent.agent.get_model", return_value=explicit) as mocked:
            agent = Agent(
                model=primary,
                auxiliary_model=aux,
                grounding=GroundingConfig(model="some-explicit:id"),
            )
        assert agent._grounding_model is explicit
        mocked.assert_called_once_with("some-explicit:id")


# =============================================================================
# Max-iterations summary routing
# =============================================================================


class TestMaxIterationsSummaryRouting:
    def test_summary_call_uses_auxiliary_model(self):
        """When max_iterations is reached and the agent emits a final summary,
        that summary call must go to the auxiliary model — saves primary budget.
        """

        # Primary keeps calling a tool forever; built-in max_iterations
        # triggers the summary call path which we want routed to aux.
        @__import__("locus.tools.decorator", fromlist=["tool"]).tool
        def echo(msg: str) -> str:  # type: ignore[no-redef]
            return f"echo: {msg}"

        # Provide enough tool-call responses for the primary loop
        # iterations; after that the labelled fallback kicks in (but the
        # summary should be routed to aux, so primary's fallback should
        # NEVER be hit).
        primary = _LabelledModel(
            "primary",
            responses=[_assist_with_tool("echo")] * 3,
        )
        aux = _LabelledModel("aux")  # uses fallback labelled responses

        agent = Agent(
            model=primary,
            auxiliary_model=aux,
            tools=[echo],
            # Force the max_iterations summary path. The user-supplied
            # MaxIterations termination wins before built-in cap; we need
            # the BUILT-IN max_iterations path for the summary call, so do
            # NOT set termination here.
            max_iterations=2,
        )
        result = agent.run_sync("loop forever")

        # The aux model should have received exactly one summary call.
        assert aux.calls >= 1, (
            f"auxiliary model wasn't called for the summary "
            f"(primary.calls={primary.calls}, aux.calls={aux.calls})"
        )
        # And the final message should reflect the aux model's reply.
        assert "[aux]" in result.message

    def test_summary_call_falls_back_to_primary_when_no_aux(self):
        """Without an auxiliary model, the summary uses the primary."""

        @__import__("locus.tools.decorator", fromlist=["tool"]).tool
        def echo2(msg: str) -> str:  # type: ignore[no-redef]
            return f"echo: {msg}"

        # 2 tool-calls for the loop iterations; primary's labelled fallback
        # then covers the max_iterations summary call.
        primary = _LabelledModel("primary", responses=[_assist_with_tool("echo2")] * 2)
        agent = Agent(model=primary, tools=[echo2], max_iterations=2)
        result = agent.run_sync("loop")
        # Primary handled the summary too — final message tagged "primary".
        assert "[primary]" in result.message
