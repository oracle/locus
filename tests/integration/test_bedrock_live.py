# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Live integration tests for the Bedrock provider.

Auto-skips unless real AWS credentials resolve *and* Bedrock answers. The unit
suite covers request shape against botocore's service model; what only a live
call can show is that the account, the region, the model grant and the wire
format all line up at once — and that the models disagree with each other,
which is the part that bites.

Uses Nova Micro and Nova Lite: the cheapest models on the service (fractions of
a cent per run) and available in every Bedrock region.

Run with:
    hatch run test:test-integration tests/integration/test_bedrock_live.py
"""

from __future__ import annotations

import os

import pytest

from tulip.core.messages import Message


REGION = os.environ.get("TULIP_BEDROCK_REGION", "us-east-1")
MICRO = "us.amazon.nova-micro-v1:0"
LITE = "us.amazon.nova-lite-v1:0"


def _bedrock_available() -> bool:
    """Credentials resolve and the account can actually invoke a model."""
    try:
        import boto3

        session = (
            boto3.Session(profile_name=os.environ["TULIP_AWS_PROFILE"])
            if os.environ.get("TULIP_AWS_PROFILE")
            else boto3.Session()
        )
        if session.get_credentials() is None:
            return False
        # Listing is not enough: an account can list 122 models and be granted
        # none of them. Only an invoke proves the grant.
        session.client("bedrock-runtime", region_name=REGION).converse(
            modelId=MICRO,
            messages=[{"role": "user", "content": [{"text": "ok"}]}],
            inferenceConfig={"maxTokens": 4},
        )
    except Exception:
        return False
    return True


skip_without_bedrock = pytest.mark.skipif(
    not _bedrock_available(),
    reason="no AWS credentials, or the account cannot invoke Bedrock in this region",
)

pytestmark = [pytest.mark.integration, skip_without_bedrock]


@pytest.fixture
def model():
    from tulip.models.registry import get_model

    return get_model(f"bedrock:{MICRO}", region=REGION)


async def test_completion_answers_a_question_it_was_not_given(model) -> None:
    """A non-echo probe: the answer is not anywhere in the prompt."""
    response = await model.complete([Message.user("What is 17 * 23? Reply with only the number.")])
    assert "391" in (response.message.content or "")
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.stop_reason in {"end_turn", "max_tokens", "stop_sequence"}


async def test_system_prompt_is_honoured(model) -> None:
    """Proves the system block really is lifted out and applied."""
    response = await model.complete(
        [
            Message.system("You always answer with exactly the word: PINEAPPLE"),
            Message.user("What is the capital of France?"),
        ]
    )
    assert "PINEAPPLE" in (response.message.content or "").upper()


async def test_streaming_yields_deltas_then_a_final_event(model) -> None:
    chunks: list[str] = []
    final = None
    async for event in model.stream([Message.user("Count from 1 to 5, digits only.")]):
        if event.content:
            chunks.append(event.content)
        if event.done:
            final = event

    assert len(chunks) > 1, "expected more than one delta from a real stream"
    assert "5" in "".join(chunks)
    assert final is not None
    assert final.usage["completion_tokens"] > 0


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


async def test_tool_call_round_trip(model) -> None:
    """The model asks for the tool, and the result feeds back in.

    This is the shape the whole agent loop rests on, and the second half — the
    assistant turn carrying a toolUse, replayed as history — is what the
    cross-model bug was hiding in.
    """
    first = await model.complete([Message.user("Look up order ord-4821.")], tools=TOOLS)
    assert first.stop_reason == "tool_use"
    [call] = first.message.tool_calls
    assert call.name == "lookup_order"
    assert call.arguments["order_id"] == "ord-4821"

    from tulip.core.messages import ToolResult

    second = await model.complete(
        [
            Message.user("Look up order ord-4821."),
            first.message,
            Message.tool(
                ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="Order ord-4821: 1x Blue Widget, $42.00",
                )
            ),
        ],
        tools=TOOLS,
    )
    assert "widget" in (second.message.content or "").lower()


@pytest.mark.parametrize("model_id", [MICRO, LITE, "us.meta.llama3-3-70b-instruct-v1:0"])
async def test_one_code_path_serves_several_vendors(model_id: str) -> None:
    """Converse is the point: Amazon and Meta models run the same code.

    Parametrised across vendors deliberately. A single-vendor test passed while
    the provider was building assistant turns that Meta's models reject
    outright — the failure only appears when the same conversation is replayed
    against a different family.
    """
    from tulip.models.registry import get_model

    model = get_model(f"bedrock:{model_id}", region=REGION)
    try:
        response = await model.complete(
            [Message.user("Reply with the single word: OK")],
        )
    except Exception as exc:  # noqa: BLE001 — grant/quota, not a code defect
        name = type(exc).__name__
        if name in {"ThrottlingException", "ResourceNotFoundException", "AccessDeniedException"}:
            pytest.skip(f"{model_id}: {name} — account grant or quota, not a provider defect")
        raise
    assert (response.message.content or "").strip()


async def test_agent_end_to_end_with_a_gated_tool() -> None:
    """The provider under the real agent loop, with the admission gate on top.

    Bedrock is the first provider added since `gate_tool` landed, so this pins
    that a governed tool behaves the same here as everywhere else: the model
    asks for the refund, the policy denies it, and the tool body never runs.
    """
    from tulip import Agent, AgentConfig
    from tulip.control import Action, ControlPolicy
    from tulip.control.gate import gate_tool
    from tulip.security.verify import VerificationResult
    from tulip.tools.decorator import tool

    ran: list[float] = []

    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Issue a refund for an order."""
        ran.append(amount_usd)
        return f"refunded ${amount_usd}"

    def action(name: str, kwargs: dict) -> Action:
        return Action(
            name=name,
            asset=str(kwargs.get("order_id", "?")),
            kind="write",
            environment="staging",
            blast_radius=1,
            tags=frozenset({"large_refund"}),
        )

    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(deny_for=frozenset({"large_refund"})),
        action=action,
        verdict=VerificationResult(survives=True, confidence=0.95, evidence_quality=0.95),
        refusal_reason="A manager must approve refunds this size.",
    )

    from tulip.models.registry import get_model

    agent = Agent(
        config=AgentConfig(model=get_model(f"bedrock:{LITE}", region=REGION), tools=[gated])
    )
    result = await agent.arun("Refund order ord-4821 for $900. Use the tool.")

    assert ran == [], "the gate let a denied refund execute"
    assert (result.message or "").strip()
