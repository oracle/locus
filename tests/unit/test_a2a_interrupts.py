# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A paused Locus run on the A2A wire: ``input-required``, the reply, the resume.

Two kinds of pause: a question (a tool that returns the interrupt marker, as ``ask_user`` does) and an approval hold from
``gate_tool(on_refusal="interrupt")``. The remote peer answers by sending a
message on the same task; for a hold the first word decides it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from locus.a2a import A2AClient, A2AClientError, A2AServer, Message, TaskState, TextPart
from locus.a2a.spec import INVALID_PARAMS, UNSUPPORTED_OPERATION
from locus.agent import Agent
from locus.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from locus.core.messages import Role
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


@tool
def ask_region(question: str, options: str | None = None) -> str:
    """Pause the run for a person: the interrupt marker any tool may return."""
    option_list = [o.strip() for o in options.split(",")] if options else None
    return json.dumps({"__interrupt__": True, "question": question, "options": option_list})


def _http(server: A2AServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )


def _message(
    content: str, *, task_id: str | None = None, metadata: dict[str, Any] | None = None
) -> Message:
    return Message(
        role="user",
        parts=[TextPart(text=content)],
        messageId=f"m-{content[:8]}",
        taskId=task_id,
        metadata=metadata,
    )


def _question_agent() -> Agent:
    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        answers = [m.content for m in messages if m.role == Role.TOOL]
        if answers:
            return text(f"Deployed to {answers[-1]}.")
        return tool_call(
            "ask_region", question="Which region?", options="eu-frankfurt-1,us-ashburn-1"
        )

    return Agent(model=FunctionModel(handler), tools=[ask_region], reflexion=False, grounding=False)


def _held_agent(store: InMemoryApprovals, performed: list[str]) -> Agent:
    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        performed.append(order_id)
        return f"refunded {amount_usd} on {order_id}"

    refund = gate_tool(
        issue_refund,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"payment"})
        ),
        action=lambda name, kwargs: Action(name=name, asset=kwargs["order_id"], kind="payment"),
        approval=store,
        on_refusal="interrupt",
        principal="svc-billing",
    )

    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        results = [m.content or "" for m in messages if m.role == Role.TOOL]
        if results:
            return text(
                "refund issued" if "refunded" in results[-1] else "refund was not performed"
            )
        return tool_call("issue_refund", order_id="o1", amount_usd=250.0)

    return Agent(model=FunctionModel(handler), tools=[refund], reflexion=False, grounding=False)


class TestQuestion:
    @pytest.mark.parametrize("binding", ["jsonrpc", "http+json"])
    async def test_ask_user_maps_to_input_required_and_the_reply_resumes(
        self, binding: str
    ) -> None:
        server = A2AServer(agent=_question_agent(), allow_unauthenticated=True)
        async with _http(server) as http:
            client = A2AClient("http://testserver", binding=binding, http_client=http)
            task = await client.send_message(_message("deploy the service"))
            assert task.status.state == TaskState.input_required
            assert task.status.message is not None
            assert task.status.message.parts[0].text == "Which region?"  # type: ignore[union-attr]
            meta = task.status.message.metadata or {}
            assert meta["kind"] == "question"
            assert meta["options"] == ["eu-frankfurt-1", "us-ashburn-1"]
            assert meta["interruptId"]
            assert task.history[-1].role == "agent"

            listed, _ = await client.list_tasks(status=TaskState.input_required)
            assert [t.id for t in listed] == [task.id]

            resumed = await client.send_message(_message("eu-frankfurt-1", task_id=task.id))
            assert resumed.id == task.id
            assert resumed.status.state == TaskState.completed
            assert resumed.artifacts[-1].parts[0].text == "Deployed to eu-frankfurt-1."  # type: ignore[union-attr]
            assert [m.role for m in resumed.history] == ["user", "agent", "user"]

    async def test_streaming_pause_and_streamed_resume(self) -> None:
        server = A2AServer(agent=_question_agent(), allow_unauthenticated=True)
        async with _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            frames = [f async for f in client.send_message_streaming(_message("deploy"))]
            assert frames[-1]["kind"] == "status-update"
            assert frames[-1]["status"]["state"] == "input-required"
            task_id = frames[0]["id"]

            snapshot = [f async for f in client.subscribe(task_id)]
            assert [f["kind"] for f in snapshot] == ["task"]
            assert snapshot[0]["status"]["state"] == "input-required"

            resumed = [
                f
                async for f in client.send_message_streaming(
                    _message("us-ashburn-1", task_id=task_id)
                )
            ]
            assert resumed[0]["kind"] == "task"
            assert resumed[0]["status"]["state"] == "input-required"
            assert resumed[-1]["status"]["state"] == "completed"
            assert any(f["kind"] == "artifact-update" for f in resumed)

    async def test_cancel_while_paused(self) -> None:
        server = A2AServer(agent=_question_agent(), allow_unauthenticated=True)
        async with _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("deploy"))
            cancelled = await client.cancel_task(task.id)
            assert cancelled.status.state == TaskState.canceled
            with pytest.raises(A2AClientError) as err:
                await client.send_message(_message("eu-frankfurt-1", task_id=task.id))
            assert err.value.code == UNSUPPORTED_OPERATION


class TestApprovalHold:
    async def test_approve_performs_the_held_call_once(self) -> None:
        store = InMemoryApprovals()
        performed: list[str] = []
        server = A2AServer(
            agent=_held_agent(store, performed), allow_unauthenticated=True, approvals=store
        )
        async with _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("refund order o1"))
            assert task.status.state == TaskState.input_required
            meta = (task.status.message.metadata if task.status.message else None) or {}
            assert meta["kind"] == "approval"
            assert meta["replies"] == ["approve", "deny"]
            assert meta["hold"]["action"] == "issue_refund"
            assert meta["hold"]["asset"] == "o1"
            approval_id = meta["approvalId"]
            assert store.get(approval_id) is not None
            assert store.get(approval_id).status == "pending"  # type: ignore[union-attr]
            assert performed == []

            with pytest.raises(A2AClientError) as unclear:
                await client.send_message(_message("maybe later", task_id=task.id))
            assert unclear.value.code == INVALID_PARAMS
            assert (await client.get_task(task.id)).status.state == TaskState.input_required

            done = await client.send_message(
                _message("approve", task_id=task.id, metadata={"decidedBy": "alice@example.com"})
            )
            assert done.status.state == TaskState.completed
            assert done.artifacts[-1].parts[0].text == "refund issued"  # type: ignore[union-attr]
            assert performed == ["o1"]
            record = store.get(approval_id)
            assert record is not None
            assert record.status == "consumed"
            assert record.decided_by == "alice@example.com"

    async def test_deny_ends_the_task_rejected_without_performing(self) -> None:
        store = InMemoryApprovals()
        performed: list[str] = []
        server = A2AServer(
            agent=_held_agent(store, performed), allow_unauthenticated=True, approvals=store
        )
        async with _http(server) as http:
            client = A2AClient("http://testserver", binding="http+json", http_client=http)
            task = await client.send_message(_message("refund order o1"))
            approval_id = (task.status.message.metadata or {})["approvalId"]  # type: ignore[union-attr]
            denied = await client.send_message(_message("deny, too large", task_id=task.id))
            assert denied.status.state == TaskState.rejected
            assert denied.status.message is not None
            assert denied.status.message.parts[0].text.startswith("Denied by a2a:")  # type: ignore[union-attr]
            assert performed == []
            assert denied.artifacts[-1].parts[0].text == "refund was not performed"  # type: ignore[union-attr]
            record = store.get(approval_id)
            assert record is not None
            assert record.status == "consumed"

    async def test_without_an_approvals_store_the_gate_decides(self) -> None:
        store = InMemoryApprovals()
        performed: list[str] = []
        server = A2AServer(agent=_held_agent(store, performed), allow_unauthenticated=True)
        async with _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("refund order o1"))
            approval_id = (task.status.message.metadata or {})["approvalId"]  # type: ignore[union-attr]
            with pytest.raises(A2AClientError) as pending:
                await client.send_message(_message("approve", task_id=task.id))
            assert pending.value.code == UNSUPPORTED_OPERATION
            assert "has not been decided" in pending.value.message
            assert performed == []
            still = await client.get_task(task.id)
            assert still.status.state == TaskState.input_required

            store.decide(approval_id, "approved", by="ops")
            done = await client.send_message(_message("approve", task_id=task.id))
            assert done.status.state == TaskState.completed
            assert performed == ["o1"]

    async def test_a_refused_decision_keeps_the_task_paused(self) -> None:
        store = InMemoryApprovals()
        performed: list[str] = []
        server = A2AServer(
            agent=_held_agent(store, performed), allow_unauthenticated=True, approvals=store
        )
        async with _http(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message("refund order o1"))
            approval_id = (task.status.message.metadata or {})["approvalId"]  # type: ignore[union-attr]
            store.decide(approval_id, "denied", by="someone-else")
            with pytest.raises(A2AClientError) as refused:
                await client.send_message(_message("approve", task_id=task.id))
            assert refused.value.code == UNSUPPORTED_OPERATION
            assert "could not be decided" in refused.value.message
            assert (await client.get_task(task.id)).status.state == TaskState.input_required
            assert performed == []


class TestMapping:
    def test_verdict_words(self) -> None:
        from locus.a2a.protocol import _verdict_of

        assert _verdict_of("Approve.") == "approved"
        assert _verdict_of("yes, go ahead") == "approved"
        assert _verdict_of("REJECTED") == "denied"
        assert _verdict_of("no") == "denied"
        assert _verdict_of("") is None
        assert _verdict_of("perhaps") is None
