# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool-call integrity: signed tool calls between model output and execution (#450).

Three layers. :class:`ToolCallSigner` alone (canonical form, chain of custody,
taint). The executor refusing a changed call through the context factory. And
the whole runtime, driven by a scripted model: a hook that changes arguments
without re-signing is refused and named on the audit trail; one that re-signs
is accepted; the streaming path signs too; the feature off changes nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.agent import Agent
from locus.agent.hook_orchestrator import HookOrchestrator
from locus.agent.run_context import RunContext
from locus.control import AuditHook, AuditTrail, ToolCallIntegrityError, ToolCallSigner
from locus.core.events import ModelChunkEvent, TerminateEvent, ThinkEvent, ToolCompleteEvent
from locus.core.messages import Message, ToolCall
from locus.hooks.builtin.guardrails import GuardrailAction, GuardrailConfig, GuardrailsHook
from locus.hooks.provider import BeforeToolCallEvent, HookPriority, HookProvider
from locus.models.base import ModelResponse
from locus.tools.decorator import tool
from locus.tools.executor import ConcurrentExecutor, SequentialExecutor, ToolContextFactory
from locus.tools.integrity import (
    AUDIT_EVENT_TYPE,
    call_digest,
    canonical_form,
    hook_label,
    signer_for,
)
from locus.tools.registry import ToolRegistry


def _call(arguments: dict[str, Any] | None = None, *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id, name="echo", arguments={"x": "hello"} if arguments is None else arguments
    )


def _run_context(signer: ToolCallSigner | None) -> RunContext:
    rc = RunContext.create(
        run_id="run-1", thread_id=None, prompt="p", metadata=None, agent_name=None, termination=None
    )
    rc.integrity = signer
    return rc


# =============================================================================
# The signer
# =============================================================================


class TestSigner:
    def test_canonical_form_is_sorted_compact_json(self) -> None:
        call = ToolCall(id="c", name="t", arguments={"b": 1, "a": {"y": 2, "x": [1, 2]}})
        assert (
            canonical_form(call)
            == b'{"arguments":{"a":{"x":[1,2],"y":2},"b":1},"id":"c","name":"t"}'
        )
        reordered = ToolCall(id="c", name="t", arguments={"a": {"x": [1, 2], "y": 2}, "b": 1})
        assert call_digest(call) == call_digest(reordered)

    def test_sign_then_verify(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        record = signer.sign(call)
        assert record.by == ("model",)
        assert record.digest == call_digest(call)
        assert record.key_id == signer.key_id
        assert signer.verify(call) == record
        assert len(signer) == 1

    def test_unsigned_call_is_not_refused(self) -> None:
        assert ToolCallSigner().verify(_call()) is None

    def test_changed_arguments_are_a_mismatch(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        record = signer.sign(call)
        changed = call.model_copy(update={"arguments": {"x": "rm -rf /"}})
        with pytest.raises(ToolCallIntegrityError) as info:
            signer.verify(changed)
        exc = info.value
        assert exc.reason == "mismatch"
        assert exc.expected_digest == record.digest
        assert exc.actual_digest == call_digest(changed)
        assert exc.signed_by == ("model",)
        assert exc.mutated_by is None
        assert exc.kind == "tool_call_integrity"
        assert "call_1" in str(exc)
        assert "echo" in str(exc)

    def test_changed_name_is_a_mismatch(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        with pytest.raises(ToolCallIntegrityError):
            signer.verify(call.model_copy(update={"name": "delete_everything"}))

    def test_resign_extends_the_chain(self) -> None:
        signer = ToolCallSigner()
        original = _call()
        signer.sign(original)
        rewritten = original.model_copy(update={"arguments": {"x": "[REDACTED]"}})
        record = signer.resign(rewritten, by="Redactor")
        assert record is not None
        assert record.by == ("model", "Redactor")
        assert record.signed_by == "Redactor"
        assert signer.verify(rewritten) == record
        # The pre-rewrite form is no longer what was vouched for.
        with pytest.raises(ToolCallIntegrityError):
            signer.verify(original)

    def test_resign_of_an_unsigned_call_is_a_no_op(self) -> None:
        signer = ToolCallSigner()
        assert signer.resign(_call(), by="Anyone") is None
        assert signer.verify(_call()) is None

    def test_resign_needs_a_name(self) -> None:
        signer = ToolCallSigner()
        signer.sign(_call())
        with pytest.raises(ValueError, match="by="):
            signer.resign(_call(), by="")

    def test_taint_is_not_cleared_by_a_later_resign(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        record = signer.sign(call)
        changed = call.model_copy(update={"arguments": {"x": "changed"}})
        signer.taint(call.id, by="HookA")
        assert signer.resign(changed, by="HookB") is None
        with pytest.raises(ToolCallIntegrityError) as info:
            signer.verify(changed)
        assert info.value.reason == "tainted"
        assert info.value.mutated_by == "HookA"
        assert info.value.expected_digest == record.digest
        assert info.value.actual_digest == call_digest(changed)
        assert "HookA" in str(info.value)

    def test_taint_of_unknown_call_is_ignored(self) -> None:
        signer = ToolCallSigner()
        signer.taint("nope", by="HookA")
        assert signer.signature("nope") is None

    def test_sign_all(self) -> None:
        signer = ToolCallSigner()
        calls = [_call(call_id="a"), _call(call_id="b")]
        signer.sign_all(calls)
        assert all(signer.verify(c) is not None for c in calls)

    def test_keys_are_per_signer(self) -> None:
        """A call signed under another run's key is unsigned here, not refused."""
        call = _call()
        ToolCallSigner().sign(call)
        assert ToolCallSigner().verify(call) is None

    def test_explicit_key_is_honoured(self) -> None:
        key = b"k" * 32
        a, b = ToolCallSigner(key=key), ToolCallSigner(key=key)
        assert a.key_id == b.key_id
        call = _call()
        record = a.sign(call)
        assert b.sign(call).mac == record.mac

    def test_checkpoint_round_trip_verifies_within_the_run(self) -> None:
        """The record covers id, name and arguments only, so a reloaded call verifies."""
        signer = ToolCallSigner()
        call = _call({"x": "héllo", "n": 3})
        signer.sign(call)
        message = Message.assistant(tool_calls=[call])
        reloaded = Message.model_validate_json(message.model_dump_json()).tool_calls[0]
        assert reloaded is not call
        assert signer.verify(reloaded) is not None
        # After a resume in a new process the key is gone: the call is unsigned.
        assert ToolCallSigner().verify(reloaded) is None

    def test_tool_call_serialisation_is_unchanged(self) -> None:
        """The signature lives on the run, not on the message types."""
        call = _call()
        ToolCallSigner().sign(call)
        assert set(call.model_dump()) == {"id", "name", "arguments", "malformed_arguments"}
        assert set(call.to_openai_format()) == {"id", "type", "function"}

    def test_on_violation_is_validated(self) -> None:
        with pytest.raises(ValueError, match="on_violation"):
            ToolCallSigner(on_violation="explode")  # type: ignore[arg-type]

    def test_signer_for_config_values(self) -> None:
        assert signer_for(False, []) is None
        refuse = signer_for(True, [])
        assert refuse is not None
        assert refuse.on_violation == "refuse"
        trail = AuditTrail()
        abort = signer_for("abort", [object(), AuditHook(trail)])
        assert abort is not None
        assert abort.on_violation == "abort"
        assert abort.trail is trail

    def test_hook_label_names_plugins(self) -> None:
        class _Plugin:
            name = "redteam"

        class PluginAdapter:
            _plugin = _Plugin()

        class PlainHook:
            pass

        assert hook_label(PlainHook()) == "PlainHook"
        assert hook_label(PluginAdapter()) == "PluginAdapter(redteam)"


# =============================================================================
# Enforcement: audit, event, abort
# =============================================================================


class TestEnforce:
    async def test_verified_and_unsigned_calls_proceed(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        assert await signer.enforce(call) is None
        assert await signer.enforce(_call(call_id="unsigned")) is None
        assert signer.refusals == []

    async def test_refusal_is_a_tool_error_with_an_audit_record(self) -> None:
        trail = AuditTrail()
        signer = ToolCallSigner(trail=trail)
        call = _call()
        record = signer.sign(call)
        changed = call.model_copy(update={"arguments": {"x": "changed"}})

        result = await signer.enforce(changed)

        assert result is not None
        assert result.error is not None
        assert result.error.startswith("ToolCallIntegrityError:")
        assert "echo" in result.error
        assert "not executed" in result.error
        assert result.tool_call_id == "call_1"
        assert result.name == "echo"
        assert len(signer.refusals) == 1

        [entry] = trail.records()
        assert entry.event_type == AUDIT_EVENT_TYPE
        assert entry.payload["tool"] == "echo"
        assert entry.payload["call_id"] == "call_1"
        assert entry.payload["expected_digest"] == record.digest
        assert entry.payload["actual_digest"] == call_digest(changed)
        assert entry.payload["signed_by"] == ["model"]
        assert entry.payload["mutated_by"] is None
        assert entry.payload["reason"] == "mismatch"
        assert entry.payload["disposition"] == "refuse"
        assert entry.payload["key_id"] == signer.key_id
        assert trail.verify()

    async def test_refusal_names_the_hook_on_the_trail(self) -> None:
        trail = AuditTrail()
        signer = ToolCallSigner(trail=trail)
        call = _call()
        signer.sign(call)
        signer.taint(call.id, by="SilentHook")
        changed = call.model_copy(update={"arguments": {"x": "changed"}})

        assert await signer.enforce(changed) is not None

        [entry] = trail.records()
        assert entry.payload["mutated_by"] == "SilentHook"
        assert entry.payload["reason"] == "tainted"

    async def test_refusal_is_published_on_the_bus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        # ``locus.observability.emit`` the attribute is the function; the
        # module is reached through ``sys.modules``.
        emit_mod = sys.modules["locus.observability.emit"]
        seen: list[tuple[str, dict[str, Any]]] = []

        async def _emit(event_type: str, /, **data: Any) -> None:
            seen.append((event_type, data))

        monkeypatch.setattr(emit_mod, "emit", _emit)
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        await signer.enforce(call.model_copy(update={"arguments": {"x": "changed"}}))

        assert [t for t, _ in seen] == ["tool.integrity.refused"]
        assert seen[0][1]["tool_name"] == "echo"
        assert seen[0][1]["tool"] == "echo"  # deprecated alias, one release
        assert {"expected_digest", "actual_digest", "mutated_by"} <= set(seen[0][1])

    async def test_abort_mode_holds_the_error_for_the_runtime(self) -> None:
        signer = ToolCallSigner(on_violation="abort")
        call = _call()
        signer.sign(call)
        result = await signer.enforce(call.model_copy(update={"arguments": {"x": "changed"}}))
        assert result is not None
        assert result.error is not None
        with pytest.raises(ToolCallIntegrityError) as info:
            signer.raise_pending_abort()
        assert info.value is signer.refusals[0]
        signer.raise_pending_abort()  # consumed

    async def test_refuse_mode_never_raises(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        await signer.enforce(call.model_copy(update={"arguments": {"x": "changed"}}))
        signer.raise_pending_abort()

    async def test_a_failing_trail_does_not_unrefuse(self) -> None:
        class _BrokenTrail:
            def record(self, event_type: str, payload: Any) -> None:
                raise OSError("disk full")

        signer = ToolCallSigner(trail=_BrokenTrail())
        call = _call()
        signer.sign(call)
        result = await signer.enforce(call.model_copy(update={"arguments": {"x": "changed"}}))
        assert result is not None
        assert result.error is not None


# =============================================================================
# The executor
# =============================================================================


def _registry() -> tuple[ToolRegistry, list[dict[str, Any]]]:
    ran: list[dict[str, Any]] = []

    @tool
    def echo(x: str) -> str:
        """Echo ``x``."""
        ran.append({"x": x})
        return x

    registry = ToolRegistry()
    registry.register(echo)
    return registry, ran


class TestExecutor:
    @pytest.mark.parametrize("executor", [SequentialExecutor(), ConcurrentExecutor()])
    async def test_intact_call_runs(self, executor: Any) -> None:
        registry, ran = _registry()
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        [result] = await executor.execute(
            [call], registry, ToolContextFactory(run_id="r", integrity=signer)
        )
        assert result.success
        assert result.content == "hello"
        assert ran == [{"x": "hello"}]

    @pytest.mark.parametrize("executor", [SequentialExecutor(), ConcurrentExecutor()])
    async def test_changed_call_is_refused_and_the_tool_does_not_run(self, executor: Any) -> None:
        registry, ran = _registry()
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        call.arguments["x"] = "changed"  # in place, after signing
        [result] = await executor.execute(
            [call], registry, ToolContextFactory(run_id="r", integrity=signer)
        )
        assert result.error is not None
        assert "ToolCallIntegrityError" in result.error
        assert ran == []
        assert len(signer.refusals) == 1

    async def test_streaming_executor_refuses_only_the_changed_sibling(self) -> None:
        registry, ran = _registry()
        signer = ToolCallSigner()
        intact, changed = _call({"x": "a"}, call_id="a"), _call({"x": "b"}, call_id="b")
        signer.sign_all([intact, changed])
        changed.arguments["x"] = "evil"
        factory = ToolContextFactory(run_id="r", integrity=signer)
        results = {}
        async for idx, result in ConcurrentExecutor().execute_streaming(
            [intact, changed], registry, factory
        ):
            results[idx] = result
        assert results[0].success
        assert results[0].content == "a"
        assert results[1].error is not None
        assert "ToolCallIntegrityError" in results[1].error
        assert ran == [{"x": "a"}]

    async def test_unsigned_call_runs_with_a_signer(self) -> None:
        registry, ran = _registry()
        [result] = await SequentialExecutor().execute(
            [_call()], registry, ToolContextFactory(run_id="r", integrity=ToolCallSigner())
        )
        assert result.success
        assert ran == [{"x": "hello"}]

    async def test_no_signer_means_no_check(self) -> None:
        registry, ran = _registry()
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        call.arguments["x"] = "changed"
        [result] = await SequentialExecutor().execute(
            [call], registry, ToolContextFactory(run_id="r")
        )
        assert result.success
        assert ran == [{"x": "changed"}]
        [result] = await SequentialExecutor().execute([call], registry, None)
        assert result.success

    def test_factory_hides_the_signer_from_dumps(self) -> None:
        factory = ToolContextFactory(run_id="r", integrity=ToolCallSigner())
        assert "integrity" not in factory.model_dump()
        assert "integrity" not in repr(factory)


# =============================================================================
# The hook seam
# =============================================================================


class _SilentHook(HookProvider):
    """Changes the arguments without re-signing: the legacy way."""

    def __init__(self, replacement: str = "changed", *, in_place: bool = False) -> None:
        self._replacement = replacement
        self._in_place = in_place

    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if self._in_place:
            event.arguments["x"] = self._replacement
        else:
            event.arguments = {**event.arguments, "x": self._replacement}


class _ResigningHook(HookProvider):
    """Rewrites the arguments and vouches for them under its own name."""

    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        event.arguments = {**event.arguments, "x": "[REDACTED]"}
        event.resign(by="Redactor")


class _SecretHook(HookProvider):
    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        event.secret_arguments = {"token": "t-123"}


class TestHookSeam:
    async def test_silent_change_taints_under_the_hook_name(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        event = await HookOrchestrator([_SilentHook()]).run_before_tool(
            call.name, call.id, call.arguments, run=_run_context(signer)
        )
        assert event.arguments["x"] == "changed"
        with pytest.raises(ToolCallIntegrityError) as info:
            signer.verify(call.model_copy(update={"arguments": event.arguments}))
        assert info.value.reason == "tainted"
        assert info.value.mutated_by == "_SilentHook"

    async def test_in_place_change_is_caught_too(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        await HookOrchestrator([_SilentHook(in_place=True)]).run_before_tool(
            call.name, call.id, call.arguments, run=_run_context(signer)
        )
        assert signer.signature(call.id).tainted_by == "_SilentHook"  # type: ignore[union-attr]

    async def test_resigning_hook_is_accepted(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        event = await HookOrchestrator([_ResigningHook()]).run_before_tool(
            call.name, call.id, call.arguments, run=_run_context(signer)
        )
        record = signer.verify(call.model_copy(update={"arguments": event.arguments}))
        assert record is not None
        assert record.by == ("model", "Redactor")

    async def test_event_exposes_the_signer(self) -> None:
        signer = ToolCallSigner()
        seen: list[Any] = []

        class _Peek(HookProvider):
            @property
            def priority(self) -> int:
                return HookPriority.BUSINESS_DEFAULT

            async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
                seen.append(event.integrity)

        await HookOrchestrator([_Peek()]).run_before_tool("echo", "c", {}, run=_run_context(signer))
        await HookOrchestrator([_Peek()]).run_before_tool("echo", "c", {}, run=_run_context(None))
        await HookOrchestrator([_Peek()]).run_before_tool("echo", "c", {})
        assert seen == [signer, None, None]

    async def test_resign_without_a_signer_is_a_no_op(self) -> None:
        event = BeforeToolCallEvent(tool_name="echo", tool_call_id="c", arguments={"x": 1})
        event.resign(by="Anyone")
        assert event.integrity is None

    async def test_secret_arguments_extend_the_signature(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        event = await HookOrchestrator([_SecretHook()]).run_before_tool(
            call.name, call.id, call.arguments, run=_run_context(signer)
        )
        merged = call.model_copy(
            update={"arguments": {**event.arguments, **event.secret_arguments}}
        )
        record = signer.verify(merged)
        assert record is not None
        assert record.by == ("model", "_SecretHook:secret_arguments")

    async def test_secret_arguments_do_not_launder_a_silent_change(self) -> None:
        signer = ToolCallSigner()
        call = _call()
        signer.sign(call)
        event = await HookOrchestrator([_SilentHook(), _SecretHook()]).run_before_tool(
            call.name, call.id, call.arguments, run=_run_context(signer)
        )
        merged = call.model_copy(
            update={"arguments": {**event.arguments, **event.secret_arguments}}
        )
        with pytest.raises(ToolCallIntegrityError) as info:
            signer.verify(merged)
        assert info.value.mutated_by == "_SilentHook"

    async def test_feature_off_leaves_the_seam_alone(self) -> None:
        event = await HookOrchestrator([_SilentHook()]).run_before_tool(
            "echo", "c", {"x": "hello"}, run=_run_context(None)
        )
        assert event.arguments == {"x": "changed"}

    async def test_unsigned_call_is_not_watched(self) -> None:
        signer = ToolCallSigner()
        await HookOrchestrator([_SilentHook()]).run_before_tool(
            "echo", "never-signed", {"x": "hello"}, run=_run_context(signer)
        )
        assert signer.signature("never-signed") is None


# =============================================================================
# The runtime, end to end
# =============================================================================


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ModelChunkEvent]:  # pragma: no cover
        raise NotImplementedError
        yield


class _StreamingScriptedModel(_ScriptedModel):
    """Delivers the same script through ``stream()``: tool calls arrive on the last chunk."""

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelChunkEvent]:
        response = await self.complete([])
        if response.message.content:
            yield ModelChunkEvent(content=response.message.content)
        yield ModelChunkEvent(
            tool_calls=list(response.message.tool_calls) or None,
            done=True,
            usage=response.usage,
            stop_reason="tool_calls" if response.message.tool_calls else "stop",
        )


def _assistant(content: str | None, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 5, "completion_tokens": 5},
    )


def _script() -> list[ModelResponse]:
    return [
        _assistant(None, [ToolCall(id="call_echo", name="echo", arguments={"x": "hello"})]),
        _assistant("Done."),
    ]


async def _drive(agent: Agent, **run_kwargs: Any) -> list[Any]:
    return [event async for event in agent.run("go", **run_kwargs)]


async def _drive_into(agent: Agent, events: list[Any]) -> None:
    """Like :func:`_drive`, keeping what was yielded before the run raised."""
    async for event in agent.run("go"):
        events.append(event)


def _complete(events: list[Any]) -> ToolCompleteEvent:
    [event] = [e for e in events if isinstance(e, ToolCompleteEvent)]
    return event


class TestAgent:
    async def test_silent_hook_change_is_refused_and_the_run_continues(self) -> None:
        registry, ran = _registry()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            hooks=[_SilentHook()],
            tool_call_integrity=True,
            max_iterations=4,
        )
        events = await _drive(agent)
        complete = _complete(events)
        assert complete.error is not None
        assert "ToolCallIntegrityError" in complete.error
        assert "_SilentHook" in complete.error
        assert ran == []
        [terminate] = [e for e in events if isinstance(e, TerminateEvent)]
        assert terminate.reason != "error"

    async def test_resigning_hook_is_accepted(self) -> None:
        registry, ran = _registry()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            hooks=[_ResigningHook()],
            tool_call_integrity=True,
            max_iterations=4,
        )
        events = await _drive(agent)
        assert _complete(events).error is None
        assert ran == [{"x": "[REDACTED]"}]

    async def test_feature_off_keeps_the_old_behaviour(self) -> None:
        registry, ran = _registry()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            hooks=[_SilentHook()],
            max_iterations=4,
        )
        events = await _drive(agent)
        assert _complete(events).error is None
        assert ran == [{"x": "changed"}]

    async def test_audit_trail_carries_both_digests_and_the_hook(self) -> None:
        registry, _ = _registry()
        trail = AuditTrail()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            hooks=[AuditHook(trail), _SilentHook()],
            tool_call_integrity=True,
            max_iterations=4,
        )
        await _drive(agent)
        [entry] = [r for r in trail.records() if r.event_type == AUDIT_EVENT_TYPE]
        signed = ToolCall(id="call_echo", name="echo", arguments={"x": "hello"})
        received = ToolCall(id="call_echo", name="echo", arguments={"x": "changed"})
        assert entry.payload["tool"] == "echo"
        assert entry.payload["call_id"] == "call_echo"
        assert entry.payload["expected_digest"] == call_digest(signed)
        assert entry.payload["actual_digest"] == call_digest(received)
        assert entry.payload["mutated_by"] == "_SilentHook"
        assert entry.payload["signed_by"] == ["model"]
        assert trail.verify()

    async def test_abort_mode_ends_the_run_with_the_typed_error(self) -> None:
        registry, ran = _registry()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            hooks=[_SilentHook()],
            tool_call_integrity="abort",
            max_iterations=4,
        )
        events: list[Any] = []
        with pytest.raises(ToolCallIntegrityError) as info:
            await _drive_into(agent, events)
        assert info.value.mutated_by == "_SilentHook"
        assert ran == []
        [terminate] = [e for e in events if isinstance(e, TerminateEvent)]
        assert terminate.reason == "error"
        assert "ToolCallIntegrityError" in (terminate.error or "")

    async def test_consumer_mutating_a_think_event_call_is_refused(self) -> None:
        """No hook involved: the call is changed between the model and the executor."""
        registry, ran = _registry()
        agent = Agent(
            model=_ScriptedModel(_script()),
            tools=list(registry),
            tool_call_integrity=True,
            max_iterations=4,
        )
        events: list[Any] = []
        async for event in agent.run("go"):
            events.append(event)
            if isinstance(event, ThinkEvent) and event.tool_calls:
                event.tool_calls[0].arguments["x"] = "evil"
        complete = _complete(events)
        assert complete.error is not None
        assert "ToolCallIntegrityError" in complete.error
        assert ran == []

    async def test_builtin_guardrail_redaction_is_accepted(self) -> None:
        registry, ran = _registry()
        script = [
            _assistant(None, [ToolCall(id="c", name="echo", arguments={"x": "mail me at a@b.co"})]),
            _assistant("Done."),
        ]
        guard = GuardrailsHook(
            GuardrailConfig(action_overrides={"pii_email": GuardrailAction.REDACT})
        )
        agent = Agent(
            model=_ScriptedModel(script),
            tools=list(registry),
            hooks=[guard],
            tool_call_integrity=True,
            max_iterations=4,
        )
        events = await _drive(agent)
        assert _complete(events).error is None
        assert len(ran) == 1
        assert "a@b.co" not in ran[0]["x"]
        assert "REDACTED" in ran[0]["x"]

    async def test_streamed_tool_calls_are_signed(self) -> None:
        registry, ran = _registry()
        chains: list[tuple[str, ...]] = []

        class _Peek(HookProvider):
            @property
            def priority(self) -> int:
                return HookPriority.BUSINESS_DEFAULT

            async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
                signer = event.integrity
                assert signer is not None
                record = signer.signature(event.tool_call_id)
                assert record is not None
                chains.append(record.by)

        agent = Agent(
            model=_StreamingScriptedModel(_script()),
            tools=list(registry),
            hooks=[_Peek(), _SilentHook()],
            tool_call_integrity=True,
            max_iterations=4,
        )
        events = await _drive(agent, stream_tokens=True)
        assert chains == [("model",)]
        complete = _complete(events)
        assert complete.error is not None
        assert "ToolCallIntegrityError" in complete.error
        assert ran == []

    async def test_each_run_has_its_own_key(self) -> None:
        registry, _ = _registry()
        keys: list[str] = []

        class _Peek(HookProvider):
            @property
            def priority(self) -> int:
                return HookPriority.BUSINESS_DEFAULT

            async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
                assert event.integrity is not None
                keys.append(event.integrity.key_id)

        class _Repeating:
            """A tool call, then an answer — on every run."""

            calls = 0

            async def complete(
                self, messages: Any, tools: Any = None, **kwargs: Any
            ) -> ModelResponse:
                self.calls += 1
                return _script()[0] if self.calls % 2 else _script()[1]

        agent = Agent(
            model=_Repeating(),
            tools=list(registry),
            hooks=[_Peek()],
            tool_call_integrity=True,
            max_iterations=4,
        )
        await _drive(agent)
        await _drive(agent)
        assert len(keys) == 2
        assert keys[0] != keys[1]
