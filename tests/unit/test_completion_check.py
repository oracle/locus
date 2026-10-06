# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``tulip.agent.completion``: a stop is not always the end of the work.

Open-weight models often end a turn on the announcement of their next step —
"Let me first check the conftest and the specific test area:" — and the loop
took that for the answer. These tests pin the detector against a corpus of
real final answers and announcements, the bounded continuation the check sends,
and how a continuation sits in history, checkpoints, compaction, the iteration
cap and the budgets.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest

from tulip.agent import Agent, CompletionCheck, Continuation, chain_verifiers
from tulip.agent.completion import (
    AUTOMATED_NOTE_KEY,
    CONTINUATION_NOTE_KEY,
    announced_step,
    edits_unchecked,
    explains_no_change,
    max_replans_for,
    requests_changes,
)
from tulip.agent.verification import FinalAnswerContext, is_ephemeral_message
from tulip.core.events import (
    FinalAnswerVerificationEvent,
    RunInfo,
    TerminateEvent,
    TulipEvent,
)
from tulip.core.messages import Message, Role, ToolCall
from tulip.core.state import ToolExecution
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.memory.compaction import CompactionTracker, ContextCompactor
from tulip.models.base import ModelResponse
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


# --------------------------------------------------------------- the corpus --

#: Replies that announce a step and stop. The first is the benchmark's own
#: (qwen3.6-35b, task gw-hold-caller, after 19 tool steps); the rest are the
#: same shape as open-weight models write it, in several languages.
ANNOUNCEMENTS = [
    "Let me first check the conftest and the specific test area:",
    "I found where the hold flag is set. Now let me look at how the caller is notified.",
    "The handler is in `gateway/calls.py`. I'll update it to keep the caller on hold.",
    "Good, the tests are in place. Next, I'll run the full suite to confirm nothing broke.",
    "Now I'll implement the confirm endpoint.",
    "I need to check how the tasks module registers its routes.",
    "I'm going to add a regression test for the empty-list case.",
    "Let's run the tests again.",
    "OK. Let me now go ahead and fix the import in `luach/tasks.py`.",
    "First, I need to read the existing confirm flow.",
    "I see the problem now. I will modify the serializer so the field is optional.",
    "Here is what I'll change:",
    "The fix needs two parts. **Let me start with the model:**",
    "Running the tests now…",
    "I'm now checking whether the migration already exists.",
    "We'll need to update the fixture as well. Let me open it.",
    "Voy a revisar el archivo de configuración.",
    "Ahora voy a ejecutar las pruebas.",
    "Vou verificar os testes agora.",
    "Je vais corriger la fonction de validation.",
    "Ich werde die Tests ausführen.",
    "Lasciami controllare il file di configurazione.",
    "Сначала проверю конфигурацию.",
    "让我先查看一下 conftest 文件。",
    "我已经找到了问题所在。接下来我来修改这个函数。",
    "首先，让我运行测试。",
    "設定ファイルを確認します。",
    "테스트를 실행해 보겠습니다.",
    "Te muestro los cambios:",
    "下面是需要修改的地方：",
]

#: Replies that end a run properly. The first block is real tulip-code final
#: answers from a playground session; the rest are the common shapes of a
#: finished turn, including the ones a loose detector would trip on.
FINAL_ANSWERS = [
    # Real tulip-code answers.
    "Hello! What would you like me to work on?",
    "Sure — I can help build a Go MCP server.\n\nA couple of choices before I edit files: do "
    "you want a **minimal MCP stdio server** scaffold in `src/` with one example tool, or do "
    "you already have a specific tool/API you want the server to expose?",
    "Fixed `src/calc.py`:\n\n- `add` now adds instead of subtracts\n- `divide` now reports "
    "division by zero instead of raising\n- `largest` now handles all-negative lists\n- Added "
    "type hints\n\nTests pass with:\n\n```bash\npython3 -m pytest tests/ -q\n```",
    "Created `src/simple_calculator.cob`.\n\nVerified:\n- Read back "
    "`src/simple_calculator.cob`\n- Ran `python3 -m pytest tests/ -q`\n- Result: `3 passed`",
    "You can run this locally:\n\n```bash\ncobc -x inventory.cob\n./inventory\n```\n\nI can "
    "keep improving the COBOL program here under `src/` if you want.",
    "I can:\n\n- Add or modify features\n- Refactor code\n- Run the test suite\n\nI’ll read "
    "the relevant code first, edit only under `src/` and `tests/`, and run tests after changes.",
    "Open work:\n- Await the user’s actual coding/task request.",
    # Common finished shapes.
    "All 42 tests pass. The hold flag is now cleared when the caller hangs up.",
    "Done. I updated `gateway/calls.py` and added `tests/test_hold.py`; `pytest -q` passes.",
    "Let me know if you want me to also update the changelog.",
    "I'll leave the formatting as is, since the project's ruff config already enforces it.",
    "I'll fix the docs in a follow-up PR if you want.",
    "Should I also update the README?",
    "Let me know if anything else needs changing!",
    "Let me walk you through it: the bug was an off-by-one in the loop bound, now fixed.",
    "Running the tests showed one failure, which I fixed in `parser.py`.",
    "Now I understand the issue: the cache key ignored the locale. It is fixed and tested.",
    "No changes were needed: the endpoint already returns 404 for a missing task.",
    "Summary of changes:\n- Added `confirm()`\n- Added two tests",
    "```python\ndef f():\n    return 1\n```",
    "The function returns the user's id.",
    "I would recommend running the integration suite before deploying.",
    "You should run `make check` before pushing.",
    "We should consider caching this later.",
    "He corregido el error y las pruebas pasan.",
    "已修复该问题，所有测试均已通过。",
    "修正しました。すべてのテストが通りました。",
    "",
    "   ",
]


@pytest.mark.parametrize("reply", ANNOUNCEMENTS)
def test_the_detector_catches_every_announcement_in_the_corpus(reply: str) -> None:
    assert announced_step(reply) is not None, reply


@pytest.mark.parametrize("reply", FINAL_ANSWERS)
def test_the_detector_passes_every_final_answer_in_the_corpus(reply: str) -> None:
    assert announced_step(reply) is None, reply


def test_the_detector_quotes_the_announcing_sentence_clipped() -> None:
    reply = "Lots of context here.\n\nNow let me run the full test suite to confirm."
    assert announced_step(reply) == "Now let me run the full test suite to confirm."
    long = "Let me check " + "x" * 400 + ":"
    snippet = announced_step(long)
    assert snippet is not None
    assert len(snippet) == 160
    assert snippet.endswith("…")


def test_a_reply_of_markup_only_announces_nothing() -> None:
    assert announced_step("**") is None


# ---------------------------------------------------- did it change anything --


@pytest.mark.parametrize(
    "prompt",
    [
        "Fix the flaky test in tests/test_hold.py",
        "Can you add a --dry-run flag to the CLI?",
        "Review the tasks module and fix any bugs you find.",
        "The confirm endpoint returns 500 for a missing task. Make it return 404.",
        "Your task is to implement hold-caller support in the gateway.",
        "please update the README",
        "- Rename `foo` to `bar` everywhere",
    ],
)
def test_requests_for_changes_are_recognised(prompt: str) -> None:
    assert requests_changes(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "What does admit() do?",
        "Why does the hold test fail?",
        "Explain the session layout.",
        "List the endpoints in the gateway.",
        "Summarise the last three commits.",
        "¿Qué hace esta función?",
    ],
)
def test_questions_do_not_request_changes(prompt: str) -> None:
    assert not requests_changes(prompt)


@pytest.mark.parametrize(
    "reply",
    [
        "No changes were needed: the endpoint already returns 404.",
        "The feature is already implemented in `gateway/hold.py`.",
        "I could not reproduce the failure.",
        "The edit was refused by the permission gate.",
        "This works as intended; it is not a bug.",
        "I left the code unchanged because the test is wrong, not the code.",
        "Nothing to change here.",
    ],
)
def test_replies_that_say_why_nothing_changed(reply: str) -> None:
    assert explains_no_change(reply)


def test_a_reply_that_just_stops_explains_nothing() -> None:
    assert not explains_no_change("I looked at the handler and the tests.")


# Task prompts in the shapes real coding tasks take. Each one asks for
# changes, and the narrow verb-first rule missed the second, fourth, fifth and
# sixth: a run that changed nothing on them was never sent back.
_TASK_PROMPTS = [
    # A request with "and" before the verb.
    "Let people ask the assistant about their email in chat and add what it finds "
    "to the calendar when they say so.",
    # "Let" + who, then a spec; no change verb leads any line.
    "Let people ask the assistant about their email in chat. Nothing is added unless "
    "the person asks.\n- mail: `search(query)` over the whole mailbox, newest first",
    # A problem statement, then the order on its own line after a parenthesis.
    "A film filed in the library directory answers 403 instead of playing.\n(In prod "
    "nothing changes there.)\nMake the library directory count as a media root.",
    # A requirement stated, not ordered.
    "The endpoint returns a flat list, so callers add rows up themselves and get it "
    "wrong. The response must contain the totals, computed by the service.",
    # An interface section naming what to add.
    "The cache keeps dead generations beside live ones.\n- `generation_dir()`: the "
    "live generation's directory.\n- `generations()`: what is present, by name.",
    # A new file in a spec list, then an acceptance line.
    "Pending holds read `args: {}`.\n- New `policy/evidence.py`: an in-process store, "
    "bounded.\nDone when: the reviewer sees the arguments.",
    "Changing a task someone else made happens silently. Tasks should get the same "
    "treatment as events: held for approval.",
    "Two pieces of per-call state are shared. Give each call its own record.",
    "Serve the arguments from this plane, at decision time.",
]

_QUESTION_PROMPTS = [
    "How is the segment cache keyed, and where is `_FORMAT` read?",
    "Where does the ingest read the mail label? Show me the function.",
    "Walk me through how `/v1/admit` decides a hold.",
    "Which tests cover the unlink path?",
    "Is the library directory a media root today? Look at the stream endpoint and tell me.",
    "Let me know what `generations()` returns for the legacy layout.",
    "Review the fallback chain and tell me whether `last_tier` is safe under concurrency.",
    "Give me a summary of how streaming usage is reported.",
    "Read src/ingest/mail.py and describe what `search(query)` returns.",
    "Can you tell me why `spending_by_category` returns a flat list?",
    "Compare the two checkpointers: what does each one keep?",
    "Find where the indexer key is read.",
    "Explain why the cache should be keyed by format.",
    "Look at the cache and tell me what must change for a new encode generation.",
    "Explain these:\n- `generation_dir()`: what does it return?",
    "The cache is keyed by format. Explain why it should be.",
    "The cache is keyed by format. Should it be keyed by path instead?",
]


@pytest.mark.parametrize("prompt", _TASK_PROMPTS)
def test_task_prompts_in_every_shape_request_changes(prompt: str) -> None:
    assert requests_changes(prompt)


@pytest.mark.parametrize("prompt", _QUESTION_PROMPTS)
def test_question_prompts_still_do_not(prompt: str) -> None:
    assert not requests_changes(prompt)


@pytest.mark.parametrize(
    "reply",
    [
        "The system requested my final answer before I could make the actual edits, "
        "so I was unable to apply them.",
        "I ran out of iterations, so the edits are not made yet; I couldn't finish.",
        "I didn't get a chance to apply the change.",
        "You asked for a final answer, so here is the plan: I cannot edit yet.",
    ],
)
def test_blaming_the_runs_own_stop_is_not_a_reason(reply: str) -> None:
    assert not explains_no_change(reply)


def _ex(name: str, error: str | None = None, **arguments: Any) -> ToolExecution:
    return ToolExecution(tool_name=name, tool_call_id=name, arguments=arguments, error=error)


def test_edits_without_a_later_check_are_unchecked() -> None:
    assert not edits_unchecked([])
    assert not edits_unchecked([_ex("read", path="a.py")])
    assert edits_unchecked([_ex("bash", command="pytest -q"), _ex("edit", path="a.py")])
    assert not edits_unchecked([_ex("edit", path="a.py"), _ex("bash", command="pytest -q")])
    assert not edits_unchecked([_ex("write", path="a.py"), _ex("bash", command="npm test")])
    assert edits_unchecked([_ex("edit", path="a.py"), _ex("bash", command="ls -la")])
    # A failed edit changed nothing to check.
    assert not edits_unchecked([_ex("edit", error="no match", path="a.py")])
    # A failing test run still counts as having checked.
    assert not edits_unchecked(
        [_ex("apply_patch", patch="..."), _ex("bash", error="exit 1", command="cargo test")]
    )


def test_an_edit_that_ran_its_check_as_then_run_is_checked() -> None:
    def fused(command: str, result: str) -> ToolExecution:
        return ToolExecution(
            tool_name="edit",
            tool_call_id="e1",
            arguments={"path": "a.py", "then_run": {"command": command}},
            result=result,
        )

    assert not edits_unchecked(
        [fused("pytest -q", "edited a.py\n\n[then_run] $ pytest -q\nexit 0")]
    )
    # Skipped, or not a check: the edit is still unchecked.
    assert edits_unchecked([fused("pytest -q", "edited a.py\n\n[then_run skipped] changed")])
    assert edits_unchecked([fused("ls", "edited a.py\n\n[then_run] $ ls\nexit 0")])
    # A malformed then_run is no command at all.
    assert edits_unchecked(
        [ToolExecution(tool_name="edit", tool_call_id="e2", arguments={"then_run": 3}, result="")]
    )


# ------------------------------------------------------------ the check alone --


def _ctx(
    *,
    run_id: str = "run-1",
    attempt: int = 0,
    max_replans: int = 3,
    executions: tuple[ToolExecution, ...] = (),
    prompt: str = "Fix the bug",
) -> FinalAnswerContext:
    return FinalAnswerContext(
        run=RunInfo(run_id=run_id),
        prompt=prompt,
        messages=(),
        tool_executions=executions,
        attempt=attempt,
        max_replans=max_replans,
    )


async def test_an_announcement_is_sent_back_as_a_continuation() -> None:
    seen: list[str] = []
    check = CompletionCheck(on_continuation=lambda reason, _ctx: seen.append(reason))
    feedback = await check("Let me first check the conftest:", _ctx())
    assert isinstance(feedback, Continuation)
    assert feedback.reason == "announced_step"
    assert "Let me first check the conftest:" in feedback
    assert "Carry out that step now" in feedback
    assert "say so plainly and state what you verified" in feedback
    assert seen == ["announced_step"]
    assert check.continuations("run-1") == 1
    assert check.continuations("other") == 0


async def test_a_finished_answer_is_accepted() -> None:
    check = CompletionCheck()
    assert await check("All tests pass; the hold flag is fixed.", _ctx()) is None


async def test_never_twice_in_a_row_for_the_same_reason() -> None:
    check = CompletionCheck()
    assert await check("Let me run the tests.", _ctx()) is not None
    # Sent back, and stopped again on the same note without calling a tool.
    assert await check("Let me run the tests.", _ctx(attempt=1)) is None
    # After real work, the same reason may send it back again.
    work = (_ex("bash", command="pytest"),)
    assert await check("Let me run the tests.", _ctx(attempt=1, executions=work)) is not None


async def test_insisting_on_changes_sends_a_no_change_stop_back_until_the_cap() -> None:
    check = CompletionCheck(
        max_nudges=3, needs_changes=lambda _draft, _ctx: True, insist_on_changes=True
    )
    for attempt in range(3):
        feedback = await check("Here is how the change would look.", _ctx(attempt=attempt))
        assert isinstance(feedback, Continuation)
        assert feedback.reason == "no_changes"
    assert await check("Here is how it would look.", _ctx(attempt=3, max_replans=5)) is None
    # A reason accepts the stop at once.
    fresh = CompletionCheck(needs_changes=lambda _draft, _ctx: True, insist_on_changes=True)
    assert await fresh("No changes were needed: it is already implemented.", _ctx()) is None


async def test_without_insisting_a_second_no_change_stop_is_accepted() -> None:
    check = CompletionCheck(needs_changes=lambda _draft, _ctx: True)
    assert await check("Here is how the change would look.", _ctx()) is not None
    assert await check("Here is how the change would look.", _ctx(attempt=1)) is None


async def test_the_nudges_are_capped_per_run() -> None:
    check = CompletionCheck(max_nudges=2)
    for i in range(2):
        work = tuple(_ex("read", path=str(n)) for n in range(i))
        assert await check("Let me look.", _ctx(executions=work)) is not None
    assert await check("Let me look.", _ctx(executions=(_ex("read"),) * 5)) is None
    # Another run has its own count.
    assert await check("Let me look.", _ctx(run_id="run-2")) is not None


async def test_no_nudge_when_no_replan_is_left_or_the_check_is_off() -> None:
    assert await CompletionCheck()("Let me look.", _ctx(attempt=3, max_replans=3)) is None
    assert await CompletionCheck(max_nudges=0)("Let me look.", _ctx()) is None
    with pytest.raises(ValueError, match="max_nudges"):
        CompletionCheck(max_nudges=-1)


async def test_no_changes_needs_the_callers_signal_and_no_explanation() -> None:
    changed: list[str] = []
    check = CompletionCheck(needs_changes=lambda _draft, _ctx: not changed)
    feedback = await check("I looked at the handler.", _ctx())
    assert isinstance(feedback, Continuation)
    assert feedback.reason == "no_changes"
    assert "no file has changed" in feedback
    assert (
        await CompletionCheck(needs_changes=lambda *_: True)(
            "No changes were needed: it already works.", _ctx()
        )
        is None
    )
    changed.append("a.py")
    assert await check("Done.", _ctx(run_id="run-2")) is None


async def test_unchecked_edits_are_nudged_once_per_run() -> None:
    check = CompletionCheck(unchecked_edits=True)
    edited = (_ex("edit", path="a.py"),)
    feedback = await check("Fixed it.", _ctx(executions=edited))
    assert isinstance(feedback, Continuation)
    assert feedback.reason == "unchecked_edits"
    more = (*edited, _ex("read"), _ex("edit", path="b.py"))
    assert await check("Fixed it.", _ctx(executions=more)) is None
    # A caller's own predicate replaces the default one.
    custom = CompletionCheck(unchecked_edits=lambda _draft, _ctx: True)
    assert await custom("Fixed it.", _ctx()) is not None


async def test_the_most_specific_reason_that_was_not_just_used_wins() -> None:
    check = CompletionCheck(needs_changes=lambda *_: True)
    first = await check("Let me check the handler.", _ctx())
    second = await check("Let me check the handler.", _ctx(attempt=1))
    assert isinstance(first, Continuation)
    assert isinstance(second, Continuation)
    assert (first.reason, second.reason) == ("announced_step", "no_changes")


def test_agent_options_carry_the_replans_the_check_needs() -> None:
    check = CompletionCheck(max_nudges=4)
    assert check.agent_options() == {
        "final_answer_verifier": check,
        "final_answer_verifier_max_replans": 4,
    }
    assert CompletionCheck(max_nudges=0).agent_options()["final_answer_verifier_max_replans"] == 1


async def test_chained_verifiers_run_in_order_until_one_rejects() -> None:
    calls: list[str] = []

    async def accept(_draft: str, _ctx: FinalAnswerContext) -> str | None:
        calls.append("accept")
        return None

    async def reject(_draft: str, _ctx: FinalAnswerContext) -> str | None:
        calls.append("reject")
        return "no"

    async def never(_draft: str, _ctx: FinalAnswerContext) -> str | None:  # pragma: no cover
        calls.append("never")
        return None

    assert chain_verifiers() is None
    assert chain_verifiers(None, accept, None) is accept
    chained = chain_verifiers(accept, None, reject, never)
    assert chained is not None
    assert await chained("x", _ctx()) == "no"
    assert calls == ["accept", "reject"]
    calls.clear()
    both = chain_verifiers(accept, accept)
    assert both is not None
    assert await both("x", _ctx()) is None
    assert calls == ["accept", "accept"]
    assert max_replans_for(3, None, 1) == 4
    assert max_replans_for(10, 3) == 10


# ------------------------------------------------------------ in the loop --


@tool
def read(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


async def _collect(stream: Any) -> list[TulipEvent]:
    return [e async for e in stream]


def _verdicts(events: list[TulipEvent]) -> list[FinalAnswerVerificationEvent]:
    return [e for e in events if isinstance(e, FinalAnswerVerificationEvent)]


def _terminate(events: list[TulipEvent]) -> TerminateEvent:
    return next(e for e in events if isinstance(e, TerminateEvent))


def _agent(model: Any, check: CompletionCheck, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=[read],
        reflexion=False,
        grounding=False,
        **check.agent_options(),
        **kwargs,
    )


ANNOUNCEMENT = "Let me first check the conftest and the specific test area:"


async def test_the_run_goes_on_after_an_announcement_and_keeps_it_in_history() -> None:
    model = ScriptedModel(
        [
            tool_call("read", path="gateway/calls.py"),
            text(ANNOUNCEMENT),
            tool_call("read", path="tests/conftest.py", call_id="c2"),
            text("Fixed: the caller stays on hold; tests pass."),
        ]
    )
    store = MemoryCheckpointer()
    agent = _agent(model, CompletionCheck(), checkpointer=store)
    events = await _collect(agent.run("Fix hold caller", thread_id="t1"))

    verdicts = _verdicts(events)
    assert [(v.passed, v.continuation, v.reason) for v in verdicts] == [
        (False, True, "announced_step"),
        (True, False, None),
    ]
    assert verdicts[0].replanning
    end = _terminate(events)
    assert end.reason == "complete"
    assert end.final_message == "Fixed: the caller stays on hold; tests pass."

    # The model saw its own announcement, then the note, on the next call.
    after_nudge = model.received_messages[2]
    assert after_nudge[-2].role == Role.ASSISTANT
    assert after_nudge[-2].content == ANNOUNCEMENT
    assert after_nudge[-1].role == Role.USER
    assert "Carry out that step now" in (after_nudge[-1].content or "")

    # Both are history, not turn-only: the checkpoint holds them.
    saved = await store.load("t1")
    assert saved is not None
    contents = [m.content for m in saved.messages]
    assert ANNOUNCEMENT in contents
    note = next(m for m in saved.messages if m.metadata.get(CONTINUATION_NOTE_KEY))
    assert note.metadata[AUTOMATED_NOTE_KEY] is True
    assert note.metadata[CONTINUATION_NOTE_KEY] == "announced_step"
    assert not any(is_ephemeral_message(m) for m in saved.messages)
    # The conversation still alternates: no two assistant messages in a row.
    roles = [m.role for m in saved.messages]
    assert all(not (a == b == Role.ASSISTANT) for a, b in pairwise(roles))


async def test_a_nudged_turn_counts_against_the_iteration_cap() -> None:
    model = ScriptedModel([text("Let me look at the tests."), text("summary after the cap")])
    events = await _collect(_agent(model, CompletionCheck(), max_iterations=1).run("Fix it"))
    end = _terminate(events)
    assert end.reason == "max_iterations"
    assert end.iterations_used == 1
    assert _verdicts(events)[0].continuation


async def test_a_nudged_turn_counts_against_the_token_budget() -> None:
    # Each scripted call reports 30 tokens: the nudged call would be over 25.
    model = ScriptedModel([text("Let me look at the tests."), text("never reached")])
    events = await _collect(_agent(model, CompletionCheck(), token_budget=25).run("Fix it"))
    end = _terminate(events)
    assert end.reason == "token_budget"
    assert end.usage["prompt_tokens"] + end.usage["completion_tokens"] == 30
    assert len(model.received_messages) == 1


async def test_the_cap_ends_the_nudging_and_the_last_reply_stands() -> None:
    model = ScriptedModel(
        [
            text("Let me look at a."),
            tool_call("read", path="a"),
            text("Let me look at b."),
            tool_call("read", path="b", call_id="c2"),
            text("Let me look at c."),
        ]
    )
    events = await _collect(_agent(model, CompletionCheck(max_nudges=2)).run("Fix it"))
    verdicts = _verdicts(events)
    assert [v.continuation for v in verdicts] == [True, True, False]
    assert verdicts[-1].passed
    assert _terminate(events).final_message == "Let me look at c."


async def test_a_continuation_after_an_empty_reply_keeps_the_summary_text() -> None:
    # An empty reply goes through the loop's "final answer requested" call;
    # the text it returns is what the check sees and what is kept.
    model = ScriptedModel(
        [
            ModelResponse(message=Message.assistant(content=""), usage={}),
            text("I'll run the tests now."),
            tool_call("read", path="a"),
            text("Done; tests pass."),
        ]
    )
    events = await _collect(_agent(model, CompletionCheck()).run("Fix it"))
    assert _verdicts(events)[0].continuation
    assert _terminate(events).final_message == "Done; tests pass."
    after = model.received_messages[2]
    assert after[-2].content == "I'll run the tests now."


async def test_a_resumed_turn_carries_the_continuation_from_its_checkpoint() -> None:
    store = MemoryCheckpointer()
    model = ScriptedModel(
        [text("Let me check the handler."), tool_call("read", path="h.py"), text("Fixed.")]
    )
    agent = _agent(model, CompletionCheck(), checkpointer=store)
    await _collect(agent.run("Fix it", thread_id="t2"))
    saved = await store.load("t2")
    assert saved is not None
    kinds = [(m.role, bool(m.metadata.get(CONTINUATION_NOTE_KEY))) for m in saved.messages]
    assert (Role.USER, True) in kinds


# --------------------------------------------------------------- compaction --


async def test_compaction_never_takes_a_continuation_note_for_the_users_request() -> None:
    def turn(i: int) -> list[Message]:
        return [
            Message.assistant(
                content=f"step {i} " + "s" * 2_000,
                tool_calls=[ToolCall(id=f"c{i}", name="read", arguments={"path": f"f{i}"})],
            ),
            Message(role=Role.TOOL, content=f"out{i} " + "o" * 400, tool_call_id=f"c{i}"),
        ]

    messages = [Message.system("SYSTEM"), Message.user("TASK")]
    for i in range(3):
        messages += turn(i)
    messages.append(Message.user("ALSO: keep the old API working"))
    for i in range(3, 6):
        messages += turn(i)
    messages.append(Message.assistant(ANNOUNCEMENT))
    messages.append(
        Message(
            role=Role.USER,
            content="[Automated note, not from the user] carry on",
            metadata={AUTOMATED_NOTE_KEY: True, CONTINUATION_NOTE_KEY: "announced_step"},
        )
    )
    for i in range(6, 16):
        messages += turn(i)

    class Summariser:
        async def complete(self, messages: list[Message], **_: Any) -> ModelResponse:
            return ModelResponse(message=Message.assistant(content="SUMMARY"), usage={})

    compactor = ContextCompactor(context_length=10_000, summary_model=Summariser())
    outcome = await compactor.compact(messages, iteration=16, tracker=CompactionTracker())
    assert outcome is not None
    assert outcome.stage == "summarize"
    kept = [m.content for m in outcome.messages]
    assert "ALSO: keep the old API working" in kept, "the user's latest request stays"


async def test_announcements_can_be_turned_off_and_old_runs_are_forgotten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tulip.agent import completion

    assert await CompletionCheck(announcements=False)("Let me look.", _ctx()) is None
    monkeypatch.setattr(completion, "_RUNS_KEPT", 2)
    check = CompletionCheck()
    for run in ("a", "b", "c"):
        await check("Let me look.", _ctx(run_id=run))
    assert check.continuations("a") == 0, "the oldest run's state was dropped"
    assert check.continuations("c") == 1


async def test_a_no_tool_calls_termination_waits_for_the_reply_after_a_nudge() -> None:
    from tulip.core.termination import NoToolCalls

    model = ScriptedModel(
        [text("Let me look at the tests."), tool_call("read", path="t"), text("Done.")]
    )
    events = await _collect(_agent(model, CompletionCheck(), termination=NoToolCalls()).run("Fix"))
    # The nudged reply did not end the run: the model was called again and acted.
    assert len(model.received_messages) >= 2
    assert any(e.tool_name == "read" for e in events if hasattr(e, "tool_name"))
