# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A completion check: a turn without tool calls is not always the end.

The loop ends a run on the first model reply that calls no tool. That is the
right rule for a model that only stops when it is done, and the wrong one for
the open-weight models that often write the *announcement* of their next step
and stop there — "Let me first check the conftest and the specific test
area:" — with the tool call never made. Unattended, the run then exits 0 with
nothing changed, and a supervisor reads that as success.

:class:`CompletionCheck` is a ``final_answer_verifier`` that rejects such a
stop and sends the model back to work with a short note::

    check = CompletionCheck(max_nudges=3)
    agent = Agent(model=..., **check.agent_options())

It sends the model back when:

``announced_step``   the reply ends by announcing an action it did not take
                     (:func:`announced_step`: a trailing ``:``, or a last
                     sentence like "Let me…", "I'll…", "Next, I…", "Voy a…",
                     "让我…");
``no_changes``       the caller's ``needs_changes`` signal says the task wants
                     changes and none were made, and the reply does not say why
                     none were needed (:func:`explains_no_change`);
``unchecked_edits``  (opt-in, once per run) files were edited and no test or
                     check command ran after the last edit
                     (:func:`edits_unchecked`).

The nudges are bounded: at most ``max_nudges`` per run, and never the same
reason twice in a row — a second stop for the same reason with no tool call in
between is accepted, since the model has heard the note and chose to stop.
``insist_on_changes=True`` lifts that rule for ``no_changes``: an unattended
run that changed nothing is sent back until it changes something, says why
nothing needed changing, or reaches the cap. Each
nudge is a :class:`Continuation`: the loop keeps the model's reply in the
conversation as an ordinary assistant message and adds the note as an automated
user-role message (not turn-only, unlike a rejected answer), so the model sees
what it said, a checkpoint holds it, and compaction treats both as history. A
nudged turn is a model call like any other: it counts against
``max_iterations`` and every budget.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tulip.tools.action_fusion import RAN, fused_command


if TYPE_CHECKING:
    from tulip.agent.verification import FinalAnswerContext, FinalAnswerVerifier
    from tulip.core.state import ToolExecution


__all__ = [
    "AUTOMATED_NOTE_KEY",
    "CONTINUATION_NOTE_KEY",
    "CompletionCheck",
    "Continuation",
    "ContinuationReason",
    "announced_step",
    "chain_verifiers",
    "edits_unchecked",
    "explains_no_change",
    "max_replans_for",
    "requests_changes",
]

#: ``Message.metadata`` key on a user-role message the loop wrote rather than
#: the user (a continuation note). Compaction never takes one for the user's
#: latest request, and a session listing does not count it as a turn.
AUTOMATED_NOTE_KEY = "tulip_automated_note"

#: ``Message.metadata`` key on a continuation note; the value is its reason.
CONTINUATION_NOTE_KEY = "tulip_continuation"

ContinuationReason = Literal["announced_step", "no_changes", "unchecked_edits"]


class Continuation(str):  # noqa: SLOT000 — str subtypes cannot have non-empty slots
    """Verifier feedback that sends the model back to unfinished work.

    A ``str``, so every verifier consumer that expects text still works. The
    loop recognises the type: the reply it answers stays in the conversation
    (a rejected *answer* is dropped instead), and the
    :class:`~tulip.core.events.FinalAnswerVerificationEvent` carries
    ``continuation=True`` and the ``reason``.
    """

    reason: str

    def __new__(cls, text: str, reason: str) -> Continuation:
        obj = super().__new__(cls, text)
        obj.reason = reason
        return obj


# ------------------------------------------------------------- the detector --


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


#: English action verbs that, after "let me" / "I'll" / "I'm going to", name a
#: step to take. A closed list on purpose: "I'll leave it as is", "Let me know"
#: and "I'll be happy to help" end real answers, and an open pattern would
#: send those back.
_VERBS = _words(
    "add address adjust analyse analyze apply attempt begin build call change check "
    "clean commit compare compile configure confirm continue create debug delete dig "
    "edit examine execute explore fetch find fix format go grep handle implement "
    "inspect install investigate launch lint list locate look make modify move open "
    "patch proceed profile query re-check re-run read rebuild recheck refactor remove "
    "rename repair replace rerun restore retry revert review rewrite run scan search "
    "see set start study take tackle test trace try update use validate verify view "
    "work write"
)

#: The -ing forms, for "I'm now checking…" / "Now running the tests…".
_GERUNDS = _words(
    "adding adjusting analysing analyzing applying building calling changing checking "
    "cleaning comparing compiling creating debugging deleting digging editing "
    "examining executing exploring fetching fixing grepping implementing inspecting "
    "installing investigating looking making modifying moving opening patching "
    "reading rebuilding refactoring removing renaming repairing replacing rerunning "
    "restoring retrying reverting reviewing rewriting running scanning searching "
    "starting testing tracing trying updating validating verifying viewing working "
    "writing"
)

_VERB = r"(?:" + "|".join(re.escape(v) for v in sorted(_VERBS, key=len, reverse=True)) + r")\b"
_GERUND = r"(?:" + "|".join(_GERUNDS) + r")\b"

#: Fillers allowed between the intent and the verb: "I'll now go ahead and run".
_FILL = (
    r"(?:(?:now|first|also|then|next|quickly|just|briefly|carefully|again|actually|"
    r"go ahead and|start by|begin by|proceed to|continue by|continue to|try to|need to|"
    r"have to|want to)\s+)*"
)

#: A sentence that announces a step, matched at its start (after connectors
#: such as "Now," or "Next," are stripped).
_ANNOUNCE = re.compile(
    r"^(?:"
    # English: Let me / Let's / I'll / I will / I'm going to / I need to / We'll …
    rf"let me\s+(?!know\b){_FILL}{_VERB}"
    rf"|let['’]?s\s+{_FILL}{_VERB}"
    r"|i(?:['’]ll| will| shall|['’]m going to| am going to|['’]m gonna|['’]m about to"
    rf"| am about to| need to| have to| should| must)\s+{_FILL}{_VERB}"
    rf"|we(?:['’]ll| will|['’]re going to| are going to)\s+{_FILL}{_VERB}"
    rf"|(?:i['’]m|i am|we['’]re|we are)\s+(?:now\s+)?{_GERUND}"
    rf"|going to\s+{_FILL}{_VERB}"
    # Spanish
    r"|(?:voy a|vamos a|déjame|dejame|permíteme|permiteme|procedo a|paso a|pasaré a)\s+\w"
    # Portuguese
    r"|(?:vou|vamos|deixe-me|deixa-me|deixa eu)\s+\w"
    # French
    r"|(?:je vais|nous allons|laissez-moi|laisse-moi|allons)\s+\w"
    # German
    r"|(?:ich werde|wir werden|lass mich|lassen sie mich|ich schaue mir|ich sehe mir"
    r"|ich prüfe|ich überprüfe)\b"
    # Italian
    r"|(?:lasciami|fammi|vado a|ora controllo|ora verifico|ora eseguo|adesso controllo)\b"
    # Russian
    r"|(?:давайте|давай|я проверю|я посмотрю|сейчас проверю|сейчас посмотрю|посмотрю|проверю)\b"
    # Chinese
    r"|(?:让我|我来|我将|我会|我先|我需要|我要|下面我|接下来我|现在我)"
    r")",
    re.IGNORECASE,
)

#: Japanese and Korean put the verb last: "…を確認します。", "…확인하겠습니다".
_ANNOUNCE_TAIL = re.compile(
    r"(?:してみます|てみます|しましょう|していきます|を確認します|を実行します|を修正します|"
    r"を読みます|を見ます|見てみます|보겠습니다|하겠습니다)"
    r"\s*[。.!！…]*$"
)

#: Connectors stripped from a sentence's start before matching: a word and a
#: separator for the languages that space their words, the bare word for those
#: that do not.
_LEAD = re.compile(
    r"^(?:(?:(?:now|next|then|first|firstly|so|okay|ok|alright|all right|great|good|perfect|"
    r"finally|and|right|got it|understood|i see|ahora|primero|luego|agora|primeiro|"
    r"maintenant|d'abord|ensuite|jetzt|zuerst|als nächstes|ora|adesso|теперь|сначала|"
    r"далее)(?:\s*[,，.!:：—–-]\s*|\s+))"
    r"|(?:首先|接下来|然后|现在|下面|まず|次に|それでは|では)[,，、\s]*)+",
    re.IGNORECASE,
)

#: An announcement that defers to a later time or to the user is a real ending:
#: "I'll fix the docs in a follow-up PR", "I'll run it if you want".
_DEFERRED = re.compile(
    r"\b(?:follow[- ]?up|later|next time|separately|in a (?:future|separate|new)|another (?:pr|"
    r"change|commit|session)|if you(?:'d| would)? (?:like|want|prefer)|if you (?:say|confirm|"
    r"agree|approve)|once you|when you|should you|on request|upon request|whenever|each time|"
    r"every time|as needed|(?:after|before) (?:any |each |every )?(?:changes?|edits?|editing))\b",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
_MARKUP = re.compile(r"^[\s>*_#`\-•]*(?:\d+[.)]\s*)?")
_TRAILING_MARKUP = re.compile(r"[\s*_`]+$")

#: A sentence that opens with a gerund ("Running the full suite…").
_GERUND_START = re.compile(rf"^{_GERUND}", re.IGNORECASE)

#: Characters after a colon inside the last sentence that make it content
#: rather than an introduction.
_CONTENT_AFTER_COLON = 20

#: How much of the announcement a nudge quotes back to the model.
_SNIPPET_CHARS = 160


def _clip(text: str, limit: int = _SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def announced_step(reply: str) -> str | None:
    """The trailing announcement of an untaken step in ``reply``, or ``None``.

    Conservative by design — a false positive costs a model call and annoys,
    a false negative costs only what the loop already did without this. A
    reply counts as announcing a step when, after trailing whitespace and
    emphasis markup are stripped:

    - it ends with a colon (``:`` or ``：``) — in any language, a reply that
      ends introducing something that never comes; or
    - its last sentence (not a question, not deferred to later or to the
      user) starts, after connectors like "Now," or "Next,", with a
      first-person intent and an action verb: "Let me check…", "I'll run…",
      "I'm going to update…", "Now running…", or the Spanish, Portuguese,
      French, German, Italian, Russian, Chinese, Japanese and Korean forms of
      the same.

    A reply that ends in a code block, a list item or a plain statement is
    never one.

    Returns:
        The announcing sentence, clipped, for the nudge to quote back.
    """
    text = reply.strip()
    if not text or text.endswith("```"):
        return None
    tail = _TRAILING_MARKUP.sub("", text)
    if not tail:
        return None
    lines = [line for line in tail.splitlines() if line.strip()]
    last_line = lines[-1].strip()
    if tail.endswith((":", "：")):
        return _clip(_MARKUP.sub("", last_line) or last_line)
    sentences = [s for s in _SENTENCE_END.split(last_line) if s.strip()]
    if not sentences:  # pragma: no cover — last_line is non-empty
        return None
    sentence = _MARKUP.sub("", sentences[-1]).strip().strip("*_")
    if not sentence or sentence.rstrip("*_ ").endswith(("?", "？")):
        return None
    if _DEFERRED.search(sentence):
        return None
    _, colon, rest = sentence.partition(":")
    if colon and len(rest.strip()) >= _CONTENT_AFTER_COLON:
        # "Let me walk through it: the bug was …" — the step is in the reply.
        return None
    bare = _LEAD.sub("", sentence)
    if _ANNOUNCE.match(bare) or _ANNOUNCE_TAIL.search(sentence):
        return _clip(sentence)
    if _GERUND_START.match(bare) and sentence.endswith(("...", "…")):
        # "Running the tests now…" trails off; "Running the tests showed X." does not.
        return _clip(sentence)
    return None


# --------------------------------------------------- did it change anything --

#: Verbs that, at the start of a sentence or clause of a request, ask for a
#: change to be made. Verbs that as often ask for information ("show",
#: "list", "check", "review", "report", "test", "build", "generate") are left
#: out: a question sent back for "no changes" is a worse failure than a task
#: missed here, which only loses a nudge.
_CHANGE_VERBS = _words(
    "accept add adjust allow bump cache change clean configure convert correct create "
    "define delete deprecate disable document drop emit enable ensure expose extend "
    "extract fix fold give guard handle hold hook implement improve introduce keep log "
    "make migrate modify move optimise optimize patch persist port protect publish "
    "record refactor refuse register reject remove rename reorganise reorganize replace "
    "resolve restore restructure retry rework rewrite route serve set split store "
    "support track update upgrade wire wrap write"
)

_POLITE = re.compile(
    r"^(?:(?:please|pls|kindly|now|then|also|and|so|ok|okay|go ahead and|can you|could you|"
    r"would you|will you|i want you to|i need you to|i'd like you to|i would like you to|"
    r"we need to|you need to|you should|your task is to|the task is to|task:|goal:|"
    r"todo:|help me)\s*[,:]?\s*)+",
    re.IGNORECASE,
)
#: Clause boundaries: a line break (specs are written one requirement a line),
#: sentence ends, and "and"/"then" joining two requests.
_CLAUSES = re.compile(r"\n+|(?<=[.!?;:])\s+|\s+(?:and then|and|then)\s+|,\s*then\s+", re.IGNORECASE)
#: A change verb, not followed by "me"/"us" ("give me a summary" asks for text).
_CHANGE_START = re.compile(
    r"^(?:" + "|".join(_CHANGE_VERBS) + r")\b(?!\s+(?:me|us)\b)",
    re.IGNORECASE,
)
#: "Let people ask…", "let the agent retry…": a change to what someone can do.
#: "Let me know" and "let's" are conversation, not a request.
_LET = re.compile(r"^let\s+(?!me\b|us\b|['’]s\b)\w+\s+\w+", re.IGNORECASE)
#: A requirement stated rather than ordered: "The response must contain…",
#: "Tasks should get the same treatment".
_REQUIREMENT = re.compile(
    r"\b(?:must|should|shall|needs? to|has to|have to)\s+(?:not\s+|also\s+|still\s+|now\s+)?"
    r"(?!be\s+(?:asked|answered|explained)\b)\w+",
    re.IGNORECASE,
)
#: Clauses that ask for an answer, whatever modal follows: "Explain why it
#: should pass", "Tell me what must change".
_ASKS = re.compile(
    r"^(?:explain|describe|summari[sz]e|tell|show|list|find|where|what|why|how|when|who|"
    r"which|is|are|does|do|did|can|could|would|should|will|walk|compare|review|read|look)\b",
    re.IGNORECASE,
)
#: An interface section: a list item that names a new function, field or
#: file and says what it is ("- `generation_dir()`: the live generation's
#: directory", "- New `policy/evidence.py`: …"), or an acceptance line.
_INTERFACE_ITEM = re.compile(
    r"^(?:[-*•]|\d+[.)])\s+(?:new\s+`|`[^`\n]+`\s*(?:[:—–]|\s-\s|\())",
    re.IGNORECASE,
)
_DONE_WHEN = re.compile(r"^(?:done when|acceptance criteria|definition of done)\b", re.IGNORECASE)


def _bare(clause: str) -> str:
    return _POLITE.sub("", _MARKUP.sub("", clause.strip())).strip()


def _orders_change(clause: str) -> bool:
    """A clause that orders a change: "Fix…", "Can you add…?", "Let people…"."""
    bare = _bare(clause)
    return bool(_CHANGE_START.match(bare) or _LET.match(bare))


def _states_requirement(clause: str) -> bool:
    """A statement (not a question, not a request to explain) of what must hold."""
    raw = clause.strip()
    if raw.endswith(("?", "？")):
        return False
    bare = _bare(raw)
    return not _ASKS.match(bare) and bool(_REQUIREMENT.search(bare))


def _specifies(line: str) -> bool:
    """An interface-section item or an acceptance line."""
    raw = line.strip()
    return not raw.endswith(("?", "？")) and bool(
        _INTERFACE_ITEM.match(raw) or _DONE_WHEN.match(raw)
    )


def requests_changes(prompt: str) -> bool:
    """Whether ``prompt`` reads as asking for changes to be made.

    True when a sentence, line or clause — after "please", "can you", "your
    task is to" and the like — asks for a change:

    - it starts with a verb that orders one: "Fix the flaky test", "Can you
      add a --dry-run flag?", "Review the module and fix what you find",
      "Serve the arguments from this plane";
    - it starts with "Let" and someone other than the speaker: "Let people
      ask luach about their email";
    - it states a requirement: "The response must contain the totals",
      "Tasks should get the same treatment";
    - it is an interface section's item naming what to add ("- `search(query)`:
      …", "- New `evidence.py`: …") or a "Done when:" line.

    A question ("what does admit() do?", "why does this fail?"), a request for
    an explanation ("Explain why the test should pass") and "Let me know…" are
    not. English only; for other languages it says False, which only turns
    the ``no_changes`` nudge off.
    """
    clauses = [c for c in _CLAUSES.split(prompt) if c.strip()]
    if any(_orders_change(c) for c in clauses):
        return True
    # A prompt that opens by asking ("Explain these:", "What…?") lists things
    # to explain, not to add, whatever shape its list takes.
    if clauses and (clauses[0].strip().endswith(("?", "？")) or _ASKS.match(_bare(clauses[0]))):
        return False
    return any(_states_requirement(c) for c in clauses) or any(
        _specifies(line) for line in prompt.splitlines()
    )


_NO_CHANGE = re.compile(
    r"\b(?:"
    r"no (?:code |source |file )?(?:changes?|modifications?|edits?|fix(?:es)?) (?:is |are |was |were )?"
    r"(?:needed|required|necessary)"
    r"|nothing (?:needs|needed|has|had) to (?:change|be changed|be fixed|be done)"
    r"|nothing to (?:change|fix|do)"
    r"|(?:is|are|was|were|it's|it’s) already (?:implemented|fixed|handled|supported|present|"
    r"in place|correct|passing|done|there|the case|configured|covered)"
    r"|already (?:passes|pass|works|exists|exist|does|handles|supports)"
    r"|did(?:n't|n’t| not) (?:need to )?(?:change|modify|edit|touch)"
    r"|(?:made|make) no (?:changes?|modifications?|edits?)"
    r"|(?:left|leaving|leave) (?:the )?(?:code|files?|source|it|them) (?:unchanged|as is|alone|untouched)"
    r"|(?:cannot|can't|can’t|could not|couldn't|couldn’t|unable to|not able to|was not able to|"
    r"wasn't able to) "
    r"|not possible|refused|was denied|were denied|permission denied|not permitted|blocked by"
    r"|(?:won't|will not|should not|shouldn't) (?:change|modify|edit)"
    r"|works as (?:intended|expected|designed)|not a bug|no bug|no issue"
    r"|(?:need|needs) (?:more information|clarification|your (?:input|decision|confirmation))"
    r")",
    re.IGNORECASE,
)


#: Not a reason: the run blaming its own stop. "The system requested my final
#: answer before I could make the edits", "I ran out of iterations" — the
#: work is undone and the loop can still give the model another turn.
_NOT_A_REASON = re.compile(
    r"\b(?:"
    r"(?:system|harness|loop|runtime|you)\s+(?:has\s+|have\s+)?(?:requested|asked|told|wants?)"
    r"\s+(?:me\s+)?(?:for\s+|to\s+(?:give|provide|write)\s+)?(?:my|a|the)\s+final\s+"
    r"(?:answer|summary|response)"
    r"|before i (?:could|was able to|had (?:a chance|time) to|got to)"
    r"|ran out of (?:time|iterations|turns|steps|budget)"
    r"|(?:iteration|turn|step) limit"
    r"|(?:was|were|am) (?:asked|told) to stop"
    r"|did(?:n't|n’t| not) (?:get|have) (?:a |the )?(?:chance|time|opportunity)"
    r")",
    re.IGNORECASE,
)


def explains_no_change(reply: str) -> bool:
    """Whether ``reply`` says why nothing was changed.

    "No changes were needed", "it is already implemented", "I could not…",
    "the edit was refused", "works as intended" — a reply that gives a reason
    is accepted even when the task looked like it asked for changes. A reply
    that blames the run's own stop ("the system requested my final answer
    before I could make the edits", "I ran out of iterations") gives none:
    that is the work left undone, not a reason it needed no doing.
    """
    if _NOT_A_REASON.search(reply):
        return False
    return bool(_NO_CHANGE.search(reply))


#: Tools that edit files, by the names the SDK's own tools and the common
#: coding-agent tool sets use.
DEFAULT_EDIT_TOOLS = frozenset(
    {
        "write",
        "write_file",
        "edit",
        "edit_file",
        "multi_edit",
        "apply_patch",
        "notebook_edit",
        "str_replace",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "create_file",
        "replace_in_file",
    }
)

#: Tools that run a command line.
DEFAULT_SHELL_TOOLS = frozenset({"bash", "shell", "run_command", "execute_command", "exec"})

#: A command line that tests, type-checks, lints, builds or runs the code.
CHECK_COMMAND = re.compile(
    r"\b(?:pytest|py\.test|tox|nox|unittest|hatch\s+(?:run\s+)?(?:test|check|lint|typecheck)|"
    r"uv\s+run|poetry\s+run|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|lint|build|check|"
    r"typecheck)|npx\s+\S+|jest|vitest|mocha|tsc|eslint|ruff|mypy|pyright|flake8|pylint|"
    r"black\s+--check|go\s+(?:test|build|vet|run)|cargo\s+(?:test|check|build|clippy|run)|"
    r"make\b|ctest|cmake\s+--build|mvn|gradle|gradlew|dotnet\s+(?:test|build)|rspec|"
    r"bundle\s+exec|phpunit|swift\s+(?:test|build)|deno\s+test|bun\s+test|node\s+--test|"
    r"bazel\s+(?:test|build)|python[0-9.]*\s+(?:-m\s+\S+|-c\s|\S+\.py)|node\s+\S+\.[cm]?js)"
)


def edits_unchecked(
    executions: Iterable[ToolExecution],
    *,
    edit_tools: Iterable[str] = DEFAULT_EDIT_TOOLS,
    shell_tools: Iterable[str] = DEFAULT_SHELL_TOOLS,
    check: re.Pattern[str] = CHECK_COMMAND,
) -> bool:
    """Whether a file was edited and no check command ran after the last edit.

    ``executions`` is the run's tool record (``FinalAnswerContext.
    tool_executions``). An edit is a successful call of one of ``edit_tools``;
    a check is a call of one of ``shell_tools`` whose ``command`` argument
    matches ``check`` — run, not passed: a failing test run still told the
    model something. An edit that carried its check as ``then_run``
    (:mod:`tulip.tools.action_fusion`) and ran it is an edit and a check at
    once; one whose command was skipped is only an edit.
    """
    edits = frozenset(edit_tools)
    shells = frozenset(shell_tools)
    unchecked = False
    for execution in executions:
        if execution.tool_name in edits and execution.error is None:
            fused = fused_command(execution.arguments)
            ran = fused is not None and RAN in (execution.result or "")
            unchecked = not (ran and check.search(fused or ""))
        elif execution.tool_name in shells and check.search(
            str(execution.arguments.get("command", ""))
        ):
            unchecked = False
    return unchecked


# ------------------------------------------------------------------ the check --

#: Runs whose nudge state is kept. A check is shared by every run of an agent;
#: past this many, the oldest run's state is dropped.
_RUNS_KEPT = 256


@dataclass
class _RunState:
    nudges: int = 0
    last_reason: str | None = None
    tools_at_last: int = -1
    unchecked_sent: bool = False


def _announced_note(snippet: str) -> str:
    return (
        "[Automated note, not from the user] Your last message ended by announcing a "
        f'next step ("{snippet}") without taking it: no tool was called, so the run would '
        "end here with the work unfinished. Carry out that step now, calling the tools it "
        "needs, and keep going until the task is done. If the task is truly complete, say "
        "so plainly and state what you verified."
    )


_NO_CHANGES_NOTE = (
    "[Automated note, not from the user] The task asks for changes, but no file has "
    "changed since this run started and your reply does not say why none were needed. "
    "Make the changes now, using the tools. If nothing needs to change, say so plainly "
    "and explain what you checked."
)

_UNCHECKED_NOTE = (
    "[Automated note, not from the user] You changed files but ran no test or check "
    "command after your last edit. Run the relevant tests or checks now and fix what "
    "they show. If there is nothing that can be run, say so plainly."
)


@dataclass
class CompletionCheck:
    """A ``final_answer_verifier`` that sends back a stop before the work is done.

    Attributes:
        max_nudges: Continuations per run, at most. 0 turns the check off.
        announcements: Send back a reply that announces an untaken step.
        needs_changes: ``(draft, ctx) -> bool``: True when the task needs
            changes and none have been made. Supplied by the caller, which
            knows what "a change" is (files on disk, rows in a table); the
            nudge is skipped when the reply explains why nothing changed.
            ``None`` turns the ``no_changes`` reason off.
        unchecked_edits: ``True`` to nudge once per run when files were edited
            and no check ran after the last edit (:func:`edits_unchecked`
            with the default tool names), or a ``(draft, ctx) -> bool`` of the
            caller's own. Off by default: not every task has a check to run.
        on_continuation: Called with ``(reason, ctx)`` for each nudge sent.
        insist_on_changes: Send a ``no_changes`` stop back every time, up to
            ``max_nudges``, until files change or the reply says why none
            needed to — rather than accepting the second such stop in a row as
            the model's choice. For unattended runs, where nobody reads the
            reply: a model that answers a "make the changes" note with more
            prose, or with a call the provider did not deliver, has not chosen
            anything.
    """

    max_nudges: int = 3
    announcements: bool = True
    needs_changes: Callable[[str, FinalAnswerContext], bool] | None = None
    unchecked_edits: bool | Callable[[str, FinalAnswerContext], bool] = False
    on_continuation: Callable[[str, FinalAnswerContext], None] | None = None
    insist_on_changes: bool = False
    _runs: OrderedDict[str, _RunState] = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_nudges < 0:
            raise ValueError("max_nudges must be >= 0")

    def _state(self, run_id: str) -> _RunState:
        state = self._runs.get(run_id)
        if state is None:
            state = self._runs[run_id] = _RunState()
            while len(self._runs) > _RUNS_KEPT:
                self._runs.popitem(last=False)
        return state

    def continuations(self, run_id: str) -> int:
        """How many continuations the run ``run_id`` has been sent."""
        state = self._runs.get(run_id)
        return state.nudges if state is not None else 0

    def _unchecked(self, draft: str, ctx: FinalAnswerContext) -> bool:
        if callable(self.unchecked_edits):
            return bool(self.unchecked_edits(draft, ctx))
        return bool(self.unchecked_edits) and edits_unchecked(ctx.tool_executions)

    def _candidates(
        self, draft: str, ctx: FinalAnswerContext, state: _RunState
    ) -> Iterable[tuple[str, str]]:
        """``(reason, note)`` for each reason that applies, most specific first."""
        if self.announcements:
            snippet = announced_step(draft)
            if snippet is not None:
                yield "announced_step", _announced_note(snippet)
        if (
            self.needs_changes is not None
            and self.needs_changes(draft, ctx)
            and not explains_no_change(draft)
        ):
            yield "no_changes", _NO_CHANGES_NOTE
        if not state.unchecked_sent and self._unchecked(draft, ctx):
            yield "unchecked_edits", _UNCHECKED_NOTE

    async def __call__(self, draft: str, ctx: FinalAnswerContext) -> str | None:
        """Accept the stop (``None``) or send the model back (:class:`Continuation`)."""
        if self.max_nudges <= 0 or ctx.attempt >= ctx.max_replans:
            # No replan is left for anyone: a nudge now would be judged and
            # ignored, so the stop stands.
            return None
        state = self._state(ctx.run.run_id)
        if state.nudges >= self.max_nudges:
            return None
        tools = len(ctx.tool_executions)
        for reason, note in self._candidates(draft, ctx, state):
            if (
                reason == state.last_reason
                and tools == state.tools_at_last
                and not (reason == "no_changes" and self.insist_on_changes)
            ):
                # The model was just sent back for this and stopped again
                # without calling a tool: it heard the note and chose to stop.
                continue
            state.nudges += 1
            state.last_reason = reason
            state.tools_at_last = tools
            if reason == "unchecked_edits":
                state.unchecked_sent = True
            if self.on_continuation is not None:
                self.on_continuation(reason, ctx)
            return Continuation(note, reason)
        return None

    def agent_options(self) -> dict[str, Any]:
        """``Agent(...)`` keyword arguments that run this check."""
        return {
            "final_answer_verifier": self,
            "final_answer_verifier_max_replans": min(10, max(1, self.max_nudges)),
        }


def chain_verifiers(*verifiers: FinalAnswerVerifier | None) -> FinalAnswerVerifier | None:
    """One ``final_answer_verifier`` that runs ``verifiers`` in order.

    The first that rejects decides; later ones run only when every earlier
    one accepted. ``None`` entries are skipped, so optional verifiers chain
    without branching; with none left the result is ``None``.

    An agent takes one ``final_answer_verifier``, and a program often has
    several — a completion check, a structured-output reminder, a ``Stop``
    hook. Every rejection spends one of the agent's replans whichever member
    made it, so give the agent the sum of theirs (:func:`max_replans_for`).
    """
    chain = tuple(v for v in verifiers if v is not None)
    if not chain:
        return None
    if len(chain) == 1:
        return chain[0]

    async def verify(draft: str, ctx: FinalAnswerContext) -> str | None:
        for verifier in chain:
            feedback = await verifier(draft, ctx)
            if feedback:
                return feedback
        return None

    return verify


def max_replans_for(*counts: int | None) -> int:
    """The replans a chain needs: the sum of its members', within the SDK's 0–10."""
    return min(10, sum(c for c in counts if c))
