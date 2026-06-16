# Locus Agentic Loop — Feature Roadmap

> Each feature is independent, implemented and tested before moving to the next.

## Current State

The ReAct loop works but has fake/stub reasoning, no production safeguards, and missing patterns that every competing framework has.

---

## Feature 1: Tool Result Truncation

**Priority:** Critical — one large tool result can blow the entire context window
**Files:** `agent/config.py`, `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- Add `max_tool_result_length: int = 32000` to `AgentConfig`
- After each tool execution in `agent.py`, truncate `result.content` if over limit
- Append `\n[OUTPUT TRUNCATED — original: {N} chars]`
- Test: tool returning 100K chars gets truncated to 32K + notice

**Status:** [x] Complete (4 tests added, 83 agent tests passing)

---

## Feature 2: Message Validation (Orphan Cleanup)

**Priority:** Critical — OCI GenAI/Cohere rejects orphaned tool calls
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- Add `_validate_messages(messages)` before each model call
- Ensure every `assistant` message with `tool_calls` has matching `tool` result messages
- Remove orphaned tool calls/results that would cause provider errors
- Test: messages with orphaned tool calls get cleaned up

**Status:** [ ] Not started

---

## Feature 3: Malformed Tool Call Recovery

**Priority:** High — Cohere on OCI GenAI sometimes outputs tool calls as text
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- When model returns no structured tool calls but response text contains `tool_name(arg="value")` patterns, parse them
- Regex extraction with case-insensitive name matching against tool registry
- Falls back gracefully — if parsing fails, treat as normal text response
- Test: model text response `search_web(query="test")` gets parsed into ToolCall

**Status:** [ ] Not started

---

## Feature 4: Config Expansion (Budgets + Limits)

**Priority:** High — foundation for other features
**Files:** `agent/config.py`, `agent/result.py`, `core/state.py`
**Test:** `tests/unit/test_agent_config.py`, `tests/unit/test_state.py`

- Raise `max_iterations` cap from 100 → 500 (default stays 20)
- Add `token_budget: int | None = None` to AgentConfig
- Add `time_budget_seconds: float | None = None` to AgentConfig
- Add `total_tokens_used`, `prompt_tokens_used`, `completion_tokens_used` to AgentState
- Add `with_token_usage()` method to AgentState
- Add `"token_budget"`, `"time_budget"` to StopReason
- Update `should_terminate` to check token budget
- Test: state with token budget terminates when exceeded

**Status:** [ ] Not started

---

## Feature 5: Real Token Tracking + Budget Enforcement

**Priority:** High — real usage tracking instead of char/4 estimate
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- After each `_get_model_response`, call `state.with_token_usage()` with real counts from `ModelResponse.usage`
- Add time budget check at top of while loop
- Test: agent with token_budget=1000 stops after exceeding; agent with time_budget=1.0 stops after 1 second

**Status:** [ ] Not started

---

## Feature 6: Auto Conversation Manager

**Priority:** High — context protection for long runs
**Files:** `agent/agent.py`, `memory/conversation.py`
**Test:** `tests/unit/test_agent.py`, `tests/unit/test_conversation.py`

- Add `_conversation_manager` private attr to Agent
- In `_initialize()`: if none specified and max_iterations > 10, auto-create `SlidingWindowManager(window_size=max(20, max_iterations*2))`
- Add `async_apply()` to SummarizingManager for async LLM summarization
- Use async_apply in `_get_model_response` when available
- Test: agent with 30+ messages gets context trimmed; async_apply calls async summarize_fn

**Status:** [ ] Not started

---

## Feature 7: Real Reflector (Replace Fake Reflexion)

**Priority:** High — the biggest quality uplift
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- Import and instantiate `Reflector` from `reasoning/reflexion.py` in `_initialize()`
- Replace `_apply_reflexion` body (lines 530-580) with `Reflector.reflect()` + `adjust_state_confidence()`
- After reflexion, inject guidance as system message `[Agent Self-Reflection]\n{guidance}` when stuck/looping
- Pass `iteration_executions` for current-iteration awareness
- Test: agent in tool loop gets guidance injected; stuck agent's confidence decreases; assessment categories correct

**Status:** [ ] Not started

---

## Feature 8: Real Grounding Evaluator (Replace Stub)

**Priority:** High — validates agent output against tool evidence
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- Import and instantiate `GroundingEvaluator` from `reasoning/grounding.py` in `_initialize()`
- Replace `_apply_grounding` body (lines 582-602) with `evaluate_with_llm()`
- Add `_extract_claims()` and `_gather_evidence()` helpers
- Call grounding before final response (when model returns no tool calls)
- On failure: inject `[Grounding Check Failed]\n{replan_guidance}` and `continue` loop
- Respects `max_replans` (default 2)
- Test: agent with ungrounded claim triggers replan; grounded response passes; max_replans respected

**Status:** [ ] Not started

---

## Feature 9: Graceful Max-Iterations

**Priority:** Medium — better UX than hard stop
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- When hitting max_iterations with pending work, inject system message asking model to summarize findings
- Do ONE more iteration for the summary (grace iteration)
- Return partial results instead of empty termination
- Test: agent hitting max_iterations produces summary instead of bare stop

**Status:** [ ] Not started

---

## Feature 10: Fix run_sync State Preservation

**Priority:** Medium — run_sync currently loses all state
**Files:** `agent/agent.py`
**Test:** `tests/unit/test_agent.py`

- Add `_last_run_state` private attr
- Write final state in `run()` finally block
- Rewrite `run_sync` to use actual final state
- Populate full `ExecutionMetrics` from real state
- Test: run_sync result has tool_executions, reasoning_steps, real token counts

**Status:** [ ] Not started

---

## Implementation Order

```
Feature 1: Tool Result Truncation        ← start here (smallest, isolated)
Feature 2: Message Validation
Feature 3: Malformed Tool Call Recovery
Feature 4: Config Expansion (Budgets)
Feature 5: Real Token Tracking
Feature 6: Auto Conversation Manager
Feature 7: Real Reflector                ← biggest quality uplift
Feature 8: Real Grounding               ← LLM-as-judge
Feature 9: Graceful Max-Iterations
Feature 10: Fix run_sync
```

Each feature: implement → unit test → integration test → commit.
