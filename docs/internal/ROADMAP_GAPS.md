# Locus — Gap Closure Roadmap

> Close the real gaps vs Strands, LangGraph, Google ADK, OpenAI SDK, CrewAI.
> One feature per branch. Unit + integration tests. PR and merge before next.

## Current State

- 65 real features, 2,673 tests passing
- 5 stubs that need real implementations
- 8 gaps vs competitors

---

## Phase 1: Agent Composition (HIGH priority)

### Gap 1: Agent-as-Tool

**What:** Wrap an Agent as a Tool so another agent can call it.
**Who has it:** Strands (`agent.as_tool()`), ADK (`transfer_to_agent`), CrewAI (delegation)
**Effort:** Low (1-2 days)
**Files:** `agent/agent.py`
**Implementation:**

- Add `Agent.as_tool(name, description)` method
- Returns a `Tool` that calls `agent.run_sync(prompt)` internally
- Parent agent sees it as a regular tool
- Option: `preserve_context=True` keeps sub-agent state between calls

**Tests:**

- Unit: agent.as_tool() returns Tool, parent calls sub-agent tool, sub-agent result flows back
- Integration: parent agent delegates research to sub-agent, uses result

**Status:** [ ] Not started

---

### Gap 2: Planning Step (Plan-then-Act)

**What:** Agent generates an explicit plan before executing, can replan mid-task.
**Who has it:** ADK (`PlanReActPlanner`), Gemini Deep Research
**Effort:** Medium (2-3 days)
**Files:** `agent/agent.py`, `agent/config.py`
**Implementation:**

- Add `planning: bool = False` to AgentConfig
- When enabled, first iteration calls model with planning prompt:
  "Generate a step-by-step plan before acting. Format as numbered list."
- Plan stored in state metadata
- After each iteration, check plan progress
- If stuck, inject "[Replan] Your plan isn't working. Create a new plan."

**Tests:**

- Unit: planning=True generates plan on first iteration, plan stored in state
- Integration: agent with planning produces structured plan then executes it

**Status:** [ ] Not started

---

## Phase 2: Multi-Agent Execution (MEDIUM priority)

### Gap 3: Swarm Orchestration

**What:** Multiple agents share a task queue and coordinate autonomously.
**Who has it:** Strands (SwarmNode), OpenAI SDK (Swarm pattern)
**Stub exists:** `multiagent/swarm.py` has SwarmTask, SharedContext — needs execution logic
**Effort:** Medium (2-3 days)
**Files:** `multiagent/swarm.py`
**Implementation:**

- Add `SwarmOrchestrator.run(tasks, agents)` that distributes tasks
- Agents claim tasks from shared queue
- Results aggregated into SharedContext
- Support parallel execution of independent tasks

**Tests:**

- Unit: tasks distributed, claimed, completed, results aggregated
- Integration: 3 agents process 5 tasks concurrently

**Status:** [ ] Not started

---

### Gap 4: Agent Handoff Execution

**What:** One agent transfers conversation to another mid-execution.
**Who has it:** OpenAI SDK (handoffs), Strands (SwarmNode transfer), ADK (transfer_to_agent)
**Stub exists:** `multiagent/handoff.py` has HandoffContext — needs execution
**Effort:** Medium (2-3 days)
**Files:** `multiagent/handoff.py`, `agent/agent.py`
**Implementation:**

- Add `handoff_to(target_agent, reason, context)` built-in tool
- When called, current agent yields HandoffEvent
- Target agent receives HandoffContext with conversation summary
- Target agent continues from where the first agent left off

**Tests:**

- Unit: handoff tool triggers HandoffEvent, context transferred
- Integration: agent A researches, hands off to agent B for analysis

**Status:** [ ] Not started

---

### Gap 5: Orchestrator Routing Logic

**What:** A supervisor agent that routes tasks to specialist agents.
**Who has it:** LangGraph (supervisor pattern), CrewAI (hierarchical), Optic
**Stub exists:** `multiagent/orchestrator.py` has OrchestratorResult — needs routing
**Effort:** Medium (2-3 days)
**Files:** `multiagent/orchestrator.py`
**Implementation:**

- Add `Orchestrator.route(prompt, specialists)` using LLM to select
- Specialist registry with name, description, capabilities
- Parallel specialist dispatch for independent sub-tasks
- Result synthesis from specialist outputs

**Tests:**

- Unit: routing selects correct specialist, parallel dispatch works
- Integration: orchestrator routes medical question to specialist, gets answer

**Status:** [ ] Not started

---

### Gap 6: Agent Composition Primitives

**What:** Declarative agent composition (sequential, parallel, loop).
**Who has it:** ADK (`SequentialAgent`, `ParallelAgent`, `LoopAgent`)
**Effort:** Medium (2-3 days)
**Files:** New: `agent/composition.py`
**Implementation:**

- `SequentialAgentPipeline(agents)` — run agents in order, pass output to next
- `ParallelAgentPipeline(agents)` — run agents concurrently, merge results
- `LoopAgent(agent, condition)` — run agent repeatedly until condition met
- All use existing Agent.run() internally

**Tests:**

- Unit: sequential passes output, parallel merges, loop respects condition
- Integration: pipeline of 3 agents processes complex task

**Status:** [ ] Not started

---

## Phase 3: Quality & Polish (MEDIUM-LOW priority)

### Gap 7: Evaluation Framework

**What:** Test agent quality systematically (expected outputs, tool patterns).
**Who has it:** ADK (built-in eval), LangSmith
**Effort:** Medium (3-4 days)
**Files:** New: `evaluation/` module
**Implementation:**

- `EvalCase(prompt, expected_tools, expected_output_contains, max_iterations)`
- `EvalRunner.run(agent, cases)` → EvalReport with pass/fail per case
- Compare actual tool calls vs expected
- Score output quality (keyword matching, LLM judge)

**Tests:**

- Unit: eval cases created, runner executes, report generated
- Integration: evaluate agent on 5 medical Q&A cases

**Status:** [ ] Not started

---

### Gap 8: Pre/Post Model Hooks

**What:** Run code before/after every model call (trim context, validate, log).
**Who has it:** LangGraph (`pre_model_hook`, `post_model_hook`)
**Effort:** Low (1 day)
**Files:** `agent/agent.py`, `hooks/provider.py`
**Implementation:**

- Add `on_before_model_call(messages, tools)` to HookProvider
- Add `on_after_model_call(response)` to HookProvider
- Call in `_get_model_response` before/after model.complete()

**Tests:**

- Unit: hooks called with correct args, can modify messages
- Integration: hook trims context before model call

**Status:** [ ] Not started

---

## Implementation Order

```
Phase 1 (do first — biggest gaps):
  Gap 1: Agent-as-Tool          ← start here
  Gap 2: Planning Step

Phase 2 (fill the stubs):
  Gap 3: Swarm Orchestration
  Gap 4: Agent Handoff
  Gap 5: Orchestrator Routing
  Gap 6: Composition Primitives

Phase 3 (polish):
  Gap 7: Evaluation Framework
  Gap 8: Pre/Post Model Hooks
```

Each gap: branch → implement → unit tests → integration tests → PR → merge.
