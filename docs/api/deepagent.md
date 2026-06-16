# DeepAgent

Two complementary primitives for long-horizon research:

- **`create_deepagent`** — a plain `Agent` with reflexion + grounding, typed
  termination, and optional filesystem / todo / subagent layers. Best for
  single-agent loops.
- **`create_research_workflow`** — a `StateGraph` with a post-execution quality
  loop: execute (ReAct) → summarize → grounding eval → replan if needed. Best
  for production research where you need verifiable, grounded summaries.

## Factory — single agent

::: locus.deepagent.factory.create_deepagent

## Research workflow — StateGraph with quality loop

::: locus.deepagent.workflow.create_research_workflow
::: locus.deepagent.workflow.make_execute_node
::: locus.deepagent.workflow.make_causal_inference_node
::: locus.deepagent.workflow.make_summarize_node
::: locus.deepagent.workflow.make_grounding_eval_node
::: locus.deepagent.workflow.make_regenerate_summary_node
::: locus.deepagent.workflow.make_replan_node
::: locus.deepagent.workflow.route_after_grounding

## Subagents

::: locus.deepagent.subagent.SubAgentDef
::: locus.deepagent.subagent.task_tool

## Todos

::: locus.deepagent.todos.TodoState
::: locus.deepagent.todos.Todo
::: locus.deepagent.todos.make_todo_tools

## Filesystem

::: locus.deepagent.tools.make_filesystem_tools
::: locus.deepagent.backends.filesystem.FilesystemBackend
::: locus.deepagent.backends.state.StateBackend

## Knowledge protocol

::: locus.deepagent.protocol.KnowledgeProvider
::: locus.deepagent.protocol.KnowledgeRow
::: locus.deepagent.protocol.ItemRef
