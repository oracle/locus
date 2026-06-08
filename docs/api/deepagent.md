# DeepAgent

Two complementary primitives for long-horizon research:

- **`create_deepagent`** — a plain `Agent` with reflexion + grounding, typed
  termination, and optional filesystem / todo / subagent layers. Best for
  single-agent loops.
- **`create_research_workflow`** — a `StateGraph` with a post-execution quality
  loop: execute (ReAct) → summarize → grounding eval → replan if needed. Best
  for production research where you need verifiable, grounded summaries.

## Factory — single agent

::: tulip.deepagent.factory.create_deepagent

## Research workflow — StateGraph with quality loop

::: tulip.deepagent.workflow.create_research_workflow
::: tulip.deepagent.workflow.make_execute_node
::: tulip.deepagent.workflow.make_causal_inference_node
::: tulip.deepagent.workflow.make_summarize_node
::: tulip.deepagent.workflow.make_grounding_eval_node
::: tulip.deepagent.workflow.make_regenerate_summary_node
::: tulip.deepagent.workflow.make_replan_node
::: tulip.deepagent.workflow.route_after_grounding

## Subagents

::: tulip.deepagent.subagent.SubAgentDef
::: tulip.deepagent.subagent.task_tool

## Todos

::: tulip.deepagent.todos.TodoState
::: tulip.deepagent.todos.Todo
::: tulip.deepagent.todos.make_todo_tools

## Filesystem

::: tulip.deepagent.tools.make_filesystem_tools
::: tulip.deepagent.backends.filesystem.FilesystemBackend
::: tulip.deepagent.backends.state.StateBackend

## Knowledge protocol

::: tulip.deepagent.protocol.KnowledgeProvider
::: tulip.deepagent.protocol.KnowledgeRow
::: tulip.deepagent.protocol.ItemRef
