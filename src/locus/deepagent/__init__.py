# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Deep-research scaffolding atop locus primitives.

A "deep agent" is a research-shaped Agent that:

- Loops with tools until it submits a typed row via a designated
  ``submit`` tool, hits a confidence threshold, or exhausts a token /
  iteration budget — all expressed as a typed termination condition.
- Reflexion + grounding evaluation are on by default so the agent
  catches its own hallucinations against the tool-call evidence trail.
- Output is a Pydantic schema (``output_schema=``) enforced by the
  model provider's strict structured-output mode — no regex JSON
  parser downstream.
- Optional checkpointer persists per-thread state so multi-day scans
  resume mid-flight after a crash.

This module is a *convenience layer* over ``locus.Agent``. It bundles
the standard knobs into a single factory and ships a Provider protocol
so projects can describe a research surface declaratively.

Quick start::

    from locus import (
        create_deepagent,
    )  # or `from locus.deepagent import create_deepagent`
    from pydantic import BaseModel


    class TableInfo(BaseModel):
        schema_owner: str
        table: str
        confidence: float


    agent = create_deepagent(
        model="oci:openai.gpt-5.5",
        tools=[list_tables, query_db, submit_research],
        system_prompt="You research Oracle tables. Submit when confident.",
        output_schema=TableInfo,
    )
    async for ev in agent.run("Research FUSION.AP_INVOICES_ALL on EJOF"):
        ...

For multi-item scans (per-item iteration over a discoverable surface),
implement :class:`KnowledgeProvider` and feed it to your scan loop.
"""

from locus.deepagent.backends import (
    BackendError,
    BackendProtocol,
    FileInfo,
    FilesystemBackend,
    Match,
    StateBackend,
)
from locus.deepagent.factory import create_deepagent
from locus.deepagent.memory import load_agents_md
from locus.deepagent.protocol import (
    Grounding,
    ItemRef,
    KnowledgeProvider,
    KnowledgeRow,
)
from locus.deepagent.subagent import SubAgentDef, task_tool
from locus.deepagent.todos import Todo, TodoState, make_todo_tools
from locus.deepagent.tools import make_filesystem_tools
from locus.deepagent.workflow import (
    KEY_CAUSAL_CHAIN,
    KEY_CAUSAL_CONFIDENCE,
    KEY_CAUSAL_HYPOTHESIS,
    KEY_EVIDENCE,
    KEY_EXECUTE_PROMPT,
    KEY_GROUNDING_FACTS,
    KEY_GROUNDING_SCORE,
    KEY_PROMPT,
    KEY_REGENERATION_COUNT,
    KEY_REPLAN_COUNT,
    KEY_STOP_REASON,
    KEY_STRUCTURED_OUTPUT,
    KEY_SUMMARY,
    KEY_UNGROUNDED_CLAIMS,
    create_research_workflow,
    make_causal_inference_node,
    make_execute_node,
    make_grounding_eval_node,
    make_regenerate_summary_node,
    make_replan_node,
    make_summarize_node,
    route_after_grounding,
)


__all__ = [
    "BackendError",
    "BackendProtocol",
    "FileInfo",
    "FilesystemBackend",
    "Grounding",
    "ItemRef",
    "KnowledgeProvider",
    "KnowledgeRow",
    "Match",
    "StateBackend",
    "SubAgentDef",
    "Todo",
    "TodoState",
    "create_deepagent",
    "create_research_workflow",
    "load_agents_md",
    "make_execute_node",
    "make_causal_inference_node",
    "make_summarize_node",
    "make_grounding_eval_node",
    "make_regenerate_summary_node",
    "make_replan_node",
    "route_after_grounding",
    "KEY_PROMPT",
    "KEY_EXECUTE_PROMPT",
    "KEY_EVIDENCE",
    "KEY_GROUNDING_FACTS",
    "KEY_CAUSAL_CHAIN",
    "KEY_CAUSAL_HYPOTHESIS",
    "KEY_CAUSAL_CONFIDENCE",
    "KEY_SUMMARY",
    "KEY_STRUCTURED_OUTPUT",
    "KEY_GROUNDING_SCORE",
    "KEY_UNGROUNDED_CLAIMS",
    "KEY_REPLAN_COUNT",
    "KEY_REGENERATION_COUNT",
    "KEY_STOP_REASON",
    "make_filesystem_tools",
    "make_todo_tools",
    "task_tool",
]
