# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""``deepagent()`` — a research-shaped Agent factory.

Bundles the standard deep-research configuration into one call:

- ``reflexion=True`` and ``grounding=True`` so the agent self-corrects
  hallucinations against the tool-call evidence trail.
- Typed termination::

      (ToolCalled(submit_tool) & ConfidenceMet(min_confidence))
       | TokenLimit(total_token_budget)   # only when budget is set
       | MaxIterations(max_iterations)

  greppable, unit-testable, and per-recipe overridable.
- ``output_schema=`` enforced — the model provider's strict
  structured-output mode rejects non-conforming submissions before
  they reach the caller.
- Optional ``checkpointer`` for resume across days / process restarts.

This is a pure convenience layer over ``tulip.Agent`` — it does not
change agent semantics. Callers who need finer control can build the
Agent directly.

Naming note (since this trips people up):

  ``total_token_budget``   — cumulative INPUT+OUTPUT tokens across
                              ALL iterations of one run. Wired into
                              ``TokenLimit(total_token_budget)``
                              termination. ``None`` (default) means
                              "no TokenLimit term" — the run is
                              bounded only by ToolCalled+Confidence
                              or MaxIterations.
  ``max_output_tokens``    — per-completion output cap. Forwarded
                              to ``AgentConfig.max_tokens`` and from
                              there to the model provider's
                              ``max_tokens`` request field on every
                              completion. ``None`` = use the model's
                              own default.

**Breaking change in 0.2.0b23**: the old ``max_tokens=`` parameter
(which used to mean total-run budget but read like a per-completion
cap to anyone familiar with OpenAI / Gemini / Anthropic SDKs) was
removed. Use ``total_token_budget=`` for run-level termination and
``max_output_tokens=`` for per-completion output cap. The old name
caused empty-output runs when callers passed ``max_tokens=65536``
expecting per-completion semantics — the run silently exited via
TokenLimit before the model could write anything.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def wire_datastores(
    datastores: dict[str, Any] | None,
    datastore_top_k: int = 5,
) -> tuple[list[Any], str]:
    """Turn ``{name: RAGRetriever}`` into auto-wired tools + a routing block.

    Used by ``create_deepagent`` and ``create_research_workflow`` to give
    both research factories an identical ``datastores=`` surface. For each
    entry, a ``search_<name>`` tool is built via
    ``tulip.rag.tools.create_rag_tool`` and a one-line routing hint is
    appended to the returned prompt block. Mirrors a common deep-research
    ``create_deep_research_agent(datastores=...)`` contract so existing
    recipes translate 1:1.

    Args:
        datastores: Optional ``{name: RAGRetriever}`` or
            ``{name: {"retriever": ..., "description": ..., "top_k": ...,
            "threshold": ...}}``. Returns ``([], "")`` when ``None`` or
            empty.
        datastore_top_k: Default ``top_k`` used when an entry doesn't set
            its own.

    Returns:
        ``(tools, routing_block)`` — the list of auto-wired tools to
        append to the agent's tool list, and a Markdown block ready to
        prepend to the system prompt (empty string when no datastores).
    """
    if not datastores:
        return [], ""

    from tulip.rag.retriever import RAGRetriever
    from tulip.rag.tools import create_rag_tool

    new_tools: list[Any] = []
    ds_lines: list[str] = []
    for ds_name, ds_value in datastores.items():
        if isinstance(ds_value, RAGRetriever):
            retriever = ds_value
            ds_desc = ""
            top_k = datastore_top_k
            threshold: float | None = None
        elif isinstance(ds_value, dict):
            retriever = ds_value["retriever"]
            ds_desc = ds_value.get("description", "")
            top_k = ds_value.get("top_k", datastore_top_k)
            threshold = ds_value.get("threshold", None)
        else:
            raise TypeError(
                f"datastores[{ds_name!r}] must be a RAGRetriever or "
                f"a dict with 'retriever' (+ optional 'description', "
                f"'top_k', 'threshold'); got {type(ds_value).__name__}."
            )

        tool_name = f"search_{ds_name}"
        tool_desc = (
            f"Search the {ds_name!r} datastore"
            + (f" ({ds_desc})" if ds_desc else "")
            + f". Returns up to {top_k} relevant documents."
        )
        # Default threshold=None (no min-score filter). The 0.5 default
        # in create_rag_tool is tuned for sentence-transformer scores;
        # raw cosine similarity over Cohere embeddings can dip below
        # that for genuinely relevant matches.
        new_tools.append(
            create_rag_tool(
                retriever,
                name=tool_name,
                description=tool_desc,
                limit=top_k,
                threshold=threshold,
            )
        )
        ds_lines.append(f"- `{tool_name}(query)`: " + (ds_desc or f"the {ds_name!r} datastore"))

    routing_block = (
        "# Datastores\n\n"
        "You have these searchable datastores. Pick the one whose "
        "description best matches each query (call multiple when the "
        "question spans them):\n\n" + "\n".join(ds_lines)
    )
    return new_tools, routing_block


def create_deepagent(
    *,
    model: str | Any,
    tools: list[Any] | None = None,
    system_prompt: str,
    output_schema: type[BaseModel] | None = None,
    submit_tool: str = "submit_research",
    min_confidence: float = 0.8,
    total_token_budget: int | None = None,
    max_output_tokens: int | None = None,
    max_iterations: int = 40,
    reflexion: bool = True,
    grounding: bool = True,
    checkpointer: Any | None = None,
    enable_filesystem: bool = False,
    backend: Any | None = None,
    enable_todos: bool = False,
    todo_state: Any | None = None,
    memory_files: list[str] | tuple[str, ...] | None = None,
    subagents: list[Any] | None = None,
    datastores: dict[str, Any] | None = None,
    datastore_top_k: int = 5,
    summarize_after_messages: int | None = None,
    summarize_keep_recent: int = 10,
    **agent_kwargs: Any,
) -> Any:
    """Construct a research-shaped ``tulip.Agent``.

    Args:
        model: A tulip model string (``"openai:gpt-4o"``) or a
            ``ModelProtocol`` instance built via
            ``tulip.models.get_model``.
        tools: All tools the agent can call — MCP-derived,
            ``@tool``-decorated Python tools, and the ``submit_tool``
            that ends the loop.
        system_prompt: The agent's identity, rules, and contract for
            calling ``submit_tool`` with the structured payload.
        output_schema: Pydantic model the agent's ``submit_tool``
            payload must validate against. Tulip uses the model
            provider's strict structured-output mode to enforce it.
        submit_tool: The tool name whose call signals "I'm done — here
            is my structured answer". Default ``submit_research``.
        min_confidence: Confidence threshold the submission must clear
            for early-exit. Default 0.8.
        total_token_budget: Cumulative INPUT+OUTPUT tokens across ALL
            iterations of one run. Wired into
            ``TokenLimit(total_token_budget)`` termination. Default
            ``None`` = no TokenLimit term in the termination algebra
            (the run is bounded only by ToolCalled+Confidence or
            MaxIterations). Set this only when you want an explicit
            cumulative ceiling — and remember that a long system prompt
            on a multi-iteration run can eat the budget quickly.
        max_output_tokens: Per-LLM-call output token budget, forwarded to
            ``AgentConfig.max_tokens`` (and from there to the model
            provider's ``max_tokens`` request field on every completion).
            Default ``None`` = use the model's own default. Set this to
            ``65_536`` for long-form Gemini 2.5 Pro outputs, etc.
        max_iterations: Cap on reasoning steps. Default 40.
        reflexion: Self-critique pass after each step. Default True.
        grounding: Citation-grounding eval against tool-call evidence.
            Default True.
        checkpointer: Optional ``tulip.memory`` checkpointer for
            resume. Default None (no persistence).
        enable_filesystem: When True, attaches the six filesystem-as-
            memory tools (``write_file``, ``read_file``, ``ls``,
            ``edit_file``, ``glob``, ``grep``) to the agent's tool list
            so it can use a scratchspace for intermediate work. Default
            False.
        backend: Optional :class:`BackendProtocol` used by the FS
            tools. Honored only when ``enable_filesystem=True``.
            Defaults to a fresh :class:`StateBackend` (in-memory,
            ephemeral, scoped to the agent run). Pass a
            :class:`FilesystemBackend` for real-disk persistence.
        enable_todos: When True, attaches ``write_todos`` and
            ``read_todos`` tools backed by an in-memory
            :class:`TodoState`. Lets the agent maintain a structured
            task list across reasoning steps. Default False.
        todo_state: Optional pre-built :class:`TodoState`. Honored
            only when ``enable_todos=True``. Pass one to inspect the
            list externally after the agent runs.
        memory_files: Optional list of ``AGENTS.md``-style Markdown
            file paths whose contents are joined and prepended to the
            ``system_prompt``. Missing paths are skipped silently
            (so defaults like ``["~/AGENTS.md", "./AGENTS.md"]`` work
            without checking each one).
        subagents: Optional list of :class:`SubAgentDef` declaring
            subagents the parent can spawn mid-run via a ``task()``
            tool. Each subagent runs as a stateless one-shot.
        datastores: Optional mapping of ``{name: RAGRetriever}`` (or
            ``{name: {"retriever": RAGRetriever, "description": str,
            "top_k": int}}``). For each entry, a ``search_{name}`` tool
            is auto-wired via ``RAGRetriever.as_tool`` and appended to
            ``tools``, and a per-store description block is prepended
            to ``system_prompt`` so the model can route queries to the
            right store. Mirrors a common deep-research
            ``create_deep_research_agent(datastores=...)`` shape.
        datastore_top_k: Default top-k for auto-wired datastore tools.
            Honored only when ``datastores`` is set and an entry doesn't
            specify its own ``top_k``. Default 5.
        summarize_after_messages: If set, attaches tulip's
            :class:`SummarizingManager` so older messages are
            condensed once the conversation exceeds this count.
            Recent ``summarize_keep_recent`` messages are always
            preserved verbatim. Default ``None`` (no summarization
            — all messages kept verbatim, may blow context on long
            runs).
        summarize_keep_recent: How many recent messages
            ``SummarizingManager`` preserves untouched. Default 10.
            Honored only when ``summarize_after_messages`` is set.
        **agent_kwargs: Forwarded to ``tulip.Agent`` for advanced
            knobs (hooks, conversation_manager, plugins, …).

    Returns:
        A configured ``tulip.Agent`` ready for ``agent.run(prompt)``
        or ``agent.run_sync(prompt)``.
    """
    from tulip.agent.agent import (
        Agent,
    )  # direct import — avoids the lazy-import-as-object mypy false positive
    from tulip.core.termination import (
        ConfidenceMet,
        MaxIterations,
        TokenLimit,
        ToolCalled,
    )

    # Reject the legacy ``max_tokens=`` name explicitly. Without this
    # check it would flow into ``**agent_kwargs`` and ``Agent()`` would
    # silently set the per-completion cap to that value — completely
    # different semantics from the old `create_deepagent(max_tokens=)`
    # behavior. The 0.2.0b23 rename was specifically to kill the
    # silent foot-gun; the loud error matches that intent.
    if "max_tokens" in agent_kwargs:
        raise TypeError(
            "create_deepagent() no longer accepts ``max_tokens=`` "
            "(removed in 0.2.0b23). Use ``total_token_budget=`` for "
            "the run-level TokenLimit termination, or "
            "``max_output_tokens=`` for the per-completion output cap "
            "on each LLM call. The old name conflicted with every "
            "LLM SDK's per-completion ``max_tokens`` field and caused "
            "silent empty-output runs when callers passed "
            "``max_tokens=65536`` expecting per-completion semantics."
        )

    # Build the termination algebra. ``TokenLimit`` only joins the
    # `or`-chain when the caller opted in to a budget; the default
    # is no token-based termination at all so a long input prompt
    # can't silently kill an otherwise-healthy run.
    #
    # Breaking change in 0.2.0b23 — the old ``max_tokens=`` kwarg
    # was removed. Use ``total_token_budget=`` (run-level cap) or
    # ``max_output_tokens=`` (per-completion cap). If callers pass
    # ``max_tokens=`` it lands in **agent_kwargs and Agent() rejects
    # it loudly, which is the right failure shape — better than the
    # historical silent foot-gun.
    base = ToolCalled(submit_tool) & ConfidenceMet(min_confidence)
    if total_token_budget is not None:
        termination = base | TokenLimit(total_token_budget) | MaxIterations(max_iterations)
    else:
        termination = base | MaxIterations(max_iterations)

    # Splice filesystem-as-memory tools into the user-supplied list
    # before constructing the Agent. The default backend is an
    # ephemeral in-memory StateBackend so callers who flip the flag
    # don't have to think about cleanup.
    final_tools = list(tools) if tools else []
    if enable_filesystem:
        from tulip.deepagent.backends import StateBackend
        from tulip.deepagent.tools import make_filesystem_tools

        fs_backend = backend if backend is not None else StateBackend()
        final_tools = [*final_tools, *make_filesystem_tools(fs_backend)]

    if enable_todos:
        from tulip.deepagent.todos import TodoState, make_todo_tools

        td_state = todo_state if todo_state is not None else TodoState()
        final_tools = [*final_tools, *make_todo_tools(td_state)]

    # Subagent dispatch: attach a single ``task()`` tool the parent
    # can call to spawn one-shot subagents mid-run. The tool's catalog
    # of available subagents is implicit in its docstring; callers
    # who want it surfaced in the parent's system prompt can do so
    # via ``memory_files`` or by appending to ``system_prompt``.
    if subagents:
        from tulip.deepagent.subagent import task_tool

        final_tools = [
            *final_tools,
            task_tool(subagents, parent_model=model),
        ]

    # Datastore auto-wiring: turn ``{name: RAGRetriever}`` into a set of
    # ``search_{name}`` tools the agent can call, and prepend a routing
    # hint block so the model picks the right store per query. Shared
    # with ``create_research_workflow`` via ``wire_datastores`` so both
    # paths produce an identical tool surface + system-prompt prefix.
    new_tools, datastore_routing_block = wire_datastores(datastores, datastore_top_k)
    final_tools = [*final_tools, *new_tools]

    # Memory files: prepend to the system prompt so AGENTS.md-style
    # instructions land in front of the recipe-specific identity
    # block. Layered so users can stack base / user / project files.
    final_system_prompt = system_prompt
    if memory_files:
        from tulip.deepagent.memory import load_agents_md

        memory_block = load_agents_md(list(memory_files))
        if memory_block:
            final_system_prompt = f"{memory_block}\n\n---\n\n{system_prompt}"

    # Datastore routing hint goes immediately after any AGENTS.md memory
    # but before the recipe-specific system prompt, so per-store
    # descriptions are visible to the model on every turn.
    if datastore_routing_block:
        final_system_prompt = f"{datastore_routing_block}\n\n---\n\n{final_system_prompt}"

    # Summarization: thin pass-through to tulip's SummarizingManager.
    # Active only when the caller asks for it; otherwise the agent
    # keeps every message (default tulip behavior). Tier-3 knob —
    # avoids reinventing what tulip already ships.
    conversation_manager: Any | None = None
    if summarize_after_messages is not None:
        from tulip.memory.conversation import SummarizingManager

        conversation_manager = SummarizingManager(
            threshold=summarize_after_messages,
            keep_recent=summarize_keep_recent,
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "tools": final_tools,
        "system_prompt": final_system_prompt,
        "max_iterations": max_iterations,
        "reflexion": reflexion,
        "grounding": grounding,
    }
    if conversation_manager is not None:
        kwargs["conversation_manager"] = conversation_manager
    if output_schema is not None:
        kwargs["output_schema"] = output_schema
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    # Forward per-completion output cap to the model. AgentConfig.max_tokens
    # lands on every chat-completion request via runtime_loop.
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(agent_kwargs)

    agent = Agent(**kwargs)
    # Tulip's Agent constructor accepts ``max_iterations`` but the
    # typed ``termination`` is the load-bearing exit criterion; attach
    # it via the public AgentConfig setter so the algebra runs.
    config = getattr(agent, "config", None)
    if config is not None and hasattr(config, "termination"):
        config.termination = termination
    return agent
