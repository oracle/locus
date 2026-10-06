# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent initialization — model, tools, executor, hooks, plugins, skills.

Extracted from ``Agent`` so the public-facade class can stay focused on
the runtime loop. The two public entry points
(:func:`initialize_agent` and :func:`register_builtin_tools`) populate
the agent's private attributes in place; they do not return anything.

Idempotent: ``initialize_agent`` is a no-op when the agent's
``_initialized`` flag is already set, matching the prior in-class
behaviour where ``_initialize()`` was called from both ``__init__`` and
``run()``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from tulip.tools.decorator import Tool
from tulip.tools.executor import ConcurrentExecutor, SequentialExecutor
from tulip.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from tulip.agent.agent import Agent

logger = logging.getLogger(__name__)


def initialize_agent(agent: Agent) -> None:
    """Resolve the model, tools, executor, hooks, plugins, and skills.

    Mutates ``agent._*`` private attributes in place. Safe to call
    multiple times — the ``_initialized`` flag short-circuits subsequent
    calls.

    ``get_model`` is looked up indirectly through ``tulip.agent.agent``
    so existing test monkey-patches like
    ``monkeypatch.setattr("tulip.agent.agent.get_model", ...)`` keep
    working after the extraction. There are 30+ such sites in the test
    suite.
    """
    if agent._initialized:
        return

    # Look up ``get_model`` via the agent module so the public
    # monkey-patch surface remains stable.
    from tulip.agent import agent as _agent_module

    get_model = _agent_module.get_model

    # --- Model -------------------------------------------------------------
    if isinstance(agent.config.model, str):
        agent._model = get_model(agent.config.model, **agent.config.model_kwargs)
    elif agent.config.model_kwargs:
        # An already-built model carries its own configuration, so these
        # would do nothing at all. Silently dropping them is the worst
        # outcome: the agent runs against the wrong endpoint and looks fine.
        raise ValueError(
            "model_kwargs applies only when `model` is a provider string; "
            f"got a {type(agent.config.model).__name__} instance together with "
            f"{sorted(agent.config.model_kwargs)}. Pass those to get_model() "
            "when building the model instead."
        )
    else:
        agent._model = agent.config.model

    # --- Tools -------------------------------------------------------------
    agent._tool_registry = ToolRegistry()
    for t in agent.config.tools:
        if isinstance(t, Tool):
            agent._tool_registry.register(t)
        else:
            raise TypeError(f"Expected Tool instance, got {type(t)}")

    # Add task_complete + ask_user in explicit completion mode.
    if agent.config.completion_mode == "explicit":
        register_builtin_tools(agent)

    # --- Executor ----------------------------------------------------------
    if agent.config.tool_execution == "concurrent":
        agent._executor = ConcurrentExecutor(max_concurrency=agent.config.max_concurrency)
    else:
        agent._executor = SequentialExecutor()

    # --- Hooks + orchestrator ---------------------------------------------
    # The orchestrator holds a reference to the same list that the
    # plugin/skill registration code below extends, so hooks added by
    # plugins are picked up at dispatch time without re-wiring.
    agent._hooks = list(agent.config.hooks)
    from tulip.agent.hook_orchestrator import HookOrchestrator

    agent._hook_orchestrator = HookOrchestrator(agent._hooks)

    # --- Plugins (bundles of hooks + tools) -------------------------------
    for plugin in agent.config.plugins:
        from tulip.hooks.plugin import Plugin, PluginAdapter

        if isinstance(plugin, Plugin):
            plugin.init_agent(agent)
            agent._hooks.append(PluginAdapter(plugin))
            for plugin_tool in plugin.get_tools():
                agent._tool_registry.register(plugin_tool)

    # --- Skills (AgentSkills.io) ------------------------------------------
    if agent.config.skills:
        from tulip.hooks.plugin import PluginAdapter
        from tulip.skills.plugin import SkillsPlugin

        skills_plugin = SkillsPlugin(skills=agent.config.skills)
        skills_plugin.init_agent(agent)
        agent._hooks.append(PluginAdapter(skills_plugin))
        agent._tool_registry.register(skills_plugin.get_activation_tool())

    # --- Multi-modal provider tools ---------------------------------------
    if (
        agent.config.web_search is not None
        or agent.config.web_fetch is not None
        or agent.config.image_generator is not None
        or agent.config.speech_provider is not None
    ):
        from tulip.providers.tools import auto_register

        auto_register(
            tool_registry=agent._tool_registry,
            web_search=agent.config.web_search,
            web_fetch=agent.config.web_fetch,
            image_generator=agent.config.image_generator,
            speech_provider=agent.config.speech_provider,
        )

    # --- Playbook enforcer hook -------------------------------------------
    # Auto-installed when ``playbook`` is set so the documented contract
    # ("PlaybookEnforcer validates tool calls against step constraints")
    # is real instead of aspirational.
    if agent.config.playbook is not None:
        from tulip.playbooks.hook import PlaybookEnforcerHook

        # Hand the enforcer the agent's skills so ``PlaybookStep.uses``
        # resolves to real capabilities — without this, ``uses`` references
        # resolved to nothing and constrained nothing on the auto-installed
        # path, silently (#172, adjacent gap).
        agent._hooks.append(
            PlaybookEnforcerHook(
                agent.config.playbook,
                skills={s.name: s for s in agent.config.skills} if agent.config.skills else None,
            )
        )

    # --- ObservationPack → obs_recall ---------------------------------------
    # Large old tool outputs leave the request as placeholders; obs_recall is
    # how the model reads their exact bytes back, so it comes with them.
    if agent.config.observation_pack.enabled:
        agent._observation_pack = _build_observation_pack(agent)
        if "obs_recall" not in agent._tool_registry:
            agent._tool_registry.register(_observation_recall_tool(agent))

    # --- Deferred tools → tool_search --------------------------------------
    # Runs after every registration path above (config tools, plugins,
    # skills, providers) so the catalog sees them all. Deferral is
    # visibility only; see tulip.tools.tool_search (#177).
    if any(t.deferred for t in agent._tool_registry) and "tool_search" not in (
        agent._tool_registry
    ):
        from tulip.tools.tool_search import create_tool_search_tool

        agent._tool_registry.register(create_tool_search_tool(agent._tool_registry))

    # --- Code mode → run_code ----------------------------------------------
    # After every registration path so in-program calls can reach them all.
    # The tool is bound to this agent's registry AND hook orchestrator: the
    # whole point is that an in-sandbox call clears the same seam a
    # loop-issued call does (#176).
    if agent.config.code_mode and "run_code" not in agent._tool_registry:
        from tulip.agent.hook_orchestrator import HookOrchestrator
        from tulip.tools.code_mode import create_code_tool

        agent._hook_orchestrator = agent._hook_orchestrator or HookOrchestrator(agent._hooks)
        agent._tool_registry.register(
            create_code_tool(agent._tool_registry, agent._hook_orchestrator)
        )

    # --- Memory manager ---------------------------------------------------
    if agent.config.memory_manager is not None:
        agent._memory_manager = agent.config.memory_manager

    # --- Conversation manager ---------------------------------------------
    from tulip.models.metadata import metadata_for, model_id_of

    configured = agent.config.model
    model_id = model_id_of(configured if isinstance(configured, str) else agent._model)
    meta = metadata_for(model_id) if model_id is not None else None
    if (
        meta is not None
        and meta.input_price_per_mtok is not None
        and meta.output_price_per_mtok is not None
    ):
        agent._model_prices = (float(meta.input_price_per_mtok), float(meta.output_price_per_mtok))
    if agent.config.max_cost_usd is not None and agent._model_prices is None:
        # Fail closed: a budget that cannot measure spend would never stop anything.
        raise ValueError(
            f"max_cost_usd needs prices for model {model_id!r}. Register them with "
            "tulip.models.metadata.register_metadata(ModelMetadata(..., "
            "input_price_per_mtok=..., output_price_per_mtok=...))."
        )
    context_window = (
        None
        if agent.config.conversation_manager is not None
        else _context_window(agent, meta.context_length if meta is not None else None)
    )
    if agent.config.conversation_manager is not None:
        agent._conversation_manager = agent.config.conversation_manager
    elif context_window is not None and agent.config.compaction.enabled:
        # The window is known, so a long run can be kept inside it: clear old
        # tool output, then summarise older history and carry on (the loop
        # drives it; see ``_compact_context``).
        from tulip.memory.compaction import ContextCompactor

        compaction = agent.config.compaction
        summary_model = compaction.summary_model
        if isinstance(summary_model, str):
            summary_model = get_model(summary_model)
        agent._conversation_manager = ContextCompactor(
            context_length=context_window,
            summary_model=summary_model if summary_model is not None else agent._model,
            trigger_fraction=compaction.trigger_fraction,
            reserved_tokens=compaction.reserved_tokens,
            tail_turns=compaction.tail_turns,
            tail_token_fraction=compaction.tail_token_fraction,
            tool_output_keep_tokens=compaction.tool_output_keep_tokens,
            summary_max_tokens=compaction.summary_max_tokens,
            min_iterations_between_summaries=compaction.min_iterations_between_summaries,
        )
    elif context_window is not None:
        # Summarising is off, so no extra model calls: stale tool output is
        # pruned and a token-budgeted tail kept on each request, which is what
        # stops one large tool result ending the run.
        from tulip.memory.compactor import LLMCompactor

        # Its cuts move in steps, so most requests keep the previous one's
        # prefix and the provider's prompt cache keeps serving it.
        agent._conversation_manager = LLMCompactor(
            context_length=context_window, slide_step=_CACHE_FRIENDLY_STEP
        )
    else:
        # Unknown window: a message window, at any iteration count. A short
        # run can still overflow on one large tool output, so say how to
        # name the window.
        from tulip.memory.conversation import SlidingWindowManager

        _warn_unknown_window(model_id)
        window = max(20, agent.config.max_iterations * 2)
        # Slides a quarter of the window at a time rather than a message per
        # request, so the start of the history stays cacheable in between.
        agent._conversation_manager = SlidingWindowManager(
            window_size=window, slide_step=max(1, window // 4)
        )

    # --- Reflexion ---------------------------------------------------------
    if agent.config.reflexion and agent.config.reflexion.enabled:
        from tulip.reasoning.reflexion import Reflector

        agent._reflector = Reflector(
            loop_threshold=agent.config.tool_loop_threshold,
            diminishing_returns=agent.config.reflexion.diminishing_returns,
        )

    # --- Auxiliary model ---------------------------------------------------
    # Resolved once. Used for grounding eval, structured-output repair,
    # and the max-iterations final-summary call so those side calls don't
    # burn primary-model budget. Falls back to the primary model when
    # ``auxiliary_model`` isn't set on the config.
    if agent.config.auxiliary_model is not None:
        aux_cfg = agent.config.auxiliary_model
        if isinstance(aux_cfg, str):
            agent._auxiliary_model = get_model(aux_cfg)
        else:
            agent._auxiliary_model = aux_cfg
    else:
        agent._auxiliary_model = agent._model

    # --- Grounding evaluator ----------------------------------------------
    if agent.config.grounding and agent.config.grounding.enabled:
        from tulip.reasoning.grounding import GroundingEvaluator

        agent._grounding_evaluator = GroundingEvaluator(
            replan_threshold=agent.config.grounding.threshold,
        )
        # Precedence: grounding.model > auxiliary_model > primary.
        if agent.config.grounding.model:
            agent._grounding_model = get_model(agent.config.grounding.model)
        else:
            agent._grounding_model = agent._auxiliary_model

    agent._initialized = True


#: Names an input context window when neither the agent config nor model
#: metadata does: one setting covers every agent in a process that talks to
#: a self-hosted model the seed table cannot know.
CONTEXT_WINDOW_ENV = "TULIP_CONTEXT_WINDOW"

# Model ids already warned about, so a process that builds many agents on the
# same unknown model logs the fallback once.
_warned_unknown: set[str] = set()


#: How many messages the default token-window manager's cuts move at a time.
_CACHE_FRIENDLY_STEP = 8


def _context_window(agent: Agent, metadata_window: int | None) -> int | None:
    """The input window to count tokens against, or ``None`` when unknown.

    Most specific first: the agent's own ``context_window``, the
    ``TULIP_CONTEXT_WINDOW`` environment variable, the model-metadata entry,
    then a window the model object reports itself (``context_window`` or
    ``context_length`` on the model or its config — a gateway binding can
    carry the ``max_model_len`` its server publishes).
    """
    if agent.config.context_window is not None:
        return agent.config.context_window
    raw = os.environ.get(CONTEXT_WINDOW_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        logger.warning("Ignoring %s=%r: it must be a positive integer.", CONTEXT_WINDOW_ENV, raw)
    if metadata_window is not None:
        return metadata_window
    model = agent._model
    for holder in (model, getattr(model, "config", None)):
        for attr in ("context_window", "context_length"):
            reported = getattr(holder, attr, None)
            # ``bool`` is an int subclass; a flag is not a window.
            if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
                return reported
    return None


def _warn_unknown_window(model_id: str | None) -> None:
    key = model_id or "<unnamed>"
    if key in _warned_unknown:
        return
    _warned_unknown.add(key)
    logger.warning(
        "No context window known for model %r; keeping a message window, which a large "
        "tool output can still overflow. Set Agent(context_window=...), %s, or register "
        "it with tulip.models.metadata.register_metadata (discover_context_length reads "
        "it from a vLLM server) to count tokens instead.",
        key,
        CONTEXT_WINDOW_ENV,
    )


def _run_for(agent: Agent, run_id: str | None) -> Any:
    """The in-flight run context with ``run_id``, if any."""
    if run_id is None:
        return None
    with agent._runs_lock:
        for rc in agent._active_runs.values():
            if rc.run_id == run_id:
                return rc
    return None


def register_builtin_tools(agent: Agent) -> None:
    """Register the explicit-completion-mode built-ins on the agent.

    Adds ``task_complete`` and ``ask_user`` to the agent's tool registry.
    The closures capture the agent so ``task_complete`` can consult
    ``require_verification`` / the run's unverified-writes flag and ``ask_user``
    can emit the special ``__interrupt__`` marker the runtime loop
    recognises.
    """
    from tulip.tools.decorator import tool as tool_decorator

    agent_ref = agent  # Closure target.

    @tool_decorator(
        name="task_complete",
        description=(
            "Signal that the current task is complete. "
            "Call this ONLY when you have verified your work "
            "(e.g., tests pass, output is correct). "
            "If you wrote files, you MUST run tests/commands first. "
            "Provide a summary of what was accomplished."
        ),
    )
    def task_complete(summary: str, status: str = "success", ctx: Any = None) -> str:
        """Signal task completion with a summary."""
        # The unverified-writes flag belongs to the run that called us (found
        # by ``ctx.run_id``), not to the agent: a concurrent run's write must
        # not block this run's completion, nor its verification unblock ours.
        run = _run_for(agent_ref, getattr(ctx, "run_id", None))
        if agent_ref.config.require_verification and run is not None and run.has_unverified_writes:
            run.has_unverified_writes = False  # Reset so it doesn't loop.
            return (
                "BLOCKED: You have unverified changes. "
                "You wrote files but haven't run tests or verification commands yet. "
                "Run tests first (e.g., run_command with pytest), then call task_complete again."
            )
        return f"Task completed ({status}): {summary}"

    @tool_decorator(
        name="ask_user",
        description=(
            "Ask the user a question and wait for their response. "
            "Use this when you need clarification, approval, or a decision "
            "from the user before proceeding."
        ),
    )
    def ask_user(question: str, options: str | None = None) -> str:
        """Ask the user a question. Pauses execution until they respond.

        Args:
            question: The question to ask
            options: Comma-separated list of options (e.g., "JWT,session,OAuth")

        Returns:
            A special marker that triggers an interrupt in the agent loop
        """
        import json

        option_list = [o.strip() for o in options.split(",")] if options else None
        return json.dumps(
            {
                "__interrupt__": True,
                "question": question,
                "options": option_list,
            }
        )

    if "task_complete" not in agent._tool_registry.tools:
        agent._tool_registry.register(task_complete)
    if "ask_user" not in agent._tool_registry.tools:
        agent._tool_registry.register(ask_user)


def _build_observation_pack(agent: Agent) -> Any:
    from tulip.memory.observation_pack import ObservationPack, SwapCostModel

    config = agent.config.observation_pack
    return ObservationPack(
        directory=config.directory,
        threshold_bytes=config.threshold_bytes,
        full_sends=config.full_sends,
        excerpt_bytes=config.excerpt_bytes,
        recall_max_bytes=config.recall_max_bytes,
        recall_max_lines=config.recall_max_lines,
        label=agent.config.name,
        cost_model=SwapCostModel(
            cache_read_cost=config.cache_read_cost,
            cache_write_cost=config.cache_write_cost,
            horizon_requests=config.horizon_requests,
            min_horizon_requests=config.min_horizon_requests,
            min_batch_bytes=config.min_batch_bytes,
        ),
    )


def _observation_recall_tool(agent: Agent) -> Any:
    """``obs_recall``: one page of an archived tool output, by id."""
    from tulip.core.events import CustomEvent
    from tulip.tools.decorator import tool as tool_decorator

    agent_ref = agent

    @tool_decorator(
        name="obs_recall",
        description=(
            "Read back the exact text of an earlier tool output that was archived to "
            "save context. Pass the id from its placeholder (obs_...) and a byte offset "
            "(0 to start, then the returned next_offset to continue), or a 1-based "
            "line to start at. Returns up to 16 KB or 400 lines per call."
        ),
        idempotent=True,
    )
    def obs_recall(id: str, offset: int = 0, line: int | None = None, ctx: Any = None) -> str:  # noqa: A002 — the id the placeholder names
        """Recall a page of an archived tool output.

        Args:
            id: The observation id from the placeholder, e.g. obs_0123456789abcdef01234567.
            offset: Byte offset to start at; use the previous call's next_offset to continue.
            line: 1-based line to start at instead of a byte offset.
        """
        pack = agent_ref._observation_pack
        if pack is None:
            raise ValueError("ObservationPack is not enabled")
        run_id = getattr(ctx, "run_id", None)
        run = _run_for(agent_ref, run_id)
        session = pack.session_key(run.thread_id if run is not None else None, run_id)
        text, details = pack.recall(session, id, offset=offset, line=line)
        if run is not None:
            run.emit(CustomEvent(name="observation_pack", data={"event": "recall", **details}))
        return str(text)

    return obs_recall
