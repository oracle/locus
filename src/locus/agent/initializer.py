# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

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

from typing import TYPE_CHECKING

from locus.tools.decorator import Tool
from locus.tools.executor import ConcurrentExecutor, SequentialExecutor
from locus.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from locus.agent.agent import Agent


def initialize_agent(agent: Agent) -> None:
    """Resolve the model, tools, executor, hooks, plugins, and skills.

    Mutates ``agent._*`` private attributes in place. Safe to call
    multiple times — the ``_initialized`` flag short-circuits subsequent
    calls.

    ``get_model`` is looked up indirectly through ``locus.agent.agent``
    so existing test monkey-patches like
    ``monkeypatch.setattr("locus.agent.agent.get_model", ...)`` keep
    working after the extraction. There are 30+ such sites in the test
    suite.
    """
    if agent._initialized:
        return

    # Look up ``get_model`` via the agent module so the public
    # monkey-patch surface remains stable.
    from locus.agent import agent as _agent_module

    get_model = _agent_module.get_model

    # --- Model -------------------------------------------------------------
    if isinstance(agent.config.model, str):
        agent._model = get_model(agent.config.model)
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
    from locus.agent.hook_orchestrator import HookOrchestrator

    agent._hook_orchestrator = HookOrchestrator(agent._hooks)

    # --- Plugins (bundles of hooks + tools) -------------------------------
    for plugin in agent.config.plugins:
        from locus.hooks.plugin import Plugin, PluginAdapter

        if isinstance(plugin, Plugin):
            plugin.init_agent(agent)
            agent._hooks.append(PluginAdapter(plugin))
            for plugin_tool in plugin.get_tools():
                agent._tool_registry.register(plugin_tool)

    # --- Skills (AgentSkills.io) ------------------------------------------
    if agent.config.skills:
        from locus.hooks.plugin import PluginAdapter
        from locus.skills.plugin import SkillsPlugin

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
        from locus.providers.tools import auto_register

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
        from locus.playbooks.hook import PlaybookEnforcerHook

        agent._hooks.append(PlaybookEnforcerHook(agent.config.playbook))

    # --- Memory manager ---------------------------------------------------
    if agent.config.memory_manager is not None:
        agent._memory_manager = agent.config.memory_manager

    # --- Conversation manager ---------------------------------------------
    if agent.config.conversation_manager is not None:
        agent._conversation_manager = agent.config.conversation_manager
    elif agent.config.max_iterations > 10:
        from locus.memory.conversation import SlidingWindowManager

        window = max(20, agent.config.max_iterations * 2)
        agent._conversation_manager = SlidingWindowManager(window_size=window)

    # --- Reflexion ---------------------------------------------------------
    if agent.config.reflexion and agent.config.reflexion.enabled:
        from locus.reasoning.reflexion import Reflector

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
        from locus.reasoning.grounding import GroundingEvaluator

        agent._grounding_evaluator = GroundingEvaluator(
            replan_threshold=agent.config.grounding.threshold,
        )
        # Precedence: grounding.model > auxiliary_model > primary.
        if agent.config.grounding.model:
            agent._grounding_model = get_model(agent.config.grounding.model)
        else:
            agent._grounding_model = agent._auxiliary_model

    agent._initialized = True


def register_builtin_tools(agent: Agent) -> None:
    """Register the explicit-completion-mode built-ins on the agent.

    Adds ``task_complete`` and ``ask_user`` to the agent's tool registry.
    The closures capture the agent so ``task_complete`` can consult
    ``require_verification`` / ``_has_unverified_writes`` and ``ask_user``
    can emit the special ``__interrupt__`` marker the runtime loop
    recognises.
    """
    from locus.tools.decorator import tool as tool_decorator

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
    def task_complete(summary: str, status: str = "success") -> str:
        """Signal task completion with a summary."""
        if agent_ref.config.require_verification and agent_ref._has_unverified_writes:
            agent_ref._has_unverified_writes = False  # Reset so it doesn't loop.
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
