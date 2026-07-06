# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Main Agent class - 100% Pydantic."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from tulip.agent.config import AgentConfig, GroundingConfig, ReflexionConfig
from tulip.agent.result import AgentResult, ExecutionMetrics, StopReason
from tulip.agent.runtime_loop import AgentRuntimeMixin
from tulip.core.events import (
    TerminateEvent,
    ToolCompleteEvent,
    TulipEvent,
)
from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.models import get_model  # noqa: F401 — re-exported for test monkey-patches
from tulip.tools.decorator import Tool
from tulip.tools.executor import ToolExecutor
from tulip.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from tulip.agent.hook_orchestrator import HookOrchestrator
    from tulip.memory.conversation import ConversationManager
    from tulip.reasoning.grounding import GroundingEvaluator
    from tulip.reasoning.reflexion import Reflector


_VALID_STOP_REASONS: frozenset[str] = frozenset(
    {
        "complete",
        "terminal_tool",
        "confidence_met",
        "max_iterations",
        "tool_loop",
        "no_tools",
        "grounding_failed",
        "token_budget",
        "time_budget",
        "interrupted",
        "error",
        "cancelled",
    }
)


def _normalize_stop_reason(raw: str | None) -> StopReason:
    """Map a free-form ``TerminateEvent.reason`` to the ``StopReason`` Literal.

    User-supplied composable termination conditions emit reasons like
    ``"text_mention:DONE"``, ``"tool_called:book_flight"``, or AND-combined
    strings like ``"confidence_met AND tool_called:book_flight"``. Map by
    membership / prefix to the closest semantic match and fall back to
    ``"complete"``.
    """
    if not raw:
        return "complete"
    if raw in _VALID_STOP_REASONS:
        return raw  # type: ignore[return-value]
    # AND combinator joins child reasons with " AND ". Take the strongest
    # signal (terminal tool) if any branch matched it; otherwise fall through.
    if "tool_called:" in raw:
        return "terminal_tool"
    if "text_mention:" in raw:
        return "complete"
    # Composite reasons that contain a known literal as a substring.
    for known in _VALID_STOP_REASONS:
        if known in raw:
            return known  # type: ignore[return-value]
    return "complete"


class Agent(AgentRuntimeMixin, BaseModel):
    """
    Primary entry point for Tulip agents.

    Manages the ReAct loop with optional Reflexion and Grounding.

    Usage:
        agent = Agent(
            model="openai:gpt-4o",  # or anthropic:claude-sonnet-4-6
            tools=[search, calculate],
            system_prompt="You are a helpful assistant.",
        )

        # Async streaming
        async for event in agent.run("What is 2+2?"):
            print(event)

        # Sync execution
        result = agent.run_sync("What is 2+2?")
        print(result.message)
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    # Configuration
    config: AgentConfig = Field(..., description="Agent configuration")

    # Private attributes (not serialized).
    #
    # Concrete model / hook / reasoning types are forward-referenced via
    # ``TYPE_CHECKING`` because their modules are loaded lazily inside
    # ``initialize_agent`` to avoid pulling in optional dependencies at
    # import time.
    _model: Any = PrivateAttr(default=None)  # ModelProtocol — provider-specific concrete type
    _tool_registry: ToolRegistry = PrivateAttr(default_factory=ToolRegistry)
    _executor: ToolExecutor = PrivateAttr(default=None)  # type: ignore[assignment]
    # ``list[Any]``: heterogeneous by design — holds both ``HookProvider``
    # subclasses and ``PluginAdapter`` (duck-typed via ``__getattr__``).
    # The orchestrator uses the same widened type.
    _hooks: list[Any] = PrivateAttr(default_factory=list)
    _hook_orchestrator: HookOrchestrator | None = PrivateAttr(default=None)
    _conversation_manager: ConversationManager | None = PrivateAttr(default=None)
    _memory_manager: Any = PrivateAttr(default=None)  # BaseMemoryManager | None
    _reflector: Reflector | None = PrivateAttr(default=None)
    _grounding_evaluator: GroundingEvaluator | None = PrivateAttr(default=None)
    _grounding_model: Any = PrivateAttr(default=None)  # ModelProtocol
    _auxiliary_model: Any = PrivateAttr(default=None)  # ModelProtocol
    _last_run_state: AgentState | None = PrivateAttr(default=None)
    _interrupt_state: AgentState | None = PrivateAttr(default=None)
    _interrupt_prompt: str | None = PrivateAttr(default=None)
    _has_unverified_writes: bool = PrivateAttr(default=False)
    _interrupt_thread_id: str | None = PrivateAttr(default=None)
    _interrupt_metadata: dict[str, Any] | None = PrivateAttr(default=None)
    _cancel_signal: threading.Event | None = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model: str | Any | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        reflexion: ReflexionConfig | bool | None = None,
        grounding: GroundingConfig | bool | None = None,
        max_iterations: int = 20,
        conversation_manager: Any | None = None,
        checkpointer: Any | None = None,
        hooks: list[Any] | None = None,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize an Agent.

        Args:
            model: Model string or ModelProtocol instance
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            reflexion: Reflexion config (True for defaults, False/None to disable)
            grounding: Grounding config (True for defaults, False/None to disable)
            max_iterations: Maximum iterations before stopping
            conversation_manager: Conversation manager for message pruning
            checkpointer: Checkpointer for state persistence
            hooks: Lifecycle hooks
            config: Full AgentConfig (overrides other params)
            **kwargs: Additional config options
        """
        # Build config from params or use provided
        if config is not None:
            agent_config = config
        else:
            # Handle reflexion
            reflexion_config = None
            if reflexion is True:
                reflexion_config = ReflexionConfig()
            elif isinstance(reflexion, ReflexionConfig):
                reflexion_config = reflexion

            # Handle grounding
            grounding_config = None
            if grounding is True:
                grounding_config = GroundingConfig()
            elif isinstance(grounding, GroundingConfig):
                grounding_config = grounding

            agent_config = AgentConfig(
                model=model or "openai:gpt-4o",
                tools=tools or [],
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                reflexion=reflexion_config,
                grounding=grounding_config,
                max_iterations=max_iterations,
                conversation_manager=conversation_manager,
                checkpointer=checkpointer,
                hooks=hooks or [],
                **kwargs,
            )

        super().__init__(config=agent_config)
        self._initialize()

    def _initialize(self) -> None:
        """Resolve model, tools, executor, hooks, plugins, skills.

        Delegates to :func:`tulip.agent.initializer.initialize_agent` so
        the public-facade class stays focused on the runtime loop. The
        method is kept on ``Agent`` (rather than removed) because
        ``run()`` calls ``self._initialize()`` early to lazy-init when
        the user constructs the agent without driving setup eagerly.
        """
        from tulip.agent.initializer import initialize_agent

        initialize_agent(self)

    @property
    def name(self) -> str | None:
        """Display name from the config (multi-agent label, logs, traces)."""
        return self.config.name

    def run_sync(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent synchronously.

        Args:
            prompt: User prompt to process
            thread_id: Optional thread ID for checkpointing
            metadata: Additional metadata for tools

        Returns:
            AgentResult with final message and state
        """

        async def _run() -> AgentResult:
            started_at = datetime.now(UTC)
            stop_reason: StopReason = "complete"
            final_message: str = ""
            tool_errors = 0

            callback = self.config.callback_handler

            async for event in self.run(prompt, thread_id=thread_id, metadata=metadata):
                # Fire callback if set
                if callback is not None:
                    callback(event)

                if isinstance(event, TerminateEvent):
                    stop_reason = _normalize_stop_reason(event.reason)
                    final_message = event.final_message or ""
                elif isinstance(event, ToolCompleteEvent):
                    if event.error:
                        tool_errors += 1

            # Use actual final state from run() instead of reconstructing
            state = self._last_run_state
            if state is None:
                state = await self._create_initial_state(prompt, thread_id, metadata)
                if final_message:
                    state = state.with_message(Message.assistant(final_message))

            # Structured-output coercion (no-op when output_schema is unset).
            parsed_obj = None
            parse_error_msg = None
            structured_message = final_message
            if self.config.output_schema is not None:
                parsed_obj, parse_error_msg, state = await self._structure_output(
                    state, final_message or ""
                )
                if parsed_obj is not None:
                    # Replace ``message`` with the canonical JSON form so callers
                    # using ``result.message`` still see a schema-valid string.
                    structured_message = parsed_obj.model_dump_json()

            # Run GSAR judgment when configured. Single-pass v1: judge
            # the final answer, surface the result on AgentResult.
            # Full Algorithm-1 outer loop (regenerate / replan) lives in
            # tulip.reasoning.gsar_evaluator and can be wired
            # explicitly when the caller wants the loop dynamics.
            gsar_judgment, gsar_score_value, gsar_decision = await self._run_gsar_judgment(
                state, structured_message or final_message
            )

            elapsed_ms = (datetime.now(UTC) - started_at).total_seconds() * 1000
            metrics = ExecutionMetrics(
                iterations=state.iteration,
                tool_calls=len(state.tool_executions),
                tool_errors=tool_errors,
                total_tokens=state.total_tokens_used,
                prompt_tokens=state.prompt_tokens_used,
                completion_tokens=state.completion_tokens_used,
                cache_creation_input_tokens=state.cache_creation_tokens_used,
                cache_read_input_tokens=state.cache_read_tokens_used,
                duration_ms=elapsed_ms,
            )

            return AgentResult.from_state(
                state=state,
                stop_reason=stop_reason,
                metrics=metrics,
                started_at=started_at,
                parsed=parsed_obj,
                parse_error=parse_error_msg,
                message=structured_message,
                gsar_judgment=gsar_judgment,
                gsar_score=gsar_score_value,
                gsar_decision=gsar_decision,
            )

        async def _run_and_close_clients() -> AgentResult:
            # Wrap _run() so any model-level httpx client is shut down
            # *inside* this asyncio.run loop. Otherwise the client's
            # connections remain bound to the loop we're about to close;
            # when ``run_sync`` is called again, the next ``asyncio.run``
            # opens a fresh loop and the old client's ``__del__`` tries
            # to ``aclose`` against the now-closed loop, raising
            # ``RuntimeError: Event loop is closed``.
            try:
                return await _run()
            finally:
                close = getattr(self.model, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001 — cleanup must never mask a real error from _run()
                        pass

                # Same reasoning for the checkpointer's connection pool.
                # A driver connection pool is bound to the asyncio loop
                # that created it. Closing it here drains the connections
                # *inside* this loop. Skipping this step means the next
                # ``run_sync`` opens a fresh loop with the old pool still
                # holding TCP handles from the dead loop — which surfaces
                # as connection-reset errors on the next save.
                ckpt = getattr(getattr(self, "config", None), "checkpointer", None)
                ckpt_close = getattr(ckpt, "close", None) if ckpt is not None else None
                if ckpt_close is not None:
                    try:
                        await ckpt_close()
                    except Exception:  # noqa: BLE001 — cleanup must never mask _run() errors
                        pass

                # Drain any background tasks the SDK spawned (httpx's TLS
                # teardown schedules ``loop.call_soon`` callbacks via
                # anyio that fire after ``client.close()`` returns). If
                # we don't await them, the loop closes mid-flight and the
                # callbacks raise "Event loop is closed" on the asyncio
                # default exception handler — visible in stderr as
                # "Task exception was never retrieved".
                try:
                    pending = [
                        t
                        for t in asyncio.all_tasks()
                        if t is not asyncio.current_task() and not t.done()
                    ]
                    if pending:
                        await asyncio.wait(pending, timeout=2.0)
                except Exception:  # noqa: BLE001 — best-effort drain; never block teardown
                    pass

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(_run_and_close_clients())
        else:
            # There's a running loop, run in a thread to avoid nesting
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run_and_close_clients())
                return future.result()

    def invoke(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Invoke the agent (alias for run_sync).

        Args:
            prompt: User prompt to process
            thread_id: Optional thread ID for checkpointing
            metadata: Additional metadata for tools

        Returns:
            AgentResult with final message and state
        """
        return self.run_sync(prompt, thread_id=thread_id, metadata=metadata)

    def cancel(self) -> None:
        """Cancel a running agent from an external thread.

        Sets a signal that the agent loop checks at each iteration.
        The agent will stop gracefully with stop_reason="cancelled".

        Thread-safe — can be called from any thread while the agent is running.

        Example:
            import threading

            def run_agent():
                result = agent.run_sync("Long task...")
                print(result.stop_reason)  # "cancelled"

            t = threading.Thread(target=run_agent)
            t.start()
            time.sleep(5)
            agent.cancel()  # Stop from main thread
            t.join()
        """
        if self._cancel_signal is None:
            self._cancel_signal = threading.Event()
        self._cancel_signal.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_signal is not None and self._cancel_signal.is_set()

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """
        Wrap this agent as a Tool for use by another agent.

        The returned tool accepts a prompt string and returns the agent's
        final response. This enables agent delegation — a parent agent
        can call a sub-agent as if it were any other tool.

        Args:
            name: Tool name (defaults to agent_id or "sub_agent")
            description: Tool description (defaults to system prompt excerpt)

        Returns:
            A Tool that runs this agent when called

        Example:
            >>> researcher = Agent(
            ...     model=model, tools=[search], system_prompt="You research topics."
            ... )
            >>> writer = Agent(model=model, tools=[researcher.as_tool("research")])
            >>> result = writer.run_sync("Write about quantum computing")
        """
        from tulip.tools.decorator import tool as tool_decorator

        agent = self
        tool_name = name or self.config.agent_id or "sub_agent"
        tool_desc = description or (
            "Delegate a task to a sub-agent. "
            "The sub-agent has its own tools and will work independently "
            "to answer your request. Send a clear, specific prompt."
        )

        @tool_decorator(name=tool_name, description=tool_desc)
        def agent_tool(prompt: str) -> str:
            """Run the sub-agent with the given prompt and return its response.

            Args:
                prompt: The task or question to delegate to the sub-agent

            Returns:
                The sub-agent's final response
            """
            result = agent.run_sync(prompt)
            if result.success:
                return result.message
            return f"Sub-agent finished with status '{result.stop_reason}': {result.message}"

        return agent_tool

    async def resume(
        self,
        response: str,
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[TulipEvent]:
        """
        Resume agent execution after an interrupt.

        When a tool calls ask_user() and the agent yields an InterruptEvent,
        call this method with the user's response to continue execution.

        Without an in-memory interrupt (a fresh process — the pod that paused
        is gone), pass ``thread_id``: the interrupted state is reloaded from
        the configured checkpointer, so a durably-checkpointed run resumes
        anywhere, not just in the process that paused it.

        Args:
            response: The user's response to the interrupt question
            thread_id: Checkpoint thread to rehydrate from when this Agent
                instance holds no in-memory interrupt (requires a checkpointer)

        Yields:
            TulipEvent instances for the remaining execution

        Example:
            >>> async for event in agent.run("Build an app"):
            ...     if isinstance(event, InterruptEvent):
            ...         answer = input(event.question)
            ...         async for event in agent.resume(answer):
            ...             handle(event)
        """
        if self._interrupt_state is not None:
            state = self._interrupt_state
            self._interrupt_state = None
            # Re-run — we pass the original prompt; the state already has the
            # full history
            prompt = self._interrupt_prompt or ""
            thread_id = self._interrupt_thread_id
            metadata = self._interrupt_metadata
            # Clear interrupt bookkeeping
            self._interrupt_prompt = None
            self._interrupt_thread_id = None
            self._interrupt_metadata = None
        else:
            # Rehydrate: no in-memory interrupt, so reload the paused state
            # from the checkpointer (the cross-process resume path).
            if thread_id is None or self.config.checkpointer is None:
                raise RuntimeError(
                    "No interrupt to resume from. Call run() first, or pass "
                    "thread_id with a configured checkpointer to rehydrate."
                )
            loaded = await self.config.checkpointer.load(thread_id)
            if loaded is None:
                raise RuntimeError(f"No checkpoint found for thread {thread_id!r} to resume from.")
            state = loaded
            prompt = ""
            metadata = None

        # Add the user's response as a tool result for ask_user
        state = state.with_message(Message.system(f"[User Response] {response}"))

        # Store for _create_initial_state to pick up
        self._last_run_state = state

        # Continue execution from the interrupted state
        async for event in self._run_from_state(state, prompt, thread_id, metadata):
            yield event

    @property
    def model(self) -> Any:
        """Get the model instance."""
        self._initialize()
        return self._model

    @property
    def tools(self) -> ToolRegistry:
        """Get the tool registry."""
        self._initialize()
        return self._tool_registry

    def add_tool(self, tool: Tool) -> None:
        """Register a tool on this agent after construction.

        Tulip compiles ``config.tools`` into the runtime ``ToolRegistry``
        once, inside ``__init__`` (via :func:`tulip.agent.initializer.
        initialize_agent`). Mutating ``self.config.tools`` directly after
        that point is a silent no-op — the model never sees the added
        tool because the registry has already been built.

        Use this method (or :meth:`add_tools`) when you want to compose a
        specialist fleet at runtime: build each specialist, wrap it via
        ``Agent.as_tool(...)``, and attach the wrappers to the
        orchestrator.

        The tool is also appended to ``self.config.tools`` so that a
        subsequent re-initialisation (e.g. after a config-driven
        clone) sees the same shape.

        Raises:
            TypeError: if ``tool`` is not a :class:`tulip.tools.Tool`
                instance. Callable functions must be wrapped with the
                :func:`@tool` decorator first.
            ValueError: if a tool with the same ``name`` is already
                registered (propagated from
                :meth:`ToolRegistry.register`).
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"Expected Tool instance (use @tool to wrap a function), got {type(tool)}"
            )
        self._initialize()
        self._tool_registry.register(tool)
        # Mirror into config so a re-initialisation reconstructs the
        # same surface. ``config.tools`` is a list[Any] by Pydantic
        # declaration, so we mutate in place rather than reassigning.
        self.config.tools.append(tool)

    def add_tools(self, tools: list[Tool]) -> None:
        """Register multiple tools at once.

        Equivalent to calling :meth:`add_tool` for each entry. If any
        single registration fails (wrong type, duplicate name), the
        whole call fails: tools registered before the failing one
        remain in the registry. Validate inputs ahead of time when
        atomic behaviour is required.
        """
        for t in tools:
            self.add_tool(t)

    @property
    def system_prompt(self) -> str:
        """Get the configured system prompt as a string.

        If the config value is a callable (dynamic prompt), it is
        coerced to its ``repr`` so this property never returns non-str.
        Use ``self.config.system_prompt`` directly to access the raw
        value (string or callable) when you need to invoke the
        dynamic form.
        """
        prompt = self.config.system_prompt
        return prompt if isinstance(prompt, str) else repr(prompt)
