# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Main Agent class - 100% Pydantic."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from locus.agent.config import AgentConfig, GroundingConfig, ReflexionConfig
from locus.agent.durability import DurabilityMode
from locus.agent.result import (
    STOP_REASONS,
    AgentResult,
    ExecutionMetrics,
    StopReason,
    normalize_stop_reason,
)
from locus.agent.run_context import (
    ARUN_RESULT_SLOT,
    PendingInterrupt,
    ResultSlot,
    RunContext,
)
from locus.agent.runtime_loop import (
    UNFINISHED_CALL_ERROR,
    AgentRuntimeMixin,
    _invocation_arguments,
    _take_iteration_checkpoints,
    close_unfinished_calls,
)
from locus.core.errors import ApprovalPendingError, GSARValidationError
from locus.core.events import (
    GroundingEvent,
    LocusEvent,
    ReflectEvent,
    TerminateEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.core.messages import Message, Role, ToolResult
from locus.core.state import AgentState
from locus.models import get_model  # noqa: F401 — re-exported for test monkey-patches
from locus.tools.decorator import Tool
from locus.tools.executor import ToolExecutor
from locus.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from locus.agent.hook_orchestrator import HookOrchestrator
    from locus.agent.subagent import SubagentResult
    from locus.memory.conversation import ConversationManager
    from locus.memory.ledger import LedgerManifest
    from locus.reasoning.grounding import GroundingEvaluator
    from locus.reasoning.reflexion import Reflector


logger = logging.getLogger(__name__)

#: Kept under their old names for code that imported them from here.
_VALID_STOP_REASONS: frozenset[str] = frozenset(STOP_REASONS)
_normalize_stop_reason = normalize_stop_reason


def _interrupt_payload(content: str | None) -> dict[str, Any] | None:
    """The interrupt marker a tool returned, or ``None`` for a real result."""
    if not content or '"__interrupt__": true' not in content:
        return None
    import json  # noqa: PLC0415

    try:
        data = json.loads(content)
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("__interrupt__"):
        return data
    return None


#: Kept under its old name for code that imported it from here.
_UNFINISHED_CALL_ERROR = UNFINISHED_CALL_ERROR


def _turn_finished(state: AgentState) -> bool:
    """Whether ``state`` ends on the model's answer rather than inside a turn."""
    if not state.messages:
        return False
    last = state.messages[-1]
    return last.role == Role.ASSISTANT and not last.tool_calls


@contextlib.asynccontextmanager
async def _closing(stream: Any) -> AsyncIterator[Any]:
    """``contextlib.aclosing`` that tolerates a stream with no ``aclose``.

    ``run`` may be overridden by something that returns a plain async
    iterator; a real generator is closed on exit so an early exit (an
    exception, a cancelled ``arun``) finalizes the run immediately.
    """
    try:
        yield stream
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


class Agent(AgentRuntimeMixin, BaseModel):
    """
    Primary entry point for Locus agents.

    Manages the ReAct loop with optional Reflexion and Grounding.

    Usage:
        agent = Agent(
            model="openai:gpt-4o",  # or anthropic:claude-sonnet-5-5
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
    # ObservationPack (``config.observation_pack``), built on initialisation.
    _observation_pack: Any = PrivateAttr(default=None)
    _model_prices: tuple[float, float] | None = PrivateAttr(default=None)
    # The model's leaked tool-call formats, resolved from its profile on first
    # use (``None`` until then); see ``AgentConfig.leaked_tool_call_formats``.
    _leaked_formats: tuple[str, ...] | None = PrivateAttr(default=None)
    _memory_manager: Any = PrivateAttr(default=None)  # BaseMemoryManager | None
    _reflector: Reflector | None = PrivateAttr(default=None)
    _grounding_evaluator: GroundingEvaluator | None = PrivateAttr(default=None)
    _grounding_model: Any = PrivateAttr(default=None)  # ModelProtocol
    _auxiliary_model: Any = PrivateAttr(default=None)  # ModelProtocol
    # Last finished run's state. Informational only: under concurrency it is
    # whichever run finished last, so nothing in the runtime reads it back.
    _last_run_state: AgentState | None = PrivateAttr(default=None)
    # Paused runs, keyed strictly by thread id (``None`` = a thread-less run).
    _interrupts: dict[str | None, PendingInterrupt] = PrivateAttr(default_factory=dict)
    # In-flight runs (keyed by ``id(RunContext)``), for ``cancel(thread_id=...)``.
    _active_runs: dict[int, RunContext] = PrivateAttr(default_factory=dict)
    _runs_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    # Legacy cancel-all flag: set by ``cancel()`` with no thread id so a
    # cancel issued before a run starts still stops it (pre-2.17 contract).
    _cancel_signal: threading.Event | None = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    # MCP tools are attached from the first run() (connecting is async);
    # this keeps that a one-time step rather than a per-run round trip.
    _mcp_attached: bool = PrivateAttr(default=False)
    # Per-server attach bookkeeping: ids of attached clients, and when a
    # failed one may be retried (a server down at the first run attaches on
    # a later run instead of never).
    _mcp_attached_ids: set[int] = PrivateAttr(default_factory=set)
    _mcp_retry_at: dict[int, float] = PrivateAttr(default_factory=dict)

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
            **kwargs: Any other :class:`~locus.agent.config.AgentConfig`
                field, forwarded verbatim. 36 options are reachable only
                this way and are therefore invisible to ``help(Agent)``,
                to editor autocomplete, and to ``inspect.signature(Agent)``
                -- Pydantic's metaclass builds ``__signature__`` from the
                explicit parameters above and drops the ``**kwargs``. They
                are real, supported, and documented; they just cannot be
                discovered by introspection. The ones most often reached
                for:

                ``termination``, ``output_schema`` (+ ``output_schema_strict``,
                ``output_schema_retries``), ``memory_manager``, ``skills``,
                ``playbook``, ``planning``, ``auxiliary_model``, ``gsar``,
                ``require_verification``, ``plugins``, ``temperature``,
                ``max_tokens``, ``token_budget``, ``time_budget_seconds``,
                ``max_concurrency``, ``tool_execution``, ``tool_result_store``,
                ``tool_result_offload``, ``web_search``, ``web_fetch``, ``image_generator``,
                ``speech_provider``, ``name``, ``agent_id``, ``metadata``.

                See :class:`~locus.agent.config.AgentConfig` for the full
                set and the type of each.
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

        Delegates to :func:`locus.agent.initializer.initialize_agent` so
        the public-facade class stays focused on the runtime loop. The
        method is kept on ``Agent`` (rather than removed) because
        ``run()`` calls ``self._initialize()`` early to lazy-init when
        the user constructs the agent without driving setup eagerly.
        """
        from locus.agent.initializer import initialize_agent

        initialize_agent(self)

    @property
    def observation_pack(self) -> Any:
        """The agent's :class:`~locus.memory.observation_pack.ObservationPack`, or ``None``.

        Its ``stats(session)`` says what it did for a thread (or a thread-less
        run, by run id).
        """
        self._initialize()
        return self._observation_pack

    @property
    def name(self) -> str | None:
        """Display name from the config (multi-agent label, logs, traces)."""
        return self.config.name

    async def arun(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        invocation_id: str | None = None,
        durability: DurabilityMode | None = None,
        from_checkpoint: str | None = None,
    ) -> AgentResult:
        """
        Run the agent asynchronously and return a single ``AgentResult``.

        This is the async, browser-safe equivalent of :meth:`run_sync`: it
        drives the same event loop (``self.run(...)``), builds the same
        ``AgentResult``, but without spawning a thread or calling
        ``asyncio.run``. That makes it usable in environments where threads
        are unavailable — notably Pyodide / WebAssembly, where ``run_sync`` cannot work.

        Unlike ``run_sync`` it does NOT close the underlying model client or
        checkpointer pool afterward — the caller owns the loop, so those
        resources stay bound to it and can be reused across ``arun`` calls.

        **Each call starts from an empty conversation** unless you give it
        somewhere to remember: prior turns are reloaded only when the agent has
        a ``checkpointer`` configured *and* you pass ``thread_id``. Calling
        ``run_sync(prompt)`` in a loop therefore produces independent turns,
        not a conversation — which is the right default for a shared agent, and
        a surprise if you expected it to follow along.

        Args:
            prompt: User prompt to process
            thread_id: Conversation to continue. Required (with a configured
                checkpointer) for this run to see earlier turns.
            metadata: Additional metadata for tools
            invocation_id: An id for this turn the caller keeps; a run issued
                again under it after a crash replays the recorded tool calls
                (see :meth:`run`).
            durability: When this run writes its iteration checkpoints
                (``"sync"``, ``"async"`` or ``"exit"``; see :meth:`run`).
            from_checkpoint: Start the turn from the thread's state at this
                checkpoint instead of its latest (see :meth:`run`).

        Returns:
            AgentResult with final message and state
        """
        started_at = datetime.now(UTC)
        stop_reason: StopReason = "complete"
        final_message: str = ""
        tool_errors = 0

        callback = self.config.callback_handler

        # Only forward ``model_kwargs`` when the caller supplied it, so an
        # existing subclass or test double that overrides ``run`` with the
        # previous signature keeps working.
        run_kwargs: dict[str, Any] = {"thread_id": thread_id, "metadata": metadata}
        if model_kwargs is not None:
            run_kwargs["model_kwargs"] = model_kwargs
        if invocation_id is not None:
            run_kwargs["invocation_id"] = invocation_id
        if durability is not None:
            run_kwargs["durability"] = durability
        if from_checkpoint is not None:
            run_kwargs["from_checkpoint"] = from_checkpoint

        grounding_score: float | None = None
        ungrounded_claims: list[str] = []
        # The runtime already counts these -- ``_reflexion_evals`` and
        # ``_grounding_evals`` -- but they are locals in a generator, so the
        # numbers never left. Counting the events it yields gets the same
        # figures without threading state through the loop.
        reflexion_evaluations = 0
        grounding_evaluations = 0

        # The run this call drives leaves its final state in ``slot`` — never
        # on the shared agent, where a concurrent run on another thread would
        # overwrite it between the last event and this read.
        slot = ResultSlot(owner=self)
        slot_token = ARUN_RESULT_SLOT.set(slot)
        try:
            async with _closing(self.run(prompt, **run_kwargs)) as events:
                async for event in events:
                    # Fire callback if set
                    if callback is not None:
                        callback(event)

                    if isinstance(event, TerminateEvent):
                        stop_reason = _normalize_stop_reason(event.reason)
                        final_message = event.final_message or ""
                    elif isinstance(event, ToolCompleteEvent):
                        if event.error:
                            tool_errors += 1
                    elif isinstance(event, ReflectEvent):
                        reflexion_evaluations += 1
                    elif isinstance(event, GroundingEvent):
                        grounding_evaluations += 1
                        # The grounding loop already ran and emitted its verdict;
                        # it simply never reached the result. Keep the last one:
                        # with `max_replans` the answer is re-grounded after each
                        # replan, and the score that describes the answer being
                        # returned is the final one.
                        grounding_score = event.score
                        ungrounded_claims = list(event.ungrounded_claims)
        finally:
            ARUN_RESULT_SLOT.reset(slot_token)

        # The actual final state of THIS run. ``None`` only when ``run`` is
        # overridden by something that never goes through the runtime loop.
        state = slot.state
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
        # locus.reasoning.gsar_evaluator and can be wired
        # explicitly when the caller wants the loop dynamics.
        gsar_judgment, gsar_score_value, gsar_decision = await self._run_gsar_judgment(
            state, structured_message or final_message
        )

        # ``fail_on_low_score`` was documented as raising GSARValidationError
        # and never read, so an agent configured to refuse un-grounded output
        # shipped it silently -- the one outcome the setting exists to prevent.
        if (
            self.config.gsar is not None
            and self.config.gsar.fail_on_low_score
            and gsar_decision is not None
            and gsar_decision != "proceed"
        ):
            raise GSARValidationError(
                f"GSAR judgment returned {gsar_decision!r}"
                + (f" (score {gsar_score_value:.2f})" if gsar_score_value is not None else "")
                + "; fail_on_low_score is set.",
                decision=gsar_decision,
                score=gsar_score_value,
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
            cost_usd=state.cost_usd_used if state.priced else None,
            duration_ms=elapsed_ms,
            reflexion_evaluations=reflexion_evaluations,
            grounding_evaluations=grounding_evaluations,
        )

        return AgentResult.from_state(
            state=state,
            stop_reason=stop_reason,
            metrics=metrics,
            started_at=started_at,
            parsed=parsed_obj,
            parse_error=parse_error_msg,
            message=structured_message,
            grounding_score=grounding_score,
            ungrounded_claims=ungrounded_claims,
            gsar_judgment=gsar_judgment,
            gsar_score=gsar_score_value,
            gsar_decision=gsar_decision,
        )

    async def drain_memory(self) -> None:
        """Wait for the memory manager's background extractions to finish.

        With ``LLMMemoryManager(extract_mode="background")`` extraction runs
        after a turn's final event; await this at graceful shutdown (and in
        tests) so none is lost — bound it with ``asyncio.timeout(...)`` if
        needed. A no-op without a memory manager or in ``"inline"`` mode.
        :meth:`run_sync` calls it before returning.
        """
        manager = self._memory_manager or self.config.memory_manager
        drain = getattr(manager, "drain", None)
        if drain is not None:
            await drain()

    def run_sync(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        invocation_id: str | None = None,
        durability: DurabilityMode | None = None,
        from_checkpoint: str | None = None,
    ) -> AgentResult:
        """
        Run the agent synchronously.

        Blocking wrapper around :meth:`arun`: it drives ``self.arun(...)`` on
        a fresh event loop (or a worker thread when a loop is already
        running, with the caller's context variables — the OpenTelemetry
        context among them — carried over) and closes the model /
        checkpointer clients bound to that loop before returning. In
        thread-less environments (Pyodide / WASM) this cannot work — call
        ``await self.arun(...)`` directly there.

        Inside a running loop, tools from an ``MCPClient`` connected on that
        loop cannot be reached from the worker thread's loop: the first call
        would wait forever. ``run_sync`` raises ``RuntimeError`` for that
        case; use ``await agent.arun(...)`` there.

        Args:
            prompt: User prompt to process
            thread_id: Optional thread ID for checkpointing
            metadata: Additional metadata for tools
            invocation_id: An id for this turn the caller keeps (see :meth:`run`).
            durability: When this run writes its iteration checkpoints
                (``"sync"``, ``"async"`` or ``"exit"``; see :meth:`run`).
            from_checkpoint: Start the turn from the thread's state at this
                checkpoint instead of its latest (see :meth:`run`).

        Returns:
            AgentResult with final message and state

        Raises:
            RuntimeError: Called inside a running event loop with MCP tools
                whose session is bound to that loop.
        """

        async def _run_and_close_clients() -> AgentResult:
            # Wrap ``self.arun`` so any model-level httpx client is shut down
            # *inside* this asyncio.run loop. Otherwise the client's
            # connections remain bound to the loop we're about to close;
            # when ``run_sync`` is called again, the next ``asyncio.run``
            # opens a fresh loop and the old client's ``__del__`` tries
            # to ``aclose`` against the now-closed loop, raising
            # ``RuntimeError: Event loop is closed``.
            try:
                return await self.arun(
                    prompt,
                    thread_id=thread_id,
                    metadata=metadata,
                    model_kwargs=model_kwargs,
                    invocation_id=invocation_id,
                    durability=durability,
                    from_checkpoint=from_checkpoint,
                )
            finally:
                # Background memory extraction is bound to this loop, which
                # closes on return: finish it here rather than lose it.
                try:
                    await self.drain_memory()
                except Exception:  # noqa: BLE001 — cleanup must never mask a real error from arun()
                    pass

                close = getattr(self.model, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001 — cleanup must never mask a real error from arun()
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
                    except Exception:  # noqa: BLE001 — cleanup must never mask arun() errors
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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(_run_and_close_clients())
        else:
            bound = self._mcp_clients_bound_to(loop)
            if bound:
                raise RuntimeError(
                    "Agent.run_sync() was called inside a running event loop with MCP "
                    f"tools whose session lives on that loop ({', '.join(bound)}). "
                    "run_sync drives the run on a worker thread's own loop, which "
                    "cannot reach a session bound to the caller's, so the first MCP "
                    "tool call would wait forever. Use `await agent.arun(...)` (or "
                    "`async for event in agent.run(...)`) here, or call run_sync "
                    "where no loop is running and let the agent connect the server "
                    "itself (mcp_servers=[client])."
                )
            # There's a running loop: run in a thread to avoid nesting. The
            # caller's context variables go with it — a run that only
            # happens to be driven from a thread keeps its trace parent.
            import concurrent.futures
            import contextvars

            context = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(context.run, asyncio.run, _run_and_close_clients())
                return future.result()

    def _mcp_clients_bound_to(self, loop: asyncio.AbstractEventLoop) -> list[str]:
        """Labels of the configured MCP clients whose live session runs on ``loop``.

        Looks at ``mcp_servers`` and at the tools an ``MCPClient.to_locus_tools``
        call produced (they carry their client).
        """
        candidates: list[Any] = list(getattr(self.config, "mcp_servers", None) or ())
        candidates.extend(getattr(t, "_source", None) for t in self.config.tools)
        seen: set[int] = set()
        labels: list[str] = []
        for client in candidates:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            bound_loop = getattr(client, "bound_loop", None)
            if callable(bound_loop) and bound_loop() is loop:
                labels.append(str(getattr(client, "label", None) or type(client).__name__))
        return labels

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

    def cancel(self, thread_id: str | None = None) -> int:
        """Cancel running agent work from any thread or task.

        Sets a signal the agent loop checks at each iteration; the run stops
        gracefully with ``stop_reason="cancelled"``.

        * ``cancel(thread_id="t-42")`` stops only the in-flight run(s) on that
          thread — the form a multi-user server wants, where one Agent serves
          every conversation.
        * ``cancel()`` (no argument) keeps the pre-2.17 meaning: stop every
          in-flight run of this agent, and if none is running yet, the next run
          to start.

        Thread-safe — can be called from any thread while the agent is running.

        Args:
            thread_id: Only cancel runs on this conversation.

        Returns:
            How many in-flight runs were signalled (a no-argument call with
            nothing running still arms the signal for the next run and
            returns 0).

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
        with self._runs_lock:
            runs = list(self._active_runs.values())
        if thread_id is not None:
            targets = [rc for rc in runs if rc.thread_id == thread_id]
        else:
            if self._cancel_signal is None:
                self._cancel_signal = threading.Event()
            self._cancel_signal.set()
            targets = runs
        for rc in targets:
            rc.cancel.set()
        return len(targets)

    @property
    def is_cancelled(self) -> bool:
        """Whether a cancel-all (``cancel()`` with no thread id) is pending."""
        return self._cancel_signal is not None and self._cancel_signal.is_set()

    @property
    def _interrupt_state(self) -> AgentState | None:
        """The most recently parked interrupt's state (back-compat view).

        Resume never reads this: paused runs are held per thread id in
        ``_interrupts`` and resumed strictly by it.
        """
        if not self._interrupts:
            return None
        return list(self._interrupts.values())[-1].state

    def pending_interrupts(self) -> list[str | None]:
        """Thread ids that have a paused run held in this process's memory.

        A thread paused in another process (or before a restart) is not
        listed; ``resume(thread_id=...)`` still rehydrates it from the
        checkpointer.
        """
        return list(self._interrupts)

    async def run_subagent(
        self,
        prompt: str,
        *,
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a focused subagent. Complete the delegated task.",
        model: str | Any | None = None,
        name: str = "subagent",
        max_iterations: int = 10,
        hooks: list[Any] | None = None,
        on_event: Callable[[LocusEvent], Awaitable[None] | None] | None = None,
        **agent_kwargs: Any,
    ) -> SubagentResult:
        """Run an isolated child agent and return its :class:`SubagentResult`.

        Thin binding of :func:`locus.agent.subagent.run_subagent` to this
        agent: the child defaults to this agent's model, and this agent's
        :meth:`cancel` stops the child — whether the call comes from one of
        this agent's tool bodies mid-run or from a harness between runs.

        Isolation is the point: the child gets a fresh conversation, its own
        system prompt, and ONLY the tools in ``tools`` — never this agent's
        toolset by inheritance. When called from a tool body during a run,
        the child's token usage folds into this run's counters, so
        ``token_budget`` and ``TerminateEvent.usage`` keep counting
        delegated spend. See :mod:`locus.agent.subagent` for the full
        contract and a harness-side ``task``-tool example.

        Args:
            prompt: The delegated task.
            tools: Explicit tool allowlist for the child (default: none).
            system_prompt: The child's system prompt.
            model: Child model override; defaults to this agent's model.
            name: Attribution label stamped on the child's events.
            max_iterations: The child's iteration cap.
            hooks: Lifecycle hooks for the child (policy attaches here).
            on_event: Callback receiving every child event (sync or async).
            **agent_kwargs: Further ``AgentConfig`` fields for the child.

        Returns:
            SubagentResult with the child's text, usage, iterations, and
            stop reason.
        """
        from locus.agent.subagent import run_subagent

        if self._cancel_signal is None:
            self._cancel_signal = threading.Event()
        return await run_subagent(
            prompt,
            model=model if model is not None else self.model,
            tools=tools,
            system_prompt=system_prompt,
            name=name,
            max_iterations=max_iterations,
            hooks=hooks,
            on_event=on_event,
            cancel_signal=self._cancel_signal,
            **agent_kwargs,
        )

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
        from locus.tools.decorator import tool as tool_decorator

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
        response: str = "",
        *,
        thread_id: str | None = None,
        perform_dangling: bool = False,
        metadata: dict[str, Any] | None = None,
        invocation_id: str | None = None,
        durability: DurabilityMode | None = None,
        from_checkpoint: str | None = None,
    ) -> AsyncGenerator[LocusEvent, None]:
        """
        Resume agent execution after an interrupt, or a run that crashed.

        When a tool calls ask_user() and the agent yields an InterruptEvent,
        call this method with the user's response to continue execution.

        With ``invocation_id`` the call instead continues a run the process
        died in: ``resume(invocation_id=..., thread_id=...)`` is
        :meth:`continue_turn` for that invocation. It takes no ``response``.
        The run picks the turn up from the thread's last checkpoint — or,
        when none of the turn's checkpoints landed (``durability="exit"``),
        from the turn's start as the tool-effect ledger recorded it — and
        every tool call the crashed attempt recorded is replayed from the
        ledger rather than run again (:mod:`locus.memory.ledger`).

        **Resume is keyed strictly by thread.** Paused runs are held in memory
        per ``thread_id``; ``resume(thread_id="A")`` only ever continues
        thread A — from memory when this process paused it, otherwise
        rehydrated from the configured checkpointer (a fresh process: the pod
        that paused is gone). It never picks up another thread's pause.
        Without ``thread_id`` the call is accepted only when exactly one run
        is paused in memory (the single-conversation back-compat form); with
        several paused it raises rather than guess.

        A resumed run continues the SAME turn: its iteration count and budgets
        carry on from the pause.

        ``perform_dangling`` closes the gap the 2.11.1 fold left open for
        approval holds. The fold answers the dangling call with the human's
        verdict TEXT — the model reads "approve" as the call's result and,
        reasonably, never re-issues a call that already returned, so nothing
        ever performs the action (measured live; see
        FINDING-an-approved-action-can-silently-not-happen.md for the first
        half of this story). With ``perform_dangling=True`` and a dangling
        call that is not ``ask_user``, the runtime RE-INVOKES that call —
        same tool, same arguments, through the normal executor — and folds
        the REAL result instead. A gated tool decides for itself under the
        caller's primed decision: approve executes, deny refuses, and either
        way the model sees what actually happened. ``ask_user`` and the
        no-dangling-call case are untouched, so answers behave exactly as
        before regardless of the flag.

        Args:
            response: The user's response to the interrupt question
            thread_id: The paused thread to continue. Resumes that thread's
                in-memory interrupt, or rehydrates it from the checkpointer.
            perform_dangling: Re-invoke a dangling non-``ask_user`` call and
                fold its real result, instead of folding ``response`` as text
            metadata: Invocation metadata for the resumed segment (what tools
                see as ``ctx.invocation_metadata`` and hooks as
                ``event.run.metadata``). Defaults to the paused run's metadata
                (in memory) or the thread's checkpointed metadata. Ephemeral
                keys (``mcp_headers``) are never checkpointed: an in-process
                resume reuses the paused run's, a cross-process resume that
                needs them passes them here again.
            invocation_id: The id of a run that crashed (its ``run_id``, or
                the ``invocation_id`` it was started with) to continue on
                ``thread_id``, replaying its recorded tool calls.
            durability: When the resumed segment writes its iteration
                checkpoints (see :meth:`run`).
            from_checkpoint: Continue from the thread's state at this
                checkpoint instead of its latest (one of
                ``checkpointer.list(thread_id)``), as if the run had paused
                there: the thread's in-memory pause, if any, is discarded,
                and the segment's first lasting save records the checkpoint
                as its parent (later saves follow on from it). The
                checkpoint itself is never deleted. Needs ``thread_id`` and
                a checkpointer.

        Raises:
            ApprovalPendingError: ``perform_dangling=True`` and the held call
                interrupted again — its approval has not been decided yet. The
                thread stays paused and nothing is folded or checkpointed.
            RuntimeError: Nothing to resume (no paused run for the thread and
                no checkpoint), or ``thread_id`` omitted while several runs
                are paused.
            LookupError: ``from_checkpoint`` names a checkpoint the thread
                does not have.

        Yields:
            LocusEvent instances for the remaining execution

        Example:
            >>> async for event in agent.run("Build an app"):
            ...     if isinstance(event, InterruptEvent):
            ...         answer = input(event.question)
            ...         async for event in agent.resume(answer):
            ...             handle(event)
        """
        if invocation_id is not None:
            if thread_id is None:
                raise RuntimeError(
                    "resume(invocation_id=...) needs thread_id: a run is found through its thread."
                )
            if response:
                raise RuntimeError(
                    "resume(invocation_id=...) continues a crashed run and takes no response; "
                    "answer an interrupt with resume(response, thread_id=...)."
                )
            if from_checkpoint is not None:
                raise RuntimeError(
                    "resume(invocation_id=...) continues a crashed run from its ledger and "
                    "cannot also start from_checkpoint; fork the thread at that checkpoint "
                    "and run on the fork instead."
                )
            async with contextlib.aclosing(
                self.continue_turn(
                    thread_id, invocation_id=invocation_id, metadata=metadata, durability=durability
                )
            ) as continued:
                async for event in continued:
                    yield event
            return

        if from_checkpoint is not None:
            if thread_id is None or self.config.checkpointer is None:
                raise RuntimeError("resume(from_checkpoint=...) needs thread_id and a checkpointer")
            # The named checkpoint supersedes whatever pause is parked in memory.
            self._take_interrupt(thread_id)
            pending = None
        else:
            pending = self._take_interrupt(thread_id)
        if pending is not None:
            state = pending.state
            # The prompt of the paused run; the state already has the history.
            prompt = pending.prompt
            thread_id = pending.thread_id
            if metadata is not None:
                run_metadata, ephemeral = self._split_run_metadata(metadata)
            else:
                run_metadata, ephemeral = pending.metadata, dict(pending.ephemeral)
        else:
            # Rehydrate: no in-memory interrupt for this thread, so reload the
            # paused state from the checkpointer (the cross-process path).
            if thread_id is None or self.config.checkpointer is None:
                raise RuntimeError(
                    "No interrupt to resume from. Call run() first, or pass "
                    "thread_id with a configured checkpointer to rehydrate."
                )
            loaded = await self.config.checkpointer.load(thread_id, from_checkpoint)
            if loaded is None:
                if from_checkpoint is not None:
                    raise LookupError(
                        f"no checkpoint {from_checkpoint!r} of thread {thread_id!r} to resume from"
                    )
                raise RuntimeError(f"No checkpoint found for thread {thread_id!r} to resume from.")
            state = loaded
            if from_checkpoint is not None:
                # The segment's final save must not delete the checkpoint it
                # descends from (an iteration save names itself in the list).
                _, state = _take_iteration_checkpoints(state)
            prompt = ""
            # A checkpoint never holds ephemeral metadata (``mcp_headers``):
            # a cross-process resume that needs it passes ``metadata=``.
            run_metadata, ephemeral = self._split_run_metadata(
                metadata if metadata is not None else dict(loaded.metadata)
            )
            # Checkpoints never carry the (ephemeral) memory block, so the
            # rest of this turn gets it re-injected, as the in-memory path has.
            # (A fresh process has not initialised the agent yet.)
            memory_manager = self._memory_manager or self.config.memory_manager
            if memory_manager is not None:
                state = await memory_manager.on_session_start(state)

        # This resumed segment is a run of its own: cancellable by thread,
        # visible to hooks as ``event.run``, isolated from concurrent runs.
        rc = self._begin_run(
            state, prompt, thread_id, run_metadata, ephemeral=ephemeral, durability=durability
        )
        if from_checkpoint is not None and rc.checkpoints is not None:
            rc.checkpoints.parent_checkpoint_id = from_checkpoint
        try:
            folded_events: list[LocusEvent] = []
            state = await self._fold_resume_response(
                state, response, rc, perform_dangling, pending, folded_events
            )
        except BaseException:
            self._end_run(rc, state)
            raise
        for folded_event in folded_events:
            yield folded_event

        # Continue execution from the interrupted state
        async with contextlib.aclosing(
            self._run_from_state(state, prompt, thread_id, run_metadata, _run=rc)
        ) as segment:
            async for event in segment:
                yield event

    async def continue_turn(
        self,
        thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        invocation_id: str | None = None,
        durability: DurabilityMode | None = None,
    ) -> AsyncGenerator[LocusEvent, None]:
        """Continue a turn that stopped before it finished, from its checkpoint.

        A process killed mid-turn (a deploy, an OOM, a lost pod) leaves the
        thread's latest checkpoint inside the turn — with
        ``checkpoint_every_n_iterations`` set, at the last iteration whose
        tool results were saved. ``run()`` on that thread would start a NEW
        turn with a new user message; this continues the SAME turn instead:
        no user message is added, iteration count and budgets carry on, and
        every call whose result is in the checkpoint stays done — the model
        is called next, with those results in front of it.

        A call the checkpoint holds without a result (the process stopped
        while it ran) is answered with an error saying its outcome is
        unknown, never re-run: re-running a side effect the first attempt
        may already have performed is the model's decision to make, with
        that error in front of it.

        With the tool-effect ledger on (``AgentConfig.tool_ledger``; the
        default for ``durability="async"`` and ``"exit"``), the continued
        segment replays: a tool call the crashed attempt recorded at the same
        step with the same arguments returns the recorded result and the tool
        does not run again; only calls with no record run. And when none of
        the turn's checkpoints landed — ``durability="exit"`` writes once, at
        the end — the turn is rebuilt from its start as the ledger's manifest
        recorded it (the prompt, the run id), and run again with replay.

        A thread paused on an interrupt in this process is answered with
        :meth:`resume`, which folds the human's response; ``continue_turn``
        refuses it.

        Args:
            thread_id: The thread whose turn to continue.
            metadata: Invocation metadata for the continued segment. Defaults
                to the thread's checkpointed metadata; ephemeral keys
                (``mcp_headers``) are never checkpointed, so a segment that
                needs them passes them here again.
            model_kwargs: Per-call model parameters, as for :meth:`run`.
            invocation_id: When given, the turn in flight on the thread must
                be this run (its ``run_id`` / the ``invocation_id`` it was
                started with), or the call raises rather than continue
                another turn.
            durability: When the continued segment writes its iteration
                checkpoints (see :meth:`run`).

        Raises:
            RuntimeError: No checkpointer, no checkpoint or ledger for the
                thread, the thread is paused on an in-memory interrupt, its
                last turn already finished with an answer (start a new one
                with ``run()``), or ``invocation_id`` is not the turn in
                flight.

        Yields:
            LocusEvent instances for the rest of the turn.
        """
        checkpointer = self.config.checkpointer
        if checkpointer is None:
            raise RuntimeError("continue_turn needs a checkpointer on the agent")
        if thread_id in self._interrupts:
            raise RuntimeError(
                f"Thread {thread_id!r} is paused on an interrupt; answer it with resume()."
            )
        loaded = await checkpointer.load(thread_id)
        manifest = await self._ledger_manifest(thread_id)
        mid_turn = loaded is not None and not _turn_finished(loaded)
        # The turn's checkpoints never landed, but its ledger did: the run
        # started, recorded tool calls, and died before its first save.
        unsaved_turn = (
            manifest is not None
            and not mid_turn
            and (loaded is None or loaded.run_id != manifest.invocation_id)
        )
        if invocation_id is not None:
            in_flight = (
                loaded.run_id
                if mid_turn and loaded is not None
                else manifest.invocation_id
                if unsaved_turn and manifest is not None
                else None
            )
            if in_flight != invocation_id:
                raise RuntimeError(
                    f"Invocation {invocation_id!r} is not in flight on thread {thread_id!r}"
                    + (f" ({in_flight!r} is)." if in_flight is not None else ".")
                )
        self._initialize()
        if mid_turn and loaded is not None:
            state = close_unfinished_calls(loaded)
            prompt = ""
            metadata_source = dict(loaded.metadata)
        elif unsaved_turn and manifest is not None:
            state = await self._create_initial_state(
                manifest.prompt, thread_id, manifest.metadata or None
            )
            state = state.model_copy(update={"run_id": manifest.invocation_id})
            prompt = manifest.prompt
            metadata_source = dict(state.metadata)
        elif loaded is None:
            raise RuntimeError(f"No checkpoint found for thread {thread_id!r} to continue.")
        else:
            raise RuntimeError(
                f"The last turn on thread {thread_id!r} already finished; "
                "call run() to start a new one."
            )
        run_metadata, ephemeral = self._split_run_metadata(
            metadata if metadata is not None else metadata_source
        )
        # Checkpoints never carry the (ephemeral) memory block; the rest of
        # the turn gets it re-injected, as a cross-process resume does.
        if self._memory_manager is not None:
            state = await self._memory_manager.on_session_start(state)

        rc = self._begin_run(
            state,
            prompt,
            thread_id,
            run_metadata,
            ephemeral=ephemeral,
            durability=durability,
            replay=True,
        )
        async with contextlib.aclosing(
            self._run_from_state(state, prompt, thread_id, run_metadata, model_kwargs, _run=rc)
        ) as segment:
            async for event in segment:
                yield event

    async def _ledger_manifest(self, thread_id: str) -> LedgerManifest | None:
        """The turn the thread's tool-effect ledger belongs to, or ``None``."""
        from locus.memory.ledger import ToolEffectLedger  # noqa: PLC0415

        try:
            return await ToolEffectLedger(self.config.checkpointer).manifest(thread_id)
        except Exception:  # noqa: BLE001 — a ledger that cannot be read is no ledger
            logger.warning(
                "could not read the tool-effect ledger of thread %s", thread_id, exc_info=True
            )
            return None

    def _take_interrupt(self, thread_id: str | None) -> PendingInterrupt | None:
        """Pop the in-memory interrupt ``resume`` should continue, if any.

        With a thread id only that thread's entry is eligible — never another
        thread's, which is how a resume for thread A once executed thread B's
        held booking. Without one, the single paused run is taken; several
        paused runs make the call ambiguous and it is refused.
        """
        if thread_id is not None:
            return self._interrupts.pop(thread_id, None)
        if not self._interrupts:
            return None
        if len(self._interrupts) > 1:
            raise RuntimeError(
                f"{len(self._interrupts)} runs are paused on this agent "
                f"(threads {sorted(map(str, self._interrupts))}); pass thread_id "
                "to say which one to resume."
            )
        key = next(iter(self._interrupts))
        return self._interrupts.pop(key)

    async def _fold_resume_response(
        self,
        state: AgentState,
        response: str,
        rc: RunContext,
        perform_dangling: bool,
        pending: PendingInterrupt | None,
        out_events: list[LocusEvent],
    ) -> AgentState:
        """Answer the paused call with ``response`` (or its real result).

        Events produced here (the performed call's start/complete pair and any
        custom events its hooks emitted) are appended to ``out_events`` and
        yielded by ``resume`` only once the fold succeeded, so a still-pending
        approval raises before the consumer sees a half-performed call.
        """
        # Fold the user's response as the TOOL RESULT of the dangling
        # ask_user call. The pause leaves that call un-folded in state, and a
        # bare system note in its place breaks the call→result rhythm the
        # model is pattern-matching on — a model that needs to ask AGAIN then
        # tends to put its follow-up question in plain text, which ends the
        # run instead of pausing it. Answering the call keeps the exchange in
        # the tool protocol, so ask → answer → ask-again holds for N turns.
        answered = {
            m.tool_call_id for m in state.messages if m.role == Role.TOOL and m.tool_call_id
        }
        # ANY unanswered call, not just ``ask_user``. The rhythm argument above
        # is name-independent: an approval hold suspends on the governed call
        # itself (``refund_customer``), so restricting the search to ask_user
        # sent every approval resume down the system-note path — and produced
        # exactly the failure the comment predicts. The model returns an empty
        # turn, the loop reads that as "finished", and an APPROVED action is
        # silently never performed while the run reports success — the worst
        # available failure for a governance feature, because the transcript
        # shows an approval and an untroubled reply, and nothing anywhere
        # records that the action did not happen.
        #
        # ask_user still wins when both are present, so the established path is
        # bit-for-bit unchanged and only the previously-broken case moves.
        dangling = None
        for msg in reversed(state.messages):
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                unanswered = [tc for tc in msg.tool_calls if tc.id not in answered]
                dangling = next(
                    (tc for tc in unanswered if tc.name == "ask_user"),
                    unanswered[0] if unanswered else None,
                )
                break
        if dangling is not None:
            folded: ToolResult | None = None
            if perform_dangling and dangling.name != "ask_user":
                # Re-invoke the held call itself — same tool, same arguments,
                # through the normal executor so error handling and result
                # stringification are exactly the loop's. A gated wrapper
                # decides under the primed decision (approve executes, deny
                # refuses); the fold then carries the REAL outcome, which is
                # the only thing that makes an approval actually happen. If
                # the tool is no longer registered, fall back to the text
                # fold rather than inventing a result.
                self._initialize()
                if self._tool_registry.get(dangling.name) is not None:
                    from locus.agent.runtime_loop import _apply_hook_result, _complete_event
                    from locus.core.messages import ToolCall as _ToolCall
                    from locus.tools.executor import ToolContextFactory

                    arguments = dict(dangling.arguments or {})
                    if rc.integrity is not None:
                        # Its provenance in THIS run is the checkpoint, not a
                        # model response; signed so the hook seam below is
                        # held to the same chain of custody as a live call.
                        rc.integrity.sign(
                            _ToolCall(id=dangling.id, name=dangling.name, arguments=arguments),
                            by="checkpoint",
                        )
                    # The same start event the loop emits, so the performed
                    # call arrives as a start/complete pair — and the start
                    # carries the ARGUMENTS, which ToolCompleteEvent does not,
                    # so a consumer compensating on this path can finally see
                    # what the approved call ran with.
                    out_events.append(
                        ToolStartEvent(
                            tool_name=dangling.name,
                            tool_call_id=dangling.id,
                            arguments=arguments,
                        )
                    )
                    # The SAME hook seam the loop runs around every tool call
                    # (#172). This is the single most consequential call in a
                    # governed run — the one a human approved — and executing
                    # it through the bare executor made it invisible to every
                    # hook: playbook trackers recorded the step as skipped,
                    # audit sinks riding the hook seam never saw it. A
                    # before-hook that cancels is a legitimate second veto and
                    # is honoured here exactly as the loop honours it.
                    before_event = await self._orch().run_before_tool(
                        dangling.name, dangling.id, arguments, run=rc
                    )
                    out_events.extend(rc.drain())
                    if before_event.cancel:
                        cancel_msg = (
                            before_event.cancel
                            if isinstance(before_event.cancel, str)
                            else "Cancelled by hook"
                        )
                        folded = ToolResult(
                            tool_call_id=dangling.id,
                            name=dangling.name,
                            content=cancel_msg,
                            error=None,
                            duration_ms=0.0,
                        )
                    else:
                        arguments = before_event.arguments
                        [invoked] = await self._executor.execute(
                            [
                                _ToolCall(
                                    id=dangling.id,
                                    name=dangling.name,
                                    arguments=_invocation_arguments(before_event),
                                )
                            ],
                            self._tool_registry,
                            ToolContextFactory(
                                run_id=state.run_id,
                                agent_id=state.agent_id,
                                iteration=state.iteration,
                                state=state,
                                invocation_metadata=rc.metadata or {},
                                ephemeral_metadata=rc.ephemeral,
                                integrity=rc.integrity,
                            ),
                        )
                        still_held = _interrupt_payload(invoked.content)
                        if still_held is not None:
                            # The gate interrupted AGAIN: nobody has decided
                            # yet. Folding the raw ``__interrupt__`` marker as
                            # the call's result (the pre-2.17 behaviour) told
                            # the model the action had "returned" and ended
                            # the turn with the booking silently never made.
                            # Leave the thread exactly as paused instead.
                            self._interrupts[rc.thread_id] = pending or PendingInterrupt(
                                state=state,
                                prompt=rc.prompt,
                                thread_id=rc.thread_id,
                                metadata=rc.metadata,
                                ephemeral=rc.ephemeral,
                            )
                            raise ApprovalPendingError(
                                f"{dangling.name!r} on thread {rc.thread_id!r} is still "
                                "awaiting a decision; record it before resuming.",
                                thread_id=rc.thread_id,
                                interrupt_id=dangling.id,
                                question=str(still_held.get("question", "")),
                                metadata=still_held.get("metadata") or {},
                            )
                        # Keep the structured result: the approved call is the
                        # one a UI most needs to render (the confirmed booking).
                        folded = invoked.model_copy(
                            update={"tool_call_id": dangling.id, "name": dangling.name}
                        )
                        after_event = await self._orch().run_after_tool(
                            folded.name,
                            folded.content if folded.success else None,
                            folded.error,
                            tool_call_id=folded.tool_call_id,
                            arguments=arguments,
                            run=rc,
                        )
                        folded = _apply_hook_result(folded, after_event)
                    # This execution happens BEFORE the loop starts streaming,
                    # so without this the call is invisible: a consumer sees
                    # the run resume and finish with no record of the action
                    # it performed. Emit the same event the loop would have,
                    # so a trace, an audit sink, and a UI all see the approved
                    # action exactly as they see any other tool call.
                    out_events.append(
                        _complete_event(
                            folded,
                            result=folded.content,
                            duration_ms=folded.duration_ms or 0.0,
                        )
                    )
                    out_events.extend(rc.drain())
            if folded is None:
                folded = ToolResult(tool_call_id=dangling.id, name=dangling.name, content=response)
            # The call is answered: the thread no longer waits on its hold.
            from locus.control.holds import without_hold  # noqa: PLC0415

            state = without_hold(state, dangling.id)
            state = state.with_message(Message.tool(folded))
        else:
            state = state.with_message(Message.system(f"[User Response] {response}"))

        return state

    # =========================================================================
    # Checkpoint history: read, edit and fork a thread
    # =========================================================================

    def _require_checkpointer(self) -> Any:
        checkpointer = self.config.checkpointer
        if checkpointer is None:
            raise RuntimeError("checkpoint history needs a checkpointer on the agent")
        return checkpointer

    async def get_state(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState:
        """The thread's state at ``checkpoint_id``, or its latest.

        Raises:
            RuntimeError: The agent has no checkpointer.
            LookupError: No such thread or checkpoint.
        """
        state = await self._require_checkpointer().load(thread_id, checkpoint_id)
        if state is None:
            where = f"checkpoint {checkpoint_id!r} of " if checkpoint_id else ""
            raise LookupError(f"no {where}thread {thread_id!r}")
        loaded: AgentState = state
        return loaded

    async def get_state_history(
        self, thread_id: str, limit: int = 10
    ) -> list[tuple[str, AgentState]]:
        """``(checkpoint_id, state)`` pairs for the thread, newest first."""
        checkpointer = self._require_checkpointer()
        history: list[tuple[str, AgentState]] = []
        for checkpoint_id in await checkpointer.list_checkpoints(thread_id, limit=limit):
            state = await checkpointer.load(thread_id, checkpoint_id)
            if state is not None:
                history.append((checkpoint_id, state))
        return history

    async def update_state(
        self,
        thread_id: str,
        *,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        """Write a new checkpoint: the given one (or the latest) plus these changes.

        The source checkpoint is never modified, so history stays intact and
        the edit can itself be inspected or forked. The next ``run`` or
        ``resume`` on the thread continues from the new checkpoint.

        Returns:
            The new checkpoint's id.
        """
        state = await self.get_state(thread_id, checkpoint_id)
        for message in messages or []:
            state = state.with_message(message)
        for key, value in (metadata or {}).items():
            state = state.with_metadata(key, value)
        new_id: str = await self._require_checkpointer().save(
            state,
            thread_id,
            metadata={"source": "update_state", "parent_checkpoint_id": checkpoint_id},
        )
        return new_id

    async def fork(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
        *,
        new_thread_id: str | None = None,
    ) -> str:
        """Start a new thread from the given checkpoint (or the latest).

        The new thread carries the conversation up to that point and nothing
        after it; running on it never changes the parent thread. This is
        :meth:`BaseCheckpointer.fork <locus.memory.BaseCheckpointer.fork>`:
        the thread's tool-effect ledger is forked with it, up to that
        checkpoint's iteration, so the fork replays nothing the parent did
        past the fork point. ``checkpointer.list(thread_id)`` names the
        checkpoints to fork at.

        Returns:
            The new thread's id.

        Raises:
            LookupError: No such thread or checkpoint.
            ValueError: ``new_thread_id`` is the parent thread.
        """
        checkpointer = self._require_checkpointer()
        if hasattr(checkpointer, "fork"):
            forked: str = await checkpointer.fork(
                thread_id, checkpoint_id, new_thread_id=new_thread_id
            )
            return forked
        # A duck-typed checkpointer without the time-travel surface: copy the
        # state by hand, as before the surface existed.
        from locus.memory.time_travel import fork_metadata, new_fork_id  # noqa: PLC0415

        state = await self.get_state(thread_id, checkpoint_id)
        forked = new_thread_id or new_fork_id(thread_id)
        if forked == thread_id:
            raise ValueError("a fork needs a thread id different from its parent")
        await checkpointer.save(state, forked, metadata=fork_metadata(thread_id, checkpoint_id))
        return forked

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

        Locus compiles ``config.tools`` into the runtime ``ToolRegistry``
        once, inside ``__init__`` (via :func:`locus.agent.initializer.
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
            TypeError: if ``tool`` is not a :class:`locus.tools.Tool`
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
