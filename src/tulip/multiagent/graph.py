# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Graph-based workflow execution for multi-agent systems.

This module provides a powerful graph execution engine supporting:
- DAG and cyclic graph execution
- Conditional edges with dynamic routing
- Parallel node execution
- Human-in-the-loop interrupts
- Map-reduce patterns via Send
- Subgraph composition
- State reducers for composable updates

Example - Basic DAG:
    from tulip.multiagent.graph import StateGraph, START, END

    graph = StateGraph()
    graph.add_node("process", process_fn)
    graph.add_node("validate", validate_fn)
    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", END)

    result = await graph.execute({"data": input_data})

Example - Conditional routing:
    def route_by_type(state):
        if state["type"] == "error":
            return "error_handler"
        return "normal_flow"

    graph.add_conditional_edges("classifier", route_by_type, {
        "error_handler": "handle_error",
        "normal_flow": "process"
    })

Example - Human-in-the-loop:
    from tulip.core.interrupt import interrupt

    async def review_node(inputs):
        approval = interrupt({"action": "delete", "id": inputs["id"]})
        if approval == "approved":
            return {"status": "deleted"}
        return {"status": "cancelled"}
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

from tulip.core.command import Command, is_command, normalize_node_output
from tulip.core.interrupt import (
    InterruptException,
    InterruptState,
    NodeExecutionContext,
)
from tulip.core.reducers import (
    Reducer,
    apply_reducers,
)
from tulip.core.send import Send, SendResult, is_send_list, normalize_sends


# =============================================================================
# Special Node Constants
# =============================================================================

START = "__START__"
END = "__END__"


# =============================================================================
# Enums and Status
# =============================================================================


class NodeStatus(StrEnum):
    """Status of a node in the graph."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class StreamMode(StrEnum):
    """Streaming output modes."""

    VALUES = "values"  # Full state after each step
    UPDATES = "updates"  # State deltas only
    NODES = "nodes"  # Node execution events
    CUSTOM = "custom"  # User-emitted data
    DEBUG = "debug"  # Maximum detail


# =============================================================================
# Result Models
# =============================================================================


class NodeResult(BaseModel):
    """Result from executing a node."""

    node_id: str
    status: NodeStatus
    output: Any = None
    error: str | None = None
    duration_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    command: Command | None = None  # If node returned a Command
    sends: list[Send] | None = None  # If node returned Sends

    model_config = {"arbitrary_types_allowed": True}

    @property
    def success(self) -> bool:
        """Whether the node executed successfully."""
        return self.status == NodeStatus.COMPLETED


class GraphResult(BaseModel):
    """Result from executing a graph."""

    graph_id: str
    success: bool
    node_results: dict[str, NodeResult] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    final_outputs: dict[str, Any] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)
    duration_ms: float | None = None
    interrupt: InterruptState | None = None  # If interrupted
    iterations: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_interrupted(self) -> bool:
        """Whether execution was interrupted for human input."""
        return self.interrupt is not None


class StreamEvent(BaseModel):
    """Event emitted during streaming execution."""

    mode: StreamMode
    node_id: str | None = None
    data: Any = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Node
# =============================================================================


class RetryPolicy(BaseModel):
    """Retry policy for node execution.

    Exponential backoff with optional jitter, matching LangGraph's pattern.

    Example:
        node = Node(
            name="api_call",
            executor=call_api,
            retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0),
        )
    """

    max_attempts: int = Field(default=3, ge=1)
    initial_interval: float = Field(default=1.0, ge=0, description="Seconds")
    backoff_factor: float = Field(default=2.0, ge=1.0)
    max_interval: float = Field(default=60.0, ge=0, description="Seconds")
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        import random

        delay = min(
            self.initial_interval * (self.backoff_factor**attempt),
            self.max_interval,
        )
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5  # noqa: S311
        return delay


class CachePolicy(BaseModel):
    """Cache policy for node execution results.

    Caches node outputs to avoid re-computation for identical inputs.

    Example:
        node = Node(
            name="expensive_lookup",
            executor=lookup,
            cache_policy=CachePolicy(ttl_seconds=300),
        )
    """

    ttl_seconds: float = Field(default=300.0, ge=0, description="Cache TTL in seconds")
    enabled: bool = True


# Simple in-memory cache for node results
_node_cache: dict[str, tuple[Any, float]] = {}


class Node(BaseModel):
    """
    A node in the execution graph.

    Wraps an agent or callable that processes inputs and produces outputs.
    Supports retry policies with exponential backoff and result caching.
    """

    id: str = Field(default_factory=lambda: f"node_{uuid4().hex[:8]}")
    name: str
    description: str = ""

    # The agent or callable to execute
    # Signature: async (inputs: dict[str, Any]) -> Any
    executor: Callable[..., Any]

    # Optional condition for execution
    # If provided, node only runs if condition returns True
    condition: Callable[[dict[str, Any]], bool] | None = None

    # Retry configuration (legacy — use retry_policy for full control)
    max_retries: int = 0
    retry_delay_ms: float = 1000

    # Enhanced retry with exponential backoff
    retry_policy: RetryPolicy | None = None

    # Cache policy for result caching
    cache_policy: CachePolicy | None = None

    # Deferred execution — runs only at graph exit
    defer: bool = False

    # Timeout in milliseconds (None = no timeout)
    timeout_ms: float | None = None

    # Whether this is a subgraph node
    is_subgraph: bool = False
    subgraph: Any | None = None  # Graph instance if is_subgraph

    model_config = {"arbitrary_types_allowed": True}

    async def execute(
        self,
        inputs: dict[str, Any],
        *,
        resume_value: Any = None,
        is_resuming: bool = False,
    ) -> NodeResult:
        """
        Execute the node with given inputs.

        Args:
            inputs: Dictionary of inputs from upstream nodes
            resume_value: Value to pass if resuming from interrupt
            is_resuming: Whether we're resuming from an interrupt

        Returns:
            NodeResult with output or error
        """
        started_at = datetime.now(UTC)
        attempts = 0

        # Determine retry limit from policy or legacy config
        max_attempts = self.retry_policy.max_attempts if self.retry_policy else self.max_retries + 1

        # Check cache
        if self.cache_policy and self.cache_policy.enabled:
            import hashlib
            import json as _json
            import time as _time

            cache_key = hashlib.sha256(  # noqa: S324
                f"{self.id}:{_json.dumps(inputs, sort_keys=True, default=str)}".encode()
            ).hexdigest()
            cached = _node_cache.get(cache_key)
            if cached is not None:
                cached_output, cached_time = cached
                if _time.time() - cached_time < self.cache_policy.ttl_seconds:
                    return NodeResult(
                        node_id=self.id,
                        status=NodeStatus.COMPLETED,
                        output=cached_output,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                        duration_ms=0.0,
                    )

        while attempts < max_attempts:
            try:
                # Check condition
                if self.condition is not None and not self.condition(inputs):
                    return NodeResult(
                        node_id=self.id,
                        status=NodeStatus.SKIPPED,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    )

                # Set up execution context for interrupt handling
                async with NodeExecutionContext(
                    node_id=self.id,
                    resume_value=resume_value,
                    is_resuming=is_resuming,
                ):
                    # Handle subgraph execution
                    if self.is_subgraph and self.subgraph is not None:
                        subgraph_result = await self.subgraph.execute(inputs)
                        # Use final_state which has all merged outputs from the subgraph
                        output = subgraph_result.final_state
                    else:
                        # Execute with optional timeout.
                        # Check the executor itself AND its __call__ method so that
                        # callable class instances with async __call__ are awaited
                        # correctly (asyncio.iscoroutinefunction returns False for
                        # instances but True for bound methods and async functions).
                        _exec_call = getattr(type(self.executor), "__call__", None)  # noqa: B004
                        _is_async = asyncio.iscoroutinefunction(self.executor) or (
                            _exec_call is not None and asyncio.iscoroutinefunction(_exec_call)
                        )
                        if _is_async:
                            coro = self.executor(inputs)
                        else:
                            # Wrap sync function
                            loop = asyncio.get_running_loop()
                            coro = loop.run_in_executor(None, lambda: self.executor(inputs))

                        if self.timeout_ms:
                            output = await asyncio.wait_for(
                                coro,
                                timeout=self.timeout_ms / 1000,
                            )
                        else:
                            output = await coro

                completed_at = datetime.now(UTC)
                duration_ms = (completed_at - started_at).total_seconds() * 1000

                # Parse output for Commands and Sends
                command = None
                sends = None

                if is_command(output):
                    command = output
                elif is_send_list(output) or isinstance(output, Send):
                    sends = normalize_sends(output)

                # Store in cache if policy set
                if self.cache_policy and self.cache_policy.enabled:
                    import time as _time

                    _node_cache[cache_key] = (output, _time.time())

                return NodeResult(
                    node_id=self.id,
                    status=NodeStatus.COMPLETED,
                    output=output,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=completed_at,
                    command=command,
                    sends=sends,
                )

            except InterruptException as e:
                # Node is requesting human input
                completed_at = datetime.now(UTC)
                duration_ms = (completed_at - started_at).total_seconds() * 1000
                return NodeResult(
                    node_id=self.id,
                    status=NodeStatus.INTERRUPTED,
                    output=e.value,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            except TimeoutError:
                completed_at = datetime.now(UTC)
                return NodeResult(
                    node_id=self.id,
                    status=NodeStatus.FAILED,
                    error=f"Node execution timed out after {self.timeout_ms}ms",
                    duration_ms=self.timeout_ms,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            except Exception as e:  # noqa: BLE001
                attempts += 1
                if attempts >= max_attempts:
                    completed_at = datetime.now(UTC)
                    duration_ms = (completed_at - started_at).total_seconds() * 1000
                    return NodeResult(
                        node_id=self.id,
                        status=NodeStatus.FAILED,
                        error=str(e),
                        duration_ms=duration_ms,
                        started_at=started_at,
                        completed_at=completed_at,
                    )

                # Calculate delay from policy or legacy config
                if self.retry_policy:
                    delay = self.retry_policy.get_delay(attempts - 1)
                else:
                    delay = self.retry_delay_ms / 1000
                await asyncio.sleep(delay)

        # Should not reach here
        return NodeResult(
            node_id=self.id,
            status=NodeStatus.FAILED,
            error="Unexpected execution path",
        )


# =============================================================================
# Edge Types
# =============================================================================


class Edge(BaseModel):
    """
    A directed edge connecting two nodes in the graph.

    Represents data flow from source to target.
    """

    source_id: str
    target_id: str

    # LangGraph-compatible aliases
    @property
    def source(self) -> str:
        return self.source_id

    @property
    def target(self) -> str:
        return self.target_id

    # Optional key mapping: source_output_key -> target_input_key
    key_mapping: dict[str, str] | None = None

    # Optional transformer to modify data as it flows
    transform: Callable[[Any], Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def apply(self, source_output: Any) -> dict[str, Any]:
        """Transform source output to target input."""
        # Apply transformation if provided
        if self.transform is not None:
            source_output = self.transform(source_output)

        # Apply key mapping
        if self.key_mapping is not None:
            if isinstance(source_output, dict):
                return {
                    target_key: source_output.get(source_key)
                    for source_key, target_key in self.key_mapping.items()
                }
            first_target = next(iter(self.key_mapping.values()), self.source_id)
            return {first_target: source_output}

        # Default: pass entire output under source node id
        return {self.source_id: source_output}


class ConditionalEdge(BaseModel):
    """
    A conditional edge with dynamic target selection.

    The router function determines which target to route to based on state.
    """

    source_id: str
    router: Callable[[dict[str, Any]], str | list[str]]
    targets: dict[str, str] = Field(default_factory=dict)  # router_result -> node_id
    default_target: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    def resolve_target(self, state: dict[str, Any]) -> list[str]:
        """Resolve the target node(s) based on state."""
        result = self.router(state)

        if isinstance(result, list):
            # Multiple targets (parallel execution)
            targets = []
            for r in result:
                if r in self.targets:
                    targets.append(self.targets[r])
                elif self.default_target:
                    targets.append(self.default_target)
                else:
                    targets.append(r)
            return targets

        # Single target
        if result in self.targets:
            target = self.targets[result]
        elif self.default_target:
            target = self.default_target
        else:
            target = result
        return [target] if target else []


# =============================================================================
# Graph Configuration
# =============================================================================


class GraphConfig(BaseModel):
    """Configuration for graph execution."""

    # Execution settings
    parallel: bool = True  # Run independent nodes in parallel
    allow_cycles: bool = False  # Allow cyclic graphs
    max_iterations: int = 100  # Max iterations for cyclic graphs

    # Interrupt settings
    interrupt_before: list[str] = Field(default_factory=list)  # Pause before these nodes
    interrupt_after: list[str] = Field(default_factory=list)  # Pause after these nodes

    # Checkpointing
    checkpointer: Any | None = None  # BaseCheckpointer
    thread_id: str | None = None

    # Store for cross-thread memory
    store: Any | None = None  # BaseStore

    # Streaming
    stream_mode: StreamMode = StreamMode.VALUES

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Streaming sink helper
# =============================================================================


# Context-local stream sink. ``StateGraph.stream()`` sets this for the
# duration of a streaming run so node bodies can emit custom events via
# :func:`emit_custom`. ``None`` outside a streaming context — emitting
# then is a silent no-op so node code can be written once and run under
# either ``stream()`` or ``execute()``.
_active_stream_sink: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_tulip_active_stream_sink", default=None
)


async def emit_custom(data: Any, *, node_id: str | None = None) -> None:
    """Emit a ``StreamEvent(mode=CUSTOM)`` from inside a graph node.

    Use this when your node wants to surface progress / partial output that
    isn't a state update. The event reaches consumers of
    ``StateGraph.stream()`` immediately; outside a streaming context it's
    a silent no-op so the same node code works under ``execute()`` too.

    Example::

        async def long_running_node(state):
            for i in range(10):
                await emit_custom({"progress": i / 10})
                await asyncio.sleep(0.1)
            return {"done": True}
    """
    sink = _active_stream_sink.get()
    if sink is None:
        return
    await sink(StreamEvent(mode=StreamMode.CUSTOM, node_id=node_id, data=data))


async def _emit_node_events(
    sink: Any,
    node_id: str,
    result: NodeResult,
    state: dict[str, Any],
    mode: StreamMode,
) -> None:
    """Push node-completion events through the streaming sink.

    Called from inside ``StateGraph.execute`` after each node completes.
    A no-op when ``sink`` is None (the default for non-streaming
    ``execute`` callers). When set, builds the appropriate ``StreamEvent``
    for the active mode and forwards it to the sink.
    """
    if sink is None:
        return
    if mode == StreamMode.VALUES:
        # Snapshot of full state after this node completed.
        await sink(StreamEvent(mode=mode, node_id=node_id, data=dict(state)))
    elif mode == StreamMode.UPDATES:
        await sink(StreamEvent(mode=mode, node_id=node_id, data=result.output))
    elif mode == StreamMode.NODES:
        await sink(StreamEvent(mode=mode, node_id=node_id, data=result))
    elif mode == StreamMode.DEBUG:
        await sink(
            StreamEvent(
                mode=mode,
                node_id=node_id,
                data={
                    "result": result.model_dump(mode="json"),
                    "state": dict(state),
                },
            )
        )
    # CUSTOM is reserved for user-emitted data; nothing automatic to forward.


# =============================================================================
# State Graph
# =============================================================================


class StateGraph(BaseModel):
    """
    A stateful graph for workflow execution.

    Supports:
    - Conditional edges with dynamic routing
    - Cycles (optional, with max iteration limit)
    - Human-in-the-loop interrupts
    - Map-reduce via Send
    - Subgraph composition
    - State reducers for composable updates
    """

    id: str = Field(default_factory=lambda: f"graph_{uuid4().hex[:8]}")
    name: str = ""
    description: str = ""

    # Graph structure
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    conditional_edges: list[ConditionalEdge] = Field(default_factory=list)

    # State schema and reducers
    state_schema: type[BaseModel] | None = None
    _reducers: dict[str, Reducer[Any]] = PrivateAttr(default_factory=dict)

    # Configuration
    config: GraphConfig = Field(default_factory=GraphConfig)

    # Internal state
    _adjacency: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _reverse_adjacency: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _entry_point: str | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize after model creation."""
        # Add virtual START and END nodes
        if START not in self.nodes:
            self.nodes[START] = Node(
                id=START,
                name="START",
                executor=lambda x: x,  # Pass-through
            )
        if END not in self.nodes:
            self.nodes[END] = Node(
                id=END,
                name="END",
                executor=lambda x: x,  # Pass-through
            )

        # Extract reducers from state schema
        if self.state_schema:
            from tulip.core.reducers import extract_reducers_from_model

            self._reducers = extract_reducers_from_model(self.state_schema)

        self._rebuild_adjacency()

    def set_entry_point(self, node_id: str) -> StateGraph:
        """Set the entry point node (after START)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        self._entry_point = node_id
        # Add edge from START to entry point
        self.add_edge(START, node_id)
        return self

    def set_finish_point(self, node_id: str) -> StateGraph:
        """Set a finish point node (before END)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        self.add_edge(node_id, END)
        return self

    def add_node(
        self,
        node_id: str | Node,
        executor: Callable[..., Any] | StateGraph | None = None,
        *,
        description: str = "",
        condition: Callable[[dict[str, Any]], bool] | None = None,
        max_retries: int = 0,
        timeout_ms: float | None = None,
        retry_policy: RetryPolicy | None = None,
        cache_policy: CachePolicy | None = None,
        defer: bool = False,
    ) -> StateGraph:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node, or a Node object (for backward compatibility)
            executor: Function or subgraph to execute (optional if node_id is a Node)
            description: Node description
            condition: Optional condition for execution
            max_retries: Retry attempts on failure
            timeout_ms: Execution timeout

        Returns:
            Self for chaining
        """
        # Support old API: add_node(Node)
        if isinstance(node_id, Node):
            node = node_id
            if node.id in self.nodes:
                raise ValueError(f"Node already exists: {node.id}")
            self.nodes[node.id] = node
            self._rebuild_adjacency()
            return self

        # New API: add_node(node_id, executor)
        if executor is None:
            raise TypeError("add_node() missing 1 required positional argument: 'executor'")

        if node_id in self.nodes:
            raise ValueError(f"Node already exists: {node_id}")

        # Check if executor is a subgraph
        is_subgraph = isinstance(executor, StateGraph)

        # When ``is_subgraph`` is True, ``executor`` is a StateGraph
        # instance; mypy can't narrow the union without an isinstance
        # check, but the precondition is enforced upstream.
        node_executor: Callable[..., Any] = (
            executor.execute  # type: ignore[union-attr]
            if is_subgraph
            else executor  # type: ignore[assignment]
        )
        node = Node(
            id=node_id,
            name=node_id,
            description=description,
            executor=node_executor,
            condition=condition,
            max_retries=max_retries,
            timeout_ms=timeout_ms,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            defer=defer,
            is_subgraph=is_subgraph,
            subgraph=executor if is_subgraph else None,
        )
        self.nodes[node_id] = node
        self._rebuild_adjacency()
        return self

    def add_edge(
        self,
        source: str | Node,
        target: str | Node,
        key_mapping: dict[str, str] | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> StateGraph:
        """
        Add a directed edge between nodes.

        Args:
            source: Source node ID or Node object
            target: Target node ID or Node object
            key_mapping: Optional key mapping for data transformation
            transform: Optional transform function

        Returns:
            Self for chaining
        """
        # Support old API: add_edge(Node, Node)
        source_id = source.id if isinstance(source, Node) else source
        target_id = target.id if isinstance(target, Node) else target

        # Allow START and END as valid nodes
        valid_sources = set(self.nodes.keys()) | {START}
        valid_targets = set(self.nodes.keys()) | {END}

        if source_id not in valid_sources:
            raise ValueError(f"Source node not found: {source_id}")
        if target_id not in valid_targets:
            raise ValueError(f"Target node not found: {target_id}")

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            key_mapping=key_mapping,
            transform=transform,
        )
        self.edges.append(edge)
        self._rebuild_adjacency()

        # Validate no cycles (unless allowed)
        if not self.config.allow_cycles and self._has_cycle():
            self.edges.pop()
            self._rebuild_adjacency()
            raise ValueError(f"Adding edge {source_id} -> {target_id} would create a cycle")

        return self

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[dict[str, Any]], str | list[str]],
        targets: dict[str, str] | None = None,
        default: str | None = None,
    ) -> StateGraph:
        """
        Add conditional edges with dynamic routing.

        Args:
            source: Source node ID
            router: Function that returns target node ID(s) based on state
            targets: Optional mapping from router return values to node IDs
            default: Default target if router returns unmapped value

        Returns:
            Self for chaining

        Example:
            def route_by_type(state):
                return "error" if state["has_error"] else "success"

            graph.add_conditional_edges("check", route_by_type, {
                "error": "handle_error",
                "success": "continue"
            })
        """
        if source not in self.nodes:
            raise ValueError(f"Source node not found: {source}")

        cond_edge = ConditionalEdge(
            source_id=source,
            router=router,
            targets=targets or {},
            default_target=default,
        )
        self.conditional_edges.append(cond_edge)
        return self

    def _rebuild_adjacency(self) -> None:
        """Rebuild adjacency lists from edges."""
        self._adjacency = defaultdict(list)
        self._reverse_adjacency = defaultdict(list)

        for edge in self.edges:
            self._adjacency[edge.source_id].append(edge.target_id)
            self._reverse_adjacency[edge.target_id].append(edge.source_id)

    def _has_cycle(self) -> bool:
        """Check if the graph has a cycle using DFS."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def _get_next_nodes(
        self,
        current_node: str,
        state: dict[str, Any],
        command: Command | None = None,
    ) -> list[str]:
        """
        Determine next nodes to execute.

        Considers:
        - Command.goto if present
        - Conditional edges
        - Regular edges
        """
        # Command takes precedence
        if command and command.has_goto:
            return command.goto_nodes

        # Check conditional edges
        for cond_edge in self.conditional_edges:
            if cond_edge.source_id == current_node:
                return cond_edge.resolve_target(state)

        # Fall back to regular edges
        return self._adjacency.get(current_node, [])

    def _apply_state_update(
        self,
        current_state: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply state update using reducers."""
        return apply_reducers(current_state, update, self._reducers)

    def _gather_inputs(
        self,
        node_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Gather inputs for a node from state and edges."""
        inputs = dict(state)  # Start with full state

        # Apply edge transformations
        for edge in self.edges:
            if edge.target_id == node_id:
                source_data = state.get(edge.source_id)
                if source_data is not None:
                    edge_inputs = edge.apply(source_data)
                    inputs.update(edge_inputs)

        return inputs

    async def execute(
        self,
        inputs: dict[str, Any] | Command | None = None,
        *,
        config: GraphConfig | None = None,
        _event_sink: Any = None,
    ) -> GraphResult:
        """
        Execute the graph.

        Args:
            inputs: Initial state or Command (for resume)
            config: Optional execution configuration
            _event_sink: Internal-use only. When provided, an async callable
                ``(StreamEvent) -> None`` invoked after each node completes
                so :meth:`stream` can yield intermediate events. Public
                callers should use :meth:`stream` instead of touching this.

        Returns:
            GraphResult with final state and outputs
        """
        start_time = datetime.now(UTC)
        cfg = config or self.config

        # Handle resume from interrupt
        resume_value = None
        resume_node = None
        if isinstance(inputs, Command) and inputs.has_resume:
            resume_value = inputs.resume
            # Load state from checkpointer if available
            if cfg.checkpointer and cfg.thread_id:
                saved_state = await cfg.checkpointer.load(cfg.thread_id)
                if saved_state:
                    inputs = saved_state.metadata.get("graph_state", {})
                    resume_node = saved_state.metadata.get("interrupted_node")
                    from tulip.observability.emit import (  # noqa: PLC0415
                        EV_CHECKPOINT_LOADED,
                        emit,
                    )

                    await emit(
                        EV_CHECKPOINT_LOADED,
                        thread_id=cfg.thread_id,
                        backend=type(cfg.checkpointer).__name__,
                        resume_node=resume_node,
                    )
            else:
                # Without checkpointer, get resume node from state
                state_data = inputs.update or {}
                resume_node = state_data.pop("__resume_node__", None)
                inputs = state_data
        elif isinstance(inputs, Command):
            inputs = inputs.update

        # Initialize state
        state: dict[str, Any] = dict(inputs or {})
        node_results: dict[str, NodeResult] = {}
        execution_order: list[str] = []
        iterations = 0

        # Determine starting node(s)
        if resume_node:
            current_nodes = [resume_node]
        else:
            current_nodes = self._adjacency.get(START, [])
            if not current_nodes and self._entry_point:
                current_nodes = [self._entry_point]

        # Main execution loop
        while current_nodes and iterations < cfg.max_iterations:
            iterations += 1
            next_nodes: list[str] = []

            # When resuming from an ``interrupt_before`` pause, the very
            # first iteration of the new ``execute()`` call re-enters the
            # gate node we paused on. Without this guard, the gate would
            # fire immediately and we'd pause again — making "approve once
            # and continue" semantics impossible. Skip the gate check for
            # the resume_node on the first iteration only; ``resume_node``
            # is cleared at the bottom of the iteration so subsequent
            # iterations gate normally.
            just_resumed_node = resume_node if iterations == 1 else None

            # Check for interrupt_before
            for node_id in current_nodes:
                if node_id == just_resumed_node:
                    continue
                if node_id in cfg.interrupt_before:
                    # Create a placeholder interrupt value for interrupt_before
                    from tulip.core.interrupt import InterruptValue

                    placeholder_interrupt = InterruptValue(
                        payload={"type": "interrupt_before", "node": node_id},
                        node_id=node_id,
                        graph_id=self.id,
                    )
                    interrupt_state = InterruptState(
                        interrupt=placeholder_interrupt,
                        node_id=node_id,
                        pending_nodes=current_nodes,
                        state_snapshot=state,
                    )

                    # Save to checkpointer if available so a follow-up
                    # ``execute(Command(resume=...))`` can recover the
                    # paused state. Without this, durable cross-process
                    # resume doesn't work for the ``interrupt_before``
                    # gate — the reason most users wire a checkpointer
                    # to a ``StateGraph`` in the first place.
                    #
                    # We pack the graph-level fields into
                    # ``AgentState.metadata`` because the resume side
                    # (``execute()`` above, line ~999) reads them off
                    # ``saved_state.metadata`` — MemoryCheckpointer and
                    # the other in-protocol backends round-trip the
                    # ``AgentState`` itself, not the ``metadata=`` kwarg.
                    if cfg.checkpointer and cfg.thread_id:
                        from tulip.core.state import AgentState  # noqa: PLC0415

                        stub_state = AgentState(
                            metadata={
                                "graph_state": state,
                                "interrupted_node": node_id,
                                "interrupt": interrupt_state.model_dump(),
                            }
                        )
                        await cfg.checkpointer.save(
                            state=stub_state,
                            thread_id=cfg.thread_id,
                        )
                        from tulip.observability.emit import (  # noqa: PLC0415
                            EV_CHECKPOINT_SAVED,
                            emit,
                        )

                        await emit(
                            EV_CHECKPOINT_SAVED,
                            thread_id=cfg.thread_id,
                            backend=type(cfg.checkpointer).__name__,
                            trigger="graph_interrupt_before",
                            interrupted_node=node_id,
                        )

                    # Include resume node in final state
                    final_state_with_resume = {**state, "__resume_node__": node_id}
                    return GraphResult(
                        graph_id=self.id,
                        success=False,
                        node_results=node_results,
                        final_state=final_state_with_resume,
                        execution_order=execution_order,
                        duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                        interrupt=interrupt_state,
                        iterations=iterations,
                    )

            # Lazy import — observability is opt-in.
            from tulip.observability.emit import (  # noqa: PLC0415
                EV_GRAPH_NODE_COMPLETED,
                EV_GRAPH_NODE_STARTED,
                emit,
            )

            # Execute current nodes (parallel if enabled)
            if cfg.parallel and len(current_nodes) > 1:
                tasks = []
                node_span_ids: dict[str, str] = {}
                for node_id in current_nodes:
                    if node_id == END:
                        continue
                    node = self.nodes[node_id]
                    node_inputs = self._gather_inputs(node_id, state)
                    is_resume = node_id == resume_node
                    span_id = uuid4().hex[:8]
                    node_span_ids[node_id] = span_id
                    await emit(
                        EV_GRAPH_NODE_STARTED,
                        graph_id=self.id,
                        node_id=node_id,
                        iteration=iterations,
                        span_id=span_id,
                        parallel=True,
                        is_resuming=is_resume,
                    )
                    tasks.append(
                        node.execute(
                            node_inputs,
                            resume_value=resume_value if is_resume else None,
                            is_resuming=is_resume,
                        )
                    )

                if tasks:
                    parallel_started = time.perf_counter()
                    results = await asyncio.gather(*tasks)
                    for node_id, result in zip(
                        [n for n in current_nodes if n != END],
                        results,
                        strict=True,
                    ):
                        node_results[node_id] = result
                        execution_order.append(node_id)
                        await emit(
                            EV_GRAPH_NODE_COMPLETED,
                            graph_id=self.id,
                            node_id=node_id,
                            span_id=node_span_ids.get(node_id),
                            status=str(result.status),
                            duration_ms=(time.perf_counter() - parallel_started) * 1000,
                            parallel=True,
                        )
                        await _emit_node_events(
                            _event_sink, node_id, result, state, cfg.stream_mode
                        )
            else:
                # Sequential execution
                for node_id in current_nodes:
                    if node_id == END:
                        continue

                    node = self.nodes[node_id]
                    node_inputs = self._gather_inputs(node_id, state)
                    is_resume = node_id == resume_node
                    span_id = uuid4().hex[:8]
                    started_at = time.perf_counter()
                    await emit(
                        EV_GRAPH_NODE_STARTED,
                        graph_id=self.id,
                        node_id=node_id,
                        iteration=iterations,
                        span_id=span_id,
                        parallel=False,
                        is_resuming=is_resume,
                    )
                    result = await node.execute(
                        node_inputs,
                        resume_value=resume_value if is_resume else None,
                        is_resuming=is_resume,
                    )
                    node_results[node_id] = result
                    execution_order.append(node_id)
                    await emit(
                        EV_GRAPH_NODE_COMPLETED,
                        graph_id=self.id,
                        node_id=node_id,
                        span_id=span_id,
                        status=str(result.status),
                        duration_ms=(time.perf_counter() - started_at) * 1000,
                        parallel=False,
                    )
                    await _emit_node_events(_event_sink, node_id, result, state, cfg.stream_mode)

            # Clear resume context after first node
            resume_node = None
            resume_value = None

            # Process results and determine next nodes
            for node_id in [n for n in current_nodes if n != END]:
                # Use a different name from the earlier ``result`` so mypy
                # doesn't widen the previously narrowed type.
                node_result = node_results.get(node_id)
                if not node_result:
                    continue
                result = node_result

                # Handle interrupt
                if result.status == NodeStatus.INTERRUPTED:
                    interrupt_state = InterruptState(
                        interrupt=result.output,
                        node_id=node_id,
                        pending_nodes=[n for n in current_nodes if n != node_id],
                        state_snapshot=state,
                    )

                    # Save to checkpointer if available. Same packing as
                    # the ``interrupt_before`` branch above — the resume
                    # path reads off ``saved_state.metadata``.
                    if cfg.checkpointer and cfg.thread_id:
                        from tulip.core.state import AgentState  # noqa: PLC0415

                        stub_state = AgentState(
                            metadata={
                                "graph_state": state,
                                "interrupted_node": node_id,
                                "interrupt": interrupt_state.model_dump(),
                            }
                        )
                        await cfg.checkpointer.save(
                            state=stub_state,
                            thread_id=cfg.thread_id,
                        )
                        from tulip.observability.emit import (  # noqa: PLC0415
                            EV_CHECKPOINT_SAVED,
                            emit,
                        )

                        await emit(
                            EV_CHECKPOINT_SAVED,
                            thread_id=cfg.thread_id,
                            backend=type(cfg.checkpointer).__name__,
                            trigger="graph_interrupt",
                            interrupted_node=node_id,
                        )

                    # Store resume node in state for checkpointer-less resumption
                    final_state_with_resume = {**state, "__resume_node__": node_id}

                    return GraphResult(
                        graph_id=self.id,
                        success=False,
                        node_results=node_results,
                        final_state=final_state_with_resume,
                        execution_order=execution_order,
                        duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                        interrupt=interrupt_state,
                        iterations=iterations,
                    )

                # Handle successful execution
                if result.success:
                    # Apply state updates
                    update, command = normalize_node_output(result.output)
                    if update:
                        state = self._apply_state_update(state, update)
                        # Store raw output under namespaced key to avoid conflicts
                        state[f"_node_{node_id}"] = result.output

                    # Handle Send (map-reduce)
                    if result.sends:
                        send_results = await self._execute_sends(result.sends, state)
                        for sr in send_results:
                            if sr.success:
                                state[sr.send_id] = sr.result

                    # Determine next nodes
                    node_next = self._get_next_nodes(node_id, state, command)
                    next_nodes.extend(node_next)

                # Check for interrupt_after
                if node_id in cfg.interrupt_after:
                    interrupt_state = InterruptState(
                        interrupt=None,  # type: ignore
                        node_id=node_id,
                        pending_nodes=next_nodes,
                        state_snapshot=state,
                    )
                    return GraphResult(
                        graph_id=self.id,
                        success=True,
                        node_results=node_results,
                        final_state=state,
                        execution_order=execution_order,
                        duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                        interrupt=interrupt_state,
                        iterations=iterations,
                    )

            # Check if we've reached END
            if END in current_nodes or END in next_nodes:
                break

            # Move to next nodes (deduplicate)
            current_nodes = list(dict.fromkeys(next_nodes))

        # Calculate duration
        end_time = datetime.now(UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Determine final outputs (from last executed nodes)
        final_outputs = {}
        for node_id in reversed(execution_order):
            if node_id in node_results and node_results[node_id].success:
                final_outputs[node_id] = node_results[node_id].output

        # Check success
        success = all(
            r.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED) for r in node_results.values()
        )

        return GraphResult(
            graph_id=self.id,
            success=success,
            node_results=node_results,
            final_state=state,
            final_outputs=final_outputs,
            execution_order=execution_order,
            duration_ms=duration_ms,
            iterations=iterations,
        )

    async def _execute_sends(
        self,
        sends: list[Send],
        state: dict[str, Any],
    ) -> list[SendResult]:
        """Execute Send operations (map pattern)."""
        results: list[SendResult] = []

        # Group by target node
        by_node: dict[str, list[Send]] = defaultdict(list)
        for send in sends:
            by_node[send.node].append(send)

        # Execute in parallel
        tasks = []
        send_ids = []
        for node_id, node_sends in by_node.items():
            if node_id not in self.nodes:
                for send in node_sends:
                    results.append(
                        SendResult(
                            send_id=send.send_id,
                            node=node_id,
                            success=False,
                            error=f"Node not found: {node_id}",
                        )
                    )
                continue

            node = self.nodes[node_id]
            for send in node_sends:
                # Merge state with send payload
                inputs = {**state, **send.payload}
                tasks.append(node.execute(inputs))
                send_ids.append((send.send_id, node_id))

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for (send_id, node_id), result in zip(send_ids, task_results, strict=True):
                # ``return_exceptions=True`` widens to NodeResult |
                # BaseException; narrow on BaseException so mypy can
                # treat the else branch as NodeResult.
                if isinstance(result, BaseException):
                    results.append(
                        SendResult(
                            send_id=send_id,
                            node=node_id,
                            success=False,
                            error=str(result),
                        )
                    )
                else:
                    results.append(
                        SendResult(
                            send_id=send_id,
                            node=node_id,
                            success=result.success,
                            result=result.output,
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )
                    )

        return results

    async def stream(
        self,
        inputs: dict[str, Any] | Command | None = None,
        *,
        config: GraphConfig | None = None,
        mode: StreamMode | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream graph execution events as nodes complete.

        Drives :meth:`execute` on a background task with an async-queue
        sink wired in, then yields each node-completion event in real time
        (instead of collecting them at the end). The final state arrives as
        the last yielded event in ``VALUES`` mode.

        Args:
            inputs: Initial state or Command (for resume).
            config: Execution configuration.
            mode: Stream mode override (``VALUES`` / ``UPDATES`` / ``NODES``
                / ``DEBUG`` / ``CUSTOM``). Defaults to ``config.stream_mode``.

        Yields:
            ``StreamEvent`` per node, then a terminal ``VALUES`` event with
            the final state when ``mode == VALUES``. Re-raises any
            exception the underlying execute raised so callers can react.
        """
        cfg = config or self.config
        # The mode argument is per-call; reflect it in cfg so the sink in
        # execute() emits the right events. We don't mutate the caller's
        # cfg — clone it.
        stream_mode = mode or cfg.stream_mode
        if mode is not None and mode != cfg.stream_mode:
            cfg = cfg.model_copy(update={"stream_mode": mode})

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        sentinel = None  # marks normal completion in the queue

        async def _sink(event: StreamEvent) -> None:
            await queue.put(event)

        async def _drive() -> GraphResult:
            # Make the sink visible to ``emit_custom`` calls inside node
            # bodies via the module-level ContextVar. Reset on exit so we
            # don't leak the sink into unrelated tasks.
            token = _active_stream_sink.set(_sink)
            try:
                return await self.execute(inputs, config=cfg, _event_sink=_sink)
            finally:
                _active_stream_sink.reset(token)
                # Signal end-of-stream regardless of success/error so the
                # consumer never deadlocks.
                await queue.put(sentinel)

        task = asyncio.create_task(_drive())

        consumer_broke_early = False
        try:
            while True:
                event = await queue.get()
                if event is sentinel:
                    break
                yield event
        except (GeneratorExit, asyncio.CancelledError):
            consumer_broke_early = True
            raise
        finally:
            # If the consumer broke out early, cancel the driver so the
            # background task doesn't leak.
            if consumer_broke_early and not task.done():
                task.cancel()

        # task is done (or was cancelled). Surface any exception execute
        # raised; otherwise emit the terminal final-state event in
        # VALUES mode.
        result = task.result()
        if stream_mode == StreamMode.VALUES:
            yield StreamEvent(mode=stream_mode, data=result.final_state)

    async def ainvoke(
        self,
        inputs: dict[str, Any] | None = None,
        config: GraphConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LangChain/LangGraph-compatible alias for execute() returning final_state."""
        result = await self.execute(inputs or {}, config=config)
        return result.final_state

    def invoke(
        self,
        inputs: dict[str, Any] | None = None,
        config: GraphConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous entry point — LangGraph parity for ``CompiledStateGraph.invoke``.

        Thin sync wrapper around :meth:`ainvoke`. Refuses to run when called
        from inside a live event loop — use :meth:`ainvoke` there instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(inputs, config=config, **kwargs))
        msg = (
            "StateGraph.invoke() called from inside a running event loop. "
            "Use `await graph.ainvoke(...)` from async code."
        )
        raise RuntimeError(msg)

    def run_sync(
        self,
        inputs: dict[str, Any] | None = None,
        config: GraphConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for :meth:`invoke` — matches the spelling used in the
        ``docs/concepts/multi-agent/graph.md`` examples."""
        return self.invoke(inputs, config=config, **kwargs)

    async def astream(
        self,
        inputs: dict[str, Any] | None = None,
        config: GraphConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """LangChain/LangGraph-compatible alias for stream()."""
        async for event in self.stream(inputs, config=config):
            yield event

    def get_graph(self) -> StateGraph:
        """LangGraph-compatible alias — returns self.

        Self carries ``.nodes``, ``.edges``, ``.draw_mermaid()``, and
        ``.draw_ascii()`` so the LangGraph chain ``compiled.get_graph().draw_mermaid()``
        works out of the box.
        """
        return self

    def draw_mermaid(self, *, direction: str = "TD") -> str:
        """Render this graph as a Mermaid flowchart.

        LangGraph parity for ``compiled.get_graph().draw_mermaid()``.
        Delegates to :func:`tulip.multiagent.visualize.draw_mermaid`.
        """
        from tulip.multiagent.visualize import draw_mermaid as _draw_mermaid

        return _draw_mermaid(self, direction=direction)

    def draw_ascii(self) -> str:
        """Render this graph as ASCII.

        LangGraph parity for ``compiled.get_graph().draw_ascii()``.
        Delegates to :func:`tulip.multiagent.visualize.draw_ascii`.
        """
        from tulip.multiagent.visualize import draw_ascii as _draw_ascii

        return _draw_ascii(self)

    def get_mermaid(self, *, direction: str = "TD") -> str:
        """Alias for :meth:`draw_mermaid` — matches the spelling used in
        ``docs/concepts/multi-agent/graph.md``."""
        return self.draw_mermaid(direction=direction)

    async def aget_state(self, config: Any = None) -> None:
        """LangGraph-compatible stub — tulip uses checkpointer.load directly."""
        return

    def compile(
        self,
        *,
        checkpointer: Any | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        store: Any | None = None,
    ) -> StateGraph:
        """
        Compile the graph with configuration.

        Args:
            checkpointer: Checkpointer for state persistence
            interrupt_before: Nodes to pause before
            interrupt_after: Nodes to pause after
            store: Store for cross-thread memory

        Returns:
            Configured graph (self)
        """
        if checkpointer:
            self.config.checkpointer = checkpointer
        if interrupt_before:
            self.config.interrupt_before = interrupt_before
        if interrupt_after:
            self.config.interrupt_after = interrupt_after
        if store:
            self.config.store = store
        return self


# =============================================================================
# Legacy Graph (DAG-only, backwards compatible)
# =============================================================================


class Graph(StateGraph):
    """
    Legacy Graph class for backwards compatibility.

    Use StateGraph for new code.
    """

    def __init__(self, **data: Any):
        # Disable cycles by default for legacy Graph
        if "config" not in data:
            data["config"] = GraphConfig(allow_cycles=False)
        super().__init__(**data)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_graph(
    name: str = "",
    description: str = "",
    allow_cycles: bool = False,
) -> StateGraph:
    """Create a new graph."""
    config = GraphConfig(allow_cycles=allow_cycles)
    return StateGraph(name=name, description=description, config=config)


def node(
    name: str,
    executor: Callable[..., Any],
    *,
    description: str = "",
    condition: Callable[[dict[str, Any]], bool] | None = None,
    max_retries: int = 0,
    timeout_ms: float | None = None,
) -> Node:
    """Create a node with the given executor."""
    return Node(
        name=name,
        description=description,
        executor=executor,
        condition=condition,
        max_retries=max_retries,
        timeout_ms=timeout_ms,
    )
