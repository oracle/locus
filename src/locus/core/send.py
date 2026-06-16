# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Send primitive for dynamic parallel execution (map-reduce patterns).

The Send class enables dynamic fan-out patterns where a router
can spawn multiple parallel node executions with different payloads.

Example - Map-reduce pattern:
    from locus.core.send import Send

    async def split_tasks(inputs):
        # Fan out to multiple workers
        return [
            Send("worker", {"task": task, "index": i})
            for i, task in enumerate(inputs["tasks"])
        ]

    async def worker(inputs):
        # Process individual task
        return {"result": process(inputs["task"])}

    async def aggregate(inputs):
        # Collect all worker results
        results = [r["result"] for r in inputs.values()]
        return {"combined": merge_results(results)}

Example - Conditional fan-out:
    async def router(inputs):
        sends = []
        if inputs["needs_analysis"]:
            sends.append(Send("analyzer", inputs))
        if inputs["needs_validation"]:
            sends.append(Send("validator", inputs))
        if not sends:
            sends.append(Send("default_handler", inputs))
        return sends
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Send(BaseModel):
    """
    Directive to send data to a specific node for parallel execution.

    When a node returns a list of Send objects, the graph executor
    spawns parallel executions of the target nodes with the given payloads.

    Attributes:
        node: Target node ID to execute
        payload: Data to pass to the target node
        send_id: Unique identifier for this send (for result tracking)
        metadata: Additional context for the send operation

    Example - Simple send:
        >>> Send("worker", {"task": "process_data"})

    Example - With tracking:
        >>> Send(
        ...     node="analyzer",
        ...     payload={"data": chunk},
        ...     metadata={"chunk_index": 0, "total_chunks": 10},
        ... )
    """

    node: str
    payload: dict[str, Any] = Field(default_factory=dict)
    send_id: str = Field(default_factory=lambda: f"send_{uuid4().hex[:8]}")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def with_payload(self, **kwargs: Any) -> Send:
        """Return new Send with additional payload data."""
        new_payload = {**self.payload, **kwargs}
        return self.model_copy(update={"payload": new_payload})

    def with_metadata(self, **kwargs: Any) -> Send:
        """Return new Send with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return self.model_copy(update={"metadata": new_metadata})


class SendResult(BaseModel):
    """
    Result from a Send operation.

    Tracks the outcome of a parallel node execution spawned by Send.

    Attributes:
        send_id: ID of the original Send
        node: Target node that was executed
        success: Whether execution succeeded
        result: Output from the node
        error: Error message if failed
        duration_ms: Execution time in milliseconds
    """

    send_id: str
    node: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class SendBatch(BaseModel):
    """
    A batch of Send operations to execute together.

    Used internally by the graph executor to group sends
    that should be processed in parallel.

    Attributes:
        sends: List of Send operations
        source_node: Node that generated these sends
        aggregator_node: Optional node to receive all results
    """

    sends: list[Send]
    source_node: str
    aggregator_node: str | None = None

    model_config = {"frozen": True}

    @property
    def target_nodes(self) -> list[str]:
        """Get unique target nodes."""
        return list({s.node for s in self.sends})

    @property
    def count(self) -> int:
        """Number of sends in batch."""
        return len(self.sends)

    def group_by_node(self) -> dict[str, list[Send]]:
        """Group sends by target node."""
        groups: dict[str, list[Send]] = {}
        for send in self.sends:
            if send.node not in groups:
                groups[send.node] = []
            groups[send.node].append(send)
        return groups


# =============================================================================
# Send Detection and Processing
# =============================================================================


def is_send(value: Any) -> bool:
    """Check if a value is a Send instance."""
    return isinstance(value, Send)


def is_send_list(value: Any) -> bool:
    """Check if a value is a list of Send instances."""
    return isinstance(value, list) and all(isinstance(v, Send) for v in value)


def normalize_sends(value: Any) -> list[Send] | None:
    """
    Normalize output to list of Sends if applicable.

    Args:
        value: Node output

    Returns:
        List of Send objects, or None if not send output

    Example:
        >>> normalize_sends(Send("node", {}))
        [Send(node="node", ...)]
        >>> normalize_sends([Send("a", {}), Send("b", {})])
        [Send(node="a", ...), Send(node="b", ...)]
        >>> normalize_sends({"result": 1})
        None
    """
    if isinstance(value, Send):
        return [value]
    if is_send_list(value):
        # ``is_send_list`` is a custom TypeGuard-style check; mypy can't
        # propagate the narrowing back to ``value`` here, so the return
        # would be Any otherwise.
        return value  # type: ignore[no-any-return]
    return None


def extract_send_results(
    results: list[SendResult],
    key: str = "result",
) -> dict[str, Any]:
    """
    Extract results from Send operations into a dict.

    Args:
        results: List of SendResult objects
        key: Key to extract from each result

    Returns:
        Dict mapping send_id to extracted value

    Example:
        >>> results = [SendResult(send_id="s1", result={"data": 1}, ...)]
        >>> extract_send_results(results)
        {"s1": {"data": 1}}
    """
    return {r.send_id: r.result for r in results if r.success}


def aggregate_send_results(
    results: list[SendResult],
    reducer: Any = None,
) -> Any:
    """
    Aggregate results from Send operations.

    Args:
        results: List of SendResult objects
        reducer: Optional reducer function (current, update) -> merged

    Returns:
        Aggregated result

    Example - Collect as list:
        >>> aggregate_send_results(results)
        [result1, result2, result3]

    Example - With reducer:
        >>> aggregate_send_results(results, reducer=lambda a, b: {**a, **b})
        {combined_dict}
    """
    successful = [r.result for r in results if r.success]

    if reducer is None:
        return successful

    if not successful:
        return None

    result = successful[0]
    for item in successful[1:]:
        result = reducer(result, item)
    return result


# =============================================================================
# Convenience Constructors
# =============================================================================


def send(node: str, **payload: Any) -> Send:
    """
    Create a Send to a target node.

    Args:
        node: Target node ID
        **payload: Data to pass to the node

    Returns:
        Send instance

    Example:
        >>> send("worker", task="process", data=[1, 2, 3])
    """
    return Send(node=node, payload=payload)


def broadcast(nodes: list[str], payload: dict[str, Any] | None = None) -> list[Send]:
    """
    Create Sends to multiple nodes with same payload.

    Args:
        nodes: List of target node IDs
        payload: Shared payload for all nodes

    Returns:
        List of Send instances

    Example:
        >>> broadcast(["worker1", "worker2", "worker3"], {"task": data})
    """
    payload = payload or {}
    return [Send(node=node, payload=payload) for node in nodes]


def scatter(
    node: str,
    items: list[Any],
    key: str = "item",
    include_index: bool = True,
) -> list[Send]:
    """
    Scatter items to same node with different payloads.

    Args:
        node: Target node ID
        items: Items to distribute
        key: Key name for each item in payload
        include_index: Whether to include index in payload

    Returns:
        List of Send instances

    Example:
        >>> scatter("processor", [data1, data2, data3])
        [Send(node="processor", payload={"item": data1, "index": 0}), ...]
    """
    sends = []
    for i, item in enumerate(items):
        payload: dict[str, Any] = {key: item}
        if include_index:
            payload["index"] = i
            payload["total"] = len(items)
        sends.append(Send(node=node, payload=payload))
    return sends
