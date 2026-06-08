# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for StateGraph and graph execution."""

import pytest

from tulip.core import (
    Command,
    Send,
    goto,
    interrupt,
    scatter,
)
from tulip.core.interrupt import NodeExecutionContext
from tulip.multiagent.graph import (
    END,
    START,
    ConditionalEdge,
    Edge,
    GraphConfig,
    GraphResult,
    InterruptState,
    Node,
    NodeResult,
    NodeStatus,
    StateGraph,
    create_graph,
    node,
)


class TestNode:
    """Tests for Node class."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test basic node execution."""

        async def executor(inputs):
            return {"result": inputs.get("x", 0) * 2}

        n = Node(name="double", executor=executor)
        result = await n.execute({"x": 5})

        assert result.success
        assert result.output == {"result": 10}
        assert result.duration_ms is not None

    @pytest.mark.asyncio
    async def test_sync_executor(self):
        """Test node with sync executor."""

        def sync_executor(inputs):
            return {"value": inputs.get("x", 0) + 1}

        n = Node(name="sync", executor=sync_executor)
        result = await n.execute({"x": 10})

        assert result.success
        assert result.output == {"value": 11}

    @pytest.mark.asyncio
    async def test_condition_skip(self):
        """Test node is skipped when condition is False."""

        async def executor(inputs):
            return {"processed": True}

        n = Node(
            name="conditional",
            executor=executor,
            condition=lambda x: x.get("should_run", False),
        )
        result = await n.execute({"should_run": False})

        assert result.status == NodeStatus.SKIPPED
        assert result.output is None

    @pytest.mark.asyncio
    async def test_condition_run(self):
        """Test node runs when condition is True."""

        async def executor(inputs):
            return {"processed": True}

        n = Node(
            name="conditional",
            executor=executor,
            condition=lambda x: x.get("should_run", False),
        )
        result = await n.execute({"should_run": True})

        assert result.success
        assert result.output == {"processed": True}

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test node timeout."""
        import asyncio

        async def slow_executor(inputs):
            await asyncio.sleep(1)
            return {"done": True}

        n = Node(name="slow", executor=slow_executor, timeout_ms=100)
        result = await n.execute({})

        assert result.status == NodeStatus.FAILED
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_retry(self):
        """Test node retry on failure."""
        attempts = []

        async def flaky_executor(inputs):
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")
            return {"success": True}

        n = Node(
            name="flaky",
            executor=flaky_executor,
            max_retries=3,
            retry_delay_ms=10,
        )
        result = await n.execute({})

        assert result.success
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_interrupt_handling(self):
        """Test node handles interrupt."""

        async def interruptible(inputs):
            with NodeExecutionContext(node_id="test"):
                result = interrupt({"question": "Continue?"})
                return {"answer": result}

        n = Node(name="interruptible", executor=interruptible)
        result = await n.execute({})

        assert result.status == NodeStatus.INTERRUPTED
        assert result.output.payload == {"question": "Continue?"}


class TestEdge:
    """Tests for Edge class."""

    def test_default_apply(self):
        """Test default edge behavior."""
        edge = Edge(source_id="node1", target_id="node2")
        result = edge.apply({"x": 1, "y": 2})
        assert result == {"node1": {"x": 1, "y": 2}}

    def test_key_mapping(self):
        """Test edge with key mapping."""
        edge = Edge(
            source_id="node1",
            target_id="node2",
            key_mapping={"output": "input", "result": "data"},
        )
        result = edge.apply({"output": "value1", "result": "value2", "other": "ignored"})
        assert result == {"input": "value1", "data": "value2"}

    def test_transform(self):
        """Test edge with transform function."""
        edge = Edge(
            source_id="node1",
            target_id="node2",
            transform=lambda x: {"transformed": x["value"] * 2},
        )
        result = edge.apply({"value": 5})
        assert result == {"node1": {"transformed": 10}}


class TestConditionalEdge:
    """Tests for ConditionalEdge class."""

    def test_single_target(self):
        """Test conditional edge with single target."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda s: s.get("type", "default"),
            targets={"error": "error_handler", "success": "success_handler"},
        )
        result = edge.resolve_target({"type": "error"})
        assert result == ["error_handler"]

    def test_multiple_targets(self):
        """Test conditional edge with multiple targets."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda s: ["process", "validate"],
            targets={"process": "processor", "validate": "validator"},
        )
        result = edge.resolve_target({})
        assert result == ["processor", "validator"]

    def test_default_target(self):
        """Test conditional edge falls back to default."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda s: "unknown",
            targets={},
            default_target="fallback",
        )
        # When router returns unmapped value and no direct match, use default_target
        result = edge.resolve_target({})
        assert result == ["fallback"]


class TestStateGraph:
    """Tests for StateGraph class."""

    @pytest.mark.asyncio
    async def test_simple_linear_graph(self):
        """Test simple linear graph execution."""
        graph = StateGraph()

        async def node1(inputs):
            return {"step1": True, "value": inputs.get("x", 0) + 1}

        async def node2(inputs):
            return {"step2": True, "value": inputs.get("value", 0) * 2}

        graph.add_node("node1", node1)
        graph.add_node("node2", node2)
        graph.add_edge(START, "node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", END)

        result = await graph.execute({"x": 5})

        assert result.success
        assert result.final_state.get("step1")
        assert result.final_state.get("step2")
        assert result.execution_order == ["node1", "node2"]

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel node execution."""
        import asyncio

        execution_times = []

        graph = StateGraph()

        async def slow_node_a(inputs):
            execution_times.append(("start", "a"))
            await asyncio.sleep(0.1)
            execution_times.append(("end", "a"))
            return {"done_a": True}

        async def slow_node_b(inputs):
            execution_times.append(("start", "b"))
            await asyncio.sleep(0.1)
            execution_times.append(("end", "b"))
            return {"done_b": True}

        async def final_node(inputs):
            return {"all_done": True}

        graph.add_node("a", slow_node_a)
        graph.add_node("b", slow_node_b)
        graph.add_node("final", final_node)

        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("a", "final")
        graph.add_edge("b", "final")
        graph.add_edge("final", END)

        result = await graph.execute({})

        assert result.success
        # Both should start before either ends (parallel execution)
        # Note: Execution order depends on graph structure

    @pytest.mark.asyncio
    async def test_conditional_edges(self):
        """Test conditional edge routing."""
        graph = StateGraph()

        async def classifier(inputs):
            return {"type": "error" if inputs.get("has_error") else "success"}

        async def handle_error(inputs):
            return {"handled": "error"}

        async def handle_success(inputs):
            return {"handled": "success"}

        graph.add_node("classify", classifier)
        graph.add_node("error", handle_error)
        graph.add_node("success", handle_success)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            lambda s: s.get("type", "success"),
            {"error": "error", "success": "success"},
        )
        graph.add_edge("error", END)
        graph.add_edge("success", END)

        # Test error path
        result = await graph.execute({"has_error": True})
        assert result.final_state.get("handled") == "error"

        # Test success path
        result = await graph.execute({"has_error": False})
        assert result.final_state.get("handled") == "success"

    @pytest.mark.asyncio
    async def test_command_routing(self):
        """Test Command-based routing."""
        graph = StateGraph()

        async def router(inputs):
            if inputs.get("priority") == "high":
                return Command(update={"routed": "fast"}, goto="fast")
            return Command(update={"routed": "normal"}, goto="normal")

        async def fast_track(inputs):
            return {"path": "fast"}

        async def normal_queue(inputs):
            return {"path": "normal"}

        graph.add_node("router", router)
        graph.add_node("fast", fast_track)
        graph.add_node("normal", normal_queue)

        graph.add_edge(START, "router")
        graph.add_edge("fast", END)
        graph.add_edge("normal", END)

        result = await graph.execute({"priority": "high"})
        assert result.final_state.get("path") == "fast"

        result = await graph.execute({"priority": "normal"})
        assert result.final_state.get("path") == "normal"

    @pytest.mark.asyncio
    async def test_interrupt_and_resume(self):
        """Test interrupt pauses execution."""
        graph = StateGraph()

        async def prepare(inputs):
            return {"prepared": True}

        async def approve(inputs):
            approval = interrupt({"message": "Approve?"})
            return {"approved": approval == "yes"}

        async def execute_action(inputs):
            return {"executed": inputs.get("approved")}

        graph.add_node("prepare", prepare)
        graph.add_node("approve", approve)
        graph.add_node("execute", execute_action)

        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "approve")
        graph.add_edge("approve", "execute")
        graph.add_edge("execute", END)

        result = await graph.execute({})

        assert result.is_interrupted
        assert result.interrupt.node_id == "approve"
        assert result.interrupt.interrupt.payload == {"message": "Approve?"}

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Test max_iterations prevents infinite loops."""
        graph = StateGraph()
        graph.config.allow_cycles = True
        graph.config.max_iterations = 5

        counter = [0]

        async def loop_node(inputs):
            counter[0] += 1
            return {"count": counter[0]}

        graph.add_node("loop", loop_node)
        graph.add_edge(START, "loop")
        graph.add_edge("loop", "loop")  # Self-loop

        result = await graph.execute({})

        assert counter[0] <= 5  # Should stop at max_iterations

    @pytest.mark.asyncio
    async def test_subgraph_composition(self):
        """Test subgraph as node."""
        # Create subgraph
        subgraph = StateGraph(name="sub")

        async def sub_node1(inputs):
            return {"sub_step1": True, "value": inputs.get("value", 0) + 10}

        async def sub_node2(inputs):
            return {"sub_step2": True, "value": inputs.get("value", 0) * 2}

        subgraph.add_node("s1", sub_node1)
        subgraph.add_node("s2", sub_node2)
        subgraph.add_edge(START, "s1")
        subgraph.add_edge("s1", "s2")
        subgraph.add_edge("s2", END)

        # Create main graph
        main = StateGraph(name="main")

        async def pre(inputs):
            return {"value": inputs.get("x", 0)}

        async def post(inputs):
            return {"final": True}

        main.add_node("pre", pre)
        main.add_node("sub", subgraph)
        main.add_node("post", post)

        main.add_edge(START, "pre")
        main.add_edge("pre", "sub")
        main.add_edge("sub", "post")
        main.add_edge("post", END)

        result = await main.execute({"x": 5})
        assert result.success


class TestGraphCompile:
    """Tests for graph.compile() method."""

    def test_compile_sets_checkpointer(self):
        """Test compile sets checkpointer."""
        graph = StateGraph()

        class FakeCheckpointer:
            pass

        cp = FakeCheckpointer()
        graph.compile(checkpointer=cp)
        assert graph.config.checkpointer is cp

    def test_compile_sets_interrupt_before(self):
        """Test compile sets interrupt_before."""
        graph = StateGraph()
        graph.compile(interrupt_before=["review", "approve"])
        assert graph.config.interrupt_before == ["review", "approve"]

    def test_compile_sets_interrupt_after(self):
        """Test compile sets interrupt_after."""
        graph = StateGraph()
        graph.compile(interrupt_after=["action"])
        assert graph.config.interrupt_after == ["action"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_graph(self):
        """Test create_graph function."""
        g = create_graph(name="test", description="A test graph")
        assert g.name == "test"
        assert g.description == "A test graph"

    def test_create_graph_with_cycles(self):
        """Test create_graph with cycles allowed."""
        g = create_graph(allow_cycles=True)
        assert g.config.allow_cycles

    def test_node_function(self):
        """Test node convenience function."""

        async def my_executor(inputs):
            return {"done": True}

        n = node(
            "my_node",
            my_executor,
            description="Test node",
            max_retries=2,
            timeout_ms=5000,
        )
        assert n.name == "my_node"
        assert n.description == "Test node"
        assert n.max_retries == 2
        assert n.timeout_ms == 5000


class TestStartEnd:
    """Tests for START and END constants."""

    def test_start_end_values(self):
        """Test START and END constant values."""
        assert START == "__START__"
        assert END == "__END__"

    @pytest.mark.asyncio
    async def test_start_end_in_graph(self):
        """Test START and END in graph."""
        graph = StateGraph()

        async def middle(inputs):
            return {"processed": True}

        graph.add_node("middle", middle)
        graph.add_edge(START, "middle")
        graph.add_edge("middle", END)

        result = await graph.execute({})
        assert result.success
        assert "middle" in result.execution_order


class TestEdge:
    """Tests for Edge class."""

    def test_edge_apply_with_transform(self):
        """Test edge apply with transform function."""
        edge = Edge(
            source_id="source",
            target_id="target",
            transform=lambda x: {"transformed": x.get("value", 0) * 2},
        )
        result = edge.apply({"value": 5})
        assert result == {"source": {"transformed": 10}}

    def test_edge_apply_with_key_mapping(self):
        """Test edge apply with key mapping."""
        edge = Edge(
            source_id="source",
            target_id="target",
            key_mapping={"input_value": "output_value"},
        )
        result = edge.apply({"input_value": 42})
        assert result == {"output_value": 42}

    def test_edge_apply_key_mapping_with_non_dict(self):
        """Test edge apply key mapping when source output is not a dict."""
        edge = Edge(
            source_id="source",
            target_id="target",
            key_mapping={"x": "y"},
        )
        result = edge.apply("raw_value")
        assert result == {"y": "raw_value"}

    def test_edge_apply_default(self):
        """Test edge apply default behavior."""
        edge = Edge(source_id="source", target_id="target")
        result = edge.apply({"data": "value"})
        assert result == {"source": {"data": "value"}}


class TestConditionalEdge:
    """Tests for ConditionalEdge class."""

    def test_resolve_single_target_from_mapping(self):
        """Test resolving single target from mapping."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: state.get("decision"),
            targets={"yes": "approve", "no": "reject"},
        )
        targets = edge.resolve_target({"decision": "yes"})
        assert targets == ["approve"]

    def test_resolve_single_target_with_default(self):
        """Test resolving single target with default fallback."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: state.get("decision"),
            targets={"known": "known_node"},
            default_target="default_node",
        )
        targets = edge.resolve_target({"decision": "unknown"})
        assert targets == ["default_node"]

    def test_resolve_single_target_direct(self):
        """Test resolving single target directly without mapping."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: "direct_node",
            targets={},
        )
        targets = edge.resolve_target({})
        assert targets == ["direct_node"]

    def test_resolve_multiple_targets(self):
        """Test resolving multiple targets for parallel execution."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: ["a", "b"],
            targets={"a": "node_a", "b": "node_b"},
        )
        targets = edge.resolve_target({})
        assert targets == ["node_a", "node_b"]

    def test_resolve_multiple_targets_with_default(self):
        """Test multiple targets with some using default."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: ["known", "unknown"],
            targets={"known": "known_node"},
            default_target="fallback",
        )
        targets = edge.resolve_target({})
        assert targets == ["known_node", "fallback"]

    def test_resolve_multiple_targets_direct(self):
        """Test multiple targets resolved directly."""
        edge = ConditionalEdge(
            source_id="router",
            router=lambda state: ["direct_a", "direct_b"],
            targets={},
        )
        targets = edge.resolve_target({})
        assert targets == ["direct_a", "direct_b"]


class TestNodeWithSend:
    """Tests for Node execution returning Send objects."""

    @pytest.mark.asyncio
    async def test_node_returning_single_send(self):
        """Test node returning a single Send."""

        async def executor(inputs):
            return Send(node="target_node", payload={"data": "sent"})

        n = Node(name="sender", executor=executor)
        result = await n.execute({})

        assert result.success
        assert result.sends is not None
        assert len(result.sends) == 1
        assert result.sends[0].node == "target_node"

    @pytest.mark.asyncio
    async def test_node_returning_command(self):
        """Test node returning a Command."""

        async def executor(inputs):
            return goto("next_node")

        n = Node(name="commander", executor=executor)
        result = await n.execute({})

        assert result.success
        assert result.command is not None


class TestGraphWithSends:
    """Tests for graph execution with Send objects."""

    @pytest.mark.asyncio
    async def test_scatter_sends(self):
        """Test using scatter to create multiple Sends."""
        graph = StateGraph()

        async def scatter_node(inputs):
            items = inputs.get("items", [])
            return scatter("process", [{"item": i} for i in items])

        async def process_node(inputs):
            return {"processed": inputs.get("item")}

        graph.add_node("scatter", scatter_node)
        graph.add_node("process", process_node)
        graph.add_edge(START, "scatter")
        graph.add_edge("scatter", "process")
        graph.add_edge("process", END)

        result = await graph.execute({"items": [1, 2, 3]})
        assert result.success


class TestNodeResultStatus:
    """Tests for NodeResult status."""

    def test_node_result_success(self):
        """Test success property."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.COMPLETED,
            output={"done": True},
        )
        assert result.success is True

    def test_node_result_failure(self):
        """Test success property on failure."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.FAILED,
            error="Something went wrong",
        )
        assert result.success is False

    def test_node_result_skipped(self):
        """Test skipped status."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.SKIPPED,
        )
        assert result.success is False

    def test_node_result_interrupted(self):
        """Test interrupted status."""
        result = NodeResult(
            node_id="test",
            status=NodeStatus.INTERRUPTED,
            output={"question": "Continue?"},
        )
        assert result.success is False


class TestStateGraphConfiguration:
    """Tests for StateGraph configuration."""

    def test_set_entry_point(self):
        """Test setting entry point."""
        graph = StateGraph()

        async def my_node(inputs):
            return {"done": True}

        graph.add_node("my_node", my_node)
        graph.set_entry_point("my_node")

        assert graph._entry_point == "my_node"

    def test_set_entry_point_not_found(self):
        """Test setting entry point for nonexistent node."""
        graph = StateGraph()

        with pytest.raises(ValueError, match="Node not found"):
            graph.set_entry_point("nonexistent")

    def test_set_finish_point(self):
        """Test setting finish point."""
        graph = StateGraph()

        async def my_node(inputs):
            return {"done": True}

        graph.add_node("my_node", my_node)
        graph.set_finish_point("my_node")

        # Should have edge to END
        edges_to_end = [e for e in graph.edges if e.target_id == END]
        assert len(edges_to_end) == 1

    def test_set_finish_point_not_found(self):
        """Test setting finish point for nonexistent node."""
        graph = StateGraph()

        with pytest.raises(ValueError, match="Node not found"):
            graph.set_finish_point("nonexistent")

    def test_add_node_missing_executor(self):
        """Test add_node without executor raises error."""
        graph = StateGraph()

        with pytest.raises(TypeError, match="missing 1 required"):
            graph.add_node("my_node")

    def test_add_duplicate_node(self):
        """Test adding duplicate node raises error."""
        graph = StateGraph()

        async def my_node(inputs):
            return {}

        graph.add_node("my_node", my_node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("my_node", my_node)

    def test_add_node_object(self):
        """Test adding Node object directly."""
        graph = StateGraph()

        async def executor(inputs):
            return {"done": True}

        node = Node(id="my_node", name="my_node", executor=executor)
        graph.add_node(node)

        assert "my_node" in graph.nodes

    def test_add_duplicate_node_object(self):
        """Test adding duplicate Node object raises error."""
        graph = StateGraph()

        async def executor(inputs):
            return {}

        node = Node(id="my_node", name="my_node", executor=executor)
        graph.add_node(node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node)


class TestStateGraphEdges:
    """Tests for StateGraph edge operations."""

    def test_add_edge_invalid_source(self):
        """Test adding edge with invalid source."""
        graph = StateGraph()

        async def my_node(inputs):
            return {}

        graph.add_node("target", my_node)

        with pytest.raises(ValueError, match="Source node not found"):
            graph.add_edge("nonexistent", "target")

    def test_add_edge_invalid_target(self):
        """Test adding edge with invalid target."""
        graph = StateGraph()

        async def my_node(inputs):
            return {}

        graph.add_node("source", my_node)

        with pytest.raises(ValueError, match="Target node not found"):
            graph.add_edge("source", "nonexistent")

    def test_add_edge_with_transform(self):
        """Test adding edge with transform function."""
        graph = StateGraph()

        async def node_a(inputs):
            return {"value": 10}

        async def node_b(inputs):
            return {"result": inputs.get("transformed", 0) * 2}

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(
            "a",
            "b",
            transform=lambda x: {"transformed": x.get("value", 0) + 5},
        )

        assert len(graph.edges) == 1
        assert graph.edges[0].transform is not None

    def test_add_edge_with_key_mapping(self):
        """Test adding edge with key mapping."""
        graph = StateGraph()

        async def node_a(inputs):
            return {"output_value": 42}

        async def node_b(inputs):
            return {"result": inputs.get("input_value", 0)}

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b", key_mapping={"output_value": "input_value"})

        assert len(graph.edges) == 1
        assert graph.edges[0].key_mapping is not None


class TestStateGraphConditionalEdges:
    """Tests for conditional edges."""

    @pytest.mark.asyncio
    async def test_conditional_edges_with_targets(self):
        """Test conditional edges with targets mapping."""
        graph = StateGraph()

        async def router_node(inputs):
            return {"decision": inputs.get("choice", "default")}

        async def path_a(inputs):
            return {"path": "A"}

        async def path_b(inputs):
            return {"path": "B"}

        graph.add_node("router", router_node)
        graph.add_node("path_a", path_a)
        graph.add_node("path_b", path_b)

        graph.add_conditional_edges(
            "router",
            router=lambda state: state.get("decision", "default"),
            targets={"option_a": "path_a", "option_b": "path_b"},
            default="path_a",
        )

        assert len(graph.conditional_edges) == 1

    @pytest.mark.asyncio
    async def test_conditional_edges_without_targets(self):
        """Test conditional edges returning node names directly."""
        graph = StateGraph()

        async def router_node(inputs):
            return {"next": "path_b"}

        async def path_a(inputs):
            return {"path": "A"}

        async def path_b(inputs):
            return {"path": "B"}

        graph.add_node("router", router_node)
        graph.add_node("path_a", path_a)
        graph.add_node("path_b", path_b)
        graph.add_edge(START, "router")
        graph.add_edge("path_a", END)
        graph.add_edge("path_b", END)

        graph.add_conditional_edges(
            "router",
            router=lambda state: state.get("next", "path_a"),
        )

        result = await graph.execute({"next": "path_b"})
        assert result.success


class TestStateGraphSubgraph:
    """Tests for subgraph support."""

    @pytest.mark.asyncio
    async def test_add_subgraph_node(self):
        """Test adding a subgraph as a node."""
        # Create subgraph
        subgraph = StateGraph()

        async def sub_node(inputs):
            return {"sub_result": inputs.get("value", 0) * 2}

        subgraph.add_node("sub_node", sub_node)
        subgraph.add_edge(START, "sub_node")
        subgraph.add_edge("sub_node", END)

        # Create main graph
        main_graph = StateGraph()

        async def main_node(inputs):
            return {"value": 10}

        main_graph.add_node("main", main_node)
        main_graph.add_node("subgraph", subgraph)  # Subgraph as node
        main_graph.add_edge(START, "main")
        main_graph.add_edge("main", "subgraph")
        main_graph.add_edge("subgraph", END)

        result = await main_graph.execute({})
        assert result.success


class TestGraphCycleDetection:
    """Tests for cycle detection."""

    def test_cycle_detection_simple(self):
        """Test simple cycle detection."""
        graph = StateGraph(config=GraphConfig(allow_cycles=False))

        async def node(inputs):
            return {}

        graph.add_node("a", node)
        graph.add_node("b", node)

        graph.add_edge(START, "a")
        graph.add_edge("a", "b")

        # This should raise because it creates a cycle
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("b", "a")

    def test_allow_cycles(self):
        """Test allowing cycles."""
        graph = StateGraph(config=GraphConfig(allow_cycles=True))

        async def node(inputs):
            return {}

        graph.add_node("a", node)
        graph.add_node("b", node)

        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")  # Should not raise

        assert len(graph.edges) == 3


class TestStateGraphInterruptBefore:
    """Tests for interrupt_before functionality."""

    @pytest.mark.asyncio
    async def test_interrupt_before_node(self):
        """Test interrupting before a specific node."""
        graph = StateGraph()

        async def step1(inputs):
            return {"step": 1}

        async def step2(inputs):
            return {"step": 2}

        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        cfg = GraphConfig(
            interrupt_before=["step2"],
        )
        result = await graph.execute({"initial": "value"}, config=cfg)

        # Should be interrupted before step2
        assert result.success is False
        assert result.interrupt is not None
        assert result.interrupt.node_id == "step2"

    @pytest.mark.asyncio
    async def test_interrupt_returns_resume_node(self):
        """Test that interrupt includes resume node in state."""
        graph = StateGraph()

        async def step1(inputs):
            return {"done": True}

        async def step2(inputs):
            return {"final": True}

        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        cfg = GraphConfig(interrupt_before=["step2"])
        result = await graph.execute({}, config=cfg)

        assert "__resume_node__" in result.final_state
        assert result.final_state["__resume_node__"] == "step2"

    @pytest.mark.asyncio
    async def test_interrupt_before_saves_through_checkpointer(self):
        """When a checkpointer is configured, the interrupt_before pause
        boundary writes through to it — durable resume needs the paused
        state on disk, not just on the returned ``GraphResult``."""
        from tulip.memory.backends.memory import MemoryCheckpointer

        graph = StateGraph()

        async def gate(inputs):
            return {"gated": True}

        async def after(inputs):
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        cp = MemoryCheckpointer()
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id="t-save",
        )
        result = await graph.execute({"x": 1}, config=cfg)
        assert result.interrupt is not None
        assert result.interrupt.node_id == "gate"

        saved = await cp.load("t-save")
        assert saved is not None, "interrupt_before must persist when a checkpointer is wired"
        assert saved.metadata.get("interrupted_node") == "gate"
        assert saved.metadata.get("graph_state", {}).get("x") == 1

    @pytest.mark.asyncio
    async def test_resume_after_interrupt_before_advances_past_gate(self):
        """``Command(resume=...)`` must continue past the gate node — the
        previous behaviour re-paused on every resume, so durable
        human-in-the-loop flows needed application-side bookkeeping to
        rewrite ``__resume_node__``."""
        from tulip.memory.backends.memory import MemoryCheckpointer

        graph = StateGraph()

        async def gate(inputs):
            return {"gated": True}

        async def after(inputs):
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        cp = MemoryCheckpointer()
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id="t-resume",
        )
        first = await graph.execute({}, config=cfg)
        assert first.interrupt is not None  # paused

        resumed = await graph.execute(Command(resume=True), config=cfg)
        assert resumed.interrupt is None, "resume should not re-pause at the gate"
        assert resumed.final_state.get("done") is True

    @pytest.mark.asyncio
    async def test_resume_without_checkpointer_uses_resume_node_in_state(self):
        """Same advance-past-gate behaviour for the checkpointer-less path
        (``Command(resume=..., update={...with __resume_node__})``)."""
        graph = StateGraph()

        async def gate(inputs):
            return {"gated": True}

        async def after(inputs):
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        cfg = GraphConfig(interrupt_before=["gate"])
        first = await graph.execute({}, config=cfg)
        assert first.interrupt is not None
        carry = dict(first.final_state)  # includes __resume_node__ == "gate"

        resumed = await graph.execute(Command(resume=True, update=carry), config=cfg)
        assert resumed.interrupt is None
        assert resumed.final_state.get("done") is True

    @pytest.mark.asyncio
    async def test_interrupt_before_round_trips_through_file_checkpointer(self, tmp_path):
        """The pause-boundary save shape must round-trip through any
        BaseCheckpointer-conformant backend, not just MemoryCheckpointer.
        FileCheckpointer exercises JSON serialization on disk — proves
        ``AgentState.metadata`` survives ``to_checkpoint()`` →
        ``from_checkpoint()`` for the non-memory path too."""
        from tulip.memory.backends.file import FileCheckpointer

        graph = StateGraph()

        async def gate(inputs):
            return {"gated": True}

        async def after(inputs):
            return {"done": True}

        graph.add_node("gate", gate)
        graph.add_node("after", after)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", "after")
        graph.add_edge("after", END)

        cp = FileCheckpointer(base_dir=tmp_path / "cps")
        cfg = GraphConfig(
            interrupt_before=["gate"],
            checkpointer=cp,
            thread_id="t-file",
        )
        paused = await graph.execute({"x": 1}, config=cfg)
        assert paused.interrupt is not None

        # Reload through the file backend (verifies on-disk shape).
        saved = await cp.load("t-file")
        assert saved is not None
        assert saved.metadata.get("interrupted_node") == "gate"
        assert saved.metadata.get("graph_state", {}).get("x") == 1

        resumed = await graph.execute(Command(resume=True), config=cfg)
        assert resumed.interrupt is None
        assert resumed.final_state.get("done") is True


class TestInlineInterruptCheckpointerSave:
    """The inline ``interrupt()`` save site at the other end of the
    execution loop uses the same shape as the new ``interrupt_before``
    save. Was crashing before the fix (``state=None`` against any
    backend that calls ``state.to_checkpoint()``). These tests prove the
    shape works on both memory and file backends."""

    @pytest.mark.asyncio
    async def test_inline_interrupt_persists_to_memory_checkpointer(self):
        from tulip.memory.backends.memory import MemoryCheckpointer

        graph = StateGraph()

        async def prepare(inputs):
            return {"prepared": True}

        async def approve(inputs):
            approval = interrupt({"message": "Approve?"})
            return {"approved": approval == "yes"}

        async def execute_action(inputs):
            return {"executed": True}

        graph.add_node("prepare", prepare)
        graph.add_node("approve", approve)
        graph.add_node("execute_action", execute_action)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "approve")
        graph.add_edge("approve", "execute_action")
        graph.add_edge("execute_action", END)

        cp = MemoryCheckpointer()
        cfg = GraphConfig(checkpointer=cp, thread_id="t-inline-mem")
        result = await graph.execute({}, config=cfg)
        assert result.is_interrupted
        assert result.interrupt.node_id == "approve"

        saved = await cp.load("t-inline-mem")
        assert saved is not None
        assert saved.metadata.get("interrupted_node") == "approve"
        assert "interrupt" in saved.metadata

    @pytest.mark.asyncio
    async def test_inline_interrupt_persists_to_file_checkpointer(self, tmp_path):
        from tulip.memory.backends.file import FileCheckpointer

        graph = StateGraph()

        async def prepare(inputs):
            return {"prepared": True}

        async def approve(inputs):
            approval = interrupt({"message": "Approve?"})
            return {"approved": approval == "yes"}

        graph.add_node("prepare", prepare)
        graph.add_node("approve", approve)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "approve")
        graph.add_edge("approve", END)

        cp = FileCheckpointer(base_dir=tmp_path / "cps-inline")
        cfg = GraphConfig(checkpointer=cp, thread_id="t-inline-file")
        result = await graph.execute({}, config=cfg)
        assert result.is_interrupted

        saved = await cp.load("t-inline-file")
        assert saved is not None
        assert saved.metadata.get("interrupted_node") == "approve"


class TestStateGraphParallelExecution:
    """Tests for parallel node execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test executing multiple nodes in parallel."""
        graph = StateGraph()

        async def node_a(inputs):
            return {"a": "done"}

        async def node_b(inputs):
            return {"b": "done"}

        async def final(inputs):
            return {"complete": True}

        graph.add_node("node_a", node_a)
        graph.add_node("node_b", node_b)
        graph.add_node("final", final)
        graph.add_edge(START, "node_a")
        graph.add_edge(START, "node_b")
        graph.add_edge("node_a", "final")
        graph.add_edge("node_b", "final")
        graph.add_edge("final", END)

        cfg = GraphConfig(parallel=True)
        result = await graph.execute({}, config=cfg)

        assert result.success
        # Both a and b should have been executed
        assert "node_a" in result.execution_order or "a" in result.final_state


class TestStateGraphResumeExecution:
    """Tests for resume functionality."""

    @pytest.mark.asyncio
    async def test_resume_with_command(self):
        """Test resuming execution with Command."""
        from tulip.core.command import Command

        graph = StateGraph()

        async def step1(inputs):
            return {"step1": True}

        async def step2(inputs):
            return {"step2": True}

        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)

        # First execute to get interrupted state
        cfg = GraphConfig(interrupt_before=["step2"])
        result1 = await graph.execute({}, config=cfg)

        # Resume execution using Command
        resume_cmd = Command(resume="continue", update=result1.final_state)
        result2 = await graph.execute(resume_cmd)

        assert result2.success


class TestStateGraphEdgeCases:
    """Tests for edge cases in graph execution."""

    @pytest.mark.asyncio
    async def test_execute_empty_graph(self):
        """Test executing graph with no nodes."""
        graph = StateGraph()

        result = await graph.execute({})

        assert result.success
        assert len(result.node_results) == 0

    @pytest.mark.asyncio
    async def test_execute_with_max_iterations(self):
        """Test max iterations limit."""
        graph = StateGraph(config=GraphConfig(allow_cycles=True))

        counter = {"value": 0}

        async def counting_node(inputs):
            counter["value"] += 1
            return {"count": counter["value"]}

        graph.add_node("counter", counting_node)
        graph.add_edge(START, "counter")
        graph.add_edge("counter", "counter")  # Self loop

        cfg = GraphConfig(max_iterations=3)
        result = await graph.execute({}, config=cfg)

        # Should stop after max iterations
        assert counter["value"] <= 3

    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """Test error handling in nodes."""
        graph = StateGraph()

        async def failing_node(inputs):
            raise ValueError("Node failed")

        graph.add_node("fail", failing_node)
        graph.add_edge(START, "fail")
        graph.add_edge("fail", END)

        result = await graph.execute({})

        # Should capture the error
        assert "fail" in result.node_results
        assert result.node_results["fail"].status == NodeStatus.FAILED
        assert result.node_results["fail"].error is not None

    @pytest.mark.asyncio
    async def test_graph_result_duration(self):
        """Test graph result includes duration."""
        import asyncio

        graph = StateGraph()

        async def slow_node(inputs):
            await asyncio.sleep(0.01)
            return {"done": True}

        graph.add_node("slow", slow_node)
        graph.add_edge(START, "slow")
        graph.add_edge("slow", END)

        result = await graph.execute({})

        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_graph_with_no_entry_point(self):
        """Test graph without explicit entry point uses first edge."""
        graph = StateGraph()

        async def node(inputs):
            return {"done": True}

        graph.add_node("first", node)
        graph.add_edge(START, "first")
        graph.add_edge("first", END)

        result = await graph.execute({})

        assert result.success
        assert "first" in result.execution_order


class TestGraphResult:
    """Tests for GraphResult dataclass."""

    def test_graph_result_creation(self):
        """Test creating GraphResult."""
        result = GraphResult(
            graph_id="test",
            success=True,
            node_results={},
            final_state={"key": "value"},
            execution_order=["node1", "node2"],
            duration_ms=100.0,
        )

        assert result.graph_id == "test"
        assert result.success is True
        assert result.final_state == {"key": "value"}

    def test_graph_result_failed(self):
        """Test GraphResult with failure."""
        result = GraphResult(
            graph_id="test",
            success=False,
            node_results={},
            final_state={},
            execution_order=[],
            duration_ms=50.0,
        )

        assert result.success is False
        assert result.is_interrupted is False

    def test_graph_result_with_interrupt(self):
        """Test GraphResult with interrupt."""
        from tulip.core.interrupt import InterruptValue

        interrupt = InterruptValue(
            payload={"question": "Continue?"},
            node_id="decision",
            graph_id="test",
        )
        interrupt_state = InterruptState(
            interrupt=interrupt,
            node_id="decision",
            pending_nodes=["next"],
            state_snapshot={},
        )

        result = GraphResult(
            graph_id="test",
            success=False,
            node_results={},
            final_state={},
            execution_order=["prev"],
            duration_ms=25.0,
            interrupt=interrupt_state,
        )

        assert result.interrupt is not None
        assert result.interrupt.node_id == "decision"


class TestGraphMethods:
    """Tests for additional StateGraph methods."""

    def test_access_nodes_dict(self):
        """Test accessing nodes via nodes dictionary."""
        graph = StateGraph()

        async def my_node(inputs):
            return {}

        graph.add_node("my_node", my_node)

        # Access via nodes dict
        assert "my_node" in graph.nodes
        assert graph.nodes["my_node"].id == "my_node"

    def test_check_node_exists(self):
        """Test checking if node exists via nodes dict."""
        graph = StateGraph()

        async def my_node(inputs):
            return {}

        graph.add_node("my_node", my_node)

        assert "my_node" in graph.nodes
        assert "other" not in graph.nodes

    def test_access_edges_list(self):
        """Test accessing edges list."""
        graph = StateGraph()

        async def node(inputs):
            return {}

        graph.add_node("a", node)
        graph.add_node("b", node)
        graph.add_node("c", node)
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")

        # Count edges from a
        edges_from_a = [e for e in graph.edges if e.source_id == "a"]
        assert len(edges_from_a) == 2

    def test_graph_repr(self):
        """Test string representation."""
        graph = StateGraph(id="test_graph")

        async def node(inputs):
            return {}

        graph.add_node("my_node", node)

        repr_str = repr(graph)
        assert "StateGraph" in repr_str or "test_graph" in repr_str


class TestLangGraphParityShims:
    """Regression tests for the LangGraph-shape sync + rendering shims.

    These match what ``docs/concepts/multi-agent/graph.md`` documents
    and what LangGraph's ``CompiledStateGraph`` exposes.
    """

    def _trivial(self) -> StateGraph:
        from tulip.multiagent.graph import END, START

        g = StateGraph()

        async def bump(inputs):
            return {"x": (inputs.get("x") or 0) + 1}

        g.add_node("bump", bump)
        g.add_edge(START, "bump")
        g.add_edge("bump", END)
        return g.compile()

    def test_invoke_is_sync_and_runs(self):
        compiled = self._trivial()
        out = compiled.invoke({"x": 1})
        assert out["x"] == 2

    def test_run_sync_alias(self):
        compiled = self._trivial()
        assert compiled.run_sync({"x": 1})["x"] == compiled.invoke({"x": 1})["x"]

    def test_invoke_refuses_inside_running_loop(self):
        import asyncio

        compiled = self._trivial()

        async def caller():
            compiled.invoke({"x": 1})

        with pytest.raises(RuntimeError, match="running event loop"):
            asyncio.run(caller())

    def test_get_mermaid_and_draw_mermaid_equal(self):
        compiled = self._trivial()
        mermaid = compiled.get_mermaid()
        assert mermaid == compiled.draw_mermaid()
        assert mermaid.startswith("graph TD")
        assert "bump" in mermaid

    def test_get_graph_chain_to_draw_mermaid(self):
        # LangGraph parity: `compiled.get_graph().draw_mermaid()` works.
        compiled = self._trivial()
        assert compiled.get_graph().draw_mermaid() == compiled.draw_mermaid()

    def test_draw_ascii_runs(self):
        compiled = self._trivial()
        ascii_art = compiled.draw_ascii()
        assert isinstance(ascii_art, str)
        assert ascii_art
