# ReAct loop

The low-level ReAct loop primitives. Most users should reach for the
high-level `Agent` API (see [Agent](agent.md)) — these classes are for
when you need to compose your own loop shape (different node order,
custom routing, batched execution).

## Loop

::: locus.loop.react.ReActLoop
::: locus.loop.react.ReActLoopConfig
::: locus.loop.react.create_react_loop

## Nodes

::: locus.loop.nodes.Node
::: locus.loop.nodes.NodeResult
::: locus.loop.nodes.ThinkNode
::: locus.loop.nodes.ExecuteNode
::: locus.loop.nodes.ReflectNode

## Router

::: locus.loop.router.Router
::: locus.loop.router.ConditionalRouter
::: locus.loop.router.NodeType
::: locus.loop.router.RouteDecision

## Runner

::: locus.loop.runner.LoopRunner
::: locus.loop.runner.BatchRunner
::: locus.loop.runner.StreamingCollector
::: locus.loop.runner.create_runner
