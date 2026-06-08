# ReAct loop

The low-level ReAct loop primitives. Most users should reach for the
high-level `Agent` API (see [Agent](agent.md)) — these classes are for
when you need to compose your own loop shape (different node order,
custom routing, batched execution).

## Loop

::: tulip.loop.react.ReActLoop
::: tulip.loop.react.ReActLoopConfig
::: tulip.loop.react.create_react_loop

## Nodes

::: tulip.loop.nodes.Node
::: tulip.loop.nodes.NodeResult
::: tulip.loop.nodes.ThinkNode
::: tulip.loop.nodes.ExecuteNode
::: tulip.loop.nodes.ReflectNode

## Router

::: tulip.loop.router.Router
::: tulip.loop.router.ConditionalRouter
::: tulip.loop.router.NodeType
::: tulip.loop.router.RouteDecision

## Runner

::: tulip.loop.runner.LoopRunner
::: tulip.loop.runner.BatchRunner
::: tulip.loop.runner.StreamingCollector
::: tulip.loop.runner.create_runner
