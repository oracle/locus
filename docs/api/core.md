# Core primitives

The shared types every other module is built on. Most consumers don't
import from `locus.core` directly — the re-exports at the package root
(`from locus import Message`, etc.) are the supported entry points —
but the typed protocols, error classes, and control-flow helpers below
are the foundation underneath every API.

## Configuration

::: locus.core.config.LocusSettings

## Protocols

The duck-typed contracts that providers, tools, and checkpointers
implement. Use these for type hints when accepting "any model" / "any
checkpointer" / "any tool" without binding to a concrete class.

::: locus.core.protocols.ModelProtocol
::: locus.core.protocols.ToolProtocol
::: locus.core.protocols.CheckpointerProtocol

## Messages

The wire format the agent loop emits to model providers and receives back.

::: locus.core.messages.Message
::: locus.core.messages.Role
::: locus.core.messages.ToolCall
::: locus.core.messages.ToolResult

## Errors

`LocusError` is the common base — every other exception below is a
subclass. Catch `LocusError` when you want one handler for the whole
SDK; catch the specific subclass when you need to recover differently
for, e.g., a checkpoint miss vs. a model throttle.

::: locus.core.errors.LocusError
::: locus.core.errors.ConfigError
::: locus.core.errors.ValidationError

### Model errors

::: locus.core.errors.ModelError
::: locus.core.errors.ModelAuthError
::: locus.core.errors.ModelResponseError
::: locus.core.errors.ModelThrottledError

### Tool errors

::: locus.core.errors.ToolError
::: locus.core.errors.ToolExecutionError
::: locus.core.errors.ToolNotFoundError
::: locus.core.errors.ToolValidationError

### Checkpoint errors

::: locus.core.errors.CheckpointError
::: locus.core.errors.CheckpointNotFoundError
::: locus.core.errors.CheckpointSerializationError

### RAG errors

::: locus.core.errors.RAGError
::: locus.core.errors.EmbeddingError
::: locus.core.errors.VectorStoreError

## Control flow

`Command` and friends are the return values graph / loop nodes use to
steer execution: `goto("node")`, `end(value)`, `resume_with(payload)`.
The agent loop also uses `interrupt()` for human-in-the-loop pauses.

### Command

::: locus.core.command.Command
::: locus.core.command.End
::: locus.core.command.Continue
::: locus.core.command.goto
::: locus.core.command.end
::: locus.core.command.resume_with
::: locus.core.command.is_command
::: locus.core.command.normalize_node_output

### Interrupt (HITL)

::: locus.core.interrupt.interrupt
::: locus.core.interrupt.InterruptException
::: locus.core.interrupt.InterruptValue
::: locus.core.interrupt.InterruptState
::: locus.core.interrupt.GraphInterrupted
::: locus.core.interrupt.InterruptHandler
::: locus.core.interrupt.AutoApproveHandler

### Send (map-reduce / fan-out)

`Send` lets a single node dispatch multiple child invocations and
gather their results. `broadcast` / `scatter` are convenience builders.

::: locus.core.send.Send
::: locus.core.send.SendBatch
::: locus.core.send.SendResult
::: locus.core.send.send
::: locus.core.send.broadcast
::: locus.core.send.scatter
::: locus.core.send.is_send
::: locus.core.send.is_send_list
::: locus.core.send.normalize_sends
::: locus.core.send.extract_send_results
::: locus.core.send.aggregate_send_results

## Reducers

Reducers combine state updates produced by parallel branches. Pick the
built-in that matches your field's semantics (`add_messages` for
`tuple[Message, ...]`, `merge_dict` for `dict`, `max_value` /
`min_value` / `last_value` for scalars).

::: locus.core.reducers.Reducer
::: locus.core.reducers.reducer
::: locus.core.reducers.get_reducer
::: locus.core.reducers.apply_reducers

### Built-in reducers

::: locus.core.reducers.add_messages
::: locus.core.reducers.merge_dict
::: locus.core.reducers.deep_merge_dict
::: locus.core.reducers.append_list
::: locus.core.reducers.unique_append_list
::: locus.core.reducers.add_numbers
::: locus.core.reducers.max_value
::: locus.core.reducers.min_value
::: locus.core.reducers.last_value
::: locus.core.reducers.first_value
::: locus.core.reducers.set_union
