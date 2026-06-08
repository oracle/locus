# Core primitives

The shared types every other module is built on. Most consumers don't
import from `tulip.core` directly — the re-exports at the package root
(`from tulip import Message`, etc.) are the supported entry points —
but the typed protocols, error classes, and control-flow helpers below
are the foundation underneath every API.

## Configuration

::: tulip.core.config.TulipSettings

## Protocols

The duck-typed contracts that providers, tools, and checkpointers
implement. Use these for type hints when accepting "any model" / "any
checkpointer" / "any tool" without binding to a concrete class.

::: tulip.core.protocols.ModelProtocol
::: tulip.core.protocols.ToolProtocol
::: tulip.core.protocols.CheckpointerProtocol

## Messages

The wire format the agent loop emits to model providers and receives back.

::: tulip.core.messages.Message
::: tulip.core.messages.Role
::: tulip.core.messages.ToolCall
::: tulip.core.messages.ToolResult

## Errors

`TulipError` is the common base — every other exception below is a
subclass. Catch `TulipError` when you want one handler for the whole
SDK; catch the specific subclass when you need to recover differently
for, e.g., a checkpoint miss vs. a model throttle.

::: tulip.core.errors.TulipError
::: tulip.core.errors.ConfigError
::: tulip.core.errors.ValidationError

### Model errors

::: tulip.core.errors.ModelError
::: tulip.core.errors.ModelAuthError
::: tulip.core.errors.ModelResponseError
::: tulip.core.errors.ModelThrottledError

### Tool errors

::: tulip.core.errors.ToolError
::: tulip.core.errors.ToolExecutionError
::: tulip.core.errors.ToolNotFoundError
::: tulip.core.errors.ToolValidationError

### Checkpoint errors

::: tulip.core.errors.CheckpointError
::: tulip.core.errors.CheckpointNotFoundError
::: tulip.core.errors.CheckpointSerializationError

### RAG errors

::: tulip.core.errors.RAGError
::: tulip.core.errors.EmbeddingError
::: tulip.core.errors.VectorStoreError

## Control flow

`Command` and friends are the return values graph / loop nodes use to
steer execution: `goto("node")`, `end(value)`, `resume_with(payload)`.
The agent loop also uses `interrupt()` for human-in-the-loop pauses.

### Command

::: tulip.core.command.Command
::: tulip.core.command.End
::: tulip.core.command.Continue
::: tulip.core.command.goto
::: tulip.core.command.end
::: tulip.core.command.resume_with
::: tulip.core.command.is_command
::: tulip.core.command.normalize_node_output

### Interrupt (HITL)

::: tulip.core.interrupt.interrupt
::: tulip.core.interrupt.InterruptException
::: tulip.core.interrupt.InterruptValue
::: tulip.core.interrupt.InterruptState
::: tulip.core.interrupt.GraphInterrupted
::: tulip.core.interrupt.InterruptHandler
::: tulip.core.interrupt.AutoApproveHandler

### Send (map-reduce / fan-out)

`Send` lets a single node dispatch multiple child invocations and
gather their results. `broadcast` / `scatter` are convenience builders.

::: tulip.core.send.Send
::: tulip.core.send.SendBatch
::: tulip.core.send.SendResult
::: tulip.core.send.send
::: tulip.core.send.broadcast
::: tulip.core.send.scatter
::: tulip.core.send.is_send
::: tulip.core.send.is_send_list
::: tulip.core.send.normalize_sends
::: tulip.core.send.extract_send_results
::: tulip.core.send.aggregate_send_results

## Reducers

Reducers combine state updates produced by parallel branches. Pick the
built-in that matches your field's semantics (`add_messages` for
`tuple[Message, ...]`, `merge_dict` for `dict`, `max_value` /
`min_value` / `last_value` for scalars).

::: tulip.core.reducers.Reducer
::: tulip.core.reducers.reducer
::: tulip.core.reducers.get_reducer
::: tulip.core.reducers.apply_reducers

### Built-in reducers

::: tulip.core.reducers.add_messages
::: tulip.core.reducers.merge_dict
::: tulip.core.reducers.deep_merge_dict
::: tulip.core.reducers.append_list
::: tulip.core.reducers.unique_append_list
::: tulip.core.reducers.add_numbers
::: tulip.core.reducers.max_value
::: tulip.core.reducers.min_value
::: tulip.core.reducers.last_value
::: tulip.core.reducers.first_value
::: tulip.core.reducers.set_union
