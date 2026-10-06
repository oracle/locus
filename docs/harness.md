# The coding harness (`tulip.harness`)

`tulip.harness` is the toolset a coding agent works with — read, write,
edit, search, a shell — written once against a workspace protocol and run
unchanged on the host, in memory, or inside a sandbox. tulip-code is a UI
over it; the Tulip gateway runs the same tools inside NVIDIA OpenShell
sandboxes.

## Build a harness

```python
from tulip import Agent
from tulip.control import ControlPolicy, gate_tool
from tulip.harness import LocalBackend, build_harness

policy = ControlPolicy(require_human_for=frozenset({"workspace.exec"}))
harness = build_harness(
    LocalBackend("."),
    wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec),
)
agent = Agent(model=model, tools=harness.tools, system_prompt=harness.prompt_fragment)
```

`build_harness(backend, *, tools=None, wrap=None, config=None)` returns a
`Harness`:

- `tools` — the built tools, already wrapped.
- `specs` — each tool's `ActionSpec`, by name.
- `prompt_fragment` — what the model needs to know: where the workspace is,
  whether commands run on the host, the read-before-edit rule, the shell's
  timeouts.
- `context` — the session state the tools share: the backend, the read
  ledger, the todo list.
- `preview(name, arguments)` — the diff a file-changing call would make,
  without making it. A gate that shows a person the change before approving
  it calls this.

`tools` picks by name. The default is `DEFAULT_TOOLS` — everything but
`apply_patch`, and without the shell tools on a backend that has no shell.
`PATCH_TOOLS` swaps `edit`, `multi_edit` and `write` for `apply_patch`, the
format the GPT family is trained on;
`tulip.models.profiles.profile_for(model).edit_format` says which a model
wants.

`HarnessConfig` holds the behaviour: whether a change needs a read first,
the shell's default and maximum timeouts, what happens at a timeout
(`"background"` keeps the command under a handle, `"kill"` stops it), how
much output the model sees, whether the model can see images, a
`ToolResultStore` for output too long to show, and the `on_exec` /
`on_change` hooks.

## The single-gate rule

Tool bodies never call a gate. `wrap` is the one place governance is
applied, and it is applied to every tool. Locally that is
`tulip.control.gate_tool`; the gateway passes its own. Both see the same
`Action` for the same call, because the labels come from
`tulip.harness.labels`:

| Kind | Tools |
| --- | --- |
| `workspace.read` | read, ls, glob, grep, bash_output, todo_write, todo_read, and `bash` lines made only of read-only programs |
| `workspace.write` | write, edit, multi_edit, apply_patch, notebook_edit |
| `workspace.exec` | every other `bash` line, write_stdin, kill_shell |
| `network` | reserved for the web tools |

A `bash` line is read with `tulip.control.shell.parse_command` — the
programs it actually runs, behind `sudo`, `env`, `xargs`, `sh -c` and inside
`$(...)` — and tagged with what it does: `exec:destructive`,
`exec:vcs-push`, `exec:force-push`, `exec:network`, `exec:remote-code`,
`exec:shutdown`, `exec:check` (tests and linters), `exec:never` (the
never-allow floor), `exec:background`. A line the parser cannot read is
`workspace.exec` with `exec:unparsed`: unknown, never harmless. Every action
also carries the tool's name as a tag and the path or command as its asset.

**Without `wrap`, the tools are ungated.** They read, write and run
commands with nothing between the model and the workspace. That is right
for a test, or for a throwaway sandbox that is itself the boundary. It is
not right for a host shell, and `build_harness` logs a warning when you ask
for that.

## Backends

All four implement `WorkspaceBackend`, which extends the deepagent
`BackendProtocol` (so a workspace also serves the deepagent filesystem
tools). Paths are relative to the backend's root or absolute inside it, and
they name the same files the backend's shell sees. Recursive walks skip
`.git`, `node_modules` and the other `SKIP_DIRS`.

| Backend | Where | Isolated | Shell |
| --- | --- | --- | --- |
| `LocalBackend(root)` | the host, files confined to `root` | no — labelled `UNISOLATED: host shell` | yes, each command in its own process group |
| `MemoryBackend(files)` | a dict | yes | no |
| `SessionBackend(session, root=...)` | any sandbox, through `SessionLike` | yes, unless you say otherwise | yes, over `exec` |
| `OpenShellBackend(client, sandbox, workspace=...)` | an NVIDIA OpenShell sandbox, root `/sandbox` | yes | yes, over `exec_stream` |

### Bring your own sandbox

A sandbox needs three methods to be a workspace:

```python
class MySandbox:
    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]: ...
    def upload_file(self, path: str, data: bytes) -> None: ...
    def download_file(self, path: str) -> bytes: ...


backend = SessionBackend(MySandbox(), root="/workspace", label="mysandbox:dev")
```

Everything else is a short POSIX shell script run with `exec`: stat, windows
of a long file, listings, search (ripgrep when the sandbox has it, `grep -P`
otherwise). Background commands are process groups the sandbox owns, started
with `setsid`, with their output in a file and their input on a FIFO, so
nothing has to stay connected while they run. The sandbox needs `sh`,
coreutils or busybox, `find`, `grep` and `awk`.

### OpenShell

```bash
pip install "tulip-agents[openshell]"
```

```python
from tulip.harness import OpenShellBackend, build_harness

backend = OpenShellBackend.connect("my-sandbox", workspace="team-a", endpoint="gateway:443")
# or: OpenShellBackend.create(workspace="team-a") — creates one, deletes it on close()
harness = build_harness(backend, wrap=gateway_gate)
```

OpenShell has no file RPC, so files move over `exec`: downloads are a `tar`
of the one file on standard output, uploads are `cat >` with the bytes on
standard input (so an executable keeps its mode), in chunks for large files.
Commands run with `execution_timeout`, so the gateway stops one that runs
too long. The `openshell` package is imported only by `connect` and
`create`; pass a client of your own to the constructor and nothing from it
is imported.

## Evidence

Every command the harness runs produces an `ExecRecord`: the command and its
output by SHA-256, the exit status, duration, whether it timed out or was
truncated, and the backend label. Digests, not text: a command line can hold
a token and its output can hold anything. Records are emitted as
`harness.exec` on the event bus (a no-op outside a run) and handed to
`HarnessConfig.on_exec`.
