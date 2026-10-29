# OCI GenAI Sandboxes

An OCI GenAI Sandbox is a managed, temporary Linux workspace in your OCI
tenancy: its own file system, network namespace and process space, created in
seconds inside a Generative AI project and ended by the service when its
lifetime runs out. `OCISandboxBackend` makes one the workspace of the
[coding harness](../harness.md): the agent's read, write, edit, search and
shell calls run inside the sandbox instead of on the machine that runs the
agent, and every call still passes through the harness's single gate.

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Install</dt><dd markdown="span">`pip install "locus-sdk[oci-sandbox]"`</dd></div>
<div markdown="block"><dt>Workspace</dt><dd markdown="span">`/workspace`, the one writable directory</dd></div>
<div markdown="block"><dt>Lifetime</dt><dd>Set at creation, up to 48 hours, extendable while it runs</dd></div>
<div markdown="block"><dt>Shapes</dt><dd markdown="span">`SMALL`, `MEDIUM`, `LARGE`; the managed `python-3.11` runtime or your own OCI Registry image</dd></div>
<div markdown="block"><dt>Regions</dt><dd markdown="span">[Where sandboxes are available](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-regions.htm); a sandbox lives in its project's region</dd></div>
</dl>

{{ locus_diagram harness-sandbox }}

## What a sandbox gives the harness

- **Isolation from the host.** Commands run in an isolated Linux runtime
  (gVisor) on Oracle-managed infrastructure. A `pip install`, a test suite or
  a script the model wrote touches nothing on the machine that runs the agent.
- **An end built in.** The sandbox expires on its own, so one that a crashed
  process leaves behind does not run on. `close()` stops and deletes it
  sooner.
- **Networking you set once.** Outbound access is the project's policy:
  Oracle-managed with allow-all, deny or selected domains, or your own
  subnet. With egress allowed, `pip install` inside the sandbox works.
- **OCI from inside, by policy.** Code in the sandbox can call OCI services
  with the sandbox's own resource principal, scoped by an IAM policy to the
  project it came from, without a credential in the workspace.
- **The same tools and the same gate.** `build_harness` does not change, the
  action labels do not change, and `wrap` still sees every call.

## Prerequisites

1. **A Generative AI project with sandboxes enabled**, in a
   [supported region](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-regions.htm).
   Enable sandboxes when you
   [create the project](https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-project.htm)
   and choose its outbound networking then; the networking mode is fixed for
   the project's life. The project's OCID is what `create()` takes.
2. **IAM.** Creating and deleting sandboxes needs `manage`; running commands
   and moving files in one that exists needs `use`:

    ```text
    allow group <group-name> to manage generative-ai-sandbox in compartment <compartment-name>
    allow group <group-name> to use generative-ai-project in compartment <compartment-name>
    ```

    The
    [permissions page](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-permissions.htm)
    lists the operation behind each verb.

3. **Authentication**, the same choices as the rest of Locus: an
   `~/.oci/config` profile with an API key or a session token, an instance
   principal, or a resource principal.
4. **The SDK with the sandbox client.** The `oci-sandbox` extra pins an
   `oci` release that ships `oci.generative_ai_sandbox`. It is imported when
   a backend is first used, so `import locus.harness` does not need it.

## Create, connect, extend, close

```python
from locus.harness import OCISandboxBackend

backend = OCISandboxBackend.create(
    "ocid1.generativeaiproject.oc1...",   # a project with sandboxes enabled
    region="eu-frankfurt-1",
    profile="DEFAULT",                     # or auth_type="instance_principal"
    expiration="PT1H",                     # ISO 8601, at most 48 hours
)
print(backend.sandbox_id)                  # sbx-...

backend.extend("PT30M")                    # the new expiry time comes back
backend.close()                            # stopped and deleted
```

`create()` makes the sandbox, waits until it is `RUNNING` and owns it:
`close()` stops and deletes it, and so does any failure between creation and
the return. `connect()` attaches to a sandbox that already exists and never
deletes it; its `close()` only ends the harness's background commands.

```python
backend = OCISandboxBackend.connect("sbx-...", region="eu-frankfurt-1")
```

| Option | Default | What it does |
| --- | --- | --- |
| `runtime` | `python-3.11` | The managed runtime, or a same-region OCI Registry image |
| `shape` | `SMALL` | `SMALL`, `MEDIUM` or `LARGE` |
| `expiration` | `PT1H` | The sandbox's lifetime as an ISO 8601 duration, sent as whole minutes; a lifetime that never ends is refused |
| `warmpool_name` | none | A warm pool of the project, for a sandbox that starts from reserved capacity |
| `display_name` | `locus-harness-…` | How the sandbox is listed in the project |
| `env` | none | Variables added to every command's environment |
| `max_output_bytes` | 900 kB | The most output one command returns |
| `ready_timeout`, `poll_interval` | 300 s, 2 s | How long `create()` waits for `RUNNING` |

Pass a `GenerativeAiSandboxClient` of your own to the constructor when the
client needs settings these options do not cover; `connect` and `create`
build one for you.

## An agent in a sandbox

```python
from locus.agent import Agent
from locus.control import ControlPolicy, gate_tool
from locus.harness import OCISandboxBackend, build_harness

backend = OCISandboxBackend.create("ocid1.generativeaiproject.oc1...", expiration="PT1H")
policy = ControlPolicy(require_human_for=frozenset({"workspace.exec"}))
harness = build_harness(
    backend,
    wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec),
)
agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=harness.tools,
    system_prompt=harness.prompt_fragment,
)
try:
    result = agent.run_sync("Add pytest tests for calc/ and run them.")
    print(result.message)
finally:
    backend.close()
```

The model writes files, runs `python -m pytest` and reads the output, all
inside the sandbox. The policy above holds every `workspace.exec` for a
person; `workspace.read` and `workspace.write` calls go through. Drop `wrap`
only when the sandbox itself is the boundary you want, as in a throwaway
run; `build_harness` warns when a host shell is left ungated, and says
nothing here, because the sandbox is isolated.

## How the service shapes the backend

The sandbox API runs commands and moves files; `OCISandboxBackend` is a thin
adapter over it, and four properties of the service decide how:

- **Commands stream.** Each command's events carry a sequence number, so
  standard output and standard error come back in the order the command
  wrote them.
- **The deadline is enforced inside.** Every command runs under `timeout`
  inside the sandbox, so a command that overruns stops at the harness's
  deadline; the service's own timeout is a later backstop. A command that
  did not finish reports no exit status and `timed_out`.
- **Output is capped before it leaves.** Each command writes to a file in
  the sandbox and at most `max_output_bytes` come back, the head and the
  most recent tail with a note of what was dropped, because the service
  returns nothing for output far above a megabyte.
- **Only `/workspace` is writable.** `/tmp` is read-only, so background
  commands keep their state in `/workspace/.locus-harness`, which recursive
  walks skip.

Files use the file API: every write carries a base64 SHA-256 the service
checks, and a write above 8 MiB goes in parts joined inside the sandbox,
because the service refuses very large bodies.

## Beyond the harness

`backend.client` is the SDK's `GenerativeAiSandboxClient`, so the rest of
the service is a call away when a workflow needs it:

- **Snapshots** save the workspace files and start a new sandbox from them,
  so a prepared environment does not have to be rebuilt each time
  ([snapshots](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-snapshots.htm)).
- **Environment variables** set at creation reach every command.
- **Custom images** from OCI Registry replace the managed runtime
  ([custom images](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-custom-images.htm));
  the harness's own commands expect GNU `timeout` and `coreutils` in the
  image.
- **Warm pools** keep idle capacity ready so `create()` returns sooner
  ([warm pools](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-warm-pools.htm)).
- **Published ports** expose a service running in the sandbox at a URL with
  no authentication of its own; publish only what you need.
- **Suspend and resume** are available on projects with workspace persistence
  enabled.

## Evidence

Every command the harness runs in the sandbox produces an `ExecRecord`, the
command and its output by digest, the exit status, duration, whether it timed
out or was truncated, and the backend label (`oci-sandbox:<region>/<id>`).
See [Evidence](../harness.md#evidence).

## Oracle documentation

- [Sandboxes in OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandboxes.htm)
- [Creating and using a sandbox](https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-sandbox.htm)
- [Runtimes and resource limits](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-limits.htm)
- [Permissions](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-permissions.htm)
- [Networking](https://docs.oracle.com/en-us/iaas/Content/generative-ai/sandbox-networking.htm)
- [Projects](https://docs.oracle.com/en-us/iaas/Content/generative-ai/projects.htm)
