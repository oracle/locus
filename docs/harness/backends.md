# Sandbox backends

The [coding harness](../harness.md) runs its tools against a workspace
protocol, and a sandbox is a workspace with three more things: a lifecycle,
declared limits and evidence hooks. `SandboxBackend` is that protocol. Every
backend in `locus.harness` implements it, the harness spec names one by a
short string, and a conformance kit checks a backend of your own against it.

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Protocol</dt><dd markdown="span">`locus.harness.SandboxBackend`, over `WorkspaceBackend`</dd></div>
<div markdown="block"><dt>Selection</dt><dd markdown="span">`SandboxSpec(backend="ssh" | "oci-container" | …)` and `open_sandbox(spec)`</dd></div>
<div markdown="block"><dt>Conformance</dt><dd markdown="span">`locus.harness.testing.assert_backend_conformance(factory)`</dd></div>
<div markdown="block"><dt>Warm pools</dt><dd markdown="span">`WarmPoolHint`, `SandboxPool`, `open_pool(spec)`</dd></div>
</dl>

| Backend | `backend=` | Install | Where it runs | Lifecycle |
| --- | --- | --- | --- | --- |
| `LocalBackend` | `local` | — | the host, files confined to a root | `stop` kills commands |
| `MemoryBackend` | `memory` | — | a dict; no shell | — |
| `SessionBackend` | — | — | anything with `exec`, `upload_file`, `download_file` | `stop` kills commands |
| `OpenShellBackend` | `openshell` | `[openshell]` | an [NVIDIA OpenShell](openshell.md) sandbox | `close` deletes a sandbox it created |
| `OCISandboxBackend` | `oci-sandbox` | `[oci-sandbox]` | an [OCI GenAI Sandbox](oci-sandbox.md) | `close` stops and deletes a sandbox it created |
| `SSHSandboxBackend` | `ssh` | `[ssh]` | any host with `sshd` | `close` disconnects |
| `OCIContainerInstanceBackend` | `oci-container` | `[oci-container]` | an OCI Container Instance, over SSH | `stop`/`start` restart the instance; `close` deletes it |
| `E2BBackend` | `e2b` | `[e2b]` | an E2B sandbox | `close` kills a sandbox it created |
| `DaytonaBackend` | `daytona` | `[daytona]` | a Daytona sandbox | `stop`/`start`; `close` deletes a sandbox it created |

## The protocol

`WorkspaceBackend` is files and a shell: `read_bytes`, `write_bytes`,
`stat`, `read_lines`, `ls`, `glob`, `grep`, `exec` with a timeout, and the
background-command methods. `SandboxBackend` adds:

```python
class SandboxBackend(WorkspaceBackend, Protocol):
    state: LifecycleState          # "starting" | "running" | "stopped" | "closed"
    limits: ResourceLimits         # cpus, memory_mb, disk_mb, max_exec_s, max_output_bytes

    def start(self) -> None: ...   # make it ready; a running sandbox is left alone
    def stop(self) -> None: ...    # end every command, release the compute, keep the sandbox
    def close(self) -> None: ...   # stop, then destroy what this backend created
    def add_exec_hook(self, hook: Callable[[ExecRecord], None]) -> Callable[[], None]: ...
```

Every transition is idempotent: `start` on a running sandbox, `stop` on a
stopped one and `close` on a closed one do nothing, so a caller that is not
sure what happened calls again. A step that raises leaves the state where it
was and is retried by the next call; `close` attempts the destroy step even
when stopping failed, because a sandbox that outlives the process is the
worse outcome.

Creation is a classmethod on each backend — `create()` provisions and owns,
`connect()` attaches to something that exists and never deletes it — because
what it takes is the one thing the backends do not share.

### Limits

`ResourceLimits` declares what the sandbox was given; `None` means
unbounded or unknown. `max_exec_s` is enforced by every backend: each `exec`
timeout is clamped to it. The others are enforced by whatever runs the
sandbox and are declared so a policy can read them. A backend that provisions
compute (`oci-container`) takes `cpus` and `memory_mb` from the spec at
creation.

```python
backend.limits = ResourceLimits(max_exec_s=300)
backend.exec("make test", timeout=3600)   # stopped at 300 s
```

### Evidence hooks

A hook receives an [`ExecRecord`](../harness.md#evidence) for every command
the backend runs for a caller — the command and its output by SHA-256, the
exit status, the duration, the backend label — whether or not the command
came through a harness tool. The tools still publish their own records on
the event bus; a hook is for whoever holds the backend: a runner's evidence
ledger, a gateway, a test. A hook runs after the command; one that raises
is logged and skipped, and the result still reaches the caller — the
command ran, and a listener's failure must not make the caller run it
again.

```python
records: list[ExecRecord] = []
remove = backend.add_exec_hook(records.append)
...
remove()
```

## Choosing a backend from a spec

`SandboxSpec` is the declarative form: the backend's name, a root, an
environment for every command, limits, a warm-pool hint and the backend's
own options. `open_sandbox` builds it.

```python
from locus.harness import ResourceLimits, SandboxSpec, WarmPoolHint, open_sandbox

spec = SandboxSpec(
    backend="oci-container",
    options={"compartment_id": "ocid1.compartment...", "subnet_id": "ocid1.subnet...",
             "bastion_id": "ocid1.bastion..."},
    limits=ResourceLimits(cpus=2, memory_mb=4096, max_exec_s=600),
    warm_pool=WarmPoolHint(ready=2),
)
backend = open_sandbox(spec)
```

`SandboxSpec.from_mapping` reads the same thing from a parsed YAML or JSON
document and refuses a key it does not know, so a typo does not fall back to
the host:

```yaml
sandbox:
  backend: ssh
  root: /srv/work
  env: {CI: "1"}
  limits: {max_exec_s: 600}
  options: {host: build-7.internal, username: agent, key_filename: ~/.ssh/agent}
```

`options` are the keyword arguments of the backend's constructor. Each
built-in name picks `connect` when the option that names an existing sandbox
is present (`instance_id`, `sandbox_id`, `sandbox`) and `create` otherwise;
`ssh` always connects. `register_backend(name, factory)` adds a name of your
own; the factory receives the whole spec.

### Warm pools

Creating a remote sandbox can take a minute, which makes a poor interactive
tool. `WarmPoolHint` says how many to have ready:

- `ready=N` is honoured on this side by `SandboxPool`: `open_pool(spec)`
  keeps `N` sandboxes created on worker threads, `acquire()` hands one over
  and starts its replacement, `release(backend)` closes it — a workspace an
  agent has worked in is not clean, and the next run expects a fresh one.
- `name` is a pool the service itself keeps, passed on at creation. OCI
  GenAI Sandboxes have [warm pools](oci-sandbox.md#beyond-the-harness) per
  project; the `oci-sandbox` backend sends the name as `warmpool_name`.

```python
pool = open_pool(spec)          # spec.warm_pool.ready sandboxes, created now
backend = pool.acquire(timeout=120)
try:
    harness = build_harness(backend, wrap=gate)
    ...
finally:
    pool.release(backend)       # closed; a replacement is already warming
pool.close()
```

## The conformance kit

`locus.harness.testing` checks a backend against the contract the tools
assume and the protocol above. It needs nothing from pytest, so it runs from
a test or from a script against a live sandbox:

```python
from locus.harness.testing import assert_backend_conformance

def test_my_backend_conforms():
    assert_backend_conformance(lambda: MyBackend.create(...))
```

The factory is called once; the backend is closed at the end whatever
happened, and `close()` is itself under test: the state must be `closed`
afterwards and a second `close()` must be quiet. Every check runs even after
one fails, and the `AssertionError` names each failed check and what it saw.
The checks, in order:

- **Files.** Bytes round-trip exactly; parent directories are created; a
  missing file is `file_not_found` and a directory is `is_directory`; no
  path resolves outside the root; `stat` hashes content; `read_lines`
  numbers, windows and cuts long lines; CRLF is kept on disk and stripped in
  windows; `ls`, `glob` and `grep` behave as the tools expect and skip
  `.git`, `node_modules` and the other `SKIP_DIRS`.
- **Shell.** `exec` returns the status and both streams in order, runs in
  the root with stdin closed, takes a working directory and environment,
  streams output, and does not take an exit status of 124 for a timeout.
- **Deadline.** A command past its timeout is killed where it runs and
  reported as `timed_out` with no exit status. (`timeouts=False` leaves the
  slow check out.)
- **Background commands.** One takes input until it is closed, can be
  killed with its children, is listed and released; a foreground job has no
  stdin.
- **Protocol.** The lifecycle transitions, hooks see every command, and a
  declared `max_exec_s` bounds `exec`.

`run_conformance(backend)` runs the same checks on an open backend without
closing it and returns a `ConformanceReport` with `passed`, `failed` and
`skipped`; a backend with no shell has the shell checks skipped, one that is
not a `SandboxBackend` the protocol checks. The kit writes under one
directory it creates in the workspace root and removes at the end.

## SSH

`pip install "locus-sdk[ssh]"`. Any host you can log into is a workspace:

```python
from locus.harness import SSHJump, SSHSandboxBackend

backend = SSHSandboxBackend.connect(
    "build-7.internal",
    username="agent",
    key_filename="~/.ssh/agent",
    root="/srv/work",                      # created if missing; the login directory when omitted
    via=SSHJump("bastion.internal", username="agent"),   # optional jump host
)
```

What the host needs: `sshd` with the `sftp` subsystem (the backend falls
back to `cat` over `exec` without it), and the tools every Linux image has —
`sh`, coreutils or busybox, `find`, `grep`, `awk`. On a host whose
non-interactive login `PATH` lacks them (macOS without Homebrew's coreutils
on the `PATH`), `path=` sets the `PATH` for every command, the backend's own
scripts included.

Three decisions are made on this side rather than left to the host:

- **A command is stopped at its deadline where it runs.** Closing an SSH
  channel does not kill the process behind it, so every command runs under a
  watchdog on the host: the command in its own process group (`setsid`
  where util-linux has it, POSIX job control elsewhere), a sibling that
  sends the group `TERM` at the deadline and `KILL` two seconds later.
- **Both streams, in order.** The command's standard error is joined to its
  standard output on the host, so a warning sits where it was written.
- **Host keys are checked.** By default the host must be in the system's or
  the user's `known_hosts`. `host_key="<type> <base64>"` pins one key for a
  host whose key arrived another way; `known_hosts="accept-unknown"` is the
  explicit opt-out for a sandbox with no other channel. A key accepted that
  way is kept in memory for the session: the backend never writes the
  user's `known_hosts`, and a file it cannot parse (an `@cert-authority`
  line) is skipped with a warning rather than failing the connection.

`connect()` labels the workspace `ssh:user@host[:port]` and treats
`localhost` as not isolated; pass `isolated=` to say otherwise.

## OCI Container Instances

`pip install "locus-sdk[oci-container]"`. An
[OCI Container Instance](https://docs.oracle.com/en-us/iaas/Content/container-instances/home.htm)
runs a container image on serverless compute in your tenancy, on a VNIC in a
subnet you choose, with the CPUs and memory you ask for. The service has no
command API, so the backend provisions the instance with the OCI SDK and
works in it over SSH:

```python
from locus.harness import OCIContainerInstanceBackend

backend = OCIContainerInstanceBackend.create(
    "ocid1.compartment...",
    "ocid1.subnet...",                 # a private subnet with a NAT gateway
    bastion_id="ocid1.bastion...",     # an OCI Bastion in the same VCN
    ocpus=1, memory_gb=2,
)
...
backend.close()                        # the instance is deleted
```

`create()` generates an SSH key pair for this backend alone, in memory, and
starts the container with a bootstrap command that installs `sshd` if the
image lacks it, authorizes the key for `root`, creates the workspace,
**prints the host's public keys to the container log** and runs `sshd` in
the foreground. The backend reads the keys back through the service API and
connects with them pinned, so there is no trust-on-first-use: a log without
a key is an error, not an unverified connection. With `bastion_id`, a
port-forwarding session to the instance's private IP is created with the
same key pair and the connection is a channel through the bastion, with the
bastion's host key pinned the same way: the key the service reports or — it
reports none for a port-forwarding session — the one in your `known_hosts`,
or else the key the endpoint presents at the first connection, read once and
kept for the session (an `INFO` line says which); that is the shape for an
instance in a private subnet, which is where a sandbox belongs. Without it, `public_ip=True` assigns a public IP on a public
subnet, or the private IP is used as is for a process that can reach it.
`known_hosts="accept-unknown"` is the explicit opt-in to connect anyway when
the container or the bastion reports no key, with a warning; nothing is
written to the user's `known_hosts` either way.

| Option | Default | What it does |
| --- | --- | --- |
| `image` | Oracle Linux 9 from Oracle's registry | Any image with `sshd` or with `dnf`, `microdnf`, `apt-get` or `apk` to install it |
| `shape`, `ocpus`, `memory_gb` | `CI.Standard.E4.Flex`, 1, 2 | The instance's compute; reported as `limits` |
| `bastion_id` | none | Connect through an OCI Bastion; the session is deleted on close |
| `bastion_session_ttl` | 3 h | How long the bastion session lives; a restart after it expired makes a new one |
| `public_ip` | `False` | Assign a public IP and connect to it |
| `ssh_port` | 22 | The port `sshd` listens on in the container |
| `known_hosts` | `"strict"` | Require the container's host keys and pin the bastion's; `"accept-unknown"` connects unverified when one is missing |
| `container_env` | none | Environment variables set on the container |
| `ready_timeout` | 600 s | For the instance to run and `sshd` to answer; installing it on the default image takes about a minute |

Prerequisites:

1. **A subnet with egress** for the bootstrap to install `sshd` — a NAT or
   internet gateway on its route table — unless the image already has it.
   For the bastion path, the subnet's security rules must admit TCP on the
   SSH port from the bastion's subnet (the VCN's CIDR does).
2. **An OCI Bastion** in the VCN for a private subnet, with this machine's
   address — as the bastion sees it, which is not always what a web lookup
   shows — in its client CIDR list. The bastion checks the list after it
   has accepted the session's key and closes the connection when the
   address is not on it; the backend retries that for `AUTH_GRACE` (90 s)
   and then raises, naming the bastion and quoting the container log, so a
   failure there says so instead of running out the ready timeout. Creating
   a bastion is a one-time step; the backend only creates sessions on it.
3. **IAM** for the user or principal that runs the backend: `manage
   compute-container-family` and `use virtual-network-family` in the
   compartment, `read availability-domains` on the tenancy when no AD is
   given, and `manage bastion-session` with `read bastion-family` for the
   bastion path.
4. **Authentication**, the same choices as the rest of the SDK: a
   `~/.oci/config` profile with an API key or a session token, an instance
   principal, or a resource principal.

`stop()` stops the instance and `start()` starts it again; **the
container's filesystem does not survive a stop** — the service restarts the
container from its image, the bootstrap runs again and the backend
reconnects to a fresh workspace, re-pinning the host keys of the new boot
(the same keys, for an image that carries its own). `close()` on a running
instance deletes it as it is, without stopping it first.
`connect(instance_id, pkey=...)` attaches to an instance you provisioned
yourself (one that runs `sshd` and accepts the key) and never deletes it.

## E2B and Daytona

`pip install "locus-sdk[e2b]"` or `"locus-sdk[daytona]"`. Two hosted sandbox
services with Python clients whose command and file calls fit
`SessionLike`; `E2BBackend` and `DaytonaBackend` are thin adapters over
them, with `create()` owning the sandbox and `connect()` attaching to one.

```python
from locus.harness import E2BBackend, DaytonaBackend

e2b = E2BBackend.create(template="base", lifetime_s=600)        # E2B_API_KEY
daytona = DaytonaBackend.create(image="python:3.12", lifetime_minutes=60)  # DAYTONA_API_KEY
```

Both take the service's API key from the environment or as `api_key=`, and
both set a lifetime on the sandbox as the backstop for one a crashed process
leaves behind. Daytona sandboxes can be stopped and started, and keep their
files across a stop; E2B sandboxes are killed on `close()`.

These two adapters are written against the clients' documented surfaces and
checked with stand-ins for them in the unit tests. Running them against the
services needs an account; the live tests under `tests/integration` do so
when `E2B_API_KEY` or `DAYTONA_API_KEY` is set, and are the check to run
before relying on either adapter.

## Writing a backend

A sandbox that can run a command and move a file is three methods away from
the protocol:

```python
from locus.harness import SessionBackend

class MySession:
    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]: ...
    def upload_file(self, path: str, data: bytes) -> None: ...
    def download_file(self, path: str) -> bytes: ...


class MyBackend(SessionBackend):
    def __init__(self, handle, **options):
        super().__init__(MySession(handle), root="/workspace", label=f"mine:{handle.id}", **options)
        self._handle = handle

    def _stop(self) -> None:      # end commands, release compute
        super()._stop()
        self._handle.pause()

    def _start(self) -> None:     # resume
        self._handle.resume()

    def _destroy(self) -> None:   # delete what create() made
        self._handle.delete()
```

`SessionBackend` builds the whole workspace over the three methods — stat,
windows, listings, search and background commands are short POSIX scripts
run through `exec` — and `SandboxMixin`, which every backend carries, turns
the three hooks into the lifecycle, records every command for the hooks and
clamps timeouts to the limits. Then run the conformance kit against it, and
`register_backend("mine", factory)` to name it in a spec.
