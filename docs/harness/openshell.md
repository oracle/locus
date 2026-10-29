# NVIDIA OpenShell

[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) runs agent sandboxes
behind a gateway, scoped to a workspace: each sandbox is a container the
gateway starts and confines, and a client reaches it only through the
gateway. `OpenShellBackend` makes one the workspace of the
[coding harness](../harness.md); the Locus gateway and the
[box runner](../runner.md) run the same tools in the same sandboxes.

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Install</dt><dd markdown="span">`pip install "locus-sdk[openshell]"`</dd></div>
<div markdown="block"><dt>Workspace</dt><dd markdown="span">`/sandbox`</dd></div>
<div markdown="block"><dt>Gateway</dt><dd markdown="span">Yours, or `examples/openshell-gateway/` on any Docker host</dd></div>
<div markdown="block"><dt>Credentials</dt><dd>Providers keep them at the gateway; the sandbox sees a placeholder</dd></div>
</dl>

{{ locus_diagram harness-sandbox }}

## Connect or create

```python
from locus.harness import OpenShellBackend, build_harness

backend = OpenShellBackend.connect("my-sandbox", workspace="team-a", endpoint="gateway:443")
# or: OpenShellBackend.create(workspace="team-a") — creates one, deletes it on close()
harness = build_harness(backend, wrap=gateway_gate)
```

`connect()` attaches to a sandbox that exists; `create()` makes one in the
workspace, waits until it is ready and deletes it on `close()`. `create()`
also takes `image=`, the container image the sandbox runs, `providers=`, the
OpenShell providers to attach, and `labels=`. The `openshell` package is
imported only by these two constructors; pass a configured `SandboxClient`
of your own to the constructor and nothing from it is imported.

## A gateway to try it

`examples/openshell-gateway/` starts a gateway with one `docker compose up -d`
on a machine with Docker Engine 28 or newer: the NVIDIA gateway image bound
to loopback, creating sandboxes as sibling containers through the host's
Docker socket. `LOCUS_OPENSHELL_ENDPOINT=127.0.0.1:8080` then points the
harness, and the live harness tests, at it.

The sandbox binary confines itself with Landlock and exits at start-up unless
the kernel is 6.2 or newer and Landlock is in the active LSM list; the
example's README has the check and the fix.

## Providers: credentials the sandbox never holds

A provider keeps a credential at the gateway. The sandbox sees a placeholder
that the sandbox proxy swaps for the real value only at the provider's
endpoints, so a command inside the sandbox can call a model without ever
holding the key. OpenShell's `oci-genai` profile does this for the OCI
Generative AI OpenAI-compatible endpoint:

```python
backend = OpenShellBackend.create(
    workspace="team-a",
    image="docker.io/library/buildpack-deps:noble-curl",   # the default image has no curl
    providers=["oci-genai"],
)
backend.exec(
    "curl -sS https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    "/openai/v1/chat/completions"
    ' -H "Authorization: Bearer $OCI_GENAI_API_KEY"'
    ' -H "Content-Type: application/json" --data @body.json',
    timeout=120,
)
```

`examples/openshell-gateway/provider.py` imports a profile and creates a
provider at the gateway.

## How files and commands move

OpenShell's client runs a command, `exec_stream`, and that is all it offers
for the sandbox's contents: there is no file RPC. So files move over `exec`
too. A download is a `tar` of the one file on standard output, read back as an
archive, so the bytes are exact and the archive says whether the path was a
directory. An upload is `cat >` with the bytes on standard input, in chunks
for large files, rather than `tar -x`, so an edit to an executable script
keeps its mode.

Commands run with `execution_timeout` set, so the gateway, not the agent's
process, stops a command that runs too long; it reports exit status 124 and
the harness reports `timed_out`.

## Evidence

Every command produces an `ExecRecord` with the backend label
`openshell:<workspace>/<sandbox>`. See [Evidence](../harness.md#evidence).
