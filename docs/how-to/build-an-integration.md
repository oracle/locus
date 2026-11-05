# Build an integration

An integration is a small module that implements the core **`SecurityAdapter`**
contract and reuses the core toolkit. The adapter pattern is general — it fits
any vendor system an agent reads from or acts on; the protocol lives in
the `locus-sdk-security` package (`locus_security`), and new domain packages are welcome.
An integration lives in your own distribution, depends one-way on
`locus-sdk`, and is discovered by explicit import.

Place it in the **domain package that fits your vendor** — `siem/`, `edr/`,
`identity/`, `threat_intel/`, `vuln/`, `compute/`, `notify/`, or `soar/` — one
canonical home per integration (add a new domain package if none fits).

## The contract

```python
from locus_security import SecurityAdapter, ToolAdapter  # the protocol + a concrete

# A SecurityAdapter is just: name, vendor, tools() -> list[Tool]
```

## Five steps

1. **A pure function** that calls the vendor API on the live path (credentials
   from the environment via `locus_security.env(...)`) and returns a
   deterministic, benign **offline sample** when no credentials are set — so it
   runs in CI with no network. Keep the same return shape on both paths.
2. **An `async @tool`** wrapper that returns `locus_security.as_json(...)`.
3. **A `*_adapter()` factory** returning a `ToolAdapter`
   (`name`, `vendor`, `_tools=[…]`).
4. **(If it asserts about an asset)** build a GSAR partition — GSAR is the SDK's
   evidence check: a claim must be backed by tool output, otherwise the agent
   abstains — with `locus_security.tool_match` / `inference_claim` and route it
   through `locus.control.ground_finding`, so an ungrounded result abstains. (For
   threat-intel indicators, `locus_security.indicator_type` maps a coarse kind —
   `"ip"` / `"domain"` / `"hash"` — to the typed enum.)
5. **An optional extra** in `pyproject.toml` (`<area>-<vendor>`) if the live
   path needs a vendor SDK. The offline path must need nothing beyond core.

## Conformance — required

Every adapter must pass the core conformance kit (`locus_security.testing`) —
the shared contract test every integration is held to:

```python
from locus_security.testing import assert_adapter_conformance, assert_tool_returns_json
from acme_locus.siem import acme_adapter, acme_tool   # your domain package

def test_conforms():
    assert_adapter_conformance(acme_adapter())

async def test_tool_json():
    await assert_tool_returns_json(acme_tool, "…")   # offline path
```

Pass the kit, and exercise the live path in `tests/test_live_paths.py`.

## Wire it in

```python
from locus_security import security_toolset
from acme_locus.siem import acme_adapter

tools = security_toolset(extra=acme_adapter().tools())
```

## Community playbooks

Drop a `Playbook` factory in your package (or a YAML loaded with
`locus.playbooks.load_playbook`) referencing your tool names plus core's bundled
tools in `expected_tools`.

## Rules

- **One-way dependency.** Import from `locus` (core); never make core depend on
  your integration.
- **Offline by default.** No test may require credentials or network.
- **BYO-credentials.** Read credentials from the environment; never hardcode them.
