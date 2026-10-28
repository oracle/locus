---
title: Quickstart
description: Install Locus, run one complete Python agent, and verify the result before adding tools or policy controls.
---

# Quickstart

This path does one thing: install Locus and run a complete agent. It uses
OpenAI throughout and ends after the first successful response. The next
guide adds a policy-controlled action.

## Requirements

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Python</dt><dd>3.11–3.14</dd></div>
<div markdown="block"><dt>Locus</dt><dd markdown="span">`locus-sdk` {{ locus_sdk_version }}</dd></div>
<div markdown="block"><dt>Provider</dt><dd>OpenAI</dd></div>
<div markdown="block"><dt>Credential</dt><dd>Provider-specific API key</dd></div>
<div markdown="block"><dt>Execution mode</dt><dd>Live model call</dd></div>
</dl>

For a credential-free run, skip to [try Locus offline](#try-locus-offline).

!!! tip "OpenRouter and Together.ai"
    The same installation works. Set `OPENROUTER_API_KEY` and use
    `model="openrouter:provider/model-id"`, or set `TOGETHER_API_KEY` and use
    `model="together:organization/model-id"`. The suffix must match the exact
    id in the provider's catalog. See [OpenAI-compatible providers](../concepts/providers/openai-compatible.md#openrouter-and-togetherai).

## 1. Create an environment

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install "locus-sdk[openai]"
    export OPENAI_API_KEY="your-key"
    ```

=== "Windows PowerShell"

    ```powershell
    py -3 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install "locus-sdk[openai]"
    $env:OPENAI_API_KEY = "your-key"
    ```

On Oracle Cloud, install `locus-sdk[oci]` instead, point at an OCI config
profile, and use `oci:` model ids. See [OCI GenAI models](oci-models.md).

## 2. Save one complete file

Create `first_agent.py`:

```python
from locus import Agent, tool


@tool
def lookup_order(order_id: str) -> dict:
    """Return the current status of an order."""
    return {
        "order_id": order_id,
        "status": "delivered",
        "amount_usd": 42.50,
    }


agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_order],
    system_prompt="Answer support questions using the order lookup.",
)

result = agent.run_sync("What happened to order ORD-7842?")
print(result.message)
```

## 3. Run it

```bash
python first_agent.py
```

The wording varies because this is a live model call. A successful result
should mention order `ORD-7842`, its `delivered` status, and usually the
`$42.50` amount returned by the tool.

The run is complete: the model chose a typed Python tool, Locus executed it,
and `AgentResult.message` returned the final answer.

## Try Locus offline

The repository examples use deterministic test doubles and local stubs. Their
printed results are simulations of control flow, not evidence of live-model
quality or vendor behavior.

```bash
git clone https://github.com/oracle/locus.git
cd locus
python -m pip install -e .
python examples/notebook_71_payment_refund_gate.py
```

That example needs no API key and demonstrates a local payment stub behind an
admission policy.

## Troubleshooting

| Problem | Check |
|---|---|
| `python` is older than 3.11 | Run `python --version`; create the environment with a newer interpreter |
| `ModuleNotFoundError: locus` | Activate the same virtual environment where you installed the package |
| Provider authentication error | Confirm `OPENAI_API_KEY` exists in the shell running the script |
| Model not available to the account | Replace `gpt-4o-mini` with an OpenAI model your account can use |
| Output does not exactly match this page | Expected: live model wording is nondeterministic; check the returned facts instead |

## Continue

[Put a policy gate around an action →](first-controlled-action.md){ .md-button .md-button--primary }

After that, explore [streaming](../concepts/streaming.md),
[persistence](persist-conversations.md), [multi-agent patterns](../concepts/multi-agent.md),
or [deployment](deploy.md) as separate topics.
