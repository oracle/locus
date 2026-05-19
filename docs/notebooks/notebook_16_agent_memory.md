# Agent Memory

Give an agent a checkpointer and every conversation turn is persisted to
a real database. Restart the process, attach a new agent to the same
ADB and the same `thread_id`, and the conversation resumes — messages,
tool history, confidence score and all.

What you'll learn:

- Building an `oracle_checkpointer` against an Autonomous Database 26ai.
- Keying conversations with `thread_id`.
- Writing a checkpoint after every iteration so a crash mid-tool-call
  still recovers.
- Loading the saved `AgentState` and inspecting it field by field.
- Running many independent threads against a single ADB.

This notebook does not fall back to in-memory storage — Oracle is the
only backend exercised here.

Run it:

```
export ORACLE_DSN=mydb_low
export ORACLE_USER=locus_app
export ORACLE_PASSWORD='<app-password>'
export ORACLE_WALLET=~/.oci/wallets/mydb
export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
.venv/bin/python examples/notebook_16_agent_memory.py
```

If those env vars are unset the script prints a skip banner and exits
0 — convenient for CI. The agent's model goes through the default OCI
Generative AI provider (canonical id: `openai.gpt-4.1` or
`meta.llama-3.3-70b-instruct`). For offline runs set
`LOCUS_MODEL_PROVIDER=mock`; OpenAI, Anthropic and Ollama also work as
the LLM.

## Source

```python
--8<-- "examples/notebook_16_agent_memory.py"
```
