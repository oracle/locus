# Agent with Tools

Plain Python functions, decorated with `@tool`, become things the agent
can call. The model decides when to use them; Locus runs them and feeds
the result back. This is what turns an LLM into an agent.

What you'll learn:

- Turning a Python function into a tool with `@tool`.
- Passing tools to `Agent(tools=[...])`.
- Watching `ToolStartEvent` and `ToolCompleteEvent` in the stream.
- Tools with optional arguments, default values, and structured return
  types.

Run it:

```
.venv/bin/python examples/notebook_03_agent_with_tools.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Here the agent works a deployment-readiness scenario: it looks up a
container image digest in the build registry, pulls a service's DNS and
health record, and only then calls go or no-go. The data is fictional —
`example.com` hostnames, placeholder digests, and made-up service names.

Prerequisite: the agent-basics notebook.

## Source

````python
--8<-- "examples/notebook_03_agent_with_tools.py"
````
