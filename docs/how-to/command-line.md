---
title: The locus command
description: "Every subcommand of the locus command line, run, resume, approvals, audit, policy, serve, mcp, a2a, codex, eval and deploy, with its actions and exit codes, read from the command itself at build time."
---

# The `locus` command

The `locus` command runs an agent, decides what it is waiting on and checks
its record. It installs with the package; `locus --help` prints this table's
first column and `locus <command> --help` the options of one command.

`AGENT`, where a command takes one, is `module:attr` or `path/to/file.py:attr`,
naming an `Agent` or a zero-argument function that returns one:

```bash
locus run app.py:agent "refund order 4821" --thread t1
locus approvals --store approvals.json list
locus resume app.py:agent --thread t1 --answer approved --perform
locus serve app.py:agent --port 8000
```

## Commands

{{ locus_cli_reference }}

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | A check or a decision failed: a gate did not pass, a verification found a problem, an approval was rejected |
| `2` | Usage error |
| `3` | The run is paused, waiting for a person; `locus resume` continues it |

## See also

- [Quickstart](quickstart.md): the first install and a first run.
- [First controlled action](first-controlled-action.md): the hold, the approval and the resume, end to end.
- [Build it with Codex](codex.md): `locus codex` and the Locus skill.
- [Deploy to OCI](deploy-oci.md): `locus deploy` from `init` to `destroy`.
