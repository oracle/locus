# Codex scenarios

A series of tasks that Codex builds with Locus from the skill alone, each with
an automatic check. Together they exercise the SDK the way an application
does: a gated tool and its audit trail, a hold a person approves and a run
that resumes, a `StateGraph` fan-out, a retrieval tool, the coding harness
behind the gate, a checkpointed conversation, an agent served over HTTP, an
orchestrator with specialists, a typed answer with a termination rule and an
evaluation, an agent on OCI Generative AI, the permission rules of a coding
agent, and a playbook with a custom hook.

| Directory | What Codex builds | Live |
|---|---|---|
| `01-gated-refund-agent` | `gate_tool` with an argument-derived `Action`, allow / hold / deny on the trail, tests on `AgentTestClient` | |
| `02-hold-and-resume` | `on_refusal="interrupt"`, `FileApprovals`, `FileCheckpointer`, `agent.resume(..., perform_dangling=True)`; the refund runs once | |
| `03-stategraph-fanout` | `StateGraph` with `Send` fan-out and a merge node, `get_mermaid()` | |
| `04-rag-grounded-tool` | `RAGRetriever` over `InMemoryVectorStore` with an embedding double, `create_rag_tool` | |
| `05-harness-gated-workspace` | `build_harness(MemoryBackend, wrap=gate_tool)`: reads pass, writes held, `preview` | |
| `06-checkpointed-conversation` | `FileCheckpointer` and `thread_id` across agent instances | |
| `07-serve-and-invoke` | `AgentServer` with an API key, tested with FastAPI's test client | |
| `08-orchestrator-specialists` | `create_orchestrator` with two `Specialist`s on doubles | |
| `09-typed-answer-eval` | `output_schema`, `MaxIterations \| ToolCalled`, `EvalRunner` | |
| `10-oci-live-agent` | an agent on `oci:openai.gpt-5.5` with one tool, one live call | yes |
| `11-permission-rules` | `PermissionRules`, `verdict_action`, `VERDICT_POLICY`, `parse_command` | |
| `12-playbook-and-hooks` | `Playbook` with `PlaybookEnforcerHook` and a custom `HookProvider` | |

## Run them

```bash
pip install "locus-sdk[oci]"                         # the locus command, with the skill
python codex/scenarios/run.py --skip-live            # the offline series
python codex/scenarios/run.py                        # with the live scenario; OCI_PROFILE and OCI_REGION set
python codex/scenarios/run.py --codex-profile oci    # Codex itself on OCI Generative AI (examples/codex-on-oci)
python codex/scenarios/run.py --only 02-hold-and-resume
```

For each scenario the runner makes an empty project directory under
`out/scenarios/<name>`, starts `locus codex run --exec` there with a short
preamble (create `.venv`, install Locus and pytest, work offline unless told
otherwise, never print a credential, report at the end) followed by the
scenario's `PROMPT.md`, and when Codex returns runs the scenario's `check.sh`
in that directory: the tests Codex wrote must pass, and the behaviour the task
asked for is exercised once more from outside. Transcripts, Codex's report and
the check output are kept under `out/scenarios/_runs/<name>`; the summary
table is printed and written to `out/scenarios/_runs/summary.md`. The exit
status is the number of scenarios that did not pass.

`--locus-install "/path/to/checkout[oci]"` makes Codex install Locus from a
checkout instead of PyPI, which is how a change to the SDK or to the skill is
tried before a release. `--parallel 3` runs three at a time; each Codex run
takes a few minutes.

## Add one

Make a directory `NN-slug/` with a `PROMPT.md` that names the files to write
and the behaviour the tests must prove, and a `check.sh` that runs from the
project root with `.venv/bin/python`, exits non-zero on failure, and prints a
line beginning `ok:` and a space for each thing it confirmed. Add a `live` file when
the task makes a model call. Keep the task buildable from the skill: it should
name Locus concepts, not file paths inside the SDK.
