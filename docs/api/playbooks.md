# Playbooks

Structured execution plans for agents — declared step sequences with
expected tools, validation criteria, and guidance hints. When attached
via `AgentConfig.playbook`, the `PlaybookEnforcer` hook gates each
tool call against the current step and auto-advances when the step's
`expected_tools` are exhausted.

## Models

::: locus.playbooks.models.Playbook
::: locus.playbooks.models.PlaybookStep
::: locus.playbooks.models.PlaybookPlan
::: locus.playbooks.models.StepExecution
::: locus.playbooks.models.StepStatus

## Loader

::: locus.playbooks.loader.load_playbook
::: locus.playbooks.loader.PlaybookLoader
::: locus.playbooks.loader.PlaybookLoadError

## Enforcer

The hook that holds the model to the playbook's step sequence.
Installed automatically when `AgentConfig.playbook` is set.

::: locus.playbooks.enforcer.PlaybookEnforcer
::: locus.playbooks.enforcer.EnforcementResult
::: locus.playbooks.enforcer.EnforcementViolation
