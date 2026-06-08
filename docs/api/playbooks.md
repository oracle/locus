# Playbooks

Structured execution plans for agents — declared step sequences with
expected tools, validation criteria, and guidance hints. When attached
via `AgentConfig.playbook`, the `PlaybookEnforcer` hook gates each
tool call against the current step and auto-advances when the step's
`expected_tools` are exhausted.

## Models

::: tulip.playbooks.models.Playbook
::: tulip.playbooks.models.PlaybookStep
::: tulip.playbooks.models.PlaybookPlan
::: tulip.playbooks.models.StepExecution
::: tulip.playbooks.models.StepStatus

## Loader

::: tulip.playbooks.loader.load_playbook
::: tulip.playbooks.loader.PlaybookLoader
::: tulip.playbooks.loader.PlaybookLoadError

## Enforcer

The hook that holds the model to the playbook's step sequence.
Installed automatically when `AgentConfig.playbook` is set.

::: tulip.playbooks.enforcer.PlaybookEnforcer
::: tulip.playbooks.enforcer.EnforcementResult
::: tulip.playbooks.enforcer.EnforcementViolation
