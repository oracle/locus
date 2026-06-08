# Termination

Composable stop conditions — combine with `|` (OR) and `&` (AND):

```python
from tulip.core.termination import MaxIterations, ToolCalled, ConfidenceMet

termination = (
    ToolCalled("submit") & ConfidenceMet(0.9)
) | MaxIterations(15)
```

## Base

::: tulip.core.termination.TerminationCondition

## Conditions

::: tulip.core.termination.MaxIterations
::: tulip.core.termination.TokenLimit
::: tulip.core.termination.TextMention
::: tulip.core.termination.TimeLimit
::: tulip.core.termination.ToolCalled
::: tulip.core.termination.ConfidenceMet
::: tulip.core.termination.NoToolCalls
::: tulip.core.termination.CustomCondition

## Composition operators

::: tulip.core.termination.OrCondition
::: tulip.core.termination.AndCondition
