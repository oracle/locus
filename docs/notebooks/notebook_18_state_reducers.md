# State Reducers

Control how state updates combine instead of overwriting each other.
By default, when two nodes write to the same field, the second one
wins. A reducer is a function attached to a field that says how to
merge an incoming update — append to a list, sum numbers, merge
dicts, keep the max, and so on.

What you'll see:

- `Annotated[type, reducer]` on a Pydantic state schema declares the rule.
- Built-in reducers: `add_messages`, `add_numbers`, `merge_dict`,
  `append_list`, `last_value`.
- `@reducer` turns any `(current, new) -> merged` function into a custom one.
- Multiple reducers on one schema — each field merges independently.
- Two LLM-producing nodes appending to the same conversation log.

Runs on the same default (mock) as the rest of the notebooks:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_18_state_reducers.py
# or, fully offline:
TULIP_MODEL_PROVIDER=mock python examples/notebook_18_state_reducers.py
```

## Source

```python
--8<-- "examples/notebook_18_state_reducers.py"
```
