---
name: edit-proposal-model
description: Use this skill when changing the Pydantic models in `src/agent/models.py` — adding fields to `AgentProposal`, introducing a new action type, changing classifications, adjusting confidence handling, or restructuring the output contract. Triggers on phrases like "add this field to the proposal", "new action type", "extend the schema", "update the agent's output", or any edit to `src/agent/models.py`. Also use proactively when reviewing a diff that touches the proposal model.
---

# Edit the agent's proposal model

`AgentProposal` is the agent's output contract. Its JSON Schema is passed to the Claude Code CLI via `--json-schema`, which uses Anthropic's constrained sampling to enforce the shape on the agent's final message. **The schema's structure determines whether the constraint actually applies** — get it wrong and the CLI silently falls back to free-form text with `structured_output=None`.

This skill is the playbook for changing the model without breaking that contract.

## Forbidden schema constructs (smoke-tested 2026-05-12)

These cause the CLI to silently ignore the schema. The agent emits arbitrary JSON, `ResultMessage.structured_output` comes back `None`, Pydantic validation fails downstream, every eval row becomes a `SCHEMA_ERR`. No error is raised; you only notice from the scorecard.

- **`oneOf` + `discriminator` together.** This is what Pydantic generates for a discriminated union (e.g. `action: UpdateStatusAction | AddNoteAction | NoOpAction = Field(discriminator="tool")`). **Don't use Pydantic discriminated unions in `AgentProposal`.** Use a flat model with optional fields + `@model_validator` (pattern below).
- **Numeric constraints** — `Field(ge=0.0, le=1.0)`, `Field(gt=0, lt=100)`. These generate `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` in the schema. Use a field description to communicate the range to the model; if you need post-validation, use a `@field_validator`.
- **String length constraints** — `Field(min_length=...)`, `Field(max_length=...)`. Same fate. Describe expectations in the field description, validate in Python if necessary.
- **Recursive schemas** — a model that references itself directly or via a cycle. Avoid.

## Allowed constructs

These work fine:

- `$defs` / `$ref` — used for enum references. Pydantic emits these automatically when you have `StrEnum` fields.
- `enum` / `Literal[...]` — closed sets for the model to choose from.
- `anyOf` — nullable fields like `str | None` produce this. Fine.
- Plain types — `int`, `float`, `str`, `bool`, `dict`, `list`.

The regression guard is in `tests/test_models.py::test_schema_has_no_forbidden_keys`. **If you add a new forbidden key, this test catches you locally before evals catch you remotely.**

## Pattern: flat model + Python validator

Instead of discriminated unions, the agent's proposal is a single class with optional per-variant fields. Cross-field rules are enforced by a Pydantic `model_validator(mode="after")` that runs right after the SDK hands us the validated dict.

```python
class AgentProposal(BaseModel):
    # shared fields
    classification: Classification
    tool: ActionTool

    # per-tool fields, all optional in the schema
    application_id: str | None = Field(default=None, description="REQUIRED when tool is update_status or add_note.")
    status: ApplicationStatus | None = Field(default=None, description="REQUIRED when tool is update_status.")
    note: str | None = Field(default=None, description="REQUIRED when tool is add_note.")
    reason: str | None = Field(default=None, description="REQUIRED when tool is no_op.")

    # ...

    @model_validator(mode="after")
    def _action_consistent(self):
        if self.tool is ActionTool.UPDATE_STATUS:
            if self.application_id is None or self.status is None:
                raise ValueError("update_status requires application_id and status")
        # ...
        return self
```

Why this works:
- The schema is flat — no `oneOf`, no `discriminator`. Constrained sampling applies.
- The model can technically emit `{tool: "update_status", reason: "..."}` (wrong combo). The Python validator catches that as a `ValidationError`, which the eval harness records as a `SCHEMA_ERR` row. That's a real signal — the prompt is failing to teach the model the cross-field rule.
- Field descriptions teach the model what each field is for. They're part of the schema the model sees during constrained generation.

## Required workflow for any model change

1. **Edit `src/agent/models.py`.** Stick to the allowed constructs above.
2. **Update `tests/test_models.py`** with cases for any new fields, validators, or enum values. Especially:
   - A round-trip test for the new shape.
   - A validator-failure test if you added cross-field rules.
   - The schema-forbidden-keys test still passes (it's automatic, but check the diff if you added anything novel).
3. **Run `uv run pytest`.** Must be green.
4. **Smoke-test against one row** before running the full eval. The cheapest check:
   ```powershell
   uv run python -m agent.loop .\samples\emails\interview-01.eml --dry-run
   ```
   Look at the printed `proposal` — does the new field show up populated? Did `structured_output` arrive (not `null`)?
5. **Run `uv run evals`.** Compare the new scorecard to the previous one. Specifically look at:
   - `n_schema_error` — a jump here means the constraint isn't applying (forbidden construct slipped in) or your new validator rejects the model's typical output.
   - `pass_rate` — went up, went down, or held?
   - `mean_cost_usd` / `mean_tool_calls` — model changes can shift behavior in subtle ways.
6. **Commit code + new scorecard snapshot together.** The commit message should state the scorecard delta in prose, e.g. "pass_rate 83% -> 100%, mean_tool_calls 3.5 -> 2.17". That's the artifact future readers diff against.

## Update checklist when you add a new field

- [ ] `models.py` — field with `default=None` if optional, type-annotated, **descriptive** docstring (the model reads it).
- [ ] `_action_consistent` validator — does the new field have cross-field rules?
- [ ] `tests/test_models.py` — round-trip + (if applicable) validator-failure test.
- [ ] `loop.py` system prompt — does the model need new instruction on when to populate the field?
- [ ] `evals/run.py` — does `evaluate_row` need to compare the new field?
- [ ] `evals/dataset.jsonl` rows — do existing rows need `expected_*` for the new field? (If yes, you're invalidating the prior scorecard — note it in the commit.)
- [ ] `writer.py` — if the new field affects what gets written to jobtrack, the writer's `_apply` needs to handle it.

## Update checklist when you add a new action type

A new action (beyond `update_status` / `add_note` / `no_op`) is a larger change:

- [ ] `models.py` — add value to `ActionTool` enum, add per-tool fields, extend `_action_consistent`.
- [ ] `loop.py` system prompt — describe when to choose the new action.
- [ ] `gating.py` — does it have special gating rules like `no_op` always drops?
- [ ] `writer.py` `_apply` — implement the MCP call.
- [ ] `evals/run.py` — `expected_action.tool` set widens; action confusion matrix grows.
- [ ] **Add labeled dataset rows** that exercise the new action. Without them, the eval can't measure whether the agent chooses it correctly.

## When NOT to use this skill

- Editing `gating.py` thresholds — that's a tuning exercise, not a schema change. Re-run evals and commit the delta but don't worry about constrained-sampling pitfalls.
- Editing the system prompt without touching `models.py` — prompt iterations are their own loop; just measure + commit + diff.
- Adding helpers to `models.py` that don't change the schema (utility functions, dump helpers, etc.).
