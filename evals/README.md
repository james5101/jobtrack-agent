# Evals

The Phase 2 feedback loop. Every prompt or guardrail change from here on runs through this harness and produces a scorecard delta. **Commit the scorecard with the change** — that delta is the learning artifact.

## Files

- `dataset.jsonl` — sanitized rows, committed.
- `dataset-real.jsonl` — unsanitized rows from real inbox emails. Gitignored. Loaded in addition to `dataset.jsonl` if it exists.
- `run.py` — the harness.
- `results/` — committed snapshots of every run, named by UTC timestamp.

## Row schema

Each line in the dataset files is one JSON object:

```json
{
  "id": "interview-01",
  "email_body": "<full .eml text>",
  "expected_classification": "interview_invite | rejection | recruiter_outreach | status_request | offer | other | spam",
  "expected_action": {
    "tool": "update_status | add_note | no_op",
    "status": "<one of jobtrack's statuses; only set when tool is update_status>"
  },
  "expected_app_id_or_null": "<jobtrack ID, or null when no match is expected>",
  "notes": "<free-text rationale for the labels>"
}
```

## Adding rows

Either hand-edit the JSONL or use the helper:

```bash
uv run python -m evals.add_row \
    --eml samples/emails/rejection-acme-real.eml \
    --id rejection-acme-real \
    --classification rejection \
    --tool update_status --status rejected \
    --app-id 2026-05-12_acme_engineer_abcd \
    --notes "form rejection; app exists in jobtrack"
```

The script picks the right dataset file from the `.eml` filename (`*-real.eml` → `dataset-real.jsonl`, else `dataset.jsonl`) and validates the labels against the closed enums. There's also a Claude skill (`.claude/skills/add-eval-row/`) that walks you through the labeling decisions interactively.

## Running

```bash
uv run evals
```

Writes a pair of files into `results/`:

- `<UTC-timestamp>.json` — aggregate scorecard.
- `<UTC-timestamp>.rows.jsonl` — per-row detail (the agent's raw output, the parsed prediction, the eval verdict).

## Pass criteria

A row passes when **all four** are true:

- `classification` matches.
- `proposed_action.tool` matches.
- `matched_application_id` matches (or both null).
- When the action is `update_status`, the `status` arg also matches.

Otherwise the row is either a `fail` (parse succeeded, prediction wrong) or a `parse_error` (couldn't extract JSON from agent output — its own bucket so prompt-formatting regressions stand out).

## Labeling conventions

- Label by what *should* happen given the email content **and the current state of the local jobtrack store.** If your jobtrack store changes between runs, eval results can drift through no fault of the prompt.
- When two labels could fit (e.g. `interview_invite` vs `recruiter_outreach` for a chatty recruiter), pick one and document the call in `notes`. The labeling decision is half the value of the dataset.
- Never edit an existing row to make a regression pass. Add a new row that captures the edge case.
- Real, unsanitized samples go in `dataset-real.jsonl`. Sanitized counterparts in `dataset.jsonl`.

## Why two dataset files

`dataset.jsonl` ships in the public repo. It cannot contain real recruiter names, emails, or company identifiers tied to live job hunts. `dataset-real.jsonl` is gitignored and holds the unsanitized rows that exercise the agent against actual inbox content. The harness loads both and the scorecard treats them identically.
