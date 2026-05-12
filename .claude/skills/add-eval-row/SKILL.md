---
name: add-eval-row
description: Use this skill when the user wants to add a labeled email to the jobtrack-agent eval dataset. Triggers on phrases like "add this to evals", "label this email", "make an eval row from <file>", "row for this rejection", or when the user drops a new `.eml` file in `samples/emails/` and mentions evals or labeling. Also use after an agent run where the user explicitly disagrees with the prediction and wants to capture the case.
---

# Add eval row

The dataset (`evals/dataset.jsonl` + `evals/dataset-real.jsonl`) is the spec for what the agent should do. Adding rows is how the user grows the feedback loop. This skill walks them through the labeling decisions and appends a row to the right dataset file via `evals/add_row.py`.

## Required inputs (gather before running the helper)

Ask the user for any of these you don't already have:

- **`.eml` path** — usually under `samples/emails/`. If the user describes an email they haven't saved yet, suggest they save it first; don't paraphrase an email into a synthetic .eml.
- **row id** — short, unique within the target dataset. Convention: `<classification>-<company>[-real]`, lowercase, hyphen-separated. Example: `rejection-stripe`, `interview-valon-real`.
- **classification** — one of: `interview_invite`, `rejection`, `recruiter_outreach`, `status_request`, `offer`, `other`, `spam`. If the email is ambiguous, *ask the user* — half the dataset's value is the user's explicit labeling decision, not your inference.
- **action tool** — `update_status`, `add_note`, or `no_op`. `no_op` is correct when there's no matching application or the email isn't actionable. Default to `no_op` if uncertain; surfacing uncertainty is the agent's goal.
- **status** (required iff tool is `update_status`) — one of: `applied`, `screening`, `interviewing`, `offer`, `rejected`, `ghosted`, `withdrawn`.
- **app id** — the jobtrack application ID this email refers to, or the literal string `null` if no application matches. Two ways to find it:
  - Use the jobtrack MCP `search_applications` tool if the connection is available in the current session.
  - Otherwise run `uv run python -m agent.loop <eml-path>` and read `matched_application_id` out of the JSON the agent emits.
- **notes** — one or two sentences capturing the labeling rationale. Especially important when the classification or action is ambiguous, because the notes are what an eval failure 6 months from now will need to interpret the row.

## Process

1. **Confirm.** Read back the labels in plain English: "*This is a rejection from $company, action update_status to rejected, matched to <id>, because <reason>. Correct?*" Don't skip this — it's the human-in-loop on labeling.

2. **Run the helper:**

   ```bash
   uv run python -m evals.add_row \
       --eml <path> \
       --id <id> \
       --classification <cls> \
       --tool <tool> \
       --status <status> \           # only when tool=update_status
       --app-id <id-or-the-string-null> \
       --notes "<notes>"
   ```

   The script auto-picks the dataset file from the .eml filename:
   - `*-real.eml` → `evals/dataset-real.jsonl` (gitignored)
   - anything else → `evals/dataset.jsonl` (committed)

3. **Suggest the next step:** "Run `uv run evals` to re-baseline. Commit the new scorecard alongside the row." Don't auto-run evals without asking — eval runs cost money.

## Hard rules

- **Never invent label values to make the script happy.** If you don't know the classification or app id, ask the user. A wrong label poisons every future eval result worse than skipping the row entirely.
- **Real personal/business data goes to `dataset-real.jsonl`.** Anything with a real recruiter name, real personal email address, or real company in the user's active job hunt should have the `-real` suffix on both the .eml filename and the row id. The auto-pick depends on it.
- **Synthetic emails go to `dataset.jsonl`.** That file commits to a public repo. Names, addresses, IDs must be fictional.
- **Never edit an existing row to make a current run pass.** If the dataset's expectation looks wrong, talk to the user first; in most cases the right move is to add a new row capturing the edge case, not to retroactively rewrite history.
- **One row per invocation.** Loops over many emails belong in a script the user writes deliberately, not in this skill.

## When NOT to use

- The user is asking about Phase 3 guardrails work, not labeling.
- The user is iterating on the system prompt and wants to re-run the existing dataset.
- The user is asking what the dataset contains or how scoring works — point them at `evals/README.md` instead.
