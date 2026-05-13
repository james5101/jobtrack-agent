# jobtrack-agent

A Claude agent that triages job-application emails and proposes updates to [jobtrack](../application-tracker) via MCP.

> Personal learning project. Optimized for skill acquisition in the production AI stack — Claude Agent SDK, evals, layered guardrails, deployment — not for shipping product.

## What it does

Single-purpose: turn one inbound email about a job application into a proposed action against the user's jobtrack store.

```
[email]
   │
   ├─► sanitize  (HTML, quoted replies, length cap, whitespace)
   ├─► Claude Agent SDK loop  (Sonnet, read-only jobtrack MCP)
   ├─► AgentProposal  (Pydantic, schema-enforced via constrained sampling)
   ├─► confidence gate  (≥0.9 apply │ 0.6–0.9 queue │ <0.6 drop)
   ├─► writer  (direct MCP call) or pending_actions/<ts>.json
   └─► audit log row
```

The agent classifies the email (`interview_invite` / `rejection` / `recruiter_outreach` / `status_request` / `offer` / `other` / `spam`), resolves which tracked application it relates to, and proposes one of three actions: `update_status`, `add_note`, or `no_op`. The output is a typed Pydantic object with a numeric confidence that drives the gate.

Out of scope: replying to emails, scheduling, drafting cover letters, anything outbound.

## Relationship to jobtrack

jobtrack lives in [`../application-tracker`](../application-tracker) (separate repo, separate CI, separate lifecycle). This agent consumes it **only through its MCP server**. No `import jobtrack`, no direct access to `~/.jobtrack/`. If a feature is missing from jobtrack's MCP surface, it gets added to jobtrack first.

The MCP server is launched as a subprocess via `uv run --project ../application-tracker jobtrack-mcp`. Override the path with `JOBTRACK_PROJECT_PATH`.

The boundary is load-bearing: forcing MCP as the only interface is the whole point of the exercise.

## Project layout

```
src/agent/
  loop.py        # agent loop + triage pipeline + CLI entrypoint
  models.py      # Pydantic AgentProposal contract (flat by design — see edit-proposal-model skill)
  sanitize.py    # input sanitation (Phase 3.1)
  gating.py      # confidence gate (Phase 3.3)
  writer.py      # direct-MCP writer + pending_actions/ queueing (Phase 3.3)
  audit.py       # append-only audit log (Phase 3.5)
  mcp_client.py  # jobtrack stdio launch config
evals/
  dataset.jsonl       # sanitized labeled rows (committed)
  adversarial.jsonl   # prompt-injection test set (committed)
  add_row.py          # CLI helper to append a labeled row
  run.py              # eval harness (uv run evals)
  results/            # committed aggregate scorecards
tests/                # pytest, pure-function coverage
.claude/skills/       # project skills (add-eval-row, edit-proposal-model)
samples/emails/       # synthetic samples; *-real.eml gitignored
```

## Phase status

| Phase | Status |
|---|---|
| 1 — Agent loop, local only | ✓ |
| 2 — Eval harness + dataset | ✓ |
| 3 — Guardrails (sanitation, structured output, gating, cost, audit, injection) | ✓ |
| 4 — Deployment (Docker, VPS, webhook, CI/CD, observability) | next |
| 5 — Human-in-the-loop UI | optional |

Every guardrail commit includes a fresh aggregate scorecard in `evals/results/` so the impact of each change is diff-able.

## Quickstart

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
```

Run the loop against a sample:

```bash
uv run python -m agent.loop samples/emails/interview-01.eml
```

`--dry-run` (or `-n`) skips the writer — the agent still runs, the gate still decides, but jobtrack is not mutated. Useful when iterating on prompts against a live store.

Run the eval suite:

```bash
uv run evals
```

Snapshots land in `evals/results/<UTC-timestamp>.json`.

Run unit tests:

```bash
uv run pytest
```

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required |
| `JOBTRACK_PROJECT_PATH` | `../application-tracker` | where jobtrack lives |
| `AGENT_MAX_BUDGET_USD` | `0.20` | hard budget cap per invocation |
| `AGENT_MAX_TURNS` | `10` | hard tool-call/turn cap |
| `AGENT_WALL_CLOCK_TIMEOUT_S` | `60` | hard wall-clock cap |

## Hard rules

- **Never bypass MCP.** No imports from jobtrack, no direct filesystem access to `~/.jobtrack/`. Missing feature → add it to jobtrack's MCP surface first.
- **Every action goes to the audit log**, including drops, cap-hits, and dry-runs. No silent behavior.
- **Eval scorecard is part of every change.** Prompt or guardrail change without a fresh scorecard is incomplete.
- **Fail closed.** Uncertain → drop or queue. Never guess on a write.
- **Secrets never in the repo.** `.env.example` is checked in; `.env`, `audit/`, `pending_actions/`, `*-real.eml`, and `*.rows.jsonl` are not.

## What NOT to build

- **Kubernetes.** A $5 VPS + systemd teaches more for the time invested.
- **LangChain / LangGraph / CrewAI.** The Claude Agent SDK + raw MCP teaches primitives; frameworks teach the framework.
- **Fine-tuning, vector DBs, multi-agent orchestration.** Premature for this scope.
- **A web app for jobtrack itself.** Scope creep on the other repo.

## Stack

Python 3.12+, [`uv`](https://docs.astral.sh/uv/) for project management, [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) for the agent runtime, [`pydantic`](https://docs.pydantic.dev/) v2 for the output contract, [`fastmcp`](https://gofastmcp.com/) for direct MCP writes outside the agent loop, [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/) for HTML sanitation, `pytest` for tests.
