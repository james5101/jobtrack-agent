"""Phase 2 eval harness.

Loads the labeled dataset, runs the agent against each row, parses the agent's
JSON output, and produces a scorecard. Committed snapshots in `results/` are
the diff-able history that gates every prompt change from here on.

Usage:
    uv run evals
    # or
    uv run python -m evals.run

Reads:
    evals/dataset.jsonl                 # sanitized, committed
    evals/dataset-real.jsonl (optional) # unsanitized, gitignored

Writes:
    evals/results/<UTC-timestamp>.json       # aggregate scorecard
    evals/results/<UTC-timestamp>.rows.jsonl # per-row detail
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.loop import MODEL, run_agent

EVALS_DIR = Path(__file__).resolve().parent
DATASET_FILES = ["dataset.jsonl", "dataset-real.jsonl"]
RESULTS_DIR = EVALS_DIR / "results"


# ---------- Dataset loading ----------

def load_dataset() -> list[dict]:
    """Concatenate every dataset file that exists. Each JSONL line is one row."""
    rows: list[dict] = []
    for name in DATASET_FILES:
        path = EVALS_DIR / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rows.append(json.loads(line))
    return rows


# ---------- Output parsing ----------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_agent_json(text: str) -> dict | None:
    """Extract the first JSON object from agent output.

    Tolerates two known failure modes from Phase 1:
    - markdown fences (```json ... ```) wrapping the JSON
    - prose before / after the JSON block

    Strategy: try the contents of the first ``` block, then fall back to
    scanning for the first balanced JSON object in the raw text. Returns
    None if nothing parses — the caller records that as a parse_error row.
    """
    candidates: list[str] = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        idx = candidate.find("{")
        while idx != -1:
            try:
                obj, _end = decoder.raw_decode(candidate, idx)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = candidate.find("{", idx + 1)
    return None


# ---------- Per-row scoring ----------

def evaluate_row(row: dict, actual: dict | None) -> dict:
    """Compare agent output (actual) against labeled expectations (row).

    A row passes when:
    - classification matches
    - proposed_action.tool matches
    - matched_application_id matches (or both null)
    - when the action is update_status, the status arg also matches

    Each component is tracked separately so the scorecard can attribute
    failures to a specific dimension, not just total pass/fail.
    """
    if actual is None:
        return {
            "pass": False,
            "parse_error": True,
            "classification_match": False,
            "tool_match": False,
            "app_id_match": False,
            "status_match": None,
        }

    expected_classification = row["expected_classification"]
    actual_classification = actual.get("classification")
    classification_match = actual_classification == expected_classification

    expected_action = row["expected_action"]
    expected_tool = expected_action["tool"]
    actual_action = actual.get("proposed_action") or {}
    actual_tool = actual_action.get("tool")
    tool_match = actual_tool == expected_tool

    expected_app = row.get("expected_app_id_or_null")
    actual_app = actual.get("matched_application_id")
    app_id_match = actual_app == expected_app

    status_match: bool | None = None
    if expected_tool == "update_status":
        expected_status = expected_action.get("status")
        actual_args = actual_action.get("args") or {}
        actual_status = actual_args.get("status")
        status_match = actual_status == expected_status

    passed = classification_match and tool_match and app_id_match
    if expected_tool == "update_status":
        passed = passed and bool(status_match)

    return {
        "pass": bool(passed),
        "parse_error": False,
        "classification_match": classification_match,
        "tool_match": tool_match,
        "app_id_match": app_id_match,
        "status_match": status_match,
    }


# ---------- Aggregation ----------

def aggregate(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {"n_rows": 0}

    n_parse = sum(1 for r in per_row if r["evaluation"]["parse_error"])
    n_pass = sum(1 for r in per_row if r["evaluation"]["pass"])
    n_fail = n - n_pass - n_parse

    class_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    action_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in per_row:
        e = r["expected"]
        a = r.get("actual") or {}
        class_confusion[e["expected_classification"]][a.get("classification") or "<parse_err>"] += 1
        action_confusion[e["expected_action"]["tool"]][(a.get("proposed_action") or {}).get("tool") or "<parse_err>"] += 1

    valid = [r for r in per_row if not r["evaluation"]["parse_error"]]
    n_valid = max(len(valid), 1)
    mean_tool_calls = sum(len(r["tool_calls"]) for r in valid) / n_valid
    mean_cost = sum((r["cost_usd"] or 0) for r in valid) / n_valid
    mean_duration = sum(r["duration_s"] for r in valid) / n_valid

    return {
        "n_rows": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_parse_error": n_parse,
        "pass_rate": n_pass / n,
        "classification_confusion": {k: dict(v) for k, v in class_confusion.items()},
        "action_confusion": {k: dict(v) for k, v in action_confusion.items()},
        "mean_tool_calls": round(mean_tool_calls, 2),
        "mean_cost_usd": round(mean_cost, 4),
        "mean_duration_s": round(mean_duration, 2),
    }


# ---------- Running ----------

async def run_row(row: dict) -> dict:
    print(f"\n=== {row['id']} ===", file=sys.stderr)
    result = await run_agent(row["email_body"])
    actual = parse_agent_json(result["text"])
    evaluation = evaluate_row(row, actual)

    return {
        "id": row["id"],
        "expected": {
            "expected_classification": row["expected_classification"],
            "expected_action": row["expected_action"],
            "expected_app_id_or_null": row.get("expected_app_id_or_null"),
        },
        "actual": actual,
        "raw_output": result["text"],
        "tool_calls": result["tool_calls"],
        "cost_usd": result["cost_usd"],
        "duration_s": result["duration_s"],
        "evaluation": evaluation,
    }


async def run_all() -> tuple[dict, list[dict]]:
    rows = load_dataset()
    if not rows:
        print(
            f"No dataset rows found. Expected at least one of: "
            f"{', '.join(str(EVALS_DIR / f) for f in DATASET_FILES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[dict] = []
    for row in rows:
        results.append(await run_row(row))
    return aggregate(results), results


# ---------- Scorecard printing ----------

def _mark(ok: bool | None) -> str:
    if ok is None:
        return "  "
    return "  " if ok else "X "


def print_scorecard(score: dict, results: list[dict]) -> None:
    print()
    print("=" * 64)
    print("SCORECARD")
    print("=" * 64)
    print(f"Rows:        {score['n_rows']}")
    print(f"Passed:      {score['n_pass']} ({score['pass_rate']:.0%})")
    print(f"Failed:      {score['n_fail']}")
    print(f"Parse error: {score['n_parse_error']}")
    print()
    print(f"Mean tool calls/row: {score['mean_tool_calls']}")
    print(f"Mean cost/row:       ${score['mean_cost_usd']}")
    print(f"Mean duration/row:   {score['mean_duration_s']}s")
    print()
    print("Per-row results:")
    for r in results:
        e = r["evaluation"]
        flag = "PASS" if e["pass"] else ("PARSE_ERR" if e["parse_error"] else "FAIL")
        print(f"  [{flag:9s}] {r['id']:30s} "
              f"cls={_mark(e['classification_match']).strip() or 'ok'} "
              f"tool={_mark(e['tool_match']).strip() or 'ok'} "
              f"app={_mark(e['app_id_match']).strip() or 'ok'} "
              f"status={'ok' if e['status_match'] is True else ('x' if e['status_match'] is False else '-')}")
    print()
    print("Classification confusion (expected -> actual):")
    for exp, actuals in score["classification_confusion"].items():
        for act, count in actuals.items():
            print(f"  {_mark(exp == act)}{exp:22s} -> {act:22s} {count}")
    print()
    print("Action confusion (expected -> actual):")
    for exp, actuals in score["action_confusion"].items():
        for act, count in actuals.items():
            print(f"  {_mark(exp == act)}{exp:15s} -> {act:15s} {count}")
    print()


# ---------- Snapshot writing ----------

def write_snapshot(score: dict, results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    score_path = RESULTS_DIR / f"{ts}.json"
    rows_path = RESULTS_DIR / f"{ts}.rows.jsonl"

    score_payload: dict[str, Any] = {
        **score,
        "timestamp": ts,
        "model": MODEL,
    }
    score_path.write_text(
        json.dumps(score_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with rows_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    return score_path


def main() -> None:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    score, results = asyncio.run(run_all())
    print_scorecard(score, results)
    path = write_snapshot(score, results)
    try:
        rel = path.relative_to(Path.cwd())
    except ValueError:
        rel = path
    print(f"Snapshot written: {rel}", file=sys.stderr)


if __name__ == "__main__":
    main()
