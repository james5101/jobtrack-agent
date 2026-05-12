"""Append a labeled eval row from an .eml file.

Usage:
    uv run python -m evals.add_row \\
        --eml samples/emails/rejection-acme-real.eml \\
        --id rejection-acme-real \\
        --classification rejection \\
        --tool update_status --status rejected \\
        --app-id 2026-05-12_acme_engineer_abcd \\
        --notes "form rejection, app exists in jobtrack"

The dataset file is picked automatically from the .eml filename:
    *-real.eml -> evals/dataset-real.jsonl (gitignored)
    everything else -> evals/dataset.jsonl  (committed)

Override with --dataset PATH if you need to.

Validation:
- classification, tool, status are checked against the closed enums in
  loop.py / jobtrack's models.py.
- --status is required iff --tool=update_status; rejected in all other cases.
- --app-id accepts the literal string "null" to mean "no matching application".
- IDs must be unique within the target dataset file (re-runs aren't a thing —
  edit existing rows by hand if you must, but generally add new rows instead).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLASSIFICATIONS = {
    "interview_invite",
    "rejection",
    "recruiter_outreach",
    "status_request",
    "offer",
    "other",
    "spam",
}
TOOLS = {"update_status", "add_note", "no_op"}
STATUSES = {
    "applied",
    "screening",
    "interviewing",
    "offer",
    "rejected",
    "ghosted",
    "withdrawn",
}

EVALS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eml", required=True, type=Path, help=".eml file to embed verbatim")
    p.add_argument("--id", dest="row_id", required=True, help="row id (must be unique in dataset)")
    p.add_argument("--classification", required=True, choices=sorted(CLASSIFICATIONS))
    p.add_argument("--tool", required=True, choices=sorted(TOOLS))
    p.add_argument(
        "--status",
        choices=sorted(STATUSES),
        help="required iff --tool=update_status; rejected otherwise",
    )
    p.add_argument(
        "--app-id",
        dest="app_id",
        required=True,
        help="jobtrack application ID, or the literal string 'null'",
    )
    p.add_argument("--notes", required=True, help="free-text labeling rationale")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="override target dataset (default: auto-pick from .eml filename)",
    )
    return p.parse_args()


def pick_dataset(eml: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    if eml.name.endswith("-real.eml"):
        return EVALS_DIR / "dataset-real.jsonl"
    return EVALS_DIR / "dataset.jsonl"


def load_existing_ids(dataset: Path) -> set[str]:
    if not dataset.exists():
        return set()
    ids: set[str] = set()
    for line in dataset.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "id" in row:
            ids.add(row["id"])
    return ids


def main() -> None:
    args = parse_args()

    # Tool/status combo validation. argparse handles enum membership; we handle
    # the cross-field rule.
    if args.tool == "update_status" and not args.status:
        sys.exit("error: --status is required when --tool=update_status")
    if args.tool != "update_status" and args.status:
        sys.exit(f"error: --status is meaningless when --tool={args.tool!r}; omit it")

    if not args.eml.exists():
        sys.exit(f"error: {args.eml} does not exist")
    email_body = args.eml.read_text(encoding="utf-8")

    if args.tool == "update_status":
        expected_action: dict = {"tool": "update_status", "status": args.status}
    else:
        expected_action = {"tool": args.tool}

    app_id: str | None = None if args.app_id.lower() == "null" else args.app_id

    target = pick_dataset(args.eml, args.dataset)
    existing_ids = load_existing_ids(target)
    if args.row_id in existing_ids:
        sys.exit(f"error: id {args.row_id!r} already present in {target.name}")

    row = {
        "id": args.row_id,
        "email_body": email_body,
        "expected_classification": args.classification,
        "expected_action": expected_action,
        "expected_app_id_or_null": app_id,
        "notes": args.notes,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        rel = target.relative_to(Path.cwd())
    except ValueError:
        rel = target
    print(f"appended {args.row_id!r} -> {rel}")


if __name__ == "__main__":
    main()
