"""Phase 3.5 guardrail — append-only audit log.

One row per agent invocation, in `audit/audit-<UTC-date>.jsonl`. The file
rotates daily by date stamp; a day's log is one file, easy to grep or
ship. Each line is a JSON-serialized `AuditEntry`.

DESIGN

- **Local-first, structured JSON.** One JSON object per line. No
  application-specific log levels, no human-targeted prose. Each entry
  is a complete record of what happened on one invocation.
- **PII redaction by content-addressing.** The audit log records the
  hash of the email body (`sha256:<16hex>`), never the raw body. That
  preserves correlatability across logs without storing recruiter names
  or personal addresses in a place where they could leak.
- **Local file, gitignored.** This is the production log on the host the
  agent runs on. A hosted aggregator (Honeycomb / Loki / Datadog) is a
  Phase 4 concern — when that lands, it tee's the same lines but with
  whatever extra redaction the destination requires. For dev/Phase 3,
  the local file is the production behavior.
- **What we DO log:** proposal in full (including reasoning text). Yes
  the reasoning may mention real names — operators need it for
  debugging. The privacy guarantee is that this file stays on the host
  unless an operator explicitly chooses to ship it elsewhere.

NOT logged by design:
- Raw email body (we have the hash).
- Raw `structured_output` from the SDK (it's the proposal pre-validation
  — same content, less typed; redundant).
- The full sanitized email text (would defeat the hash-only contract).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .gating import GateDecision
from .models import AgentProposal

AUDIT_DIR = Path(__file__).resolve().parents[2] / "audit"


class AuditEntry(BaseModel):
    """One row in the audit log. Pydantic gives us model_dump_json() for
    serialization and a stable wire shape that the harness, tests, and
    downstream pipelines can rely on."""

    timestamp: str = Field(description="UTC ISO 8601, second precision (sortable).")
    trigger: str = Field(description="Where this invocation came from. e.g. 'cli', 'cli-dry-run', 'webhook'.")
    email_hash: str = Field(description="sha256:<16hex> of the raw inbound email body.")

    model: str = Field(description="Model identifier used for the run.")

    # The agent's structured output, validated and dumped. None when the SDK
    # didn't produce a usable proposal (cap_hit, schema_err, or sdk_error).
    proposal: dict[str, Any] | None = None

    # SDK / validation state. At most one of cap_hit / validation_error
    # should be set; result_subtype carries the SDK's own classification.
    result_subtype: str | None = None
    cap_hit: str | None = Field(default=None, description="'turns' | 'budget' | 'timeout' | None")
    validation_error: str | None = None

    # The gate's verdict. None when no proposal existed to gate.
    decision: str | None = Field(default=None, description="'apply' | 'queue' | 'drop' | None")

    # What the writer actually did (or didn't). Shape varies by decision —
    # apply rows include the jobtrack tool result; queue rows include the
    # pending-actions file path; drop rows include a reason.
    execution: dict[str, Any] | None = None

    # Metrics. Even when the run failed, these tell us *how* it failed.
    tool_call_count: int = 0
    tool_call_names: list[str] = Field(default_factory=list)
    cost_usd: float | None = None
    duration_s: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def audit_file_for(date_str: str | None = None, *, base: Path | None = None) -> Path:
    """Path to the audit file for a given UTC date (default: today).

    Exposed so tests and operators can target a specific day's log without
    walking the directory. Date-based rotation keeps each file bounded by
    a day's traffic, makes "what happened on X" trivial, and is friendly
    to `tail -F` watchers.
    """
    base = base or AUDIT_DIR
    date_str = date_str or _today_utc()
    return base / f"audit-{date_str}.jsonl"


def write_audit(entry: AuditEntry, *, base: Path | None = None) -> Path:
    """Append one entry to today's audit file. Returns the path written to.

    Single append per call; the line is JSON + newline so concurrent
    appenders never interleave mid-line (Linux/POSIX semantics; Windows
    same in practice for small writes). For very high throughput we'd
    switch to a queue + dedicated writer — out of scope here.
    """
    base = base or AUDIT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = audit_file_for(base=base)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(entry.model_dump_json() + "\n")
    return path


def build_entry(
    *,
    trigger: str,
    email_hash: str,
    model: str,
    triage_result: dict[str, Any],
) -> AuditEntry:
    """Construct an AuditEntry from a `triage()` result dict.

    Centralizes the mapping so the same data lands in audit identically
    whether the trigger is a CLI invocation or (future) a webhook handler.
    """
    proposal: AgentProposal | None = triage_result.get("proposal")
    decision: GateDecision | None = triage_result.get("decision")
    tool_calls = triage_result.get("tool_calls") or []

    return AuditEntry(
        timestamp=_now_iso(),
        trigger=trigger,
        email_hash=email_hash,
        model=model,
        proposal=proposal.model_dump(mode="json") if proposal else None,
        result_subtype=triage_result.get("result_subtype"),
        cap_hit=triage_result.get("cap_hit"),
        validation_error=triage_result.get("validation_error"),
        decision=decision.value if decision else None,
        execution=triage_result.get("execution"),
        tool_call_count=len(tool_calls),
        tool_call_names=[tc.get("name", "") for tc in tool_calls],
        cost_usd=triage_result.get("cost_usd"),
        duration_s=float(triage_result.get("duration_s") or 0.0),
    )
