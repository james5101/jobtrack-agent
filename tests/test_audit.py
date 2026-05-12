"""Unit tests for src/agent/audit.py.

Pure file I/O tests. We bypass `triage()` entirely and exercise the audit
module directly so the tests don't need the SDK or a jobtrack subprocess.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.audit import (
    AuditEntry,
    audit_file_for,
    build_entry,
    write_audit,
)
from agent.models import (
    ActionTool,
    AgentProposal,
    ApplicationStatus,
    Classification,
)


def _valid_proposal() -> AgentProposal:
    return AgentProposal.model_validate({
        "classification": Classification.INTERVIEW_INVITE,
        "matched_application_id": "x",
        "tool": ActionTool.UPDATE_STATUS,
        "application_id": "x",
        "status": ApplicationStatus.SCREENING,
        "confidence": 0.92,
        "reasoning": "Recruiter intro call.",
    })


# ---------- File rotation by date ----------

def test_audit_file_path_uses_today_by_default(tmp_path):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = audit_file_for(base=tmp_path)
    assert path == tmp_path / f"audit-{today}.jsonl"


def test_audit_file_path_uses_supplied_date(tmp_path):
    path = audit_file_for("2026-01-15", base=tmp_path)
    assert path == tmp_path / "audit-2026-01-15.jsonl"


# ---------- Append semantics ----------

def test_write_audit_appends_one_line(tmp_path):
    entry = AuditEntry(
        timestamp="2026-05-12T18:00:00Z",
        trigger="cli",
        email_hash="sha256:abc",
        model="claude-sonnet-4-6",
    )
    path = write_audit(entry, base=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["trigger"] == "cli"
    assert parsed["email_hash"] == "sha256:abc"


def test_write_audit_multiple_calls_append(tmp_path):
    for i in range(3):
        write_audit(
            AuditEntry(
                timestamp=f"2026-05-12T18:00:0{i}Z",
                trigger="cli",
                email_hash=f"sha256:row{i}",
                model="m",
            ),
            base=tmp_path,
        )
    path = audit_file_for(base=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert [r["email_hash"] for r in parsed] == ["sha256:row0", "sha256:row1", "sha256:row2"]


# ---------- build_entry from a triage result ----------

def test_build_entry_full_proposal_and_execution():
    proposal = _valid_proposal()
    # Mock triage_result mirrors what loop.triage produces.
    triage_result = {
        "proposal": proposal,
        "result_subtype": "success",
        "cap_hit": None,
        "validation_error": None,
        "decision": None,  # set below — emulate gate
        "execution": {"decision": "apply", "executed": True},
        "tool_calls": [{"name": "mcp__jobtrack__find_application_by_company", "input": {}}],
        "cost_usd": 0.022,
        "duration_s": 18.4,
    }
    # We pass decision via the result dict so build_entry can use it; emulate
    # GateDecision via a stand-in object with a .value attribute.
    from agent.gating import GateDecision
    triage_result["decision"] = GateDecision.APPLY

    entry = build_entry(
        trigger="cli",
        email_hash="sha256:abc",
        model="claude-sonnet-4-6",
        triage_result=triage_result,
    )
    assert entry.proposal is not None
    assert entry.proposal["confidence"] == 0.92
    assert entry.decision == "apply"
    assert entry.execution == {"decision": "apply", "executed": True}
    assert entry.tool_call_count == 1
    assert entry.tool_call_names == ["mcp__jobtrack__find_application_by_company"]
    assert entry.cost_usd == 0.022
    assert entry.duration_s == 18.4


def test_build_entry_handles_no_proposal_run():
    """A cap-hit or schema-error run still gets one entry — that's the
    failure mode operators need to see in the log."""
    triage_result = {
        "proposal": None,
        "result_subtype": "max_turns",
        "cap_hit": "turns",
        "validation_error": None,
        "decision": None,
        "execution": None,
        "tool_calls": [{"name": "mcp__jobtrack__find_application_by_company", "input": {}}],
        "cost_usd": None,
        "duration_s": 6.5,
    }
    entry = build_entry(
        trigger="cli",
        email_hash="sha256:abc",
        model="claude-sonnet-4-6",
        triage_result=triage_result,
    )
    assert entry.proposal is None
    assert entry.cap_hit == "turns"
    assert entry.result_subtype == "max_turns"
    assert entry.decision is None
    assert entry.execution is None
    assert entry.tool_call_count == 1
    assert entry.cost_usd is None


# ---------- Serialization round-trip ----------

def test_entry_serializes_and_reloads(tmp_path):
    entry = build_entry(
        trigger="cli",
        email_hash="sha256:1234",
        model="claude-sonnet-4-6",
        triage_result={
            "proposal": _valid_proposal(),
            "result_subtype": "success",
            "cap_hit": None,
            "validation_error": None,
            "decision": None,
            "execution": None,
            "tool_calls": [],
            "cost_usd": 0.02,
            "duration_s": 18.0,
        },
    )
    path = write_audit(entry, base=tmp_path)
    text = path.read_text(encoding="utf-8")
    reparsed = AuditEntry.model_validate(json.loads(text.strip()))
    assert reparsed.email_hash == "sha256:1234"
    assert reparsed.proposal is not None
    assert reparsed.proposal["tool"] == "update_status"
