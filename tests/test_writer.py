"""Unit tests for src/agent/writer.py.

The `_apply` path needs a live jobtrack MCP server to be meaningful — that's
integration test territory and out of scope here. We test the queue and drop
paths, which are pure file I/O, plus the dispatch logic.
"""
import asyncio
import json

import pytest

from agent.gating import GateDecision
from agent.models import (
    ActionTool,
    AgentProposal,
    ApplicationStatus,
    Classification,
)
from agent.writer import _email_hash, _queue, execute_proposal


def _valid_proposal(tool: ActionTool, confidence: float = 0.95) -> AgentProposal:
    base = {
        "classification": "interview_invite",
        "matched_application_id": "2026-05-12_valon_x",
        "tool": tool.value,
        "confidence": confidence,
        "reasoning": "test",
    }
    if tool is ActionTool.UPDATE_STATUS:
        base["application_id"] = "2026-05-12_valon_x"
        base["status"] = ApplicationStatus.SCREENING.value
    elif tool is ActionTool.ADD_NOTE:
        base["application_id"] = "2026-05-12_valon_x"
        base["note"] = "test note"
    else:  # NO_OP
        base["classification"] = "spam"
        base["matched_application_id"] = None
        base["reason"] = "test"
    return AgentProposal.model_validate(base)


# ---------- _email_hash ----------

def test_email_hash_is_stable_and_truncated():
    h1 = _email_hash("hello world")
    h2 = _email_hash("hello world")
    assert h1 == h2
    assert h1 is not None and h1.startswith("sha256:")
    # truncated to 16 hex chars after the prefix
    assert len(h1.split(":")[1]) == 16


def test_email_hash_returns_none_for_none():
    assert _email_hash(None) is None


# ---------- _queue ----------

def test_queue_writes_file_with_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.writer.PENDING_DIR", tmp_path)

    p = _valid_proposal(ActionTool.UPDATE_STATUS, confidence=0.75)
    record = _queue(p, email_body="raw email text")

    assert record["decision"] == "queue"
    assert record["executed"] is False
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["proposal"]["tool"] == "update_status"
    assert payload["email_hash"].startswith("sha256:")
    assert payload["queued_at"] == record["at"]


def test_queue_filename_uses_app_id_or_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.writer.PENDING_DIR", tmp_path)

    matched = _valid_proposal(ActionTool.UPDATE_STATUS, confidence=0.75)
    record = _queue(matched, email_body="x")
    assert "2026-05-12_valon_x" in record["queued_path"]


# ---------- execute_proposal dispatch ----------

def test_execute_drop_for_no_op_returns_record_without_io(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.writer.PENDING_DIR", tmp_path)

    p = _valid_proposal(ActionTool.NO_OP, confidence=0.95)
    record = asyncio.run(execute_proposal(p, GateDecision.DROP, email_body="x"))

    assert record["decision"] == "drop"
    assert record["executed"] is False
    assert "no_op proposal" in record["reason"]
    # nothing got written
    assert list(tmp_path.glob("*.json")) == []


def test_execute_queue_routes_to_queue_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.writer.PENDING_DIR", tmp_path)

    p = _valid_proposal(ActionTool.UPDATE_STATUS, confidence=0.75)
    record = asyncio.run(execute_proposal(p, GateDecision.QUEUE, email_body="x"))

    assert record["decision"] == "queue"
    assert record["executed"] is False
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_execute_drop_for_low_confidence_explains_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.writer.PENDING_DIR", tmp_path)

    p = _valid_proposal(ActionTool.UPDATE_STATUS, confidence=0.3)
    record = asyncio.run(execute_proposal(p, GateDecision.DROP, email_body="x"))

    assert record["decision"] == "drop"
    assert "0.30" in record["reason"]
    assert "below queue threshold" in record["reason"]
