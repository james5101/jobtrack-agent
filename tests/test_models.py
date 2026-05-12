"""Unit tests for the agent's output Pydantic models.

Focus areas:
- Round-trip via model_validate / model_dump for each action shape.
- Cross-field rules enforced by `_action_consistent`.
- The Phase 1 wrong-field-name regression (e.g. `id` instead of `application_id`).
- The generated JSON Schema is free of constructs the CLI silently ignores
  (`$defs`, `oneOf`, `discriminator`, numeric/string constraints).
"""
import json

import pytest
from pydantic import ValidationError

from agent.models import (
    ActionTool,
    AgentProposal,
    ApplicationStatus,
    Classification,
)


# ---------- Round-trip ----------

def test_round_trip_update_status():
    p = AgentProposal.model_validate({
        "classification": "interview_invite",
        "matched_application_id": "2026-05-10_valon_x",
        "tool": "update_status",
        "application_id": "2026-05-10_valon_x",
        "status": "screening",
        "confidence": 0.92,
        "reasoning": "Recruiter intro call invite.",
    })
    assert p.classification is Classification.INTERVIEW_INVITE
    assert p.tool is ActionTool.UPDATE_STATUS
    assert p.status is ApplicationStatus.SCREENING
    assert p.note is None
    assert p.reason is None


def test_round_trip_no_op():
    p = AgentProposal.model_validate({
        "classification": "spam",
        "matched_application_id": None,
        "tool": "no_op",
        "application_id": None,
        "status": None,
        "note": None,
        "reason": "Generic marketing email; no application reference.",
        "confidence": 0.95,
        "reasoning": "Newsletter signature; nothing actionable.",
    })
    assert p.tool is ActionTool.NO_OP
    assert p.application_id is None


def test_round_trip_add_note():
    p = AgentProposal.model_validate({
        "classification": "recruiter_outreach",
        "matched_application_id": "2026-05-10_valon_x",
        "tool": "add_note",
        "application_id": "2026-05-10_valon_x",
        "status": None,
        "note": "Recruiter is asking about availability next week.",
        "reason": None,
        "confidence": 0.7,
        "reasoning": "Logistics note; no state change yet.",
    })
    assert p.tool is ActionTool.ADD_NOTE
    assert "availability" in p.note


# ---------- Cross-field rules ----------

def test_update_status_requires_application_id():
    with pytest.raises(ValidationError, match="application_id is required when tool=update_status"):
        AgentProposal.model_validate({
            "classification": "interview_invite",
            "matched_application_id": None,
            "tool": "update_status",
            "status": "screening",
            "confidence": 0.9,
            "reasoning": "",
        })


def test_update_status_requires_status():
    with pytest.raises(ValidationError, match="status is required when tool=update_status"):
        AgentProposal.model_validate({
            "classification": "interview_invite",
            "matched_application_id": "x",
            "tool": "update_status",
            "application_id": "x",
            "confidence": 0.9,
            "reasoning": "",
        })


def test_add_note_requires_note():
    with pytest.raises(ValidationError, match="note is required when tool=add_note"):
        AgentProposal.model_validate({
            "classification": "recruiter_outreach",
            "matched_application_id": "x",
            "tool": "add_note",
            "application_id": "x",
            "confidence": 0.7,
            "reasoning": "",
        })


def test_no_op_requires_reason():
    with pytest.raises(ValidationError, match="reason is required when tool=no_op"):
        AgentProposal.model_validate({
            "classification": "spam",
            "matched_application_id": None,
            "tool": "no_op",
            "confidence": 0.9,
            "reasoning": "",
        })


# ---------- Enum closure ----------

def test_bad_classification_fails():
    with pytest.raises(ValidationError):
        AgentProposal.model_validate({
            "classification": "interview_request",  # not in enum
            "matched_application_id": None,
            "tool": "no_op",
            "reason": "x",
            "confidence": 0.5,
            "reasoning": "",
        })


def test_bad_status_fails():
    with pytest.raises(ValidationError):
        AgentProposal.model_validate({
            "classification": "interview_invite",
            "matched_application_id": "x",
            "tool": "update_status",
            "application_id": "x",
            "status": "phone_screen",  # not in enum
            "confidence": 0.9,
            "reasoning": "",
        })


def test_bad_tool_fails():
    with pytest.raises(ValidationError):
        AgentProposal.model_validate({
            "classification": "interview_invite",
            "matched_application_id": "x",
            "tool": "delete_application",  # not in enum
            "confidence": 0.5,
            "reasoning": "",
        })


# ---------- Generated JSON Schema is CLI-safe ----------

def test_schema_has_no_forbidden_keys():
    """Anthropic's constrained sampling silently ignores schemas with these.
    Confirmed by smoke test 2026-05-12. Keep them out of the wire schema.
    """
    schema_dump = json.dumps(AgentProposal.model_json_schema())
    # $defs / $ref themselves are fine — used for enum references and work.
    # The breakers (smoke-tested 2026-05-12) are `oneOf` + `discriminator`,
    # plus numeric/length constraints.
    for forbidden in (
        '"oneOf"',
        '"discriminator"',
        '"minimum"',
        '"maximum"',
        '"minLength"',
        '"maxLength"',
        '"exclusiveMinimum"',
        '"exclusiveMaximum"',
    ):
        assert forbidden not in schema_dump, f"schema must not contain {forbidden}"
