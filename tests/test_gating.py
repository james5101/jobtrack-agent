"""Unit tests for src/agent/gating.py — the confidence gate."""
import pytest

from agent.gating import (
    THRESHOLD_APPLY,
    THRESHOLD_QUEUE,
    GateDecision,
    gate_proposal,
)
from agent.models import (
    ActionTool,
    AgentProposal,
    ApplicationStatus,
    Classification,
)


def _update_status(confidence: float) -> AgentProposal:
    """Helper: build a valid update_status proposal with the given confidence."""
    return AgentProposal.model_validate({
        "classification": Classification.INTERVIEW_INVITE,
        "matched_application_id": "x",
        "tool": ActionTool.UPDATE_STATUS,
        "application_id": "x",
        "status": ApplicationStatus.SCREENING,
        "confidence": confidence,
        "reasoning": "test",
    })


def _no_op(confidence: float) -> AgentProposal:
    return AgentProposal.model_validate({
        "classification": Classification.SPAM,
        "matched_application_id": None,
        "tool": ActionTool.NO_OP,
        "reason": "test",
        "confidence": confidence,
        "reasoning": "test",
    })


def test_high_confidence_writes_apply():
    assert gate_proposal(_update_status(0.95)) is GateDecision.APPLY


def test_boundary_apply_threshold_is_inclusive():
    """confidence exactly at THRESHOLD_APPLY (0.9) should still APPLY."""
    assert gate_proposal(_update_status(THRESHOLD_APPLY)) is GateDecision.APPLY


def test_mid_confidence_queues():
    assert gate_proposal(_update_status(0.75)) is GateDecision.QUEUE


def test_boundary_queue_threshold_is_inclusive():
    """confidence exactly at THRESHOLD_QUEUE (0.6) should QUEUE, not DROP."""
    assert gate_proposal(_update_status(THRESHOLD_QUEUE)) is GateDecision.QUEUE


def test_just_below_queue_threshold_drops():
    assert gate_proposal(_update_status(THRESHOLD_QUEUE - 0.01)) is GateDecision.DROP


def test_low_confidence_drops():
    assert gate_proposal(_update_status(0.2)) is GateDecision.DROP


def test_no_op_always_drops_regardless_of_confidence():
    """no_op has nothing to apply or queue. Even at 0.99 confidence we DROP —
    confident in *not* doing anything is the same as not doing anything."""
    assert gate_proposal(_no_op(0.99)) is GateDecision.DROP
    assert gate_proposal(_no_op(0.5)) is GateDecision.DROP
    assert gate_proposal(_no_op(0.1)) is GateDecision.DROP
