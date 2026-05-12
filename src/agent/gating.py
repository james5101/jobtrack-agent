"""Phase 3.3 guardrail — confidence-based action gating.

After the agent emits a validated proposal, the gate decides whether to:
- APPLY: execute the write against jobtrack (high confidence).
- QUEUE: write the proposal to pending_actions/ for human review (medium).
- DROP: do nothing (low confidence; honor the agent's uncertainty).

This module is intentionally pure — no I/O, no async, no MCP. It's a
single function over an `AgentProposal`. The execution / queueing /
logging lives in `writer.py`. Splitting them is what lets the eval
harness measure gate decisions without ever mutating jobtrack.

Thresholds are constants here. The README's spec is the starting point;
production thresholds get tuned against the eval set, not vibes. When
you change a threshold, re-run evals and commit the scorecard delta.
"""
from __future__ import annotations

from enum import StrEnum

from .models import ActionTool, AgentProposal

# Tunable thresholds. Lower THRESHOLD_APPLY = more aggressive auto-apply;
# higher = more conservative. Anything below THRESHOLD_QUEUE is dropped.
THRESHOLD_APPLY = 0.9
THRESHOLD_QUEUE = 0.6


class GateDecision(StrEnum):
    APPLY = "apply"
    QUEUE = "queue"
    DROP = "drop"


def gate_proposal(p: AgentProposal) -> GateDecision:
    """Map a validated proposal to a decision.

    Special case: `no_op` proposals are always DROP. There is nothing to
    apply (no target ID) and nothing useful to queue — the agent has
    already decided not to act. Funneling no_op through DROP keeps callers
    simple and downstream audit logs consistent.
    """
    if p.tool is ActionTool.NO_OP:
        return GateDecision.DROP
    if p.confidence >= THRESHOLD_APPLY:
        return GateDecision.APPLY
    if p.confidence >= THRESHOLD_QUEUE:
        return GateDecision.QUEUE
    return GateDecision.DROP
