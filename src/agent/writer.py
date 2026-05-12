"""Phase 3.3 — execute, queue, or drop a gated proposal.

This module is the *only* place in the codebase that calls jobtrack's
write tools. The agent loop has those tools in its `disallowed_tools`
list and cannot reach them; instead, after the agent produces a
validated proposal and `gating.gate_proposal` returns APPLY, the writer
opens a direct MCP connection to jobtrack — no LLM involved — and calls
the tool with the proposal's validated args.

Why direct MCP rather than re-using the agent loop:
- Determinism. The write is a function of the validated proposal, not of
  another LLM call.
- Speed and cost. No second model invocation.
- Safety. The agent's behavior under structured-output mode shifted
  toward "do X then report" (Phase 3.2 finding); we keep write tools
  completely out of the agent's reach.

Decisions:
- APPLY: call jobtrack's update_status / add_note over MCP.
- QUEUE: write the proposal as JSON to pending_actions/ for human review.
- DROP: no-op besides the returned record.

Each path returns a dict describing what happened, intended to feed the
Phase 3.5 audit log.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from .gating import GateDecision
from .mcp_client import JOBTRACK_SERVER_NAME, jobtrack_mcp_config
from .models import ActionTool, AgentProposal

PENDING_DIR = Path(__file__).resolve().parents[2] / "pending_actions"


def _email_hash(body: str | None) -> str | None:
    if body is None:
        return None
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


async def execute_proposal(
    p: AgentProposal,
    decision: GateDecision,
    *,
    email_body: str | None = None,
) -> dict[str, Any]:
    """Dispatch on the gate decision. Returns a record suitable for audit.

    `email_body` is optional and feeds the queue payload's hash + the
    audit trail in Phase 3.5. Pass it from the caller; the writer doesn't
    re-read the email.
    """
    if decision is GateDecision.APPLY:
        return await _apply(p, email_body=email_body)
    if decision is GateDecision.QUEUE:
        return _queue(p, email_body=email_body)
    return {
        "decision": decision.value,
        "executed": False,
        "reason": (
            "no_op proposal" if p.tool is ActionTool.NO_OP
            else f"confidence {p.confidence:.2f} below queue threshold"
        ),
    }


async def _apply(p: AgentProposal, *, email_body: str | None) -> dict[str, Any]:
    """Open a jobtrack MCP connection and call the appropriate write tool."""
    config = jobtrack_mcp_config()
    transport = StdioTransport(command=config["command"], args=config["args"])

    if p.tool is ActionTool.UPDATE_STATUS:
        tool_name = f"mcp__{JOBTRACK_SERVER_NAME}__update_status"
        # NOTE: jobtrack's update_status takes `query` (id or company), not
        # `application_id`. We pass the application_id from the validated
        # proposal — it's an exact ID so jobtrack will resolve strictly.
        args: dict[str, Any] = {"query": p.application_id, "status": p.status.value}
    elif p.tool is ActionTool.ADD_NOTE:
        tool_name = f"mcp__{JOBTRACK_SERVER_NAME}__add_note"
        args = {"query": p.application_id, "note": p.note}
    else:
        raise ValueError(f"_apply called with non-write tool: {p.tool}")

    # fastmcp.Client expects bare tool names (no `mcp__<server>__` prefix)
    # when talking directly to a single server. The agent-side uses the
    # namespaced form because it can see many servers; here we're one.
    bare_tool = tool_name.split("__")[-1]

    async with Client(transport) as client:
        result = await client.call_tool(bare_tool, args)
        data = getattr(result, "data", None) or getattr(result, "content", None)

    return {
        "decision": "apply",
        "executed": True,
        "tool": p.tool.value,
        "args": args,
        "result": data,
        "email_hash": _email_hash(email_body),
        "at": _now_iso(),
    }


def _queue(p: AgentProposal, *, email_body: str | None) -> dict[str, Any]:
    """Write the proposal as JSON to pending_actions/. Phase 5's optional
    human-in-the-loop UI consumes these files; here we just persist them."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    ts = _now_iso()
    filename = f"{ts}_{p.matched_application_id or 'no-match'}.json"
    path = PENDING_DIR / filename

    payload = {
        "queued_at": ts,
        "email_hash": _email_hash(email_body),
        "proposal": p.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "decision": "queue",
        "executed": False,
        "queued_path": str(path),
        "email_hash": _email_hash(email_body),
        "at": ts,
    }
