"""Phase 3.2 guardrail — Pydantic models for the agent's output contract.

The JSON Schema generated via `AgentProposal.model_json_schema()` is passed
to the Claude Code CLI as `--json-schema`, which uses Anthropic's constrained
sampling to enforce the structure on the final assistant message. Tool use
during the loop stays free-form; only the final output is validated.

DESIGN NOTE — flat over discriminated union.

The natural Pydantic shape is `action: UpdateStatusAction | AddNoteAction |
NoOpAction` with a discriminator. The CLI's constrained sampling, however,
silently ignores schemas using `oneOf` + OpenAPI's `discriminator` keyword
together — `structured_output` comes back None and the model emits arbitrary
text. Smoke-tested 2026-05-12; the failure is silent. ($defs/$ref alone do
work — the flat schema below uses $defs for the enums.)

So the agent's output is encoded as a flat object: a single `tool` field
plus optional per-tool fields (`application_id`, `status`, `note`,
`reason`). A `model_validator` enforces the cross-field rules in Python
right after the SDK returns. The wire schema stays simple and the CLI's
constrained sampling actually applies.

Why this is still the right shape downstream:

- Wrong field names still fail (the Phase 1 `id` vs `application_id` bug).
- `confidence` is a typed float, ready for Phase 3.3's gating thresholds.
- `matched_application_id` is independent of the action target — the agent
  can say "I think this is about Valon, but I'm not confident enough to
  write, so tool=no_op" without losing the match guess.

Avoided schema constructs (silently ignored by the CLI when present):
- oneOf + discriminator (together — the discriminated-union pattern)
- Numeric constraints (minimum/maximum)
- String length constraints (minLength/maxLength)
- Recursive schemas

$defs/$ref do work — used here for the enum references.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Classification(StrEnum):
    INTERVIEW_INVITE = "interview_invite"
    REJECTION = "rejection"
    RECRUITER_OUTREACH = "recruiter_outreach"
    STATUS_REQUEST = "status_request"
    OFFER = "offer"
    OTHER = "other"
    SPAM = "spam"


class ApplicationStatus(StrEnum):
    """Mirror of jobtrack's `ApplicationStatus` enum (see application-tracker/
    src/jobtrack/models.py). Closed set — jobtrack rejects anything else.
    """

    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    REJECTED = "rejected"
    GHOSTED = "ghosted"
    WITHDRAWN = "withdrawn"


class ActionTool(StrEnum):
    UPDATE_STATUS = "update_status"
    ADD_NOTE = "add_note"
    NO_OP = "no_op"


class AgentProposal(BaseModel):
    """The complete output the agent emits, schema-enforced by the SDK and
    validated by `_action_consistent` for cross-field correctness."""

    classification: Classification = Field(description="The kind of email this is.")
    matched_application_id: str | None = Field(
        description=(
            "The jobtrack ID the email is about, or null if no application matches. "
            "May be set even when tool=no_op (low-confidence match)."
        ),
    )

    tool: ActionTool = Field(
        description="The proposed action. One of: update_status, add_note, no_op.",
    )

    # Per-tool fields. Only the ones matching `tool` should be populated.
    # Enforced in Python below — the schema alone permits any combination.
    application_id: str | None = Field(
        default=None,
        description=(
            "Jobtrack application ID to act on. REQUIRED when tool is "
            "update_status or add_note. Leave null when tool is no_op."
        ),
    )
    status: ApplicationStatus | None = Field(
        default=None,
        description=(
            "REQUIRED when tool is update_status. Leave null otherwise. "
            "Must be one of jobtrack's statuses."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "REQUIRED when tool is add_note (the note text to append). "
            "Leave null otherwise."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "REQUIRED when tool is no_op (one short sentence on why). "
            "Leave null otherwise."
        ),
    )

    confidence: float = Field(
        description=(
            "Confidence in the proposed action, 0.0 (no idea) to 1.0 (certain). "
            "Use ~0.9+ only when the email is unambiguous AND the application "
            "match is exact. Use ~0.5 when the action is plausible but the email "
            "is ambiguous. Use <0.4 when you would prefer no_op."
        ),
    )
    reasoning: str = Field(
        description="One or two sentences explaining the decision.",
    )

    @model_validator(mode="after")
    def _action_consistent(self) -> AgentProposal:
        if self.tool is ActionTool.UPDATE_STATUS:
            if self.application_id is None:
                raise ValueError("application_id is required when tool=update_status")
            if self.status is None:
                raise ValueError("status is required when tool=update_status")
        elif self.tool is ActionTool.ADD_NOTE:
            if self.application_id is None:
                raise ValueError("application_id is required when tool=add_note")
            if self.note is None:
                raise ValueError("note is required when tool=add_note")
        elif self.tool is ActionTool.NO_OP:
            if self.reason is None:
                raise ValueError("reason is required when tool=no_op")
        return self
