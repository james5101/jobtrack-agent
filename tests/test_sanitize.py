"""Unit tests for src/agent/sanitize.py.

Pure-function tests — no MCP, no SDK, no network. Should run in milliseconds.
"""
from pathlib import Path

import pytest

from agent.sanitize import (
    MAX_CHARS,
    cap_length,
    normalize_whitespace,
    sanitize_email,
    strip_html,
    strip_quoted_replies,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "emails"


# ---------- strip_html ----------

def test_strip_html_passes_plain_text_unchanged():
    text = "Hi James,\n\nThanks for applying. -Jane"
    assert strip_html(text) == text


def test_strip_html_does_not_trigger_on_stray_angle_brackets():
    """An email address or `a < b` comparison should not be treated as HTML."""
    text = "Email me at <jane@example.com>. Also a < b is fine."
    assert strip_html(text) == text


def test_strip_html_extracts_text_from_real_html():
    html = "<html><body><p>Hi <b>James</b>,</p><p>You're in.</p></body></html>"
    out = strip_html(html)
    assert "James" in out
    assert "You're in" in out
    assert "<p>" not in out
    assert "<b>" not in out


def test_strip_html_drops_script_and_style():
    html = (
        "<html><head><style>.a{color:red}</style></head>"
        "<body><script>alert(1)</script><p>visible</p></body></html>"
    )
    out = strip_html(html)
    assert "visible" in out
    assert "alert" not in out
    assert "color" not in out


# ---------- strip_quoted_replies ----------

def test_strip_quoted_replies_gmail_marker():
    text = (
        "Thanks, looking forward to it!\n\n"
        "On Mon, May 11, 2026 at 2:30 PM Jane <jane@x.com> wrote:\n"
        "> Hi James,\n"
        "> Are you free Thursday?\n"
    )
    out = strip_quoted_replies(text)
    assert "Thanks, looking forward to it!" in out
    assert "wrote:" not in out
    assert "Are you free" not in out


def test_strip_quoted_replies_outlook_block():
    text = (
        "Got it, thanks!\n\n"
        "From: Jane Recruiter <jane@x.com>\n"
        "Sent: Monday, May 11, 2026 2:30 PM\n"
        "To: James Noonan\n"
        "Subject: Re: Hiring\n\n"
        "Hi James, please confirm..."
    )
    out = strip_quoted_replies(text)
    assert "Got it, thanks!" in out
    assert "please confirm" not in out


def test_strip_quoted_replies_caret_quoted_block():
    text = (
        "Sounds good.\n\n"
        "> Hi James,\n"
        "> Are you free Thursday?\n"
        "> Jane\n"
    )
    out = strip_quoted_replies(text)
    assert "Sounds good." in out
    assert "Are you free" not in out


def test_strip_quoted_replies_leaves_text_with_no_marker_alone():
    text = "Just a normal email body. No replies, no quoting.\nSee you Tuesday."
    assert strip_quoted_replies(text) == text


def test_strip_quoted_replies_does_not_fire_on_single_quoted_line():
    """A lone `>` line (e.g. a single citation) should not trigger truncation."""
    text = "James said:\n> Yes, Thursday works.\nGreat — see you then."
    assert strip_quoted_replies(text) == text


# ---------- cap_length ----------

def test_cap_length_short_input_passes_through():
    text = "short"
    assert cap_length(text) == text


def test_cap_length_truncates_long_input_with_marker():
    text = "x" * (MAX_CHARS + 500)
    out = cap_length(text)
    assert len(out) < len(text)
    assert "truncated" in out
    assert str(MAX_CHARS + 500) in out  # original length is reported


# ---------- normalize_whitespace ----------

def test_normalize_whitespace_canonicalizes_line_endings():
    assert normalize_whitespace("a\r\nb\rc") == "a\nb\nc"


def test_normalize_whitespace_collapses_blank_line_runs():
    text = "para 1\n\n\n\npara 2"
    assert normalize_whitespace(text) == "para 1\n\npara 2"


def test_normalize_whitespace_strips_per_line_trailing_space():
    text = "hello   \nworld  \n"
    assert normalize_whitespace(text) == "hello\nworld"


# ---------- sanitize_email ----------

def test_sanitize_email_is_idempotent():
    text = "<p>Hi James,</p>\n\n\n<p>Thanks for applying.</p>"
    once = sanitize_email(text)
    twice = sanitize_email(once)
    assert once == twice


def test_sanitize_email_preserves_signal_in_interview_sample():
    """Regression guard: sanitation must not strip the signals the agent uses
    (company name, role title, sender domain) from a real-shape email."""
    raw = (SAMPLES / "interview-01.eml").read_text(encoding="utf-8")
    out = sanitize_email(raw)
    assert "Valon" in out
    assert "Senior Infrastructure Engineer" in out
    assert "jane@valon.com" in out


def test_sanitize_email_preserves_signal_in_rejection_sample():
    raw = (SAMPLES / "rejection-01.eml").read_text(encoding="utf-8")
    out = sanitize_email(raw)
    assert "Stripe" in out
    assert "Site Reliability Engineer" in out


def test_sanitize_email_drops_html_keeps_text():
    raw = (
        "<html><body><p>Hi James,</p>"
        "<p>Thanks for applying to Acme.</p></body></html>"
    )
    out = sanitize_email(raw)
    assert "Acme" in out
    assert "<p>" not in out
