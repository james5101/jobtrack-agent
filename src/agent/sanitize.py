"""Phase 3.1 guardrail — input sanitation.

Cleans an inbound email before it reaches the model. Cheap, deterministic,
no API calls. The four operations the README spec'd, in order:

1. strip_html        — extract visible text if the body is HTML.
2. strip_quoted_replies — drop reply chains (>quoted, "On X wrote:", Outlook).
3. cap_length        — hard ceiling so pathological emails can't blow context.
4. normalize_whitespace — collapse blank-line runs, canonical line endings.

Each function is pure (str -> str), idempotent, and safe to call on input
that doesn't need that operation. The orchestrator `sanitize_email` chains
all four.

Conservative bias throughout: removing real signal is worse than leaving
some noise. When a heuristic isn't confident, leave the text alone.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

MAX_CHARS = 8000

# HTML detection: matches an opening tag of a block-level element. Plain
# strings with stray `<` characters (e.g. email addresses, code snippets)
# won't trigger this — they need an actual tag name.
_HTML_MARKER = re.compile(
    r"<\s*(html|body|div|p|br|a|span|table|td|tr|th|h[1-6]|strong|em|li|ul|ol)\b",
    re.IGNORECASE,
)

# Gmail-style reply marker: "On <date>, <name> wrote:" anchored to a line.
_GMAIL_REPLY = re.compile(
    r"\n+On .{1,200}? wrote:\s*\n",
    re.IGNORECASE,
)

# Outlook-style block: From/Sent/To/Subject header cluster on consecutive lines.
_OUTLOOK_REPLY = re.compile(
    r"\n+From:\s.{1,200}?\nSent:\s.{1,200}?\nTo:\s.{1,200}?\n(?:Cc:\s.{1,200}?\n)?Subject:\s.{1,200}?\n",
    re.IGNORECASE | re.DOTALL,
)

# Two or more consecutive lines starting with `>` (quoted-text convention).
_QUOTED_BLOCK = re.compile(r"\n(?:>+\s?.*\n){2,}")


def strip_html(text: str) -> str:
    """Return visible text if the input looks like HTML; otherwise pass through.

    Uses the marker regex as a gate to avoid running BeautifulSoup on plain
    text — bs4 is forgiving and will happily parse anything, but doing so
    on a plain-text email with a stray `<` would still allocate work.
    """
    if not _HTML_MARKER.search(text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    # Drop script/style entirely — their content is never user-facing text.
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def strip_quoted_replies(text: str) -> str:
    """Truncate at the first detected reply-chain marker.

    Conservative: only acts on high-confidence patterns. The earliest match
    wins so a thread with both `>` quoting and a Gmail "On X wrote:" header
    gets cut at the header (which is upstream of the > lines).
    """
    earliest = len(text)
    for pattern in (_GMAIL_REPLY, _OUTLOOK_REPLY, _QUOTED_BLOCK):
        m = pattern.search(text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest].rstrip()


def cap_length(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate at max_chars with a marker noting the original length."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated; original was {len(text)} chars]"


def normalize_whitespace(text: str) -> str:
    """Canonicalize line endings, strip per-line trailing whitespace, collapse
    runs of 3+ newlines into a single paragraph break."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_email(raw: str) -> str:
    """Run all four operations in order. Idempotent."""
    text = strip_html(raw)
    text = strip_quoted_replies(text)
    text = cap_length(text)
    text = normalize_whitespace(text)
    return text
