"""Unit tests for src/agent/loop.py helpers.

Most of `loop.py` is the async agent loop that requires the live SDK + jobtrack
MCP. We don't unit-test that here. What we DO test is the small, pure pieces
that have caught us before — specifically the runtime-exception classifier
that maps Phase 3.4 cap errors to a (cap_hit, result_subtype) tuple.
"""
from agent.loop import _classify_runtime_exception


def test_classify_max_turns_exception():
    e = Exception("Claude Code returned an error result: Reached maximum number of turns (1)")
    cap_hit, subtype = _classify_runtime_exception(e)
    assert cap_hit == "turns"
    assert subtype == "max_turns"


def test_classify_max_turns_short_form():
    e = Exception("max turns reached")
    cap_hit, subtype = _classify_runtime_exception(e)
    assert cap_hit == "turns"


def test_classify_budget_exception_variants():
    """Defensive matching across plausible SDK message phrasings."""
    for msg in (
        "Reached maximum budget of $0.01",
        "max_budget exceeded",
        "Maximum cost reached",
        "max cost limit hit",
    ):
        cap_hit, _ = _classify_runtime_exception(Exception(msg))
        assert cap_hit == "budget", f"failed to classify {msg!r}"


def test_classify_unknown_exception_falls_through():
    """An exception that isn't cap-related must not be silently miscategorized."""
    e = Exception("Connection refused")
    cap_hit, subtype = _classify_runtime_exception(e)
    assert cap_hit is None
    assert subtype.startswith("sdk_error:")
    assert "Connection refused" in subtype


def test_classify_is_case_insensitive():
    e = Exception("REACHED MAXIMUM NUMBER OF TURNS (5)")
    cap_hit, _ = _classify_runtime_exception(e)
    assert cap_hit == "turns"
