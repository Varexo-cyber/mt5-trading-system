"""What the adviser costs, from recorded tokens rather than guesses.

"Is this burning my credit" had no answer anywhere in the interface, and the
number that matters most — how much of each prompt is served from cache — is
not visible outside the API response at all. If the cache hit rate collapses,
something has started varying the system prompt and every call is paying full
price for a prefix that used to be nearly free.
"""

from __future__ import annotations

import pytest

from dashboard.ai_exchange import call_cost, pair_ai_reviews, spend_summary

SONNET = "claude-sonnet-5"


def usage(fresh: int = 0, written: int = 0, read: int = 0, out: int = 0) -> dict[str, int]:
    return {
        "input_tokens": fresh,
        "cache_creation_input_tokens": written,
        "cache_read_input_tokens": read,
        "output_tokens": out,
    }


def exchange(model: str = SONNET, **counts: int) -> dict[str, object]:
    return {"decision": {"model": model, "usage": usage(**counts)}}


# ------------------------------------------------------------- per call ---


def test_plain_input_and_output_are_priced() -> None:
    # 1M input at $2 plus 1M output at $10.
    assert call_cost(SONNET, usage(fresh=1_000_000, out=1_000_000)) == pytest.approx(12.0)


def test_a_cache_read_costs_a_tenth() -> None:
    assert call_cost(SONNET, usage(read=1_000_000)) == pytest.approx(0.2)


def test_a_cache_write_costs_a_quarter_more() -> None:
    """Which is why the break-even is two requests, not one."""
    assert call_cost(SONNET, usage(written=1_000_000)) == pytest.approx(2.5)


def test_caching_is_cheaper_from_the_second_call() -> None:
    """The whole justification, as arithmetic: write once, read after."""
    uncached = 2 * call_cost(SONNET, usage(fresh=100_000))
    cached = call_cost(SONNET, usage(written=100_000)) + call_cost(SONNET, usage(read=100_000))
    assert cached < uncached


def test_an_unknown_model_reports_nothing_rather_than_a_guess() -> None:
    """A plausible wrong number is worse than no number."""
    assert call_cost("some-future-model", usage(fresh=1000)) is None


def test_a_call_with_no_usage_is_not_priced() -> None:
    assert call_cost(SONNET, {}) is None


# --------------------------------------------------------------- summary ---


def test_the_summary_adds_up() -> None:
    summary = spend_summary(
        [
            exchange(fresh=1_000_000, out=100_000),
            exchange(read=1_000_000, out=100_000),
        ]
    )
    assert summary["calls"] == 2
    # 2.00 + 1.00 fresh/output, then 0.20 + 1.00 cached/output.
    assert summary["usd"] == pytest.approx(4.2)
    assert summary["usd_per_call"] == pytest.approx(2.1)


def test_the_cache_rate_counts_the_whole_prompt() -> None:
    """`input_tokens` is the uncached remainder, not the prompt size. Dividing
    by it alone would report a caching win that is really a measurement error.
    """
    summary = spend_summary([exchange(fresh=1_000, written=0, read=9_000)])
    assert summary["input_tokens"] == 10_000
    assert summary["cache_hit_rate"] == pytest.approx(0.9)


def test_no_calls_does_not_divide_by_zero() -> None:
    summary = spend_summary([])
    assert summary["calls"] == 0
    assert summary["usd_per_call"] == 0.0
    assert summary["cache_hit_rate"] == 0.0


def test_rows_without_usage_are_skipped_not_counted() -> None:
    """Older ledger rows predate token recording. Counting them would drag the
    per-call average toward zero and make the spend look better than it is."""
    summary = spend_summary([{"decision": {"model": SONNET}}, exchange(fresh=1_000_000)])
    assert summary["calls"] == 1
    assert summary["usd_per_call"] == pytest.approx(2.0)


def test_an_unpriced_model_still_counts_as_a_call() -> None:
    """The call happened and the tokens were spent; only the price is unknown."""
    summary = spend_summary([exchange(model="mystery", fresh=1_000)])
    assert summary["calls"] == 1
    assert summary["priced_calls"] == 0
    assert summary["usd"] == 0.0


# ---------------------------------------------------------------- status ---


def test_a_near_miss_is_not_reported_as_a_veto() -> None:
    """ "This is a bad trade" and "this is fine, I am just not sure enough" call
    for opposite responses, and both used to render as VETO."""
    rows = [
        {"cycle_id": "c1", "event": "pretrade_request", "symbol": "EURUSD"},
        {
            "cycle_id": "c1",
            "event": "pretrade_response",
            "decision": {"approved": False, "said_yes": True, "confidence": 0.42},
        },
    ]
    assert pair_ai_reviews(rows)[0]["status"] == "UNDER THRESHOLD"


def test_a_real_veto_still_reads_as_a_veto() -> None:
    rows = [
        {"cycle_id": "c1", "event": "pretrade_request", "symbol": "EURUSD"},
        {
            "cycle_id": "c1",
            "event": "pretrade_response",
            "decision": {"approved": False, "said_yes": False, "confidence": 0.90},
        },
    ]
    assert pair_ai_reviews(rows)[0]["status"] == "VETO"


def test_an_approval_is_unaffected() -> None:
    rows = [
        {"cycle_id": "c1", "event": "pretrade_request", "symbol": "EURUSD"},
        {
            "cycle_id": "c1",
            "event": "pretrade_response",
            "decision": {"approved": True, "said_yes": True, "confidence": 0.71},
        },
    ]
    assert pair_ai_reviews(rows)[0]["status"] == "APPROVED"
