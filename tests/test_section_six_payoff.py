"""Fed a coin, the table must say coin.

`section6.cmd --sweep` walked the cost gate and reported a win rate of 42% at
every gate from 5 to 40 while the cost band fell from 28% of R to 3.5%. So the
entire difference between -0.332R and -0.064R was the spread, and the
detector's opinion contributed nothing measurable either way.

That sweep cannot ask the question underneath it. The lane targets 1.4 spans
against a 1.0 span stop, so a driftless walk resolves in its favour
1/(1+1.4) = 41.7% of the time — and the detector managed 42%. The banner
already carried that conclusion in R ("it is worth +0.03R") without anything
putting the number beside the one it has to beat.

`--payoff` moves the TARGET instead of the gate and prints the achieved rate
next to chance. Which makes it a tool that can invent an edge, and it did,
twice, before this test existed:

    fed a pure random walk, version one printed  "Positive edge +16.7%"
    after counting only barrier-resolved trades  "+7.8% at 2.0R over 34 trades"

The first mixed clock exits into a first-touch model. The second was real
arithmetic on a sample too small to mean anything: the standard error of a 33%
rate over 34 draws is 8.1%, so +7.8% is one sigma.

A tool built to say whether a detector beats chance must not manufacture an
edge out of noise. That is worse than the question going unanswered, because
an unanswered question does not get acted on.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestOrder
from core.types import Direction
from scripts.backtest_section_six import _sigmas, payoff_sweep, render_payoff

RATIOS = (0.5, 1.0, 1.4, 2.0)


def _coin_flip_market(seed: int = 7, bars: int = 6000):  # type: ignore[no-untyped-def]
    """A driftless random walk and entries that ignore it.

    Directions alternate on the index, so nothing about the entry can carry
    information about what comes next. Any edge this produces is noise by
    construction, which is the whole point of a control.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-06-01", periods=bars, freq="1min", tz=UTC)
    price = 100 + np.cumsum(rng.normal(0, 0.01, bars))
    minute = pd.DataFrame(
        {"open": price, "high": price + 0.02, "low": price - 0.02, "close": price, "spread": 0},
        index=index,
    )
    orders = [
        BacktestOrder(
            symbol="COIN",
            decided_at=index[i].to_pydatetime(),
            direction=Direction.LONG if i % 2 else Direction.SHORT,
            entry=float(price[i]),
            stop_loss=float(price[i]) - 0.10 if i % 2 else float(price[i]) + 0.10,
            take_profit=float(price[i]) + 0.14 if i % 2 else float(price[i]) - 0.14,
        )
        for i in range(0, bars - 600, 20)
    ]
    return orders, minute


class TestTheControlRun:
    def test_a_random_walk_is_never_reported_as_an_edge(self) -> None:
        """The regression, stated as the sentence the operator would read."""
        orders, minute = _coin_flip_market()

        report = render_payoff(payoff_sweep(orders, minute, RATIOS), "control")

        assert "That is chance" in report
        assert "Positive edge" not in report
        assert "outside chance" not in report

    @pytest.mark.parametrize("seed", [1, 7, 42, 99])
    def test_it_holds_across_seeds_and_not_just_the_lucky_one(self, seed: int) -> None:
        """One seed proves the arithmetic ran, not that it is right. A control
        that passes on a single draw is the same mistake as a two-sigma bar
        applied to a sample of thirty."""
        orders, minute = _coin_flip_market(seed=seed)

        report = render_payoff(payoff_sweep(orders, minute, RATIOS), f"seed {seed}")

        # THE TOOL'S OWN VERDICT, not a per-row bar of my choosing. Seed 1
        # produces +2.1 sigma at 0.5R, and that is the finding rather than a
        # test failure: four looks at a 5% test fire on noise about once every
        # ten runs. The bar has to know how many times the question was asked,
        # which is what `_significance_bar` is for.
        assert "That is chance" in report, report

    def test_every_row_counts_only_barrier_resolved_trades(self) -> None:
        """The first fault, pinned. A clock exit touched neither barrier, so
        counting it against a first-touch probability compares two different
        populations — and it inflated the edge by roughly nine points."""
        orders, minute = _coin_flip_market()

        rows = payoff_sweep(orders, minute, RATIOS)

        # Every reported rate is a share of trades that hit TP or SL, so it can
        # never exceed 1 and the count falls as the target moves away.
        counts = [trades for _r, trades, *_rest in rows]
        assert counts == sorted(counts, reverse=True)
        for _ratio, _trades, won, *_rest in rows:
            assert 0.0 <= won <= 1.0


class TestTheChanceBaseline:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.5, 2 / 3), (1.0, 0.5), (1.4, 1 / 2.4), (2.0, 1 / 3), (3.0, 0.25)],
    )
    def test_chance_is_the_first_touch_probability(self, ratio: float, expected: float) -> None:
        """`1/(1+ratio)` for a driftless walk between a stop at -1 and a target
        at +ratio. The live lane is 1.4, which puts chance at 41.7% — and the
        30-day sweep measured 42%."""
        orders, minute = _coin_flip_market()

        rows = payoff_sweep(orders, minute, (ratio,))

        assert rows[0][3] == pytest.approx(expected)


class TestTheErrorBar:
    def test_a_real_edge_at_a_real_sample_does_clear_two_sigma(self) -> None:
        """The other direction. A test that only proves the tool says no would
        pass on a tool that says no to everything."""
        # 60% against a 41.7% baseline over 300 trades is about 6.5 sigma.
        assert _sigmas(0.60, 1 / 2.4, 300) > 2.0

    def test_the_same_gap_on_a_small_sample_does_not(self) -> None:
        assert _sigmas(0.60, 1 / 2.4, 8) < 2.0

    def test_an_empty_or_impossible_sample_is_zero_and_not_infinite(self) -> None:
        """A divide-by-zero here would print `inf sigma` and read as the
        strongest finding the tool has ever produced."""
        assert _sigmas(0.5, 0.5, 0) == 0.0
        assert _sigmas(0.5, 0.0, 100) == 0.0
        assert _sigmas(0.5, 1.0, 100) == 0.0
