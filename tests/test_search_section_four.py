"""The search must come back empty on noise, or it is a story generator.

Every number this produces will be used to decide whether real money goes on a
new strategy, and the account's whole history is of measurements that looked
right and were not. So the tests here are mostly about the search FAILING
correctly: a coin flip must be refused, a train-only fit must be refused, and
the multiple-testing bar must actually rise with the size of the grid.

The one test that checks it can find something uses a synthetic market with a
planted, unmistakable edge -- because a search that can only say no is equally
useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.search_section_four import (
    CANDIDATES,
    HORIZON,
    WARMUP,
    Cell,
    Trades,
    bonferroni_sigma,
    build_parser,
    random_control,
    resolve,
    stats,
    verdict,
)


def _walk(bars: int = 3000, drift: float = 0.0, seed: int = 5) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=bars, freq="30min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 + np.cumsum(rng.normal(drift, 0.1, bars)), index=index)
    span = np.abs(rng.normal(0.12, 0.03, bars))
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + span,
            "low": close - span,
            "close": close,
        },
        index=index,
    )


class TestTheBarRisesWithTheGrid:
    """Testing forty cells and keeping the best finds a 2-sigma result on pure
    noise most of the time. The correction is not a formality."""

    def test_one_hypothesis_is_about_two_sigma(self) -> None:
        assert 1.9 <= bonferroni_sigma(1) <= 2.1

    def test_forty_cells_demand_much_more(self) -> None:
        assert bonferroni_sigma(40) > 3.2

    def test_it_is_monotone_in_the_size_of_the_grid(self) -> None:
        bars = [bonferroni_sigma(n) for n in (1, 5, 20, 100, 500)]

        assert bars == sorted(bars)

    def test_an_empty_grid_does_not_hand_out_a_free_pass(self) -> None:
        assert bonferroni_sigma(0) >= 2.0


class TestSigmaIsClusteredByDay:
    def _trades(self, n: int, per: float, days: int, correlated: bool) -> Trades:
        import random

        rng = random.Random(11)
        out = Trades()
        for day in range(days):
            shared = per if rng.random() < 0.6 else -per
            for _k in range(n // days):
                value = shared if correlated else (per if rng.random() < 0.6 else -per)
                out.r.append(value)
                out.day.append(day)
                out.when.append(pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=day))
        return out

    def test_agreeing_trades_inside_a_day_count_for_less(self) -> None:
        """Sixteen markets breaking on one morning are ONE observation."""
        _t1, _e1, loose, _n1 = stats(self._trades(400, 1.0, 40, correlated=False))
        _t2, _e2, tight, _n2 = stats(self._trades(400, 1.0, 40, correlated=True))

        assert tight < loose

    def test_no_trades_is_zero_and_not_a_crash(self) -> None:
        assert stats(Trades()) == (0.0, 0.0, 0.0, 0)

    def test_a_single_day_reports_no_sigma_rather_than_infinity(self) -> None:
        one = Trades(r=[1.0, 1.0], day=["d", "d"], when=[pd.Timestamp("2025-01-01")] * 2)

        assert stats(one)[2] == 0.0


class TestTheResolverDoesNotFlatterTheCandidate:
    def test_a_bar_holding_both_barriers_is_a_loss(self) -> None:
        """The order inside a bar is unknowable, and assuming the good one is
        how every backtest this account has produced went wrong."""
        index = pd.date_range("2025-01-01", periods=WARMUP + 5, freq="30min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.2,
                "low": 99.8,
                "close": 100.0,
            },
            index=index,
        )
        # The bar after the signal spans far beyond both barriers.
        frame.iloc[WARMUP + 1, frame.columns.get_loc("high")] = 200.0
        frame.iloc[WARMUP + 1, frame.columns.get_loc("low")] = 1.0
        signals = np.zeros(len(frame), dtype=int)
        signals[WARMUP] = 1

        found = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0)

        assert found.r == [-1.0]

    def test_resolution_starts_on_the_bar_after_the_entry(self) -> None:
        """Same-bar look-ahead read +0.487R where the truth was +0.347R once.
        The entry bar's own extremes may not resolve the trade it created."""
        index = pd.date_range("2025-01-01", periods=WARMUP + 6, freq="30min", tz="UTC")
        frame = pd.DataFrame(
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0}, index=index
        )
        # The SIGNAL bar itself reaches far above the target.
        frame.iloc[WARMUP, frame.columns.get_loc("high")] = 500.0
        signals = np.zeros(len(frame), dtype=int)
        signals[WARMUP] = 1

        found = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0)

        assert found.r == [], "the entry bar resolved its own trade"

    def test_cost_is_charged_to_both_outcomes(self) -> None:
        frame = _walk()
        signals = np.zeros(len(frame), dtype=int)
        signals[WARMUP : WARMUP + 50] = 1

        free = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0)
        charged = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.10)

        assert len(free) == len(charged)
        assert sum(charged.r) == pytest.approx(sum(free.r) - 0.10 * len(free))

    def test_an_unresolved_trade_is_dropped_rather_than_marked_to_market(self) -> None:
        """A trade that reached neither barrier has not answered the question,
        and scoring it at zero would dilute every real outcome toward zero."""
        index = pd.date_range("2025-01-01", periods=WARMUP + HORIZON + 5, freq="30min", tz="UTC")
        frame = pd.DataFrame(
            {"open": 100.0, "high": 100.01, "low": 99.99, "close": 100.0}, index=index
        )
        signals = np.zeros(len(frame), dtype=int)
        signals[WARMUP] = 1

        assert resolve(frame, signals, stop_atr=5.0, ratio=5.0, cost_r=0.0).r == []


class TestItRefusesNoise:
    def _cell(self, train: Trades, test: Trades, control: Trades) -> Cell:
        cell = Cell("candidate", "M30", "index")
        cell.train, cell.test, cell.control = train, test, control
        return cell

    def _flat(self, n: int, per: float, days: int = 60) -> Trades:
        out = Trades()
        for i in range(n):
            out.r.append(per)
            out.day.append(i % days)
            out.when.append(pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i % days))
        return out

    def test_a_coin_flip_is_refused(self) -> None:
        cell = self._cell(self._flat(400, 0.0), self._flat(300, 0.0), self._flat(400, 0.0))

        ok, why = verdict(cell, bar=3.3)

        assert not ok
        assert "net of control" in why

    def test_a_candidate_that_only_matches_the_control_is_refused(self) -> None:
        """The harness's own bias is not an edge. If a candidate reads exactly
        what random entries read on the same bars, it has found nothing."""
        cell = self._cell(self._flat(400, 0.07), self._flat(300, 0.07), self._flat(400, 0.07))

        ok, why = verdict(cell, bar=3.3)

        assert not ok
        assert "net of control" in why

    def test_a_train_only_fit_is_refused(self) -> None:
        cell = self._cell(self._flat(400, 0.5), self._flat(300, -0.5), self._flat(400, 0.0))

        ok, why = verdict(cell, bar=1.0)

        assert not ok
        assert "train only" in why

    def test_too_few_trades_is_refused_however_good_it_looks(self) -> None:
        cell = self._cell(self._flat(40, 5.0), self._flat(20, 5.0), self._flat(400, 0.0))

        ok, why = verdict(cell, bar=1.0)

        assert not ok
        assert "too few trades" in why

    def test_the_holdout_must_reach_two_sigma_on_its_own(self) -> None:
        """Passing on train and merely leaning the right way afterwards is how
        an overfit looks from the inside."""
        train = self._flat(400, 0.5)
        test = Trades()
        for i in range(300):
            test.r.append(0.5 if i % 3 else -8.0)
            test.day.append(i % 60)
            test.when.append(pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i % 60))
        cell = self._cell(train, test, self._flat(400, 0.0))

        ok, why = verdict(cell, bar=1.0)

        assert not ok
        assert "holdout" in why


class TestItCanStillFindSomething:
    def test_a_planted_edge_clears_every_bar(self) -> None:
        """A search that can only say no is as useless as one that only says
        yes. Same shape as the refusal tests, with a real edge in both halves."""
        cell = Cell("planted", "M30", "index")
        for target, n in ((cell.train, 400), (cell.test, 300)):
            for i in range(n):
                target.r.append(0.6 if i % 4 else -1.0)
                target.day.append(i % 60)
                target.when.append(pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i % 60))
        cell.control = Trades(
            r=[0.0] * 400,
            day=[i % 60 for i in range(400)],
            when=[pd.Timestamp("2025-01-01", tz="UTC")] * 400,
        )

        ok, why = verdict(cell, bar=2.0)

        assert ok, why


class TestTheCandidatesAreWellFormed:
    @pytest.mark.parametrize("name", sorted(CANDIDATES))
    def test_each_returns_one_direction_per_bar(self, name: str) -> None:
        frame = _walk()

        signals = CANDIDATES[name](frame)

        assert len(signals) == len(frame)
        assert set(np.unique(signals)) <= {-1, 0, 1}

    @pytest.mark.parametrize("name", sorted(CANDIDATES))
    def test_none_of_them_fires_on_every_bar(self, name: str) -> None:
        """A detector that always fires is a random entry with extra steps."""
        frame = _walk()

        fired = int(np.count_nonzero(CANDIDATES[name](frame)))

        assert fired < len(frame) * 0.6, f"{name} fires on {fired}/{len(frame)} bars"

    def test_the_paired_directions_really_are_opposites(self) -> None:
        """`gap_fade` must be `gap_continuation` reversed, or the pair is not
        measuring one mechanism two ways."""
        frame = _walk()

        for forward, backward in (
            ("gap_continuation", "gap_fade"),
            ("streak_reversal", "streak_continuation"),
            ("close_position_in_range", "close_position_fade"),
            ("prior_day_break", "prior_day_fade"),
        ):
            assert np.array_equal(CANDIDATES[forward](frame), -CANDIDATES[backward](frame))

    def test_the_control_is_reproducible_and_sparse(self) -> None:
        frame = _walk()

        first = random_control(frame, seed=7)
        again = random_control(frame, seed=7)
        other = random_control(frame, seed=8)

        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)
        assert 0.2 < np.count_nonzero(first) / len(first) < 0.4


class TestTheLauncher:
    def test_the_parser_accepts_comma_and_space_forms(self) -> None:
        """cmd splits on commas, so both arrive in practice."""
        parsed = build_parser().parse_args(["--days", "365", "--clocks", "M15", "M30"])

        assert parsed.days == 365
        assert parsed.clocks == ["M15", "M30"]


def _realistic(bars: int = 4000, freq: str = "30min", seed: int = 3) -> pd.DataFrame:
    """Bars with the three properties a naive random walk flattens away.

    THE FIRST FIXTURE IN THIS FILE HAD NONE OF THEM and six of twelve
    candidates produced exactly zero trades on it: constant spread means the
    close always sits mid-bar, so `close_position_in_range` never fires;
    constant volatility means `volatility_contraction` never sees a quiet
    stretch and `range_expansion` never sees a big bar; and `open == previous
    close` means there is never a gap.

    None of that was a defect in the candidates. It was a fixture that could
    not express what they look for, and it would have read as "these detectors
    are broken".
    """
    index = pd.date_range("2025-01-01", periods=bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, bars)))
    vol = 8 * vol / vol.mean()
    step = rng.normal(0, 1, bars) * vol
    day = index.normalize().to_numpy()
    fresh = np.r_[False, day[1:] != day[:-1]]
    close = 15000 + np.cumsum(step)
    open_ = np.r_[close[0], close[:-1]].copy()
    # A real session gap: the open jumps AWAY from the previous close.
    open_[fresh] = np.r_[close[0], close[:-1]][fresh] + rng.normal(0, 4, fresh.sum()) * vol[fresh]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.6, 0.4, bars)) * vol
    low = np.minimum(open_, close) - np.abs(rng.normal(0.6, 0.4, bars)) * vol
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


class TestEveryCandidateCanActuallyFire:
    """A detector that never fires is not a detector that failed, and on a
    flat fixture six of these produced nothing at all."""

    @pytest.mark.parametrize("name", sorted(CANDIDATES))
    def test_it_fires_on_bars_that_contain_what_it_looks_for(self, name: str) -> None:
        frame = _realistic()

        fired = int(np.count_nonzero(CANDIDATES[name](frame)))

        assert fired > 0, f"{name} never fires even on bars built to contain its pattern"

    def test_the_gap_detector_reads_a_hand_made_gap(self) -> None:
        """Checked directly rather than through a generator, because the
        generator built the gap back out twice: setting the fresh-day open to
        `close - step` reproduces the previous close exactly, so the gap was
        zero and the detector looked broken."""
        from scripts.search_section_four import gap_continuation

        index = pd.date_range("2025-01-01", periods=60, freq="30min", tz="UTC")
        close = np.full(60, 100.0)
        open_ = np.full(60, 100.0)
        high, low = close + 1.0, close - 1.0
        open_[30], high[30], close[30] = 103.0, 104.0, 103.5

        signals = gap_continuation(
            pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)
        )

        assert signals[30] == 1

    def test_the_report_says_when_a_candidate_never_fired(self) -> None:
        """Silence means a threshold that does not match this feed -- my
        mistake, and fixable. Failure means the mechanism does not pay -- an
        answer. They must not print the same way."""
        import inspect

        from scripts import search_section_four

        source = inspect.getsource(search_section_four._report)

        assert "NEVER FIRED" in source
        assert "TOO THIN TO JUDGE" in source


class TestTheCostModelIsTheAccountsOwn:
    """I reimplemented `_cost_share` here and got it dimensionally wrong.

        pip = spec.point * 10.0
        commission_price = (per_side * 2.0) * pip / 10.0

    Commission is account currency per lot; multiplying it by a tenth of a pip
    does not convert it to price. It needs the instrument's pip VALUE, which
    depends on contract size and quote currency -- exactly what
    `spec.money_per_lot` and `spec.pips_to_price` already know.

    It surfaced as gold reading a 62% cost share on H1 against 0.2% on M30.
    Same instrument, same formula, and cost MUST fall on a slower clock
    because the stop is wider. A number that moves 300x the wrong way is not a
    property of gold.

    `PositionSizer._cost_share` is what the account actually charges, and its
    own docstring warns that "two definitions of the same cost would
    eventually disagree". I wrote the second one anyway.
    """

    def test_it_calls_the_sizer_rather_than_recomputing(self) -> None:
        import inspect

        from scripts import search_section_four

        source = inspect.getsource(search_section_four._cost_share)
        body = source.split('"""')[-1]

        assert "sizer._cost_share(" in body
        assert "spec.point" not in body, "the hand-rolled conversion is back"

    def test_cost_falls_as_the_stop_widens(self) -> None:
        """THE PROPERTY THE BUG VIOLATED. A wider stop carries the same fixed
        commission and slippage over more distance, so the share must fall --
        monotonically, at every width."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from risk.position_sizer import PositionSizer
        from scripts.search_section_four import _cost_share

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        sizer = PositionSizer(settings)

        class Spec:
            """Minimal stand-in: the two conversions the real cost uses."""

            asset_class = type("A", (), {"value": "forex"})()
            point = 0.00001

            @staticmethod
            def money_per_lot(distance: float) -> float:
                return distance * 100_000.0

            @staticmethod
            def pips_to_price(pips: float) -> float:
                return pips * 0.0001

        shares = [_cost_share(sizer, Spec(), stop) for stop in (0.0005, 0.0010, 0.0020, 0.0040)]

        assert shares == sorted(shares, reverse=True), shares
        assert all(0.0 < s < 1.0 for s in shares[1:]), shares

    def test_the_report_prints_the_share_and_flags_an_impossible_one(self) -> None:
        """It is the number the whole search turns on and it was invisible --
        the 62% only showed up indirectly, as a random control reading
        -0.6184 R."""
        import inspect

        from scripts import search_section_four

        source = inspect.getsource(search_section_four._report)

        assert "WHAT A ROUND TRIP COSTS" in source
        assert "SUSPECT" in source


class TestOppositeDirectionsCannotBothPay:
    """`gap_fade` is literally `-gap_continuation`, so within ONE cell their
    expectancies must sum to minus twice the cost:

        continuation   2p - 1 - c
        fade           1 - 2p - c
        sum                  -2c

    The owner read the search result as both being strong. They appeared in
    the top four on DIFFERENT clocks and asset classes, which is not a
    contradiction -- but it is also not a mechanism, because a real gap effect
    points the same way everywhere. This test pins the arithmetic so the claim
    can be checked rather than argued.
    """

    def test_the_pair_sums_to_minus_twice_the_cost_on_the_same_bars(self) -> None:
        from scripts.search_section_four import CANDIDATES, resolve

        frame = _realistic(6000)
        cost = 0.05

        forward = resolve(
            frame, CANDIDATES["gap_continuation"](frame), stop_atr=1.0, ratio=1.0, cost_r=cost
        )
        backward = resolve(
            frame, CANDIDATES["gap_fade"](frame), stop_atr=1.0, ratio=1.0, cost_r=cost
        )

        assert len(forward) > 50 and len(backward) > 50
        each = sum(forward.r) / len(forward) + sum(backward.r) / len(backward)

        assert each < 0.0, "opposite entries on one market cannot both pay"
        assert each == pytest.approx(-2 * cost, abs=0.25)
