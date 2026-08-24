"""Section six: the fast-bot shape, with the candle demoted to a trigger.

Watch M1, jump in when a candle goes, jump straight back out. What makes those
bots lose is that the candle is the whole thesis, and a green minute happens
about half the time. Here M15 decides the side, M5 must not contradict it, and
the M1 close is only the moment that standing agreement becomes actionable.

Two tests matter more than the rest: the extreme-volume refusal, because that is
the release candle that takes these accounts apart, and the news blackout, which
is enforced by the rules already in force rather than by a second copy.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from analysis.momentum_scalp import MomentumScalp, read_candle, slope_direction
from config.schema import MomentumScalpConfig
from core.types import MarketContext, Series, Timeframe

BASE = 2400.0


def frame(
    closes: list[float],
    *,
    body: float = 1.0,
    wick: float = 0.2,
    volume: float = 100.0,
    last_volume: float | None = None,
    last_body: float | None = None,
    last_close_at: float = 0.6,
    freq: str = "1min",
) -> pd.DataFrame:
    """Bars whose final candle can be shaped independently of the rest."""
    rows = []
    for i, close in enumerate(closes):
        final = i == len(closes) - 1
        size = last_body if final and last_body is not None else body
        open_ = close - size
        low = min(open_, close) - wick
        high = max(open_, close) + wick
        if final:
            span = max(1e-9, high - low)
            low = close - last_close_at * span
            high = low + span
            open_ = close - size
        rows.append(
            {
                "open": open_,
                "high": max(high, open_, close),
                "low": min(low, open_, close),
                "close": close,
                "tick_volume": (last_volume if final and last_volume is not None else volume),
                "spread": 20,
                "real_volume": 0,
            }
        )
    index = pd.date_range("2026-08-24T09:00", periods=len(rows), freq=freq, tz=UTC)
    return pd.DataFrame(rows, index=index)


def context(
    *,
    m1_closes: list[float],
    m5_up: bool = True,
    m15_up: bool = True,
    **candle,
) -> MarketContext:
    step = 0.6
    m5 = [BASE + (i * step if m5_up else -i * step) for i in range(40)]
    m15 = [BASE + (i * step if m15_up else -i * step) for i in range(40)]
    now = pd.Timestamp("2026-08-24T09:40", tz=UTC).to_pydatetime()
    return MarketContext(
        symbol="XAUUSD",
        now=now,
        series={
            Timeframe.M1: Series("XAUUSD", Timeframe.M1, frame(m1_closes, **candle), now),
            Timeframe.M5: Series("XAUUSD", Timeframe.M5, frame(m5, freq="5min"), now),
            Timeframe.M15: Series("XAUUSD", Timeframe.M15, frame(m15, freq="15min"), now),
        },
        tick=None,
    )


def rising(n: int = 40, step: float = 0.5) -> list[float]:
    return [BASE + i * step for i in range(n)]


def falling(n: int = 40, step: float = 0.5) -> list[float]:
    return [BASE - i * step for i in range(n)]


class TestTheCandleRead:
    def test_a_doji_has_almost_no_body(self) -> None:
        read = read_candle(frame(rising(), body=1.0, last_body=0.01, wick=2.0), 30)

        assert read is not None
        assert read.body_share < 0.1

    def test_a_big_candle_reads_as_a_multiple_of_the_recent_ones(self) -> None:
        read = read_candle(frame(rising(), body=1.0, last_body=5.0), 30)

        assert read is not None
        assert read.body_multiple == pytest.approx(5.0, rel=0.2)

    def test_volume_is_measured_against_its_own_median(self) -> None:
        read = read_candle(frame(rising(), volume=100.0, last_volume=900.0), 30)

        assert read is not None
        assert read.volume_multiple == pytest.approx(9.0, rel=0.1)

    def test_too_little_history_is_no_read(self) -> None:
        assert read_candle(frame(rising(5)), 30) is None


class TestSlope:
    def test_it_is_a_sign_and_nothing_more(self) -> None:
        assert slope_direction(pd.Series(np.arange(20.0) + 100.0), 6) == 1
        assert slope_direction(pd.Series(120.0 - np.arange(20.0)), 6) == -1
        assert slope_direction(pd.Series([100.0] * 20), 6) == 0


class TestTheRefusalsThatMatter:
    def test_an_event_candle_is_refused_however_good_it_looks(self) -> None:
        """THE ONE THAT KEEPS THIS OUT OF THE DITCH. A minute carrying many
        times its own normal activity is a release, a headline or a stop
        cascade. It is the strongest-looking candle such a bot will ever print,
        and what follows it is a spread that takes the account apart."""
        signal = MomentumScalp().analyze(
            context(m1_closes=rising(), last_body=4.0, last_volume=2000.0)
        )

        assert signal.score == 0.0
        assert "an event, not momentum" in signal.reasoning

    def test_a_mostly_wick_candle_is_refused(self) -> None:
        signal = MomentumScalp().analyze(context(m1_closes=rising(), last_body=0.2, wick=6.0))

        assert signal.score == 0.0
        assert "mostly wick" in signal.reasoning

    def test_an_ordinary_candle_is_not_a_move(self) -> None:
        signal = MomentumScalp().analyze(context(m1_closes=rising(), last_body=1.0))

        assert signal.score == 0.0
        assert "rather than a minute" in signal.reasoning

    def test_a_candle_closing_on_its_extreme_is_the_last_buyer(self) -> None:
        signal = MomentumScalp().analyze(
            context(m1_closes=rising(), last_body=4.0, last_close_at=0.99)
        )

        assert signal.score == 0.0
        assert "last buyer" in signal.reasoning

    def test_two_out_of_three_is_a_disagreement(self) -> None:
        """A green M1 inside a falling M5 is not a majority. On a trade lasting
        minutes it is a coin flip with costs attached."""
        signal = MomentumScalp().analyze(context(m1_closes=rising(), last_body=4.0, m5_up=False))

        assert signal.score == 0.0
        assert "disagreement" in signal.reasoning


class TestWhenItDoesFire:
    def test_all_three_agreeing_produces_a_long(self) -> None:
        signal = MomentumScalp().analyze(context(m1_closes=rising(), last_body=4.0))

        assert signal.score > 0
        assert signal.details["confirm_direction"] == 1
        assert signal.details["bias_direction"] == 1

    def test_all_three_agreeing_downward_produces_a_short(self) -> None:
        signal = MomentumScalp().analyze(
            context(
                m1_closes=falling(),
                last_body=-4.0,
                last_close_at=0.4,
                m5_up=False,
                m15_up=False,
            )
        )

        assert signal.score < 0

    def test_a_bigger_body_scores_higher(self) -> None:
        small = MomentumScalp().analyze(context(m1_closes=rising(), last_body=1.8))
        large = MomentumScalp().analyze(context(m1_closes=rising(), last_body=5.0))

        assert abs(large.score) > abs(small.score)

    def test_the_reading_says_the_volume_was_normal(self) -> None:
        """The refusal is the headline of this module, so a firing signal has
        to state that the test was run and passed."""
        signal = MomentumScalp().analyze(context(m1_closes=rising(), last_body=4.0))

        assert "not an event" in signal.reasoning
        assert "volume_multiple" in signal.details

    def test_disabling_it_silences_it(self) -> None:
        module = MomentumScalp(MomentumScalpConfig(enabled=False))

        assert module.analyze(context(m1_closes=rising(), last_body=4.0)).score == 0.0


class TestTheNewsBlackoutIsTheAccountsAndNotACopy:
    """The module reads bars and has no calendar, on purpose. A second copy of
    the news rules beside it would be a copy that eventually disagrees, and the
    direction it would disagree in is "traded through a release nobody meant to
    trade through"."""

    def _runner(self, recorded: list):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Tick
        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        ctx = context(m1_closes=rising(), last_body=4.0)
        live = ctx.__class__(
            symbol=ctx.symbol,
            now=ctx.now,
            series=ctx.series,
            tick=Tick(ctx.symbol, ctx.now, BASE, BASE + 0.2),
        )
        runner._cycle_contexts = {"XAUUSD": live}
        runner.recorder = SimpleNamespace(  # type: ignore[assignment]
            has_unresolved_shadow_trade=lambda *_: False,
            record_shadow_trade=lambda **kw: recorded.append(kw) or 1,
        )
        return runner, [MomentumScalp().analyze(live)]

    def test_a_clear_market_is_recorded(self) -> None:
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {"minutes_to_news": 90.0})

        assert len(recorded) == 1
        assert str(recorded[0]["blocked_by"]) == "SECTION_6_OBSERVED"

    def test_a_news_blackout_records_nothing(self) -> None:
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NEWS_BLACKOUT, {})

        assert recorded == []

    def test_an_unreachable_calendar_records_nothing_either(self) -> None:
        """Fail-closed, matching the account. No data is no trade, and a paper
        section that quietly kept going would be measuring a strategy nobody
        would ever run."""
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NEWS_CALENDAR_UNAVAILABLE, {})

        assert recorded == []

    def test_a_release_inside_the_clearance_window_records_nothing(self) -> None:
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {"minutes_to_news": 3.0})

        assert recorded == []

    def test_in_small_out_small(self) -> None:
        """The owner's whole description of this: a hair wrong and out, a
        little right and also out. So the target is SMALLER than the stop, and
        that is a deliberate choice about what a scalp is rather than an
        oversight about reward-to-risk."""
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {"minutes_to_news": 90.0})

        plan = recorded[0]
        reward = abs(plan["tp"] - plan["entry_price"])
        risk = abs(plan["entry_price"] - plan["sl"])
        assert reward < risk


class TestItObservesAndCannotTrade:
    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_it_is_weighted_but_never_live(self) -> None:
        from core.types import TradingMode

        confluence = self._confluence()

        assert confluence.weights["momentum_scalp"] > 0
        assert "momentum_scalp" not in confluence.live_enabled_modules
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["momentum_scalp"] == 0.0

    def test_it_shares_the_momentum_family_rather_than_inventing_one(self) -> None:
        """It reads the same fact the other momentum readers read — price moved,
        hard, just now. Its own label would let it "corroborate" impulse_break
        on one observation seen twice, which is what families exist to stop."""
        from analysis.evidence_families import family_for

        assert family_for("momentum_scalp") == "momentum"
        assert family_for("impulse_break") == "momentum"


class TestTheConfigCannotBeIncoherent:
    def test_saturation_must_exceed_the_minimum_body(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            MomentumScalpConfig(minimum_body_multiple=4.0, body_saturation_multiple=3.0)

    def test_confidence_may_not_narrow_to_nothing(self) -> None:
        with pytest.raises(ValueError, match="below base_confidence"):
            MomentumScalpConfig(base_confidence=0.9, maximum_confidence=0.4)
