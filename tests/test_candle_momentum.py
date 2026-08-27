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

from analysis.candle_momentum import CandleMomentum, read_candle, slope_direction
from config.schema import CandleMomentumConfig
from core.types import MarketContext, Series, Timeframe

BASE = 2400.0


def frame(
    closes: list[float],
    *,
    body: float = 1.0,
    wick: float = 0.5,
    volume: float = 100.0,
    last_volume: float | None = None,
    last_body: float | None = None,
    last_close_at: float = 0.85,
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


def _chain(*filters):  # type: ignore[no-untyped-def]
    """A stand-in for `FilterChain` that behaves the way the real one does.

    Section six now runs the WHOLE entry chain rather than reaching past it for
    the calendar alone, so a double that only answers `find` no longer
    describes the object the lane talks to. Iterating and returning the first
    refusal is the entire contract; getting it wrong here would make these
    tests pass against a lane that never asked anything.
    """
    from types import SimpleNamespace

    def check(ctx):  # type: ignore[no-untyped-def]
        collected: dict = {}
        for filter_ in filters:
            verdict = filter_.check(ctx)
            collected.update(getattr(verdict, "data", {}) or {})
            if not verdict.passed:
                return verdict, collected
        return SimpleNamespace(
            passed=True, filter_name="chain", detail="clear", reason=None, data=collected
        ), collected

    return SimpleNamespace(find=lambda _kind: filters[0], filters=filters, check=check)


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


def bars_from(closes: list[float], *, bar_range: float = 1.0) -> pd.DataFrame:
    """Bars with a known average high-low range, so the floor is computable."""
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + bar_range / 2 for c in closes],
            "low": [c - bar_range / 2 for c in closes],
            "close": closes,
        }
    )


class TestSlope:
    """WHERE SECTION SIX TOOK THE WRONG SIDE.

    The original read two closes and returned a sign with no minimum, so a
    change of one hundred-thousandth of a percent counted as a trend. On a flat
    chart the sign is a coin flip, which made "M1, M5 and M15 all agree" a
    one-in-four coincidence rather than evidence -- and the trade that followed
    was a lone M1 candle with nothing behind it.
    """

    def test_it_is_still_a_sign_when_the_chart_really_moves(self) -> None:
        assert slope_direction(bars_from(list(np.arange(20.0) + 100.0)), 6) == 1
        assert slope_direction(bars_from(list(120.0 - np.arange(20.0))), 6) == -1
        assert slope_direction(bars_from([100.0] * 20), 6) == 0

    def test_a_move_smaller_than_the_bars_themselves_is_not_a_direction(self) -> None:
        """THE DEFECT, ASSERTED. Six bars drifting up by a thousandth apiece,
        on bars that are a full point tall: the old code called that a trend
        and let a scalp through on it."""
        closes = [100.0 + i * 0.001 for i in range(20)]

        assert slope_direction(bars_from(closes, bar_range=1.0), 6, 0.0) == 1
        assert slope_direction(bars_from(closes, bar_range=1.0), 6, 0.5) == 0

    def test_the_floor_is_in_the_instruments_own_units(self) -> None:
        """A pip count would switch this off on gold or on EURUSD depending on
        which one it was tuned for. The same drift is a trend on quiet bars and
        noise on wide ones."""
        closes = [100.0 + i * 0.2 for i in range(20)]

        assert slope_direction(bars_from(closes, bar_range=0.5), 6, 0.5) == 1
        assert slope_direction(bars_from(closes, bar_range=10.0), 6, 0.5) == 0

    def test_a_flat_chart_is_zero_and_not_floating_point_noise(self) -> None:
        """`polyfit` on twenty identical closes returns about 1e-15, not zero,
        so `travel > 0` reported a confident +1 on a chart that had not moved
        at all -- this function's original defect, reappearing inside its own
        repair."""
        assert slope_direction(bars_from([100.0] * 20), 6) == 0
        assert slope_direction(bars_from([4021.37] * 20), 6) == 0

    def test_one_spike_bar_can_no_longer_invert_the_whole_window(self) -> None:
        """Two points meant whichever bar sat at -(bars + 1) decided the answer
        by itself. With `bias_bars` at 4 that is a quarter of the evidence
        resting on one close.

        Least squares is not immune to an outlier -- a big enough one still
        moves the fit -- but it weighs the bar instead of handing it the
        casting vote, and that is the difference this test pins.
        """
        spiked = [107.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]

        # Endpoint arithmetic reads a FALL across a window that plainly rises,
        # because the one bar it happens to start from sits above the last.
        assert spiked[0] > spiked[-1]
        assert slope_direction(bars_from(spiked), 6) == 1


class TestM1HasToAgreeWithItself:
    """THE OWNER'S COMPLAINT, PINNED: "it is a buy and dumb section six does a
    sell."

    M1 was read as a single candle -- body, wick, volume -- and never as a
    direction. So one green minute inside a FALLING M1 sequence was a buy, as
    long as M5 and M15 happened to point up: the side agreed with the slower
    charts and disagreed with the chart the trade was actually taken on.
    """

    def test_one_green_minute_inside_a_falling_m1_chart_is_refused(self) -> None:
        signal = CandleMomentum().analyze(
            context(m1_closes=[*falling(39), BASE - 38 * 0.5 + 4.0], last_body=4.0)
        )

        assert signal.score == 0.0
        assert "against its own chart" in signal.reasoning
        assert signal.details["m1_direction"] == -1

    def test_a_quiet_stretch_before_the_push_is_still_allowed(self) -> None:
        """Not symmetric with M5 and M15 on purpose. Those must AGREE -- they
        are the thesis. M1 need only not contradict, because the first minute
        of a real push often follows a flat stretch, and refusing those removes
        the setups this module exists for."""
        signal = CandleMomentum().analyze(
            context(m1_closes=[*([BASE] * 39), BASE + 4.0], last_body=4.0)
        )

        assert signal.details["m1_direction"] == 0
        assert signal.score > 0

    def test_the_trigger_candle_is_not_allowed_to_vouch_for_itself(self) -> None:
        """The trigger is by construction a big bar and it is the newest one,
        so a fit that included it would be dragged its own way and the test
        would confirm itself. The reading has to come from the bars BEFORE it.

        Same falling chart as the first case; the only thing arguing upward is
        the trigger candle. If it were included the window could read +1 and
        the refusal would vanish.
        """
        # 4.5, not 12: a body that large is now refused outright as an
        # event before the M1 read is reached, and this test is about the
        # M1 read.
        closes = [*falling(39), BASE - 38 * 0.5 + 4.5]
        signal = CandleMomentum().analyze(context(m1_closes=closes, last_body=4.5))

        assert signal.details["m1_direction"] == -1
        assert signal.score == 0.0


class TestTheRefusalsThatMatter:
    def test_an_event_candle_is_refused_however_good_it_looks(self) -> None:
        """THE ONE THAT KEEPS THIS OUT OF THE DITCH. A minute carrying many
        times its own normal activity is a release, a headline or a stop
        cascade. It is the strongest-looking candle such a bot will ever print,
        and what follows it is a spread that takes the account apart."""
        signal = CandleMomentum().analyze(
            context(m1_closes=rising(), last_body=4.0, last_volume=2000.0)
        )

        assert signal.score == 0.0
        assert "an event, not momentum" in signal.reasoning

    def test_a_mostly_wick_candle_is_refused(self) -> None:
        signal = CandleMomentum().analyze(
            context(m1_closes=rising(), last_body=0.2, wick=6.0, last_close_at=0.5)
        )

        assert signal.score == 0.0
        assert "mostly wick" in signal.reasoning

    def test_an_ordinary_candle_is_not_a_move(self) -> None:
        signal = CandleMomentum().analyze(context(m1_closes=rising(), last_body=1.4, wick=0.15))

        assert signal.score == 0.0
        assert "rather than a minute" in signal.reasoning

    def test_a_candle_closing_on_its_extreme_is_the_last_buyer(self) -> None:
        signal = CandleMomentum().analyze(
            context(m1_closes=rising(), last_body=4.0, last_close_at=0.99)
        )

        assert signal.score == 0.0
        assert "last buyer" in signal.reasoning

    def test_two_out_of_three_is_a_disagreement(self) -> None:
        """A green M1 inside a falling M5 is not a majority. On a trade lasting
        minutes it is a coin flip with costs attached."""
        signal = CandleMomentum().analyze(context(m1_closes=rising(), last_body=4.0, m5_up=False))

        assert signal.score == 0.0
        assert "disagreement" in signal.reasoning


class TestWhenItDoesFire:
    def test_all_three_agreeing_produces_a_long(self) -> None:
        signal = CandleMomentum().analyze(context(m1_closes=rising(), last_body=4.0))

        assert signal.score > 0
        assert signal.details["confirm_direction"] == 1
        assert signal.details["bias_direction"] == 1

    def test_all_three_agreeing_downward_produces_a_short(self) -> None:
        signal = CandleMomentum().analyze(
            context(
                m1_closes=falling(),
                last_body=-4.0,
                last_close_at=0.15,
                m5_up=False,
                m15_up=False,
            )
        )

        assert signal.score < 0

    def test_a_bigger_body_scores_higher(self) -> None:
        small = CandleMomentum().analyze(context(m1_closes=rising(), last_body=1.8))
        # Just under `maximum_body_multiple`, which refuses at 5.0.
        large = CandleMomentum().analyze(context(m1_closes=rising(), last_body=4.5))

        assert abs(large.score) > abs(small.score)

    def test_the_reading_says_the_volume_was_normal(self) -> None:
        """The refusal is the headline of this module, so a firing signal has
        to state that the test was run and passed."""
        signal = CandleMomentum().analyze(context(m1_closes=rising(), last_body=4.0))

        assert "not an event" in signal.reasoning
        assert "volume_multiple" in signal.details

    def test_disabling_it_silences_it(self) -> None:
        module = CandleMomentum(CandleMomentumConfig(enabled=False))

        assert module.analyze(context(m1_closes=rising(), last_body=4.0)).score == 0.0


class _NewsDouble:
    """Stands in for the account's `NewsFilter` at the shape the lane uses."""

    name = "news"

    def __init__(self, *, minutes_to_news: float | None = None, blocked: str = "") -> None:
        self.minutes_to_news = minutes_to_news
        self.blocked = blocked
        self.asked: list[str] = []

    def check(self, ctx):  # type: ignore[no-untyped-def]
        from filters.base import FilterVerdict
        from risk.reasons import Reason

        self.asked.append(ctx.symbol)
        if self.blocked:
            return FilterVerdict.block("news", Reason.NEWS_BLACKOUT, self.blocked)
        return FilterVerdict.allow("news", "clear", minutes_to_news=self.minutes_to_news)


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
            open_shadow_count=lambda *_: 0,
            record_shadow_trade=lambda **kw: recorded.append(kw) or 1,
        )
        runner.broker = SimpleNamespace(  # type: ignore[assignment]
            spec=lambda _s: SimpleNamespace(
                asset_class=SimpleNamespace(value="metal"),
                currency_base="XAU",
                currency_profit="USD",
            )
        )
        runner.clock = SimpleNamespace(now=lambda: ctx.now)  # type: ignore[assignment]
        # THE CALENDAR THE LANE NOW ASKS ITSELF. `minutes_to_news` used to
        # arrive only in `extra`, which the NO_SIGNAL path does not populate.
        runner._news = _NewsDouble(minutes_to_news=90.0)  # type: ignore[attr-defined]
        runner.filters = _chain(runner._news)  # type: ignore[assignment]
        return runner, [CandleMomentum().analyze(live)]

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

    def test_the_loser_is_cut_fast_and_the_winner_is_not(self) -> None:
        """The exits were the wrong way round and the arithmetic was not close.

        "Get out fast" points in opposite directions on the two sides of a
        trade. Cutting the LOSER fast takes the required hit rate from 69% to
        41%; cutting the WINNER fast takes it back to 72%, because the spread
        does not shrink with the target. So the stop is a fraction of the
        trigger candle and the target is larger than it."""
        from risk.reasons import Reason

        recorded: list = []
        runner, signals = self._runner(recorded)

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {"minutes_to_news": 90.0})

        plan = recorded[0]
        reward = abs(plan["tp"] - plan["entry_price"])
        risk = abs(plan["entry_price"] - plan["sl"])
        assert reward > risk


class TestItIsLiveAndBraked:
    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_it_trades_its_own_lane_and_not_the_confluence_vote(self) -> None:
        """MOVED OUT OF THE VOTE ON 25 AUGUST, and this is not a demotion.

        The confluence score is a weighted mean of (raw score x confidence)
        over the agreeing modules. This module's ceiling is 45 x 0.75 = 33.75
        against a bar of 45, so it could not open a trade alone -- and joining
        a strong reader made things worse rather than better, because a mean
        is dragged down by its weakest term: market_structure alone scores 70,
        and 56.4 with this agreeing.

        That is not a threshold anyone can set correctly. A scalp's evidence is
        small and short-lived because that is what a scalp is, and an engine
        built to weigh swing evidence will always price it low.

        So it gets its own route to an order, and it is absent from
        `live_enabled_modules` precisely so it cannot do both -- vote in
        section one AND open its own trade on the same reading.
        """
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        module = settings.analysis.candle_momentum

        # NOT asserted: that the lane is switched ON. It was, and this line
        # used to check it, and that made a business decision into a test.
        # `section6.cmd` measured the lane for the first time on 26 August --
        # 1,681 trades, -0.304R each -- and it was switched off the same hour;
        # this assertion went red for a change that was exactly right. Whether
        # the account runs section six follows the evidence and will move
        # again. What may NEVER move is the line below it: the module must
        # stay out of `live_enabled_modules` so it cannot both vote in section
        # one and open its own trade on the same reading. That is the
        # invariant, and it holds whether the lane is on or off.
        # Risk-sized like every other route to an order. A fixed lot is a
        # quantity, not a risk: 0.01 of gold behind a 3.41 dollar stop is
        # EUR 2.91, and 0.01 of an index is something else entirely. The lot
        # ceiling is off (0.0) so the risk model decides outright.
        assert module.risk_pct == 1.0
        assert module.fixed_lots == 0.0
        assert "candle_momentum" not in settings.analysis.confluence.live_enabled_modules
        # Its own stop stays, and it is the strict one: this section takes
        # several trades a day, so ten losses in a row is half a day paying
        # for the same fault.
        breaker = settings.risk.section_breakers["candle_momentum"]
        assert breaker.enabled and breaker.losing_streak == 6

    def test_its_ceiling_really_is_below_the_bar(self) -> None:
        """The arithmetic the lane exists for, asserted rather than asserted
        about. If a future change lifts this module's score above the
        threshold, the lane is no longer the only way it can trade and that
        decision deserves to be re-argued."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        module = settings.analysis.candle_momentum

        ceiling = module.base_score * module.maximum_confidence

        assert ceiling < settings.analysis.confluence.score_threshold

    def test_it_shares_the_momentum_family_rather_than_inventing_one(self) -> None:
        """It reads the same fact the other momentum readers read — price moved,
        hard, just now. Its own label would let it "corroborate" impulse_break
        on one observation seen twice, which is what families exist to stop."""
        from analysis.evidence_families import family_for

        assert family_for("candle_momentum") == "momentum"
        assert family_for("impulse_break") == "momentum"


class TestTheConfigCannotBeIncoherent:
    def test_saturation_must_exceed_the_minimum_body(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            CandleMomentumConfig(minimum_body_multiple=4.0, body_saturation_multiple=3.0)

    def test_confidence_may_not_narrow_to_nothing(self) -> None:
        with pytest.raises(ValueError, match="below base_confidence"):
            CandleMomentumConfig(base_confidence=0.9, maximum_confidence=0.4)


class TestTheCandleMustPayForItself:
    """The refusal that was documented in the module docstring and not built,
    which the arithmetic then said was the most important one here.

    With a target smaller than the stop the break-even hit rate starts at 58%
    before a cent of cost. Adding a realistic spread:

        XAUUSD, M1 range 1.50, spread 0.25  ->  67% needed
        SPX500, M1 range 1.00, spread 0.50  ->  84%, unreachable

    That last case is not a thin edge. It is no edge at any hit rate, and the
    only correct response is to refuse the market rather than try harder on the
    entry.
    """

    @staticmethod
    def _with_spread(spread: float):  # type: ignore[no-untyped-def]
        from core.types import Tick

        ctx = context(m1_closes=rising(), last_body=4.0)
        return ctx.__class__(
            symbol=ctx.symbol,
            now=ctx.now,
            series=ctx.series,
            tick=Tick(ctx.symbol, ctx.now, BASE, BASE + spread),
        )

    def test_a_wide_spread_against_a_small_candle_is_refused(self) -> None:
        signal = CandleMomentum().analyze(self._with_spread(3.0))

        assert signal.score == 0.0
        assert "spreads wide" in signal.reasoning

    def test_a_normal_spread_lets_the_setup_through(self) -> None:
        signal = CandleMomentum().analyze(self._with_spread(0.05))

        assert signal.score > 0

    def test_the_reading_records_how_many_spreads_the_target_was(self) -> None:
        """So the next person can see whether a refusal was marginal or
        hopeless, instead of only that it happened."""
        signal = CandleMomentum().analyze(self._with_spread(3.0))

        assert "target_in_spreads" in signal.details
        assert signal.details["target_in_spreads"] < 5.0

    def test_no_quote_does_not_crash_the_reading(self) -> None:
        """The module is replayed in backtests with no tick at all. A missing
        spread must mean the guard does not run, never that the module dies."""
        signal = CandleMomentum().analyze(context(m1_closes=rising(), last_body=4.0))

        assert signal.score > 0

    def test_the_shipped_geometry_needs_a_hit_rate_a_filter_can_reach(self) -> None:
        """The whole reason the exits were flipped. At the configured clearance
        the break-even sits near a third, which a selective multi-timeframe
        filter can plausibly beat — where the original 67% could not be
        reasoned about at all."""
        config = CandleMomentumConfig()
        spread = 1.0
        span = spread * config.minimum_target_spreads / config.target_candle_spans
        win = config.target_candle_spans * span - spread
        loss = config.stop_candle_spans * span + spread

        assert loss / (win + loss) < 0.45


class TestWhereItIsAllowedToTradeAndHowMuchAtOnce:
    """Two limits the owner asked for, and one of them is arithmetic rather
    than preference: a scalp's whole margin is a few spreads wide, and a fixed
    fee per lot is not a cost that fits inside it."""

    def _runner(self, *, asset_class: str, open_count: int = 0):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Tick
        from runner.service import JarvisRunner

        recorded: list = []
        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        ctx = context(m1_closes=rising(), last_body=4.0)
        live = ctx.__class__(
            symbol=ctx.symbol,
            now=ctx.now,
            series=ctx.series,
            tick=Tick(ctx.symbol, ctx.now, BASE, BASE + 0.05),
        )
        runner._cycle_contexts = {"XAUUSD": live}
        runner.recorder = SimpleNamespace(  # type: ignore[assignment]
            has_unresolved_shadow_trade=lambda *_: False,
            open_shadow_count=lambda *_: open_count,
            record_shadow_trade=lambda **kw: recorded.append(kw) or 1,
        )
        runner.broker = SimpleNamespace(  # type: ignore[assignment]
            spec=lambda _s: SimpleNamespace(
                asset_class=SimpleNamespace(value=asset_class),
                currency_base="XAU",
                currency_profit="USD",
            )
        )
        runner.clock = SimpleNamespace(now=lambda: live.now)  # type: ignore[assignment]
        runner._news = _NewsDouble(minutes_to_news=90.0)  # type: ignore[attr-defined]
        runner.filters = _chain(runner._news)  # type: ignore[assignment]
        return runner, [CandleMomentum().analyze(live)], recorded

    def _observe(self, runner, signals):  # type: ignore[no-untyped-def]
        from risk.reasons import Reason

        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {"minutes_to_news": 90.0})

    def test_a_zero_commission_class_is_allowed(self) -> None:
        runner, signals, recorded = self._runner(asset_class="metal")

        self._observe(runner, signals)

        assert len(recorded) == 1

    def test_forex_is_refused_because_it_pays_a_fee_per_lot(self) -> None:
        """EUR 5.50 a lot against a margin measured in a few spreads. The gate
        is DERIVED from the commission table rather than typed out, so the two
        cannot fall out of step — and the direction they would fall out of step
        in is exactly this."""
        runner, signals, recorded = self._runner(asset_class="forex")

        self._observe(runner, signals)

        assert recorded == []

    def test_every_other_zero_commission_class_is_allowed_too(self) -> None:
        for asset_class in ("index", "crypto", "commodity", "stock"):
            runner, signals, recorded = self._runner(asset_class=asset_class)
            self._observe(runner, signals)
            assert len(recorded) == 1, asset_class

    def test_the_two_position_cap_is_honoured_on_paper(self) -> None:
        """A paper section with a live concurrency limit has to respect it on
        paper too, or the record measures the returns of a book the account has
        no room to hold."""
        runner, signals, recorded = self._runner(asset_class="metal", open_count=2)

        self._observe(runner, signals)

        assert recorded == []

    def test_one_open_position_still_leaves_room(self) -> None:
        runner, signals, recorded = self._runner(asset_class="metal", open_count=1)

        self._observe(runner, signals)

        assert len(recorded) == 1

    def test_the_cap_sits_inside_the_accounts_own_concurrency_limit(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.candle_momentum.max_concurrent == 2
        assert settings.analysis.candle_momentum.max_concurrent < 4


class TestTheLaneAsksTheCalendarItselfOnThePathItActuallyTakes:
    """The blackout checks were real, present, tested -- and unreachable.

    WHAT HAPPENED. Section six opened a trade about twenty minutes before a
    red-folder release, inside a window the account blocks for sixty.

    `_scalp_plan` has two calendar guards. The first refuses when `reason` is a
    news block; the second refuses when `extra["minutes_to_news"]` is inside
    the lane's own clearance. Both were covered by tests -- which passed the
    reason and the minutes IN BY HAND. Nothing tested whether either ever
    arrives.

    Neither does, on the route this lane actually takes. It is called from
    `_record_skip`, so it runs on setups the main path refused, and the
    commonest refusal is NO_SIGNAL from the confluence engine -- which sits
    BEFORE `self.filters.check`. On that path the news filter has not run, so
    `reason` cannot be a news block and `extra` carries no `minutes_to_news`.
    The second guard then read None and skipped itself.

    Two guards, both present, both green, and the lane had no calendar
    protection at all on its main route to an order.
    """

    def _runner(self, news):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Tick
        from runner.service import JarvisRunner

        recorded: list = []
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
            open_shadow_count=lambda *_: 0,
            record_shadow_trade=lambda **kw: recorded.append(kw) or 1,
        )
        runner.broker = SimpleNamespace(  # type: ignore[assignment]
            spec=lambda _s: SimpleNamespace(
                asset_class=SimpleNamespace(value="metal"),
                currency_base="XAU",
                currency_profit="USD",
            )
        )
        runner.clock = SimpleNamespace(now=lambda: live.now)  # type: ignore[assignment]
        runner.filters = _chain(news)  # type: ignore[assignment]
        return runner, [CandleMomentum().analyze(live)], recorded

    def _observe_on_the_no_signal_path(self, runner, signals):  # type: ignore[no-untyped-def]
        from risk.reasons import Reason

        # Exactly what `_record_skip` hands it: the confluence engine's own
        # refusal, and an `extra` with no calendar data in it because the news
        # filter has not run yet.
        runner._observe_scalp(1, "XAUUSD", signals, Reason.NO_SIGNAL, {})

    def test_a_blackout_stops_it_even_though_the_reason_says_no_signal(self) -> None:
        """The live failure. Nothing in the arguments mentions news; the
        calendar does."""
        news = _NewsDouble(blocked="high-impact USD event: Non-Farm Payrolls")
        runner, signals, recorded = self._runner(news)

        self._observe_on_the_no_signal_path(runner, signals)

        assert recorded == [], "traded through a red folder"
        assert news.asked == ["XAUUSD"], "the calendar was never consulted"

    def test_a_release_inside_the_lanes_own_runway_stops_it_too(self) -> None:
        """`minutes_to_news` now comes off the verdict instead of out of an
        `extra` that never contained it."""
        news = _NewsDouble(minutes_to_news=3.0)
        runner, signals, recorded = self._runner(news)

        self._observe_on_the_no_signal_path(runner, signals)

        assert recorded == []

    def test_a_genuinely_clear_market_still_trades(self) -> None:
        """The guard must not become a way of refusing everything."""
        news = _NewsDouble(minutes_to_news=90.0)
        runner, signals, recorded = self._runner(news)

        self._observe_on_the_no_signal_path(runner, signals)

        assert len(recorded) == 1

    def test_an_unreadable_calendar_refuses(self) -> None:
        """The account's standing rule: no data is not permission."""

        class _Broken:
            name = "news"

            def check(self, ctx):  # type: ignore[no-untyped-def]
                raise RuntimeError("calendar cache is corrupt")

        runner, signals, recorded = self._runner(_Broken())

        self._observe_on_the_no_signal_path(runner, signals)

        assert recorded == []

    def test_a_missing_news_filter_refuses(self) -> None:
        """A lane that cannot find the calendar may not assume it is clear."""
        runner, signals, recorded = self._runner(None)

        self._observe_on_the_no_signal_path(runner, signals)

        assert recorded == []


class TestTheGeometryHoldsTogetherAsOneShape:
    """Four numbers that are only correct in relation to each other.

    The owner asked for "70-80% and a euro picked up each time", and for as
    much winning and as little losing as possible. Those are two different
    requests and only the second one is expectancy. Given as a target of 0.5R
    the first request produces ~71% and LOSES money, because at that payoff one
    loss eats 2.2 wins and the break-even rate is 73.3%.

    So it is delivered at the EXIT instead: the target stays where the
    arithmetic wants it and the claim banks the euro off trades that peak and
    stall. Two different trades, two different exits, and the runners are not
    cut to buy a nicer-looking hit rate.

    Every number below is derived from the others. Change one alone and the
    shape stops meaning what its comments say it means.
    """

    def config(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.candle_momentum

    def test_the_gate_buys_a_known_cost_and_that_is_what_it_is_for(self) -> None:
        """`minimum_target_spreads` reads as a target rule and IS a cost rule.
        target = 1.4 span and stop = 1.0 span, so the gate fixes how many
        spreads wide 1R is, and one spread is the round trip."""
        config = self.config()
        span_in_spreads = config.minimum_target_spreads / config.target_candle_spans
        stop_in_spreads = span_in_spreads * config.stop_candle_spans
        cost_share = 1.0 / stop_in_spreads

        assert stop_in_spreads == pytest.approx(10.0)
        assert cost_share == pytest.approx(0.10)

    def test_the_claim_banks_about_a_euro_and_not_about_a_dime(self) -> None:
        """The floor was 2.0 spreads. Behind a 10-spread stop that is 0.1R --
        EUR 0.21 -- which banks nothing worth banking and then lets the rest
        run back to the stop. Six spreads peak, minus the one the round trip
        costs, is half an R."""
        config = self.config()
        stop_in_spreads = (
            config.minimum_target_spreads / config.target_candle_spans * config.stop_candle_spans
        )
        banked_r = (config.scalp_claim_minimum_spreads - 1.0) / stop_in_spreads

        assert banked_r == pytest.approx(0.5)

    def test_the_claim_is_below_the_target_so_both_exits_can_happen(self) -> None:
        """If the claim armed above the target it could never fire -- the
        target would be hit first, every time, and the rule would be dead code
        that looks alive."""
        config = self.config()
        target_in_spreads = config.minimum_target_spreads

        assert config.scalp_claim_minimum_spreads < target_in_spreads

    def test_a_runner_is_not_cut_at_the_claim_floor(self) -> None:
        """The whole reason the euro is taken at the exit and not at the
        target: the leash grows with the peak, so a trade that reaches 20
        spreads may give back 8 before anything is claimed. Small win secured,
        big win left alone."""
        config = self.config()
        peak = 20.0
        leash = max(config.scalp_claim_spreads, peak * config.scalp_giveback_share)

        assert leash == pytest.approx(8.0)
        assert leash > config.scalp_claim_spreads
