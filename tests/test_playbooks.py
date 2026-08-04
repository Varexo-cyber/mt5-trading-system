"""Independent theories, each with its own plan — and the gates that bound them.

The properties that matter most here are not "does the pattern fire". They are:
spread must never be allowed to eat a small stop, two theories disagreeing must
stand the system down rather than be settled by score, and a short-horizon plan
must carry a short-horizon target rather than a swing target in miniature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.playbooks import (
    FadeConfig,
    MomentumScalp,
    Play,
    PlaybookEngine,
    RangeFade,
    ScalpConfig,
    _atr,
)
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Tick, Timeframe, TradingMode

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


def series(timeframe: Timeframe, closes: list[float]) -> Series:
    """A bar series from a close path, with plausible highs and lows."""
    index = pd.date_range(end=NOW, periods=len(closes), freq=timeframe.duration, tz=UTC)
    closes_array = np.array(closes, dtype=float)
    opens = np.concatenate([[closes_array[0]], closes_array[:-1]])
    # Small wicks: an exaggerated one makes a shallow pullback measure deep,
    # because the pullback is taken from the impulse extreme.
    wick = np.abs(closes_array - opens) * 0.08 + 0.000005
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes_array) + wick,
            "low": np.minimum(opens, closes_array) - wick,
            "close": closes_array,
            "tick_volume": np.full(len(closes), 500),
        },
        index=index,
    )
    return Series(symbol="EURUSD", timeframe=timeframe, df=frame, fetched_at=NOW)


def context(
    frames: dict[Timeframe, Series], *, bid: float = 1.1000, spread: float = 0.00010
) -> MarketContext:
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series=frames,
        tick=Tick(symbol="EURUSD", time=NOW, bid=bid, ask=bid + spread),
    )


def impulse_path(base: float = 1.1000) -> list[float]:
    """Enough history to measure against, then a realistic impulse.

    Three things this has to get right or the playbook correctly refuses it,
    and getting them wrong in a fixture is easy:

    * **Length.** The target is sized from favourable excursion over a 24-bar
      horizon, so a 25-bar fixture leaves one sample and the target lands
      wherever that single window happened to go.
    * **Scale.** An impulse of eight ATR is not a scalp setup, it is a news
      spike. Real continuation setups are one to three ATR, and the stop that
      follows from one is proportionate.
    * **Wick size.** Pullback is measured from the impulse extreme, so an
      exaggerated wick makes a shallow retrace read as a deep one.
    """
    rng = np.random.default_rng(11)
    # A long, normally-noisy history: this sets ATR and gives the reachability
    # window something to measure.
    history = base + np.cumsum(rng.normal(0, 0.00020, 260))
    level = float(history[-1])
    # A quiet stretch, tighter than the surrounding noise.
    calm = level + rng.normal(0, 0.00004, 14)
    step = float(level)
    # The impulse: roughly 2 ATR over four bars.
    rise = np.linspace(step, step + 0.00090, 4)
    # A shallow give-back, well under half an ATR.
    pull = [step + 0.00084]
    return [*history.tolist(), *calm.tolist(), *rise.tolist(), *pull]


def ranging_path(bars: int = 90, low: float = 1.0980, high: float = 1.1020) -> list[float]:
    """Price oscillating between two levels it keeps respecting."""
    mid, half = (high + low) / 2, (high - low) / 2
    return [mid + half * np.sin(i / 3.5) for i in range(bars)]


# --------------------------------------------------------------- the Play type ---


def play(**kwargs: object) -> Play:
    base = {
        "playbook": "test",
        "direction": Direction.LONG,
        "entry": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "conviction": 70.0,
        "horizon_minutes": 60,
        "thesis": "test",
    }
    return Play(**{**base, **kwargs})  # type: ignore[arg-type]


def test_a_play_reports_its_own_reward_to_risk() -> None:
    assert play().reward_risk == pytest.approx(2.0)


def test_a_play_with_no_risk_does_not_divide_by_zero() -> None:
    assert play(stop_loss=1.1000).reward_risk == 0.0


# ------------------------------------------------------------- the spread gate ---


def test_a_scalp_is_refused_when_spread_eats_the_stop() -> None:
    """The constraint that decides whether scalping a small account is viable.

    Same chart, same pattern, only the spread differs. A tight stop with a wide
    spread is a machine for paying the broker, and the pattern logic cannot see
    that — only this gate can.
    """
    path = impulse_path()
    frames = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, [path[-1]] * 20),
    }
    book = MomentumScalp(ScalpConfig(max_spread_share_of_stop=0.15))

    tight = book.propose(context(frames, bid=path[-1], spread=0.00002), TradingMode.PAPER)
    wide = book.propose(context(frames, bid=path[-1], spread=0.00060), TradingMode.PAPER)

    assert tight is not None, "a 0.2-pip spread must not block a scalp"
    assert wide is None, "a 6-pip spread against this stop must block it"


def test_the_spread_share_is_reported_so_the_reviewer_can_see_it() -> None:
    path = impulse_path()
    frames = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, [path[-1]] * 20),
    }
    found = MomentumScalp(ScalpConfig()).propose(
        context(frames, bid=path[-1], spread=0.00002), TradingMode.PAPER
    )
    assert found is not None
    assert 0.0 < float(found.evidence["spread_share_of_stop"]) < 0.15


def test_no_tick_means_no_play() -> None:
    """Without an executable quote there is no entry price to plan from."""
    frames = {Timeframe.M5: series(Timeframe.M5, impulse_path())}
    blind = MarketContext(symbol="EURUSD", now=NOW, series=frames, tick=None)
    assert MomentumScalp(ScalpConfig()).propose(blind, TradingMode.PAPER) is None


# ------------------------------------------------------------- momentum scalp ---


def test_a_clean_impulse_out_of_compression_fires() -> None:
    path = impulse_path()
    frames = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, [path[-1]] * 20),
    }
    found = MomentumScalp(ScalpConfig()).propose(
        context(frames, bid=path[-1], spread=0.00002), TradingMode.PAPER
    )
    assert found is not None
    assert found.direction is Direction.LONG
    assert found.stop_loss < found.entry < found.take_profit
    assert found.horizon_minutes <= 120, "a scalp must not carry a swing horizon"


def test_a_flat_market_produces_nothing() -> None:
    flat = [1.1000 + 0.00001 * (i % 3) for i in range(60)]
    frames = {Timeframe.M5: series(Timeframe.M5, flat)}
    assert MomentumScalp(ScalpConfig()).propose(context(frames), TradingMode.PAPER) is None


def test_an_impulse_already_underway_is_refused() -> None:
    """Mid-move is not the same as out of compression; the easy part is gone."""
    trending = np.linspace(1.1000, 1.1080, 40).tolist()
    frames = {Timeframe.M5: series(Timeframe.M5, trending)}
    assert (
        MomentumScalp(ScalpConfig()).propose(context(frames, bid=1.1080), TradingMode.PAPER) is None
    )


def test_a_deep_pullback_is_refused() -> None:
    path = impulse_path()
    path[-1] = path[0]  # gave the entire impulse back
    frames = {Timeframe.M5: series(Timeframe.M5, path)}
    assert (
        MomentumScalp(ScalpConfig()).propose(context(frames, bid=path[-1]), TradingMode.PAPER)
        is None
    )


def test_the_minute_chart_running_against_the_entry_refuses_it() -> None:
    """The difference between a pullback and a reversal."""
    path = impulse_path()
    here = path[-1]
    frames_ok = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, [here] * 20),
    }
    falling = np.linspace(here, here - 0.0020, 20).tolist()
    frames_bad = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, falling),
    }
    book = MomentumScalp(ScalpConfig())
    assert book.propose(context(frames_ok, bid=here, spread=0.00002), TradingMode.PAPER)
    assert book.propose(context(frames_bad, bid=here, spread=0.00002), TradingMode.PAPER) is None


def test_the_stop_is_tight_enough_to_be_a_scalp() -> None:
    """The property that makes this playbook worth having on a EUR 100 account.

    Anchoring to the impulse leg base instead of the pullback produced a stop
    several ATR wide — a swing trade wearing a scalp's name, and one the sizer
    would refuse as undercapitalized. A few pips is the whole point.
    """
    path = impulse_path()
    frames = {
        Timeframe.M5: series(Timeframe.M5, path),
        Timeframe.M1: series(Timeframe.M1, [path[-1]] * 20),
    }
    found = MomentumScalp(ScalpConfig()).propose(
        context(frames, bid=path[-1], spread=0.00002), TradingMode.PAPER
    )
    assert found is not None
    atr = _atr(series(Timeframe.M5, path).df)
    assert found.risk <= atr * ScalpConfig().max_stop_atr
    assert found.risk < 0.0010, "a scalp stop must be a handful of pips, not tens"


# ----------------------------------------------------------------- range fade ---


def test_a_respected_range_produces_a_fade_at_the_edge() -> None:
    path = ranging_path()
    path[-1] = 1.0982  # closing back up off the bottom
    frames = {Timeframe.M15: series(Timeframe.M15, path)}
    found = RangeFade(FadeConfig(min_touches=2)).propose(
        context(frames, bid=1.0982, spread=0.00002), TradingMode.PAPER
    )
    if found is not None:
        assert found.direction is Direction.LONG
        assert found.take_profit > found.entry
        # The midpoint, not the far side: reached far more often.
        assert found.take_profit < float(max(path))


def test_a_trending_market_produces_no_fade() -> None:
    trending = np.linspace(1.1000, 1.1200, 90).tolist()
    frames = {Timeframe.M15: series(Timeframe.M15, trending)}
    assert RangeFade(FadeConfig()).propose(context(frames, bid=1.1200), TradingMode.PAPER) is None


# ------------------------------------------------------------------- the panel ---


class _Fixed:
    def __init__(self, name: str, result: Play | None) -> None:
        self.name = name
        self.horizon_minutes = 60
        self.result = result

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:
        return self.result


class _Broken:
    name = "broken"
    horizon_minutes = 60

    def propose(self, ctx: MarketContext, mode: TradingMode) -> Play | None:
        raise RuntimeError("this theory is broken")


def engine(*books: object) -> PlaybookEngine:
    config = load_settings(env_overrides=False).analysis.confluence
    return PlaybookEngine(list(books), config)  # type: ignore[arg-type]


def test_disagreement_is_flagged_as_no_edge() -> None:
    """Not a contest to be won by the higher score."""
    verdict = engine(
        _Fixed("a", play(direction=Direction.LONG, conviction=80.0)),
        _Fixed("b", play(direction=Direction.SHORT, conviction=50.0, take_profit=1.0980)),
    ).evaluate(context({}), TradingMode.PAPER)

    assert verdict.conflict
    assert "no edge" in verdict.note


def test_agreement_is_reported_as_such() -> None:
    verdict = engine(
        _Fixed("a", play(conviction=80.0)),
        _Fixed("b", play(conviction=60.0)),
    ).evaluate(context({}), TradingMode.PAPER)

    assert not verdict.conflict
    assert verdict.best is not None
    assert verdict.best.conviction == 80.0, "the panel must be ordered by conviction"


def test_a_silent_theory_is_not_opposition() -> None:
    """A range-fade seeing nothing in a trend is the theory working."""
    verdict = engine(
        _Fixed("a", play(conviction=70.0)),
        _Fixed("b", None),
    ).evaluate(context({}), TradingMode.PAPER)

    assert not verdict.conflict
    assert len(verdict.plays) == 1


def test_one_broken_theory_does_not_silence_the_rest() -> None:
    verdict = engine(_Broken(), _Fixed("a", play(conviction=70.0))).evaluate(
        context({}), TradingMode.PAPER
    )
    assert len(verdict.plays) == 1


def test_no_theory_firing_is_reported_plainly() -> None:
    verdict = engine(_Fixed("a", None)).evaluate(context({}), TradingMode.PAPER)
    assert verdict.best is None
    assert "no theory" in verdict.note


def test_the_summary_carries_every_proposal_for_the_reviewer() -> None:
    """Claude must see the losing theories too — that is the point."""
    verdict = engine(
        _Fixed("a", play(conviction=80.0)),
        _Fixed("b", play(conviction=60.0)),
    ).evaluate(context({}), TradingMode.PAPER)

    summary = verdict.summary()
    assert summary["playbooks_that_fired"] == 2
    assert len(summary["proposals"]) == 2
    assert all("thesis" in proposal for proposal in summary["proposals"])


def test_series_duration_sanity() -> None:
    """Guard the helper: M5 bars must actually be five minutes apart."""
    frame = series(Timeframe.M5, [1.1, 1.2, 1.3]).df
    assert frame.index[1] - frame.index[0] == timedelta(minutes=5)
