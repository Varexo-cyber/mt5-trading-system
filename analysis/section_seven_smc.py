"""Section seven: replayable multi-timeframe gold liquidity-sweep reversal."""

from __future__ import annotations

import pandas as pd

from config.schema import SectionSevenSmcConfig
from core.types import MarketContext, Signal, Timeframe


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


class SectionSevenGoldSmc:
    """Trade a sweep only after displacement leaves and price retests an FVG."""

    name = "section_seven_gold_smc"

    def __init__(self, config: SectionSevenSmcConfig | None = None) -> None:
        self.config = config or SectionSevenSmcConfig()
        self._last_trade_day: dict[str, object] = {}

    def analyze(self, ctx: MarketContext) -> Signal:
        cfg = self.config
        if not cfg.enabled or ctx.symbol not in cfg.allowed_symbols:
            return Signal.neutral(self.name, "section seven SMC disabled for this market")
        clock = Timeframe.parse(cfg.timeframe)
        series = ctx.series.get(clock)
        required = (Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4)
        if series is None or any(ctx.series.get(frame) is None for frame in required):
            return Signal.neutral(self.name, "S7 needs entry, M15, M30, H1 and H4 closed bars")
        frame = series.df
        if len(frame) < cfg.liquidity_lookback + 8:
            return Signal.neutral(self.name, "S7 needs more closed bars for swing structure")
        today = frame.index[-1].date()
        if self._last_trade_day.get(ctx.symbol) == today:
            return Signal.neutral(self.name, "S7 already produced its one setup for this UTC day")

        open_ = frame["open"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        unit = _atr(frame.iloc[:-3])
        if not pd.notna(unit) or unit <= 0:
            return Signal.neutral(self.name, "S7 ATR is unavailable")

        # Four closed bars, in order: sweep, displacement/BOS, FVG confirmation,
        # then the retest.  Entering immediately on the CHOCH was the defective
        # first design: it bought the expansion rather than its return to value.
        history = slice(-(cfg.liquidity_lookback + 4), -4)
        prior_high = float(high.iloc[history].max())
        prior_low = float(low.iloc[history].min())
        swept_high = float(high.iloc[-4]) > prior_high and float(close.iloc[-4]) < prior_high
        swept_low = float(low.iloc[-4]) < prior_low and float(close.iloc[-4]) > prior_low
        if swept_high == swept_low:
            return Signal.neutral(self.name, "no single-sided closed liquidity sweep")

        direction = -1 if swept_high else 1
        micro = cfg.choch_lookback
        if direction < 0:
            choch_level = float(low.iloc[-(micro + 4) : -4].min())
            changed = float(close.iloc[-3]) < choch_level
            sweep_extreme = float(high.iloc[-4])
            gap_low, gap_high = float(high.iloc[-2]), float(low.iloc[-4])
        else:
            choch_level = float(high.iloc[-(micro + 4) : -4].max())
            changed = float(close.iloc[-3]) > choch_level
            sweep_extreme = float(low.iloc[-4])
            gap_low, gap_high = float(high.iloc[-4]), float(low.iloc[-2])
        if not changed:
            return Signal.neutral(self.name, "sweep had no displacement close through structure")
        displacement_body = abs(float(close.iloc[-3] - open_.iloc[-3]))
        if displacement_body < cfg.displacement_body_atr * unit:
            return Signal.neutral(self.name, "structure break was not displacement")
        if gap_high - gap_low < cfg.fvg_minimum_atr * unit:
            return Signal.neutral(self.name, "displacement left no measurable FVG")

        tolerance = cfg.retest_tolerance_atr * unit
        touched = (
            float(low.iloc[-1]) <= gap_high + tolerance
            and float(high.iloc[-1]) >= gap_low - tolerance
        )
        held = (
            float(close.iloc[-1]) >= gap_low
            if direction > 0
            else float(close.iloc[-1]) <= gap_high
        )
        if not touched or not held:
            return Signal.neutral(self.name, "FVG was not retested and defended")

        h1 = ctx.series[Timeframe.H1].df
        h1_high = float(h1["high"].astype(float).iloc[-cfg.context_lookback :].max())
        h1_low = float(h1["low"].astype(float).iloc[-cfg.context_lookback :].min())
        midpoint = (h1_high + h1_low) / 2.0
        entry = ctx.tick.mid if ctx.tick is not None else float(close.iloc[-1])
        if (direction < 0 and entry < midpoint) or (direction > 0 and entry > midpoint):
            return Signal.neutral(self.name, "sweep is not in H1 premium/discount territory")

        aligned = 0
        against = 0
        for timeframe in required:
            closes = ctx.series[timeframe].df["close"].astype(float)
            middle = float(closes.iloc[-cfg.context_lookback :].median())
            bias = 1 if float(closes.iloc[-1]) > middle else -1
            aligned += int(bias == direction)
            against += int(bias == -direction)
        if against >= 3:
            return Signal.neutral(self.name, "three higher clocks oppose the reversal")

        stop = sweep_extreme - direction * cfg.stop_buffer_atr * unit
        if direction * (entry - stop) <= 0:
            return Signal.neutral(self.name, "S7 structural stop is behind the entry")
        liquidity_target = prior_high if direction > 0 else prior_low
        risk = abs(entry - stop)
        reward_r = direction * (liquidity_target - entry) / risk
        if reward_r < cfg.minimum_liquidity_reward_r:
            return Signal.neutral(self.name, "opposing liquidity offers less than the required R")
        self._last_trade_day[ctx.symbol] = today
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning="liquidity sweep, displacement/BOS, FVG retest and opposing liquidity",
            invalidation_price=stop,
            key_levels=(gap_low, gap_high, choch_level, midpoint, liquidity_target),
            details={
                "timeframe": cfg.timeframe,
                "htf_aligned": aligned,
                "liquidity_target": liquidity_target,
                "liquidity_reward_r": reward_r,
            },
        )
