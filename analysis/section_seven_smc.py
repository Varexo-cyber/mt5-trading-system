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
    """Fade a swept liquidity pool after a closed-bar structure change."""

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

        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        history = slice(-(cfg.liquidity_lookback + 2), -2)
        prior_high = float(high.iloc[history].max())
        prior_low = float(low.iloc[history].min())
        swept_high = float(high.iloc[-2]) > prior_high and float(close.iloc[-2]) < prior_high
        swept_low = float(low.iloc[-2]) < prior_low and float(close.iloc[-2]) > prior_low
        if swept_high == swept_low:
            return Signal.neutral(self.name, "no single-sided closed liquidity sweep")

        direction = -1 if swept_high else 1
        micro = cfg.choch_lookback
        if direction < 0:
            choch_level = float(low.iloc[-(micro + 2) : -2].min())
            changed = float(close.iloc[-1]) < choch_level
            sweep_extreme = float(high.iloc[-2])
        else:
            choch_level = float(high.iloc[-(micro + 2) : -2].max())
            changed = float(close.iloc[-1]) > choch_level
            sweep_extreme = float(low.iloc[-2])
        if not changed:
            return Signal.neutral(self.name, "liquidity swept but no closed change of character")

        h1 = ctx.series[Timeframe.H1].df
        h1_high = float(h1["high"].astype(float).iloc[-cfg.context_lookback :].max())
        h1_low = float(h1["low"].astype(float).iloc[-cfg.context_lookback :].min())
        midpoint = (h1_high + h1_low) / 2.0
        entry = ctx.tick.mid if ctx.tick is not None else float(close.iloc[-1])
        if (direction < 0 and entry < midpoint) or (direction > 0 and entry > midpoint):
            return Signal.neutral(self.name, "sweep is not in H1 premium/discount territory")

        aligned = 0
        for timeframe in required:
            closes = ctx.series[timeframe].df["close"].astype(float)
            ema = closes.ewm(span=20, adjust=False).mean()
            slope = float(ema.iloc[-1] - ema.iloc[-4])
            aligned += int(slope * -direction > 0)
        if aligned == len(required):
            return Signal.neutral(self.name, "all higher clocks still support the swept move")

        unit = _atr(frame)
        if not pd.notna(unit) or unit <= 0:
            return Signal.neutral(self.name, "S7 ATR is unavailable")
        stop = sweep_extreme - direction * cfg.stop_buffer_atr * unit
        if direction * (entry - stop) <= 0:
            return Signal.neutral(self.name, "S7 structural stop is behind the entry")
        self._last_trade_day[ctx.symbol] = today
        return Signal(
            module=self.name,
            score=cfg.score * direction,
            confidence=cfg.confidence,
            reasoning="liquidity sweep plus closed CHOCH in H1 premium/discount",
            invalidation_price=stop,
            key_levels=(prior_high if direction < 0 else prior_low, choch_level, midpoint),
            details={"timeframe": cfg.timeframe, "htf_aligned": aligned},
        )
