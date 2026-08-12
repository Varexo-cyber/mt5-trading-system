"""Combine independent analysis modules into one auditable trade idea."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.schema import ConfluenceConfig, HorizonProfileConfig
from core.types import AnalysisModule, Direction, MarketContext, Signal, Timeframe, TradingMode


@dataclass(frozen=True, slots=True)
class TradeIdea:
    symbol: str
    approved: bool
    direction: Direction | None
    score: float
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reason: str
    signals: tuple[Signal, ...]
    setup_family: str = "swing_confluence"
    horizon: str = "swing"
    planning_timeframe: str = "H1"
    expected_horizon_minutes: int = 1440


class ConfluenceEngine:
    """Weighted agreement with live-module allowlisting and structural stops."""

    def __init__(self, modules: list[AnalysisModule], config: ConfluenceConfig) -> None:
        self.modules = modules
        self.config = config
        #: symbol -> (bar fingerprint, signals). One entry per symbol, so a
        #: whole broker catalogue costs a few hundred small tuples.
        self._signal_cache: dict[str, tuple[tuple[tuple[str, object], ...], tuple[Signal, ...]]] = (
            {}
        )

    def _bar_fingerprint(self, ctx: MarketContext) -> tuple[tuple[str, object], ...] | None:
        """Identity of the closed bars this evaluation would read.

        None means "do not cache": an empty frame is a transient condition, and
        freezing a verdict taken during one would outlive its cause.
        """
        marks: list[tuple[str, object]] = []
        for timeframe, series in ctx.series.items():
            frame = series.df
            if frame is None or frame.empty:
                return None
            marks.append((timeframe.value, frame.index[-1]))
        return tuple(sorted(marks)) if marks else None

    def _signals(self, ctx: MarketContext) -> tuple[Signal, ...]:
        """Module analysis, recomputed only once a new bar has closed.

        THE WHOLE JUSTIFICATION IS PURITY, so it is stated where it can be
        checked: no module in `analysis/modules.py` reads `ctx.tick` or
        `ctx.now`. Each one reads closed bars out of `ctx.series` and nothing
        else, so identical frames give identical signals. This returns the same
        answer, not a cheaper approximation of one.

        It matters because the live journal showed a single symbol reporting
        confluence score 38.8 eighty-two times in twelve hours. That score can
        only move when a bar closes, and the fastest frame in the ladder is M5,
        so eleven of every twelve evaluations recomputed a number that could
        not have changed — for every symbol in the catalogue, on one vCPU
        shared with the one-second position guard.

        The bars were already cached; `DataManager._bar_closed_since` does
        that. What this removes is the pandas, not the I/O.
        """
        if not self.config.cache_signals_per_bar:
            return tuple(module.analyze(ctx) for module in self.modules)
        fingerprint = self._bar_fingerprint(ctx)
        if fingerprint is None:
            return tuple(module.analyze(ctx) for module in self.modules)
        cached = self._signal_cache.get(ctx.symbol)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        signals = tuple(module.analyze(ctx) for module in self.modules)
        self._signal_cache[ctx.symbol] = (fingerprint, signals)
        return signals

    def evaluate(self, ctx: MarketContext, mode: TradingMode) -> TradeIdea:
        signals = self._signals(ctx)
        if ctx.tick is None:
            return self._reject(ctx, signals, "no executable quote")

        regime = next(
            (s.details.get("regime") for s in signals if s.module == "volatility_regime"), None
        )
        if regime == "extreme":
            return self._reject(ctx, signals, "extreme volatility regime")

        allowed_live = set(self.config.live_enabled_modules)
        weighted: list[tuple[Signal, float]] = []
        for signal in signals:
            weight = self.config.weights.get(signal.module, 0.0)
            if mode.is_live and signal.module not in allowed_live:
                weight = 0.0
            if weight > 0 and signal.score and signal.confidence >= self.config.minimum_confidence:
                weighted.append((signal, weight))
        if not weighted:
            suffix = "; no modules validated for live" if mode.is_live and not allowed_live else ""
            return self._reject(ctx, signals, f"no weighted directional evidence{suffix}")

        positive = sum(weight for signal, weight in weighted if signal.score > 0)
        negative = sum(weight for signal, weight in weighted if signal.score < 0)
        direction = Direction.LONG if positive > negative else Direction.SHORT
        agreeing = [
            (signal, weight) for signal, weight in weighted if signal.score * int(direction) > 0
        ]
        agreement = sum(weight for _, weight in agreeing) / sum(weight for _, weight in weighted)
        if len(agreeing) < self.config.minimum_directional_modules:
            return self._reject(ctx, signals, "too few independent directional modules")
        if agreement < self.config.minimum_agreement_ratio:
            return self._reject(
                ctx, signals, f"directional agreement {agreement:.1%} below threshold"
            )

        # Is the premise of this setup contradicted by a measurement?
        #
        # `market_regime` sorts the market into trend_up, trend_down, range,
        # transition or extreme. It has always been computed, always been sent
        # to the reviewer, and never read here — the check at the top of this
        # method reads `volatility_regime`, which answers a different question.
        #
        # A trend-continuation module firing while the classifier measures a
        # range is not weak evidence, it is evidence against. The reviewer said
        # so three times in one session, once as "the regime module explicitly
        # flags 'range' with low efficiency (0.08 H1, 0.11 H4) — this is chop,
        # not a trend", and each time the account paid to hear it.
        #
        # Deliberately not a module count. The prompt tells the reviewer to
        # ignore those and it correctly does: a zero from a module looking for
        # something else is the absence of evidence. "range" is a positive
        # finding. And only continuation setups are caught — a liquidity sweep
        # is a range setup and belongs in a range.
        if self.config.refuse_trend_continuation_in_range:
            regime_reading = next(
                (
                    signal.details.get("regime")
                    for signal in signals
                    if signal.module == "market_regime"
                ),
                None,
            )
            continuation = set(self.config.trend_continuation_modules)
            if regime_reading == "range" and all(
                signal.module in continuation for signal, _ in agreeing
            ):
                firing = ", ".join(sorted(signal.module for signal, _ in agreeing))
                return self._reject(
                    ctx,
                    signals,
                    f"the regime classifier measures a range while the only firing "
                    f"module(s) ({firing}) assert a trend is continuing",
                )

        denominator = sum(weight for _, weight in agreeing)
        score = (
            sum(abs(signal.score) * signal.confidence * weight for signal, weight in agreeing)
            / denominator
        )
        confidence = sum(signal.confidence * weight for signal, weight in agreeing) / denominator
        if score < self.config.score_threshold:
            return self._reject(
                ctx, signals, f"confluence score {score:.1f} below threshold", score, confidence
            )

        horizon, setup_family = self._classify_horizon(agreeing)
        profile = self.config.horizon_profiles[horizon]

        against_the_tide = self.higher_timeframe_conflict(
            ctx,
            direction,
            timeframes=profile.htf_trend_timeframes,
            threshold=profile.htf_trend_veto,
            minimum_conflicts=profile.minimum_htf_conflicts,
        )
        if against_the_tide is not None:
            return self._reject(ctx, signals, against_the_tide, score, confidence)

        adverse = self._entry_timing_conflict(
            ctx, direction, timeframes=profile.entry_timing_timeframes
        )
        if adverse is not None:
            return self._reject(ctx, signals, adverse, score, confidence)

        entry = ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid
        planning_timeframe = Timeframe.parse(profile.planning_timeframe)
        series = ctx.series.get(planning_timeframe)
        if series is None:
            return self._reject(
                ctx,
                signals,
                f"{planning_timeframe.value} history missing for {horizon} planning",
                score,
                confidence,
            )
        frame = series.df
        atr = self._atr(frame)
        candidates = [
            signal.invalidation_price
            for signal, _ in agreeing
            if signal.invalidation_price is not None
            and (
                (direction is Direction.LONG and signal.invalidation_price < entry)
                or (direction is Direction.SHORT and signal.invalidation_price > entry)
            )
        ]
        if candidates:
            structural = min(candidates) if direction is Direction.LONG else max(candidates)
            stop = (
                structural - atr * 0.25 if direction is Direction.LONG else structural + atr * 0.25
            )
            # A structural level close to entry does not make a tight stop
            # correct — it makes it a stop measured inside the noise. There was
            # no floor here at all, and the invalidation price of a nearby M15
            # swing sits a few pips away, so the stop came out at a fraction of
            # an ATR and was taken out by ordinary chop rather than by the trade
            # being wrong. That is not a tail case: it is what the adviser
            # rejected almost every setup for, in those words, all day.
            #
            # The floor pushes the stop out rather than rejecting the setup. If
            # the wider stop then prices the trade out of the account, the sizer
            # says so — which is the honest answer, and a far better one than
            # shrinking the stop until the arithmetic fits.
            floor = atr * self.config.min_stop_atr
            if abs(entry - stop) < floor:
                stop = entry - floor * int(direction)
        else:
            stop = entry - atr * self.config.atr_stop_multiple * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return self._reject(
                ctx, signals, "could not construct a positive stop distance", score, confidence
            )
        target, target_note = self._reachable_target(ctx, entry, risk, direction, profile=profile)
        if target is None:
            return self._reject(ctx, signals, target_note, score, confidence)
        return TradeIdea(
            symbol=ctx.symbol,
            approved=True,
            direction=direction,
            score=score,
            confidence=confidence,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            reason=(
                f"{len(agreeing)} modules agree ({agreement:.0%}); "
                f"{horizon} plan on {planning_timeframe.value}; target {target_note}"
            ),
            signals=signals,
            setup_family=setup_family,
            horizon=horizon,
            planning_timeframe=planning_timeframe.value,
            expected_horizon_minutes=int(
                planning_timeframe.duration.total_seconds() / 60 * profile.target_horizon_bars
            ),
        )

    def _reachable_target(
        self,
        ctx: MarketContext,
        entry: float,
        risk: float,
        direction: Direction,
        *,
        profile: HorizonProfileConfig | None = None,
    ) -> tuple[float | None, str]:
        """Place the target where this market actually goes, not where R says.

        `entry + 2R` is arithmetic. It never asks whether the instrument travels
        that far, so a slow market gets a target it reaches once a month and the
        trade becomes, in practice, a bet on the stop not being hit — the reward
        half of the reward-to-risk never arrives.

        So the distance is also measured against the instrument's own history:
        how far it has moved in the proposed direction within `horizon_bars`,
        taken at a percentile rather than a maximum so one violent week does not
        set the expectation. The target is the smaller of the two.

        A floor protects the other side. Shrinking the target indefinitely would
        buy a high hit rate with trades that cannot pay for their own spread, so
        below `minimum_r` the setup is rejected outright rather than sized down
        into something not worth taking.
        """
        config = self.config
        planned = risk * config.target_r_multiple
        planning_timeframe = (
            Timeframe.parse(profile.planning_timeframe) if profile else Timeframe.H1
        )
        horizon = profile.target_horizon_bars if profile else config.target_horizon_bars
        signal = ctx.series.get(planning_timeframe)
        if signal is None or len(signal.df) < horizon * 3:
            return entry + planned * int(direction), "no history to bound the target"

        frame = signal.df.tail(400)
        closes = frame["close"].to_numpy()
        extremes = (frame["high"] if direction is Direction.LONG else frame["low"]).to_numpy()
        windows = len(closes) - horizon
        if windows <= 0:
            return entry + planned * int(direction), "no history to bound the target"

        # Favourable excursion: how far price ran our way from each starting bar.
        runs = [
            (
                extremes[start + 1 : start + 1 + horizon].max() - closes[start]
                if direction is Direction.LONG
                else closes[start] - extremes[start + 1 : start + 1 + horizon].min()
            )
            for start in range(windows)
        ]
        typical = float(np.quantile(runs, config.target_reach_quantile))

        if typical <= 0:
            return None, "this market has not moved in this direction over the horizon"

        distance = min(planned, typical)
        achieved_r = distance / risk
        if achieved_r < config.minimum_r_multiple:
            return None, (
                f"a reachable target is only {achieved_r:.2f}R — this market travels "
                f"{typical:.5f} in {horizon} bars against a {risk:.5f} stop, below the "
                f"{config.minimum_r_multiple:.2f}R minimum"
            )
        note = "planned" if distance >= planned else f"trimmed to {achieved_r:.2f}R"
        return entry + distance * int(direction), note

    def higher_timeframe_conflict(
        self,
        ctx: MarketContext,
        direction: Direction,
        *,
        timeframes: tuple[str, ...] | None = None,
        threshold: float | None = None,
        minimum_conflicts: int = 1,
    ) -> str | None:
        """Refuse a trade taken straight into an established higher-timeframe trend.

        There was a timing gate for the timeframes *below* the bias and nothing
        at all above it, so the engine happily proposed shorts on indices in
        multi-week uptrends that had just broken to fresh highs. The adviser
        rejected them one after another in exactly those words — "countertrend
        short against a clear, accelerating multi-timeframe uptrend" — which is
        a finding worth encoding rather than paying for once per candidate.

        The drift is a least-squares slope across the window, divided by
        `sqrt(bars) * ATR` — roughly how far a market wanders over that many
        bars for no reason at all. That makes the threshold dimensionless and
        comparable across instruments and window lengths, and it is the same
        normalisation `analysis/position_health.py` uses, so "a strong trend"
        means one thing in this codebase rather than two.

        Deliberately one-sided, like the timing gate: a flat or mildly opposed
        higher timeframe is not an objection, and this never creates a signal.
        Counter-trend trades are not banned in principle — trading *into* a
        strong, still-accelerating trend is what is banned, because the setup
        has to be right about the turn and about its timing at once.
        """
        conflicts: list[tuple[str, float, str]] = []
        wanted = self.config.htf_trend_timeframes if timeframes is None else timeframes
        floor = self.config.htf_trend_veto if threshold is None else threshold
        for timeframe in wanted:
            series = ctx.series.get(Timeframe(timeframe))
            bars = self.config.htf_trend_lookback
            if series is None or len(series.df) < bars + 2:
                continue
            frame = series.df
            atr = self._atr(frame)
            if atr <= 0:
                continue
            closes = frame["close"].tail(bars).to_numpy(dtype=float)
            slope = float(np.polyfit(np.arange(len(closes), dtype=float), closes, 1)[0])
            drift = slope * bars / atr / float(np.sqrt(bars))
            against = drift * -int(direction)
            if against >= floor:
                trend = "uptrend" if drift > 0 else "downtrend"
                conflicts.append((timeframe, against, trend))
        if len(conflicts) >= minimum_conflicts:
            evidence = ", ".join(
                f"{timeframe} {trend} ({against:.2f})" for timeframe, against, trend in conflicts
            )
            return (
                f"{direction.name.lower()} straight into {len(conflicts)} established "
                f"higher-timeframe trend(s): {evidence}; need {minimum_conflicts}, "
                f"limit {floor:.2f} each"
            )
        return None

    def _entry_timing_conflict(
        self,
        ctx: MarketContext,
        direction: Direction,
        *,
        timeframes: tuple[str, ...] | None = None,
    ) -> str | None:
        """Refuse an entry the immediate price action is moving against.

        The engine went straight from an H4/H1 bias to an entry at the current
        ask, with nothing between. The plan was always "higher timeframe bias,
        middle timeframe zone, lower timeframe timing" and the timing step did
        not exist, so a long was proposed at whatever price happened to be
        printing — including while the last hour was selling into it.

        Claude caught exactly this and nothing else. Eleven of the first twelve
        reviews were vetoes, and the recurring sentence was "lower-timeframe
        (M15/M5) price action is falling into the entry, directly opposing the
        long". Encoding that here is cheaper than paying for the same finding
        once per candidate, and it is a real gate rather than a stricter
        threshold: it rejects on evidence that contradicts the setup, not on the
        setup being merely unremarkable.

        Deliberately one-sided. It never *creates* a signal, and a flat lower
        timeframe is not an objection — only a move materially against the
        proposed direction is.
        """
        wanted = self.config.entry_timing_timeframes if timeframes is None else timeframes
        for timeframe in wanted:
            series = ctx.series.get(Timeframe(timeframe))
            if series is None or len(series.df) < 20:
                continue
            frame = series.df
            atr = self._atr(frame)
            if atr <= 0:
                continue
            bars = self.config.entry_timing_lookback
            move = float(frame["close"].iloc[-1]) - float(frame["close"].iloc[-1 - bars])
            adverse_atr = -(move * int(direction)) / atr
            if adverse_atr > self.config.entry_timing_max_adverse_atr:
                return (
                    f"{timeframe} price is moving against the {direction.name.lower()}: "
                    f"{adverse_atr:.2f} ATR adverse over the last {bars} closed bars, "
                    f"limit {self.config.entry_timing_max_adverse_atr:.2f}"
                )
        return None

    def _classify_horizon(
        self,
        agreeing: list[tuple[Signal, float]],
    ) -> tuple[str, str]:
        """Name the plan from the evidence that actually created it.

        Evidence off a fast chart is an intraday plan. Once H1 structure or
        H4/H1 momentum also carries the direction, the market has supplied a
        swing thesis and the slower planning authority is appropriate.

        The fast set used to be the single literal "liquidity_sweep", and
        adding a module without extending it is a silent and expensive
        mistake: `drift_continuation` measures eight M15 bars and was handed a
        swing plan — H1 planning authority, a target twenty-four hours out —
        for a signal whose mechanism expires in about two hours. The list now
        lives in config beside the modules it names.
        """
        modules = {signal.module for signal, _weight in agreeing}
        if modules and modules <= set(self.config.intraday_modules):
            signal = max(agreeing, key=lambda pair: abs(pair[0].score) * pair[1])[0]
            timeframe = str(signal.details.get("timeframe", "M15"))
            return "intraday", f"{signal.module}_{timeframe.lower()}"
        if "market_structure" in modules:
            return "swing", "market_structure_swing"
        if "trend_momentum" in modules:
            return "swing", "trend_momentum_swing"
        return "swing", "+".join(sorted(modules)) or "swing_confluence"

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> float:
        previous = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _reject(
        ctx: MarketContext,
        signals: tuple[Signal, ...],
        reason: str,
        score: float = 0.0,
        confidence: float = 0.0,
    ) -> TradeIdea:
        """A rejected idea, carrying the score it reached where one was computed.

        Returning a flat zero threw away the only number that distinguishes
        "the modules saw nothing" from "the modules saw something and the
        threshold is out of reach". Both land in the journal as NO_SIGNAL, and
        with the score blanked the two are indistinguishable — which is exactly
        the question an operator asks after a day with no trades.
        """
        return TradeIdea(ctx.symbol, False, None, score, confidence, 0.0, 0.0, 0.0, reason, signals)
