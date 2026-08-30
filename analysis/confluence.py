"""Combine independent analysis modules into one auditable trade idea."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.target_reach import FirstTouchOutcomes, first_touch_outcomes
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

    def score_of(self, agreeing: list[tuple[Signal, float]]) -> float:
        """The confluence score: the strongest agreeing reading, raised by how
        strongly it is corroborated.

        ONE DEFINITION, AND IT HAD TO BECOME ONE. This arithmetic existed twice
        -- once at the end of `evaluate` and once inside `readiness`, which
        decides which horizon owns the proposal. Both were the same weighted
        MEAN, so a second reader that AGREED could only pull the average toward
        its own value:

            market_structure alone                        70.0
            market_structure + candle_momentum agreeing    51.9

        Repairing the first copy left the second one selecting exactly what the
        defect always selected: at a fixed bar, a score that falls as agreement
        rises prefers the group where one detector is loud and the rest are
        silent. So the quick/intraday/swing decision kept picking the lonely
        group even once the final score no longer did.

        Two copies of one rule is how that happens. There is now one.

        The premium is weighted by each corroborating module's own strength
        relative to the best, so a token nod earns a token premium; a bounded
        cap stops a crowd of weak modules outscoring one strong, well
        corroborated pair. A single agreeing module scores exactly
        `|score| x confidence`, which is what the mean gave it, so the
        threshold keeps its meaning.
        """
        strengths = sorted(
            (abs(signal.score) * signal.confidence for signal, _ in agreeing), reverse=True
        )
        best = strengths[0] if strengths else 0.0
        if best <= 0.0:
            return 0.0
        corroboration = min(
            self.config.max_corroboration_bonus,
            self.config.corroboration_bonus_per_module
            * sum(other / best for other in strengths[1:]),
        )
        return best * (1.0 + corroboration)

    def evaluate(self, ctx: MarketContext, mode: TradingMode) -> TradeIdea:
        signals = self._signals(ctx)
        if ctx.tick is None:
            return self._reject(ctx, signals, "no executable quote")

        regime = next(
            (s.details.get("regime") for s in signals if s.module == "volatility_regime"), None
        )
        if regime == "extreme":
            return self._reject(ctx, signals, "extreme volatility regime")

        # REGIMES THIS ACCOUNT DOES NOT TRADE, on its own four-day record.
        #
        # 90 closed trades, sliced by what `market_regime` read at entry:
        #
        #     transition   44 trades  59% won  -20.86 EUR   -0.47 a trade
        #     trend_down   11 trades  55% won  -11.75 EUR   -1.07 a trade
        #     trend_up     21 trades  90% won  +22.44 EUR   +1.07 a trade
        #     range         6 trades 100% won   +9.57 EUR   +1.59 a trade
        #     extreme       8 trades 100% won   +9.25 EUR   +1.16 a trade
        #
        # `transition` is half the book and all of the damage. It is the
        # classifier's leftover branch, and what it MEANS is that the two
        # timeframes disagree about direction — so half these trades were taken
        # into a market the system itself could not read. At 22 trades this
        # bucket was +2.41 EUR and was left alone for exactly that reason; at
        # 44 it is -20.86, which is the measurement arriving rather than an
        # opinion changing.
        #
        # `trend_down` is deliberately absent. It loses over these same four
        # days, but it is a real trend rather than chop: refusing it refuses
        # every short in a falling market, which is the thing this account has
        # asked for most. Eleven trades is also too thin to condemn a
        # direction — `transition` looked profitable at 22 as well. The
        # scorecard splits regime by direction now, so when the sample is big
        # enough this can be narrowed to whichever side actually loses.
        #
        # A hard refusal rather than a discount: a discount only removes a
        # module's contribution, and here the objection is to the market, not
        # to any one reader of it.
        blocked = set(self.config.refused_regimes)
        if blocked:
            market_regime = next(
                (s.details.get("regime") for s in signals if s.module == "market_regime"),
                None,
            )
            if market_regime in blocked:
                return self._reject(
                    ctx,
                    signals,
                    f"this account does not trade the {market_regime} regime: it has "
                    f"lost money over every window measured so far, and the objection "
                    f"is to the market rather than to any one module reading it",
                )

        allowed_live = set(self.config.live_enabled_modules)
        effective = self.config.effective_weights(mode)
        weighted: list[tuple[Signal, float]] = []
        for signal in signals:
            weight = effective.get(signal.module, 0.0)
            if weight > 0 and signal.score and signal.confidence >= self.config.minimum_confidence:
                weighted.append((signal, weight))
        if not weighted:
            suffix = "; no modules validated for live" if mode.is_live and not allowed_live else ""
            return self._reject(ctx, signals, f"no weighted directional evidence{suffix}")

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
        # THE RANGE DOES NOT GO AWAY BECAUSE A SECOND MODULE IS STANDING THERE.
        #
        # This used to test `all(...)` over the AGREEING modules: the setup was
        # refused only when every one of them was a continuation module. Any
        # other detector joining in switched the check off completely, so a
        # module asserting "the trend continues" inside a measured range was
        # counted at full strength for having company.
        #
        # NZDJPY LONG, 18 August. Regime `range`. `trend_momentum` at +65,
        # describing itself as "H1 bullish with H4 neutral, unconfirmed by the
        # bias timeframe", and `impulse_break` at +60 — not on the continuation
        # list, so the check never ran. 3.83% of the account went long into a
        # range and never printed a positive tick.
        #
        # It drops the contradicted CONTRIBUTION and not the setup: a veto on
        # the whole idea would throw away the other modules' evidence too, and
        # they were not the ones contradicted. And it runs HERE, before any gate
        # judges the evidence, so the module count, the lone-module floor, the
        # agreement ratio and the score all see the honest set. Ordering it
        # after them let a setup clear the lone-module floor on a module that
        # was about to be discounted.
        discounted = ""
        discounted_regime = ""
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
            # `transition` counts too, and it is the sharper objection of the
            # two. `range` means both timeframes are quiet over a long window,
            # so an hour of clean travel inside it is ordinary. `transition` is
            # the classifier's leftover branch and what it MEANS is that the two
            # timeframes disagree — which is precisely what a continuation claim
            # denies.
            #
            # Every loser opened and read on 17-18 August was in one of the two.
            # Six of six, not one in a trend: HK50, XAUJPY, EURAUD and NDX100 in
            # `transition`, UK100 and NZDJPY in `range`.
            contradicted = set(self.config.continuation_contradicting_regimes)
            if regime_reading in contradicted and continuation:
                honest = [pair for pair in weighted if pair[0].module not in continuation]
                if len(honest) != len(weighted):
                    discounted = ", ".join(
                        sorted(
                            pair[0].module for pair in weighted if pair[0].module in continuation
                        )
                    )
                    if not honest:
                        return self._reject(
                            ctx,
                            signals,
                            f"the regime classifier measures a {regime_reading} while the "
                            f"only firing module(s) ({discounted}) assert a trend is "
                            f"continuing",
                        )
                    discounted_regime = str(regime_reading)
                    weighted = honest

        direction, agreeing, agreement, conflict = self._resolve_direction(weighted)
        if discounted:
            note = (
                f"{discounted} discounted as trend-continuation in a measured "
                f"{discounted_regime}"
            )
            conflict = f"{conflict}; {note}" if conflict else note
        if len(agreeing) < self.config.minimum_directional_modules:
            return self._reject(ctx, signals, "too few independent directional modules")
        # A lone detector has to be sure of itself.
        #
        # For one module the score IS `|raw| x confidence` — the weight cancels,
        # because numerator and denominator both run over the agreeing modules.
        # So a detector firing at the bare `minimum_confidence` floor produces a
        # score the threshold cannot tell apart from a convinced one, and there
        # is no second reading to corroborate or contradict it.
        #
        # HK50 SHORT on 17 August: `impulse_break` alone at exactly 0.45,
        # scoring 60 x 0.45 = 27.0 against a 26.0 bar. EUR 3.13 of a EUR 182
        # account on one unconvinced opinion, not a single positive tick, and
        # -0.56R at the close.
        #
        # Aimed at lone AND unconvinced rather than at lone. Requiring two
        # modules would also refuse a single detector reading 0.90, which is the
        # strongest single piece of evidence this engine can produce, and would
        # cost far more setups than the one failure mode being removed.
        #
        # ONE FLOOR FOR EIGHT DETECTORS WAS A BLUNT INSTRUMENT. Measured over 90
        # days on five symbols, dropping it from 0.65 to 0.55 more than doubles
        # the setups formed, 71 to 153. But of the ~1,174 refusals it releases,
        # 473 are `fast_ema_cross` and 414 are `liquidity_sweep` — and those two
        # have the worst and the best records in the module backtest
        # respectively. A single number cannot let one through and hold the
        # other, so it was doing both or neither.
        #
        # The per-module table is empty by default and the global value applies
        # to everything, so this changes nothing until a detector has earned an
        # entry. Earned means a number from `scripts/backtest_modules.py`, not
        # an argument.
        solo_module = agreeing[0][0].module if len(agreeing) == 1 else ""
        lone_floor = self.config.lone_floor_for(solo_module)
        if len(agreeing) == 1 and lone_floor > 0:
            solo = agreeing[0][0]
            if solo.confidence < lone_floor:
                return self._reject(
                    ctx,
                    signals,
                    f"{solo.module} is the only detector pointing this way and reads "
                    f"{solo.confidence:.2f}, under the {lone_floor:.2f} a lone module needs; "
                    f"one unconvinced opinion is not corroborated evidence",
                )
        if agreement < self.config.minimum_agreement_ratio:
            return self._reject(
                ctx, signals, f"directional agreement {agreement:.1%} below threshold"
            )

        denominator = sum(weight for _, weight in agreeing)
        # THE ENGINE IS CALLED CONFLUENCE AND THIS LINE PUNISHED CONFLUENCE.
        #
        # The score was a weighted MEAN over the agreeing modules, so a second
        # reader that AGREES could only pull the average toward its own value.
        # Against the threshold of 45:
        #
        #     market_structure alone                        70.0
        #     market_structure + candle_momentum agreeing    51.9
        #
        # Two readers pointing the same way scored eighteen points LOWER than
        # one reader alone. Corroboration was a penalty, and the penalty grew
        # with how much corroboration there was.
        #
        # WHAT THAT SELECTED FOR is the part that matters. At a fixed bar, an
        # engine whose score falls as agreement rises does not merely
        # undervalue corroborated setups -- it systematically prefers the ones
        # where exactly ONE detector is loud and every other is silent. Those
        # are the least corroborated readings available. All eight detectors
        # came back at 54-57% with an average win of 0.68R against an average
        # loss of 1.04R: one shape, indistinguishable from each other and
        # barely distinguishable from a coin.
        #
        # This was known and written down without the conclusion being drawn.
        # `candle_momentum`'s docstring says "joining a strong reader makes
        # matters worse rather than better", and the answer at the time was to
        # give that module its own lane AROUND the vote rather than to ask why
        # agreement was being taxed.
        #
        # THE FIX: THE SCORE IS THE STRONGEST READING, RAISED BY HOW STRONGLY
        # IT IS CORROBORATED.
        #
        # A premium on top of the mean was tried first and was not enough: at
        # 15% a corroborated pair came out at 59.7 against 70.0 alone, so the
        # penalty survived its own repair and only got smaller. The property
        # has to hold by construction rather than by tuning, and the
        # construction is that agreement may not dilute the reading that
        # earned the score.
        #
        # The premium is weighted by each corroborating module's OWN strength
        # relative to the best, so a token nod earns a token premium and a
        # second strong reader earns a real one. Without that, any module
        # scraping past its floor would buy the full bonus, and "find anything
        # that agrees" would become the strategy.
        #
        # The scale is untouched where it matters: a single agreeing module
        # scores exactly `|score| x confidence`, which is what the plain mean
        # gave it, so the threshold keeps its meaning and nothing that trades
        # today stops trading. Zero restores the old arithmetic for one module
        # and removes the premium for the rest.
        score = self.score_of(agreeing)
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
        # The stop the chart asked for, before the fee schedule gets a say. It
        # is kept because the two are measured over different amounts of time
        # and the difference between them is what says how much.
        natural_risk = abs(entry - stop)
        cost_floor = self._cost_floor(ctx)
        if natural_risk < cost_floor:
            stop = entry - cost_floor * int(direction)
        risk = abs(entry - stop)
        if risk <= 0:
            return self._reject(
                ctx, signals, "could not construct a positive stop distance", score, confidence
            )
        horizon_bars = self._horizon_bars(risk, natural_risk, profile, frame=frame)
        target, target_note = self._reachable_target(
            ctx,
            entry,
            risk,
            direction,
            profile=profile,
            horizon=horizon_bars,
            setup_family=setup_family,
        )
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
                # The dissent travels with the trade rather than killing it.
                # The reviewer is the right place to weigh "the hourly chart
                # disagrees" — it has the whole payload and this engine does
                # not — and hiding it would be asking for an opinion on half
                # the evidence.
                + (f"; {conflict}" if conflict else "")
            ),
            signals=signals,
            setup_family=setup_family,
            horizon=horizon,
            planning_timeframe=planning_timeframe.value,
            # The same horizon the target was measured over, not the profile's
            # nominal one. Everything downstream derives its window from this
            # number — the runner's reach gate, its survival gate and its
            # runway check all divide it by the planning timeframe — so a
            # target measured over one span and judged over another is the
            # disagreement this whole line of work keeps running into.
            expected_horizon_minutes=int(
                planning_timeframe.duration.total_seconds() / 60 * horizon_bars
            ),
        )

    def _horizon_bars(
        self,
        risk: float,
        natural_risk: float,
        profile: HorizonProfileConfig | None,
        *,
        frame: pd.DataFrame | None = None,
    ) -> int:
        """Bars to allow the target, once the fee schedule has moved the stop.

        THE BUG THIS FIXES WAS MINE AND IT RAN LIVE. Flooring the stop at what
        the costs demand makes 1R a longer distance — on a quiet instrument a
        two-pip stop becomes seven — while the window it had to be covered in
        stayed at the profile's nominal bars. Measured on a synthetic walk at
        0.0002 per bar, the same market that cleared at +0.21R expected with
        the chart's own stop was REFUSED outright at the cost floor: nothing
        between 1.00R and 3.00R was reached even once in twenty-four bars.
        A change made to produce more setups was producing fewer.

        Price does not travel in a straight line, so covering a distance d
        takes `(d / ATR)^2` bars and not `d / ATR` — the same square law the
        runner's runway check already uses, and the reason a stop 3.4x wider
        needs eleven times the time rather than three.

        Capped, because the square law grows fast and an unbounded window stops
        describing the trade anyone would actually sit in — and bounded by the
        profile again, so a chart whose own stop already clears the costs is
        measured over exactly the bars it always was.
        """
        base = profile.target_horizon_bars if profile else self.config.target_horizon_bars

        # AND THE WINDOW HAS TO BE LONG ENOUGH TO SEE THE PLAN AT ALL.
        #
        # The stretch above only fires when the FEE SCHEDULE widened the stop.
        # A stop that was born wide — a structural swing stop reaching back to
        # the last swing low, which is routinely eight to sixteen ATR — got the
        # profile's flat 24 bars and could not be measured in them:
        #
        #     stop      target at 0.75R    bars the square law needs
        #     4x ATR          3.0 ATR                     9   fits
        #     6x ATR          4.5 ATR                    20   fits
        #     8x ATR          6.0 ATR                    36   does not
        #     16x ATR        12.0 ATR                   144   does not
        #     32x ATR        24.0 ATR                   576   does not
        #
        # Past six ATR every window expires with neither the target nor the
        # stop touched, and an expired window is charged a round trip, so the
        # expectancy converges on minus the cost however good the market is.
        # That is 1,683 refusals an hour reading "no target pays" about plans
        # nothing ever measured — the ruler was too short, not the market too
        # slow.
        #
        # So the window is sized to the distance it has to watch, by the same
        # square law the runway check uses. It cannot invent a trade: the
        # measurement still has to come back positive, and `filters.runway`
        # still refuses a plan that cannot finish before the session does. All
        # this buys is the chance to be measured.
        if frame is not None and len(frame) > 1 and self.config.fit_horizon_to_the_plan:
            closes = frame["close"].to_numpy(dtype=float)
            per_bar = float(np.abs(np.diff(closes[-400:])).mean())
            if per_bar > 0:
                needed = (risk * self.config.minimum_r_multiple / per_bar) ** 2
                # AND NEVER FURTHER THAN THE HISTORY CAN ACTUALLY MEASURE.
                #
                # `_reachable_target` needs three times the horizon in bars and
                # cuts its own sample to the last 400; past either of those it
                # gives up and returns the planned distance UNMEASURED. So a
                # stretch that outran the data did not widen the window, it
                # removed the gate — a test caught this approving a trade it
                # was written to refuse, with a 34,560-minute horizon and
                # "no history to bound the target" in the reason.
                #
                # Half the usable sample, so every window still has as many
                # siblings as it has bars. A plan that needs more than that is
                # one this history cannot judge, and it must keep being refused
                # for that reason rather than waved through for want of a
                # measurement.
                usable = min(len(frame), 400)
                affordable = min(len(frame) // 3, usable // 2)
                ceiling = min(int(base * self.config.max_plan_horizon_stretch), affordable)
                base = max(base, min(int(needed), max(ceiling, 0)))

        if natural_risk <= 0 or risk <= natural_risk:
            return base
        stretch = min((risk / natural_risk) ** 2, self.config.max_cost_horizon_stretch)
        return max(base, int(base * stretch))

    @staticmethod
    def _cost_floor(ctx: MarketContext) -> float:
        """The narrowest stop this instrument's costs allow, or zero.

        Supplied by the runner, which is the only layer that knows the broker's
        fee schedule, and read defensively: an engine running in a backtest or
        a unit test gets zero and plans exactly as it always did.

        It belongs here rather than only at execution because a stop that is
        going to be widened invalidates everything downstream of it — the
        target distance, how often that distance is reached, and the expectancy
        that decides whether the setup exists at all.
        """
        try:
            value = float(ctx.meta.get("min_stop_for_costs", 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0
        return value if value > 0 else 0.0

    def _reachable_target(
        self,
        ctx: MarketContext,
        entry: float,
        risk: float,
        direction: Direction,
        *,
        profile: HorizonProfileConfig | None = None,
        horizon: int | None = None,
        setup_family: str = "",
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
        # A FAMILY MAY NAME ITS OWN RATIO. One number cannot be right for two
        # different trades: `impulse_retest` was measured at every ratio from
        # 0.75 to 3.0 and nets +0.28R at 1:1 against +0.016R at 3:1, so sending
        # it out under the account's 3.0 would trade a measured strategy at
        # nearly zero. Matched on substring because `setup_family` is
        # `{module}_{timeframe}`.
        #
        # AND THE OVERRIDE HAS TO BE THE EXIT, not a suggestion the search
        # below is free to overrule -- which is exactly what it was.
        #
        # `planned` was consulted on ONE branch, the fallback taken when
        # `_first_touch_outcomes` returns None. On the branch the code actually
        # takes, the search swept `candidate` up to
        # `max(target_r_multiple, minimum_r_multiple)` -- the ACCOUNT's 3.0 --
        # and picked whatever distance its own model liked. The family's 1.0
        # bounded nothing. The dry run of 30 August shows the damage directly:
        # `order_block` winners averaged 1.33R where the shipped plan says 1.00,
        # and the hit rate came in at 34% against a measured 62%.
        #
        # It cannot be left as a mere ceiling either. The search maximises
        # expectancy per bar under a first-touch model, and that model is the
        # one whose bias this account has already measured: a random entry
        # reads +0.073R at 3:1 because a bar registers a barrier when its
        # extreme crosses it, and the overshoot is proportionally larger on the
        # nearer barrier. The bias grows with the ratio, so a free search is
        # pulled toward the far target on purpose.
        #
        # A named family is not a market to be surveyed. 18,828 resolved trades
        # measured ONE exit; any other exit is a different strategy that has
        # never been measured. So when a family names its ratio, that is the
        # target and there is no search.
        ratio = config.target_r_multiple
        named_family = False
        for family, override in config.target_r_multiple_by_family.items():
            if family in setup_family:
                ratio = override
                named_family = True
                break
        planned = risk * ratio
        if named_family:
            return (
                entry + planned * int(direction),
                f"{ratio:.2f}R fixed by the {setup_family} family, as measured",
            )
        planning_timeframe = (
            Timeframe.parse(profile.planning_timeframe) if profile else Timeframe.H1
        )
        # `horizon` overrides the profile when the caller has already worked out
        # how long this particular stop needs; see `_horizon_bars`.
        if horizon is None:
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

        # PICK THE DISTANCE THAT PAYS, INSTEAD OF THE ONE THE MULTIPLE ASKS FOR.
        #
        # Everything needed for this was already here and one line threw it
        # away: `runs` is the whole empirical distribution of how far this
        # market travels, and the code reduced it to a single quantile, capped
        # the plan at it, and never asked the question that decides the trade —
        # how OFTEN is this particular distance reached, and does that beat what
        # it costs to find out.
        #
        # A fixed multiple cannot ask that. At 1.2R a market needs 45.5% to
        # break even before costs, and whether it delivers 30% or 60% is a
        # property of the instrument that the multiple never consults. So a
        # market reaching 1.2R half the time and one reaching it a fifth of the
        # time were given the same target, and a market that pays handsomely at
        # 1.0R but not at 1.2R was refused outright for missing a floor.
        #
        # Both failures cost trades AND cost money, which is why fixing it is
        # not a loosening: the search below can only ever choose a distance with
        # a positive measured expectancy, and there are distances a fixed
        # multiple refuses that clear that bar comfortably.
        outcomes = self._first_touch_outcomes(frame, closes, direction, risk, horizon)
        if outcomes is None:
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

        # THE WHOLE BILL, IN R. The spread comes from the live quote;
        # commission and slippage come from the runner, which is the only layer
        # that knows the fee schedule.
        #
        # This counted the spread alone, on the reasoning that the sizer
        # charges commission again later and counting it twice would refuse
        # trades the sizer is about to accept. That holds at 1.0R, where
        # commission is a small part of the reward. It breaks below it: at
        # 0.6R the reward is 40% smaller and the commission is unchanged, so
        # the estimate was most optimistic at exactly the distances the search
        # is now allowed to consider. Choosing a short target on an optimistic
        # cost is how "shrinking the target buys a high hit rate on trades that
        # cannot pay for their own spread" actually happens.
        #
        # Zero when the runner did not supply it — a backtest, a unit test —
        # which is the old behaviour and not a claim that the trade is free.
        spread = getattr(ctx.tick, "spread", 0.0) or 0.0
        try:
            overhead = float(ctx.meta.get("round_trip_cost_price", 0.0))
        except (AttributeError, TypeError, ValueError):
            overhead = 0.0
        cost_r = ((spread + max(overhead, 0.0)) / risk) if risk > 0 else 0.0

        # HOW LONG A DISTANCE TAKES, so the search can prefer sooner over bigger.
        #
        # Price does not travel in a straight line: covering d takes (d/speed)^2
        # bars, the same square law the runway check uses. Typical absolute
        # bar-to-bar movement is the speed; it needs no ATR machinery and it is
        # measured on the frame already in hand.
        moves = np.abs(np.diff(closes))
        per_bar = float(moves.mean()) if moves.size else 0.0

        best_r, best_reach, best_edge, best_bars = 0.0, 0.0, 0.0, 0.0
        best_rate = 0.0
        step = 0.05
        candidate = config.minimum_r_multiple
        ceiling = max(config.target_r_multiple, config.minimum_r_multiple)
        while candidate <= ceiling + 1e-9:
            # Expected R of one trade at this distance, scoring each window as
            # what it was.
            #
            # THIS LINE USED TO READ, AND IT WAS WRONG:
            #
            #     edge = hit * (candidate - cost_r) - (1 - hit) * (1 + cost_r)
            #
            # Two outcomes where there are three. Everything that did not reach
            # the target was charged the full stop, including every window in
            # which price drifted sideways and the horizon simply expired
            # without the stop ever being touched.
            #
            # That is not a rounding error. On a synthetic walk carrying a real
            # edge, 46% of windows expire at a 1R target, and charging them
            # -1.15R each subtracts about half an R from every evaluation: the
            # market reads -0.15R and is refused where the truth is +0.31R. It
            # is also worst exactly where this account lives, because the
            # expired share grows with the target and with a short horizon.
            #
            # It does NOT manufacture edge. On a driftless walk the corrected
            # form is still negative at every distance from 0.6R to 3.0R
            # (-0.13R to -0.39R), which is the check that matters: a coin flip
            # must stay refused.
            odds = outcomes.expectancy_r(distance=candidate * risk, risk=risk, cost_r=cost_r)
            edge, hit = odds.expected_r, odds.reach
            # AND THE TARGET HAS TO BE THE EXIT, not decoration.
            #
            # Pricing expired windows honestly opens a hole the two-outcome
            # form did not have: on a market that only drifts, every window
            # ends a little in front, nothing is ever stopped, and the distance
            # is never touched. Expectancy comes out positive with a reach of
            # zero. That is not a plan — this system leaves at the target or at
            # the stop, and on such a market neither ever arrives.
            #
            # So the classic break-even rate is still enforced, on the
            # population it was derived for: the windows that actually resolved
            # one way or the other.
            if not odds.target_is_the_exit(
                reward_risk=candidate,
                cost_r=cost_r,
                minimum_resolved=config.target_minimum_resolved_windows,
            ):
                candidate += step
                continue
            # EXPECTANCY PER BAR, NOT PER TRADE, and the difference decides
            # whether this account trades at all.
            #
            # Maximising edge per trade systematically prefers the far target,
            # because a bigger target needs a lower hit rate to pay. With
            # `target_r_multiple` at 3.0 the search kept choosing 3R — and 3R on
            # an H1 plan is about nine hours of travel under the square law,
            # against eight or nine hours in a whole session. 56 of 140 live
            # setups died on INSUFFICIENT_RUNWAY: not refused on their merits,
            # refused because the target the search picked could not be reached
            # before the day ended.
            #
            # A session has a deadline, so what matters is expectancy per unit
            # of TIME. 1R reached in an hour beats 3R that needs nine hours you
            # do not have, and the money is in the account either way.
            bars_needed = (candidate * risk / per_bar) ** 2 if per_bar > 0 else 1.0
            rate = edge / max(bars_needed, 1e-9) if config.prefer_sooner_targets else edge
            if edge > 0 and rate > best_rate:
                best_r, best_reach, best_edge = candidate, hit, edge
                best_bars, best_rate = bars_needed, rate
            candidate += step

        if best_edge <= 0.0:
            worst = config.minimum_r_multiple
            floor = outcomes.expectancy_r(distance=worst * risk, risk=risk, cost_r=cost_r)
            expired = 1.0 - float(floor.resolved_windows) / max(outcomes.windows, 1)
            # TWO REFUSALS THAT LOOK IDENTICAL AND ARE NOT.
            #
            # "The market resolved these windows and the plan lost" is a
            # judgement about the trade. "Almost nothing resolved" is a
            # statement about the MEASUREMENT: within this horizon price
            # neither reached the target nor took the stop, so the window
            # cannot answer the question at all. Both used to print as "no
            # target pays on this market", which sends every diagnosis after it
            # to the wrong place — one says fix the plan, the other says the
            # horizon is too short for the stop being planned.
            #
            # The second is not rare here. Same market, same edge, stop widened
            # from 4x to 32x ATR against a fixed 24-bar horizon: resolved
            # windows go 2,915 -> 7, the expired share 2% -> 99.8%, and the
            # expectancy +0.31R -> -0.12R purely because every expired window
            # is charged a round trip. The market did not get worse; the ruler
            # got too short.
            if floor.resolved_windows < config.target_minimum_resolved_windows:
                bars = horizon or 0
                return None, (
                    f"this horizon cannot judge a {risk:.5g} stop: in {bars} bars only "
                    f"{floor.resolved_windows} of {outcomes.windows} windows reached "
                    f"{worst:.2f}R or the stop, {expired:.0%} simply ran out of time. "
                    f"The stop is too wide for the time the plan has, not the target too far"
                )
            return None, (
                f"no target between {worst:.2f}R and {ceiling:.2f}R pays on this market: "
                f"at {worst:.2f}R it goes our way {floor.resolved_reach:.0%} of the "
                f"{floor.resolved_windows} windows that resolved ({expired:.0%} expired), "
                f"worth {floor.expected_r:+.2f}R a trade at a {cost_r:.0%}-of-risk round trip"
            )

        distance = best_r * risk
        note = (
            f"{best_r:.2f}R, reached first {best_reach:.0%} of the time here "
            f"for {best_edge:+.2f}R expected"
            + (f" in about {best_bars:.0f} bars" if best_bars else "")
        )
        return entry + distance * int(direction), note

    @staticmethod
    def _first_touch_outcomes(
        frame: pd.DataFrame,
        closes: np.ndarray,
        direction: Direction,
        risk: float,
        horizon: int,
    ) -> FirstTouchOutcomes | None:
        """Per window: how it ended — target first, stop first, or neither.

        `runs` measures the favourable excursion and ignores the stop, so a
        window where price dropped a full R and only then rallied counts as a
        win at any distance below that rally. That overstates every reach rate
        it produces, and the overstatement is largest exactly where it matters —
        volatile markets that whip both ways.

        The third outcome is why this returns the whole record and not the
        excursion array. A window that expired flat is not a loss and must not
        be priced as one; without `stopped` there is no way to tell the two
        apart, and the search was pricing every one of them as a full stop.

        The walk itself lives in `analysis.target_reach` because the runner
        needs the same measurement one step later, against the stop the order
        will actually carry rather than the one this engine proposed. Two
        copies of a statistic that must agree is how the two gates came to
        disagree in the first place.

        None when the frame lacks the columns or the history to do it, so the
        caller keeps its previous behaviour rather than inventing a number.
        """
        return first_touch_outcomes(
            frame,
            risk=risk,
            bars_ahead=horizon,
            long=direction is Direction.LONG,
            closes=closes,
        )

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

    @staticmethod
    def _vote(pool: list[tuple[Signal, float]]) -> tuple[Direction, list[tuple[Signal, float]]]:
        """Which way the measured evidence points, and which modules agree.

        A module weight says how much its method is trusted; it is not the
        evidence itself. The old vote counted a barely firing 0.50-confidence
        read exactly like the same module at 0.90 and ignored the magnitude of
        its score. That made weak slow states outvote strong fresh events. The
        same strength used by the score now chooses direction symmetrically.
        """

        def strength(pair: tuple[Signal, float]) -> float:
            signal, weight = pair
            return abs(signal.score) * signal.confidence * weight

        positive = sum(strength(pair) for pair in pool if pair[0].score > 0)
        negative = sum(strength(pair) for pair in pool if pair[0].score < 0)
        direction = Direction.LONG if positive > negative else Direction.SHORT
        return direction, [
            (signal, weight) for signal, weight in pool if signal.score * int(direction) > 0
        ]

    def _resolve_direction(
        self, weighted: list[tuple[Signal, float]]
    ) -> tuple[Direction, list[tuple[Signal, float]], float, str]:
        """Pick a direction without making modules on different clocks argue.

        THE BUG THIS REPLACES cost 3,894 refusals in one day and is the direct
        cause of "why did it never take a short". The vote used to be a single
        weight sum across every firing module. On a market trending up on H4/H1
        while selling hard for the last two hours that produced:

            trend_momentum     LONG   weight 1.0
            drift_continuation SHORT  weight 0.7
            -> direction LONG, agreement 1.0/1.7 = 58.8%, below the 60% floor
            -> the whole setup discarded. No long, no short, nothing.

        And had it scraped past 60% it would have taken the LONG, into the
        selling, and died one gate later on "M15 price is moving against the
        long" — which is why simply lowering the agreement floor gains nothing
        and was not done.

        THOSE TWO MODULES ARE NOT CONTRADICTING EACH OTHER. `trend_momentum`
        reads 20/50 EMAs on H4 and H1 and answers "where is this going over
        days". `drift_continuation` measures eight M15 bars and answers "where
        is it going over the next two hours". A multi-day uptrend with an hour
        of hard selling is not a paradox, it is a pullback — and a pullback is
        a tradeable short with a two-hour horizon and a target twelve M15 bars
        out. The engine was treating one question as two votes on another.

        So the vote now runs inside each horizon group. When the groups agree,
        the whole set carries the trade exactly as before, and the agreement
        ratio is stronger for having been checked. When they disagree, the
        intraday group wins and the swing modules are simply not part of that
        trade — they were answering a different question and their dissent is
        recorded in the reason rather than used as a veto.

        WHY THE FAST SIDE WINS a disagreement, which is the one genuinely
        contestable choice here: its evidence is measured rather than inferred
        (a drift over eight bars, a separation in ATR, against an EMA
        alignment), its stop is nearer so being wrong is cheaper, and its
        horizon expires in hours rather than days so the slower thesis is still
        intact afterwards. What it is NOT is a claim that fast evidence is
        better. Taking the swing side instead would mean entering against a
        move currently underway, which the entry-timing gate refuses anyway.

        WHAT STILL STANDS between this and an order: the intraday profile's own
        higher-timeframe veto still refuses a trade fighting BOTH H4 and D1,
        the score threshold, entry timing on M5/M1, the target base rate, every
        filter, the sizer, and Claude.
        """
        quick_names = set(self.config.quick_modules)
        intraday_names = set(self.config.intraday_modules) - quick_names
        quick = [pair for pair in weighted if pair[0].module in quick_names]
        intraday = [pair for pair in weighted if pair[0].module in intraday_names]
        swing = [
            pair
            for pair in weighted
            if pair[0].module not in quick_names and pair[0].module not in intraday_names
        ]

        def scored(pool: list[tuple[Signal, float]]) -> tuple[Direction, list, float]:
            direction, agreeing = self._vote(pool)

            def evidence(pair: tuple[Signal, float]) -> float:
                return abs(pair[0].score) * pair[0].confidence * pair[1]

            total = sum(evidence(pair) for pair in pool)
            agreed = sum(evidence(pair) for pair in agreeing)
            return direction, agreeing, (agreed / total if total else 0.0)

        def readiness(result: tuple[Direction, list, float]) -> tuple[bool, float]:
            """Whether one horizon can stand on its own, and its actual score.

            A firing module is not automatically an executable setup. Previously
            the fastest firing group owned the entire symbol even when its
            confidence-discounted score was below the common threshold. That
            hid a fully qualified slower plan and mislabeled the symbol
            ``NO_SIGNAL``. Apply the existing gates per horizon before deciding
            which clock owns the proposal; no threshold is relaxed here.
            """
            _direction, agreeing, agreement = result
            # The SAME arithmetic as the final score, because it was a second
            # copy of it and repairing only the other one left this deciding
            # which horizon owns the symbol on the defect.
            score = self.score_of(agreeing)
            qualified = (
                len(agreeing) >= self.config.minimum_directional_modules
                and agreement >= self.config.minimum_agreement_ratio
                and score >= self.config.score_threshold
            )
            return qualified, score

        def join_notes(*notes: str) -> str:
            return "; ".join(note for note in notes if note)

        # A closed M1/M5 trigger is a complete entry event. Mixing it with an
        # M15 or H1 state used to change its stop and expiry, or erase a quick
        # short under a heavier slow long. Give quick evidence its own vote;
        # the quick horizon profile still checks H1/H4 conflict afterwards.
        weak_quick_note = ""
        if quick:
            quick_result = scored(quick)
            direction, agreeing, agreement = quick_result
            slower = [*intraday, *swing]
            quick_ready, quick_score = readiness(quick_result)
            if quick_ready or not slower:
                if not slower:
                    return direction, agreeing, agreement, ""
                slower_directions = {self._vote([pair])[0].name for pair in slower}
                slower_modules = ", ".join(sorted(signal.module for signal, _ in slower))
                return (
                    direction,
                    agreeing,
                    agreement,
                    (
                        f"quick {direction.name} owns the executable plan; slower "
                        f"{','.join(sorted(slower_directions))} context from {slower_modules} is "
                        "reported separately; different horizons are not averaged into one trade"
                    ),
                )
            weak_quick_note = (
                f"quick evidence scored {quick_score:.1f} below the independent "
                f"{self.config.score_threshold:.1f} requirement, so it did not hide a "
                "qualified slower setup"
            )

        slower = [*intraday, *swing]
        if not intraday or not swing:
            direction, agreeing, agreement = scored(slower)
            return direction, agreeing, agreement, weak_quick_note

        fast_result = scored(intraday)
        slow_result = scored(swing)
        fast_direction, _fast_agreeing, _fast_agreement = fast_result
        slow_direction, _slow_agreeing, _slow_agreement = slow_result
        if fast_direction is slow_direction:
            direction, agreeing, agreement = scored(slower)
            return direction, agreeing, agreement, weak_quick_note

        fast_ready, fast_score = readiness(fast_result)
        slow_ready, slow_score = readiness(slow_result)
        if slow_ready and not fast_ready:
            direction, agreeing, agreement = slow_result
            return (
                direction,
                agreeing,
                agreement,
                join_notes(
                    weak_quick_note,
                    f"swing {direction.name} owns the plan because its {slow_score:.1f} score "
                    f"qualified while the opposing intraday read scored only {fast_score:.1f}",
                ),
            )

        direction, agreeing, agreement = fast_result
        dissenting = ", ".join(sorted(signal.module for signal, _ in swing))
        return (
            direction,
            agreeing,
            agreement,
            join_notes(
                weak_quick_note,
                (
                    f"intraday {direction.name} taken over a {slow_direction.name} reading from "
                    f"{dissenting} on the slower charts; different horizons, not a contradiction"
                ),
            ),
        )

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
        if modules and modules <= set(self.config.quick_modules):
            signal = max(agreeing, key=lambda pair: abs(pair[0].score) * pair[1])[0]
            timeframe = str(signal.details.get("timeframe", "M5"))
            return "quick", f"{signal.module}_{timeframe.lower()}"
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
