"""Cross-market context that ranks ideas without becoming another trade gate.

The deterministic engine answers: "is there a setup on this chart?"  This
module answers the different question: "how does that setup compare with the
other setups visible right now?"  Its output is deliberately advisory.  It can
change ordering and enrich Claude's briefing, but it cannot approve a rejected
idea, alter prices, resize risk, or block an otherwise valid candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analysis.confluence import TradeIdea
from config.schema import AssetClassRoutingConfig
from core.instrument import AssetClass
from core.types import MarketContext, Signal, Timeframe

ASSET_PLAYBOOKS: dict[AssetClass, dict[str, object]] = {
    AssetClass.FOREX: {
        "market_clock": "Asia, London, New York and rollover liquidity",
        "primary_drivers": ("relative rates", "currency strength", "macro releases"),
        "useful_styles": ("trend continuation", "liquidity sweep", "session breakout"),
    },
    AssetClass.STOCK: {
        "market_clock": "listed exchange hours, opening gap and closing auction",
        "primary_drivers": ("earnings", "sector strength", "index beta", "gap/volume"),
        "useful_styles": ("opening continuation", "gap reaction", "trend continuation"),
    },
    AssetClass.CRYPTO: {
        "market_clock": "continuous 24/7 liquidity with a distinct weekend regime",
        "primary_drivers": ("risk appetite", "liquidity", "momentum", "weekend depth"),
        "useful_styles": ("momentum", "volatility expansion", "liquidity sweep"),
    },
    AssetClass.INDEX: {
        "market_clock": "cash-session opens over a nearly continuous CFD market",
        "primary_drivers": ("market breadth", "rates", "volatility", "risk appetite"),
        "useful_styles": ("trend continuation", "opening range", "failed breakout"),
    },
    AssetClass.METAL: {
        "market_clock": "Asia, London metals and New York liquidity",
        "primary_drivers": ("real yields", "USD", "risk-off demand"),
        "useful_styles": ("trend continuation", "liquidity sweep", "level reaction"),
    },
    AssetClass.COMMODITY: {
        "market_clock": "exchange session, daily settlement and scheduled reports",
        "primary_drivers": ("USD", "inventory", "seasonality", "supply shocks"),
        "useful_styles": ("trend continuation", "breakout", "level reaction"),
    },
    AssetClass.UNKNOWN: {
        "market_clock": "unknown broker product schedule",
        "primary_drivers": ("unclassified",),
        "useful_styles": ("observation only",),
    },
}


@dataclass(frozen=True, slots=True)
class MarketObservation:
    symbol: str
    asset_class: str
    regime: str
    h1_move_atr: float
    h4_move_atr: float
    d1_move_atr: float
    direction_votes: int
    atr_percentile: float | None
    last_h1_bar: str

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OpportunityIntelligence:
    modifier: float
    regime: str
    asset_context: dict[str, object]
    reasons: tuple[str, ...]
    thesis: str
    scout_alignment: float = 0.0
    learned_alignment: float = 0.0

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


def observe_market(
    context: MarketContext,
    asset_class: AssetClass,
    signals: tuple[Signal, ...],
) -> MarketObservation:
    """Build a scale-free point-in-time observation from closed bars only."""
    regime_signal = next((signal for signal in signals if signal.module == "market_regime"), None)
    regime = str(regime_signal.details.get("regime", "unknown")) if regime_signal else "unknown"
    atr_percentile = None
    if regime_signal is not None and regime_signal.details.get("atr_percentile") is not None:
        atr_percentile = float(regime_signal.details["atr_percentile"])

    h1 = _normalised_move(context, Timeframe.H1, 12)
    h4 = _normalised_move(context, Timeframe.H4, 8)
    d1 = _normalised_move(context, Timeframe.D1, 5)
    moves = (h1, h4, d1)
    votes = sum(1 if value > 0.15 else -1 if value < -0.15 else 0 for value in moves)
    h1_series = context.series.get(Timeframe.H1)
    return MarketObservation(
        symbol=context.symbol,
        asset_class=asset_class.value,
        regime=regime,
        h1_move_atr=round(h1, 4),
        h4_move_atr=round(h4, 4),
        d1_move_atr=round(d1, 4),
        direction_votes=votes,
        atr_percentile=round(atr_percentile, 4) if atr_percentile is not None else None,
        last_h1_bar=h1_series.last_bar_time.isoformat() if h1_series is not None else "",
    )


def assess_opportunity(
    idea: TradeIdea,
    observation: MarketObservation,
    asset_class: AssetClass,
    *,
    cap: float,
    routing: AssetClassRoutingConfig | None = None,
) -> OpportunityIntelligence:
    """Return a bounded ranking modifier; never a pass/fail decision."""
    policy = routing or AssetClassRoutingConfig()
    reasons: list[str] = []
    modifier = 0.0
    sign = int(idea.direction) if idea.direction is not None else 0
    regime = observation.regime

    if regime in {"trend_up", "trend_down"} and sign:
        regime_sign = 1 if regime == "trend_up" else -1
        if regime_sign == sign:
            modifier += policy.trend_alignment_bonus
            reasons.append("trade direction agrees with the H1/H4 persistence regime")
        else:
            modifier -= policy.countertrend_penalty
            reasons.append("trade is counter to the H1/H4 persistence regime")
    elif regime == "range":
        reversal = any(
            signal.module in {"liquidity_sweep", "level_reaction"} and signal.score * sign > 0
            for signal in idea.signals
        )
        if reversal:
            modifier += policy.range_reversal_bonus
            reasons.append("reversal evidence matches the measured range regime")
        else:
            reasons.append("range regime offers no extra support to this directional setup")
    elif regime == "transition":
        modifier -= policy.transition_penalty
        reasons.append("market is transitioning; ranking is reduced for ambiguity")
    elif regime == "extreme":
        modifier -= policy.extreme_penalty
        reasons.append("realised volatility is extreme")

    supporting = {
        signal.module
        for signal in idea.signals
        if sign and signal.score * sign > 0 and signal.confidence > 0
    }
    affinities = [
        policy.module_affinity[name] for name in supporting if name in policy.module_affinity
    ]
    if affinities:
        affinity = sum(affinities) / len(affinities)
        modifier += affinity
        reasons.append(
            f"{asset_class.value} routing adds {affinity:+.1f} for " + ", ".join(sorted(supporting))
        )

    if sign and observation.direction_votes * sign > 0:
        alignment = min(3, abs(observation.direction_votes))
        modifier += float(alignment * 1.5)
        reasons.append(f"{alignment}/3 measured horizons lean with the proposal")
    elif sign and observation.direction_votes * sign < 0:
        modifier -= 2.0
        reasons.append("the broader closed-bar drift leans against the proposal")

    modifier = max(-cap, min(cap, modifier))
    profile = dict(ASSET_PLAYBOOKS.get(asset_class, ASSET_PLAYBOOKS[AssetClass.UNKNOWN]))
    profile["asset_class"] = asset_class.value
    thesis = (
        f"{idea.direction.name if idea.direction else 'WAIT'} {idea.symbol} from "
        f"{idea.reason}; regime={regime}; " + "; ".join(reasons)
    )
    return OpportunityIntelligence(
        modifier=round(modifier, 2),
        regime=regime,
        asset_context=profile,
        reasons=tuple(reasons),
        thesis=thesis,
    )


def apply_cross_market_context(
    intelligence: OpportunityIntelligence,
    idea: TradeIdea,
    observation: MarketObservation,
    asset_class: AssetClass,
    world: dict[str, object],
    *,
    routing: AssetClassRoutingConfig,
    cap: float,
) -> OpportunityIntelligence:
    """Compare one valid setup with its peers without creating a signal.

    FX uses relative currency strength. Stocks, indices and crypto use the
    breadth of their own family. The adjustment remains ordering-only and is
    deliberately absent when the cross-section is too small or mixed.
    """
    if idea.direction is None:
        return intelligence
    sign = int(idea.direction)
    modifier = intelligence.modifier
    reasons = list(intelligence.reasons)

    if asset_class is AssetClass.FOREX:
        bare = _bare_fx(observation.symbol)
        strongest = set(world.get("strongest_currencies", []))
        weakest = set(world.get("weakest_currencies", []))
        if bare is not None:
            base, quote = bare[:3], bare[3:]
            aligned = (base in strongest and quote in weakest and sign > 0) or (
                base in weakest and quote in strongest and sign < 0
            )
            opposed = (base in weakest and quote in strongest and sign > 0) or (
                base in strongest and quote in weakest and sign < 0
            )
            if aligned:
                modifier += routing.cross_market_bonus
                reasons.append("relative currency strength confirms the FX direction")
            elif opposed:
                modifier -= routing.cross_market_penalty
                reasons.append("relative currency strength leans against the FX direction")
    elif asset_class in {
        AssetClass.STOCK,
        AssetClass.INDEX,
        AssetClass.CRYPTO,
        AssetClass.METAL,
        AssetClass.COMMODITY,
    }:
        by_asset = world.get("by_asset", {})
        counts = by_asset.get(asset_class.value, {}) if isinstance(by_asset, dict) else {}
        markets = int(counts.get("markets", 0)) if isinstance(counts, dict) else 0
        up = int(counts.get("up", 0)) if isinstance(counts, dict) else 0
        down = int(counts.get("down", 0)) if isinstance(counts, dict) else 0
        directional = up + down
        if markets >= 3 and directional >= 3:
            up_share = up / directional
            majority = routing.cross_market_majority
            aligned = (sign > 0 and up_share >= majority) or (sign < 0 and up_share <= 1 - majority)
            opposed = (sign > 0 and up_share <= 1 - majority) or (sign < 0 and up_share >= majority)
            if aligned:
                modifier += routing.cross_market_bonus
                reasons.append(f"{asset_class.value} breadth confirms the direction")
            elif opposed:
                modifier -= routing.cross_market_penalty
                reasons.append(f"{asset_class.value} breadth leans against the direction")

    modifier = max(-cap, min(cap, modifier))
    return OpportunityIntelligence(
        modifier=round(modifier, 2),
        regime=intelligence.regime,
        asset_context=intelligence.asset_context,
        reasons=tuple(reasons),
        thesis=(
            f"{idea.direction.name} {idea.symbol} from {idea.reason}; "
            f"regime={intelligence.regime}; " + "; ".join(reasons)
        ),
        scout_alignment=intelligence.scout_alignment,
        learned_alignment=intelligence.learned_alignment,
    )


def world_state(observations: list[MarketObservation]) -> dict[str, object]:
    """Aggregate one scan into a compact global dashboard/Claude briefing."""
    by_asset: dict[str, dict[str, int]] = {}
    by_regime: dict[str, int] = {}
    risk_votes: list[int] = []
    currency_strength: dict[str, float] = {}
    for row in observations:
        asset = by_asset.setdefault(row.asset_class, {"markets": 0, "up": 0, "down": 0})
        asset["markets"] += 1
        asset["up"] += 1 if row.direction_votes > 0 else 0
        asset["down"] += 1 if row.direction_votes < 0 else 0
        by_regime[row.regime] = by_regime.get(row.regime, 0) + 1
        if row.asset_class in {"stock", "index", "crypto"} and row.direction_votes:
            risk_votes.append(1 if row.direction_votes > 0 else -1)
        if row.asset_class == "forex":
            bare = _bare_fx(row.symbol)
            if bare is not None:
                base, quote = bare[:3], bare[3:]
                impulse = row.h1_move_atr + 0.5 * row.h4_move_atr
                currency_strength[base] = currency_strength.get(base, 0.0) + impulse
                currency_strength[quote] = currency_strength.get(quote, 0.0) - impulse

    breadth = sum(1 for vote in risk_votes if vote > 0) / len(risk_votes) if risk_votes else 0.5
    if len(risk_votes) >= 3 and breadth >= 0.62:
        tone = "risk_on"
    elif len(risk_votes) >= 3 and breadth <= 0.38:
        tone = "risk_off"
    else:
        tone = "mixed"
    currencies = sorted(currency_strength.items(), key=lambda item: item[1], reverse=True)
    return {
        "markets_observed": len(observations),
        "risk_tone": tone,
        "risk_breadth_up_pct": round(100.0 * breadth, 1),
        "by_asset": by_asset,
        "by_regime": by_regime,
        "strongest_currencies": [code for code, _ in currencies[:3]],
        "weakest_currencies": [code for code, _ in currencies[-3:]],
        "note": (
            "Descriptive closed-bar context only. It changes ordering and Claude context, "
            "never eligibility, size, stop, target or a hard filter."
        ),
    }


def scout_market_snapshot(
    context: MarketContext, observation: MarketObservation
) -> dict[str, object]:
    """Compact comparable evidence for one cross-market Claude scout call.

    The final reviewer still receives the deeper bar ladder. The scout's job is
    selection, so scale-free summaries are both cheaper and easier to compare
    across an FX rate near 1.10 and an index near 40,000.
    """
    frames: dict[str, object] = {}
    for timeframe in (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5):
        series = context.series.get(timeframe)
        if series is None or len(series.df) < 16:
            continue
        frame = series.df
        atr = _atr(frame)
        close = frame["close"].astype(float)
        recent = close.tail(8)
        anchor = float(recent.iloc[0])
        normalized = [round((float(value) - anchor) / atr, 3) if atr else 0.0 for value in recent]
        window = frame.tail(30)
        low = float(window["low"].min())
        high = float(window["high"].max())
        last = float(close.iloc[-1])
        frames[timeframe.value] = {
            "last_close": last,
            "atr14": round(atr, 6),
            "last_8_closes_from_anchor_atr": normalized,
            "range_position_pct": (
                round(100.0 * (last - low) / (high - low), 1) if high > low else 50.0
            ),
            "ema10_above_ema30": bool(
                close.ewm(span=10, adjust=False).mean().iloc[-1]
                > close.ewm(span=30, adjust=False).mean().iloc[-1]
            ),
            "last_closed_bar": series.last_bar_time.isoformat(),
        }
    tick = context.tick
    return {
        **observation.safe_dict(),
        "timeframes": frames,
        "spread_bps": (
            round(tick.spread / tick.mid * 10_000, 3) if tick is not None and tick.mid > 0 else None
        ),
    }


def _normalised_move(context: MarketContext, timeframe: Timeframe, lookback: int) -> float:
    series = context.series.get(timeframe)
    if series is None or len(series.df) < max(lookback + 1, 16):
        return 0.0
    frame = series.df
    atr = _atr(frame)
    if atr <= 0:
        return 0.0
    close = frame["close"].astype(float)
    return float(close.iloc[-1] - close.iloc[-(lookback + 1)]) / atr


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(ranges.rolling(period).mean().iloc[-1])
    return value if np.isfinite(value) else 0.0


def _bare_fx(symbol: str) -> str | None:
    letters = "".join(character for character in symbol.upper() if character.isalpha())
    return letters[:6] if len(letters) >= 6 else None
