"""Separate directional conviction from the price available for execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite

from config.schema import EntryQualityConfig
from core.data_manager import atr
from core.instrument import AssetClass
from core.types import Direction, MarketContext, Timeframe


class EntryTimingDecision(StrEnum):
    ENTER_NOW = "ENTER_NOW"
    WAIT_RETEST = "WAIT_RETEST"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class EntryTimingAssessment:
    """Closed-bar evidence for whether a valid idea is tradeable *now*."""

    decision: EntryTimingDecision
    reason_code: str
    detail: str
    timeframe: str
    favourable_extension_atr: float | None = None
    single_bar_body_atr: float | None = None
    ema_distance_atr: float | None = None
    directional_range_location: float | None = None
    last_bar_adverse_atr: float | None = None

    @property
    def passed(self) -> bool:
        return self.decision is EntryTimingDecision.ENTER_NOW

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewDriftAssessment:
    """Whether the executable price still matches the snapshot Claude judged."""

    decision: EntryTimingDecision
    detail: str
    drift_atr: float | None
    signed_favourable_drift_atr: float | None
    latency_seconds: float
    timeframe: str

    @property
    def passed(self) -> bool:
        return self.decision is EntryTimingDecision.ENTER_NOW

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_entry_quality(
    context: MarketContext,
    direction: Direction,
    asset_class: AssetClass,
    config: EntryQualityConfig,
) -> EntryTimingAssessment:
    """Refuse both chasing and a pullback that has not finished.

    A close near a range edge is not itself bad: every real breakout is there.
    It becomes a chase only when paired with excessive ATR-normalised drift,
    candle body or EMA distance. Refusals are `WAIT_RETEST`, not vetoes, so a
    later non-extended turn can use the unchanged directional thesis.
    """

    timeframe = Timeframe.parse(config.timeframe)
    if not config.enabled:
        return EntryTimingAssessment(
            EntryTimingDecision.ENTER_NOW,
            "DISABLED",
            "entry-quality timing is disabled",
            timeframe.value,
        )

    try:
        frame = context.bars(timeframe).df
    except (KeyError, ValueError) as exc:
        return EntryTimingAssessment(
            EntryTimingDecision.DATA_UNAVAILABLE,
            "ENTRY_TIMING_DATA_UNAVAILABLE",
            f"{timeframe.value} entry-quality data unavailable: {exc}",
            timeframe.value,
        )
    minimum = (
        max(
            config.ema_period,
            config.range_lookback_bars,
            config.extension_bars + 1,
        )
        + 1
    )
    if len(frame) < minimum:
        return EntryTimingAssessment(
            EntryTimingDecision.DATA_UNAVAILABLE,
            "ENTRY_TIMING_DATA_UNAVAILABLE",
            f"{timeframe.value} entry-quality data unavailable: need {minimum} closed bars, "
            f"received {len(frame)}",
            timeframe.value,
        )
    reference = float(atr(frame, period=14))
    if not isfinite(reference) or reference <= 0.0:
        return EntryTimingAssessment(
            EntryTimingDecision.DATA_UNAVAILABLE,
            "ENTRY_TIMING_DATA_UNAVAILABLE",
            f"{timeframe.value} entry-quality ATR is unavailable or non-positive",
            timeframe.value,
        )

    closes = frame["close"].astype(float)
    opens = frame["open"].astype(float)
    sign = int(direction)
    last = float(closes.iloc[-1])
    extension = sign * (last - float(closes.iloc[-1 - config.extension_bars])) / reference
    body = sign * (last - float(opens.iloc[-1])) / reference
    last_move = sign * (last - float(closes.iloc[-2])) / reference
    last_adverse = max(0.0, -last_move)

    recent = frame.tail(config.range_lookback_bars)
    low = float(recent["low"].min())
    high = float(recent["high"].max())
    width = high - low
    raw_location = (last - low) / width if width > 0.0 else 0.5
    directional_location = raw_location if direction is Direction.LONG else 1.0 - raw_location

    ema = float(closes.ewm(span=config.ema_period, adjust=False).mean().iloc[-1])
    ema_distance = sign * (last - ema) / reference
    asset = asset_class.value
    extension_limit = config.max_favourable_extension_atr[asset]
    body_limit = config.max_single_bar_body_atr[asset]
    ema_limit = config.max_ema_distance_atr[asset]

    metrics = {
        "favourable_extension_atr": extension,
        "single_bar_body_atr": body,
        "ema_distance_atr": ema_distance,
        "directional_range_location": directional_location,
        "last_bar_adverse_atr": last_adverse,
    }
    rounded = {key: round(value, 3) for key, value in metrics.items()}

    if last_adverse > config.max_last_bar_adverse_atr:
        return EntryTimingAssessment(
            EntryTimingDecision.WAIT_RETEST,
            "PULLBACK_STILL_ACTIVE",
            f"latest closed {timeframe.value} bar is still moving {last_adverse:.2f} ATR "
            f"against the {direction.name}; wait for the pullback to turn",
            timeframe.value,
            **rounded,
        )

    at_extreme = directional_location >= config.directional_extreme_location
    breaches: list[str] = []
    if at_extreme and extension > extension_limit:
        breaches.append(
            f"{config.extension_bars}-bar move {extension:.2f}>{extension_limit:.2f} ATR"
        )
    if at_extreme and body > body_limit:
        breaches.append(f"last body {body:.2f}>{body_limit:.2f} ATR")
    if at_extreme and ema_distance > ema_limit:
        breaches.append(f"EMA distance {ema_distance:.2f}>{ema_limit:.2f} ATR")
    if breaches:
        return EntryTimingAssessment(
            EntryTimingDecision.WAIT_RETEST,
            "DIRECTIONAL_MOVE_OVEREXTENDED",
            f"{direction.name} price is at {directional_location:.0%} of its directional "
            f"{config.range_lookback_bars}-bar range after "
            + ", ".join(breaches)
            + "; wait for a retest",
            timeframe.value,
            **rounded,
        )

    return EntryTimingAssessment(
        EntryTimingDecision.ENTER_NOW,
        "ENTRY_PRICE_NOT_EXTENDED",
        f"{timeframe.value} entry is not extended and the latest closed bar is not "
        f"materially opposing the {direction.name}",
        timeframe.value,
        **rounded,
    )


def assess_review_drift(
    context: MarketContext,
    direction: Direction,
    reviewed_entry: float,
    executable_entry: float,
    latency_seconds: float,
    config: EntryQualityConfig,
) -> ReviewDriftAssessment:
    """Bind an AI approval to the price shape it actually reviewed."""

    timeframe = Timeframe.parse(config.timeframe)
    if latency_seconds > config.max_review_latency_seconds:
        return ReviewDriftAssessment(
            EntryTimingDecision.WAIT_RETEST,
            f"AI response took {latency_seconds:.1f}s, above the "
            f"{config.max_review_latency_seconds:.1f}s review-age limit",
            None,
            None,
            latency_seconds,
            timeframe.value,
        )
    try:
        reference = float(atr(context.bars(timeframe).df, period=14))
    except (KeyError, ValueError) as exc:
        return ReviewDriftAssessment(
            EntryTimingDecision.DATA_UNAVAILABLE,
            f"cannot bind AI review to fresh {timeframe.value} price: {exc}",
            None,
            None,
            latency_seconds,
            timeframe.value,
        )
    if not isfinite(reference) or reference <= 0.0:
        return ReviewDriftAssessment(
            EntryTimingDecision.DATA_UNAVAILABLE,
            f"cannot bind AI review to fresh {timeframe.value} price: ATR unavailable",
            None,
            None,
            latency_seconds,
            timeframe.value,
        )

    signed = (executable_entry - reviewed_entry) * int(direction) / reference
    drift = abs(signed)
    # The two halves are not the same event and no longer share a limit. Price
    # running ON in the proposed direction means paying more than the price
    # that was approved — chasing, and held to the tight number. Price coming
    # BACK means a better fill and a shorter stop distance; what it might also
    # mean is a level that is failing, and that question belongs to the two
    # adverse-travel gates that were built for it and sit at 1.00 ATR. This one
    # was answering it first, at a quarter of that, on an absolute value.
    chasing = signed > 0
    limit = config.max_review_price_drift_atr if chasing else config.max_review_pullback_drift_atr
    if drift > limit:
        side = "with" if chasing else "against"
        return ReviewDriftAssessment(
            EntryTimingDecision.WAIT_RETEST,
            f"price moved {drift:.2f} ATR {side} the {direction.name} during review, above "
            f"the {limit:.2f} ATR binding limit; analyse the "
            "new price next cycle instead of chasing it",
            round(drift, 3),
            round(signed, 3),
            latency_seconds,
            timeframe.value,
        )
    return ReviewDriftAssessment(
        EntryTimingDecision.ENTER_NOW,
        f"fresh executable price remains within {drift:.2f} ATR of the reviewed entry",
        round(drift, 3),
        round(signed, 3),
        latency_seconds,
        timeframe.value,
    )
