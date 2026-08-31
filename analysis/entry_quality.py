"""Separate directional conviction from the price available for execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite

from config.schema import EntryQualityConfig
from core.data_manager import atr
from core.instrument import AssetClass
from core.types import Direction, MarketContext, Signal, Timeframe


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
    last_bar_directional_atr: float | None = None
    executable_gap_atr: float | None = None
    reference_atr: float | None = None

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
    *,
    executable_price: float | None = None,
    signals: tuple[Signal, ...] = (),
    adverse_bar_is_setup_mechanism: bool = False,
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
    closed_last = float(closes.iloc[-1])
    # Closed bars decide whether a setup exists. The live quote decides
    # whether that setup is still offered at a sensible price. Keeping those
    # jobs separate is important: using a forming candle as signal evidence is
    # look-ahead, while ignoring its price at execution is how ETHUSD was
    # bought at the top of a live spike that the last closed M5 bar could not
    # yet contain.
    last = (
        float(executable_price)
        if executable_price is not None and isfinite(float(executable_price))
        else closed_last
    )
    executable_gap = sign * (last - closed_last) / reference
    extension = sign * (last - float(closes.iloc[-1 - config.extension_bars])) / reference
    # Candle shape remains closed-bar evidence. Combining a live quote with the
    # closed candle's open would manufacture a body that never existed.
    body = sign * (closed_last - float(opens.iloc[-1])) / reference
    last_move = sign * (closed_last - float(closes.iloc[-2])) / reference
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
    live_gap_limit = config.max_live_favourable_gap_atr[asset]

    metrics = {
        "favourable_extension_atr": extension,
        "single_bar_body_atr": body,
        "ema_distance_atr": ema_distance,
        "directional_range_location": directional_location,
        "last_bar_adverse_atr": last_adverse,
        "last_bar_directional_atr": last_move,
        "executable_gap_atr": executable_gap,
        "reference_atr": reference,
    }
    rounded = {key: round(value, 3) for key, value in metrics.items()}

    if last_adverse > config.max_last_bar_adverse_atr and not adverse_bar_is_setup_mechanism:
        return EntryTimingAssessment(
            EntryTimingDecision.WAIT_RETEST,
            "PULLBACK_STILL_ACTIVE",
            f"latest closed {timeframe.value} bar is still moving {last_adverse:.2f} ATR "
            f"against the {direction.name}; wait for the pullback to turn",
            timeframe.value,
            **rounded,
        )

    # OVER-EXTENSION IS A REFUSAL WHEREVER IT SITS IN THE RANGE.
    #
    # These four tests were each written `at_extreme and <limit breached>`, so
    # the whole over-extension check only existed for a price already at 80% of
    # its twelve-bar range. That left two openings, and both are the same trade:
    #
    #   at 79% of the range, NO limit applied at all -- a market could have run
    #   five ATR from its EMA and the gate had nothing to say;
    #
    #   at 100% of the range on a calm bar, every limit was clear -- which is
    #   buying the highest price of the last hour and calling it timed.
    #
    # WHAT SAYS THIS IS WRONG rather than merely arguable. Over 1,970 closed
    # trades, 197 ever peaked above +1.00R. That is 10%. A coin flip entering
    # at random with the same 1R stop reaches +1R before -1R about half the
    # time. Ten against fifty is not a weak edge -- it is an entry taken
    # systematically at the point where the move has already happened, and the
    # give-back table's whole shape follows from it: 80% of trades reach 0.30R
    # and almost none reach 1.00R.
    #
    # So the limits apply always. They are not tight -- 2.75 ATR of travel in
    # three bars, 2.25 ATR from the EMA -- and a market past them has made its
    # move whether or not it happens to sit at the top of a twelve-bar window.
    #
    # `at_extreme` keeps exactly one job, which is the one it is good at:
    # TIGHTENING. `max_extreme_single_bar_body_atr` is less than half
    # `max_single_bar_body_atr`, because a one-bar thrust into the range edge
    # is a different event from the same bar in the middle of a range.
    at_extreme = directional_location >= config.directional_extreme_location
    breaches: list[str] = []
    if executable_gap > live_gap_limit:
        breaches.append(
            f"live quote ran {executable_gap:.2f}>{live_gap_limit:.2f} ATR beyond the "
            f"latest closed {timeframe.value} bar"
        )
    if extension > extension_limit:
        breaches.append(
            f"{config.extension_bars}-bar move {extension:.2f}>{extension_limit:.2f} ATR"
        )
    if body > body_limit:
        breaches.append(f"last body {body:.2f}>{body_limit:.2f} ATR")
    extreme_body_limit = config.max_extreme_single_bar_body_atr[asset]
    if at_extreme and body > extreme_body_limit:
        breaches.append(f"one-bar thrust at the range edge {body:.2f}>{extreme_body_limit:.2f} ATR")
    if ema_distance > ema_limit:
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

    stale_impulses = [
        signal
        for signal in signals
        if signal.module == "impulse_break"
        and signal.score * sign > 0.0
        and int(signal.details.get("bars_since_impulse", 0)) > config.stale_impulse_after_bars
    ]
    if stale_impulses and last_move < config.stale_impulse_min_resumption_atr:
        oldest = max(int(signal.details.get("bars_since_impulse", 0)) for signal in stale_impulses)
        return EntryTimingAssessment(
            EntryTimingDecision.WAIT_RETEST,
            "STALE_IMPULSE_AWAITING_RESUMPTION",
            f"the supporting impulse is {oldest} bars old and the latest closed "
            f"{timeframe.value} bar resumed only {last_move:.2f} ATR; wait for at least "
            f"{config.stale_impulse_min_resumption_atr:.2f} ATR of fresh confirmation",
            timeframe.value,
            **rounded,
        )

    return EntryTimingAssessment(
        EntryTimingDecision.ENTER_NOW,
        "ENTRY_PRICE_NOT_EXTENDED",
        f"executable {timeframe.value} entry is not extended and the latest closed bar is not "
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
