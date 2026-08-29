"""Keep a valid directional thesis alive until its entry is actually timely.

The confluence engine answers *which way*. Entry quality answers whether the
current quote is late or whether a pullback is still moving. This module joins
those observations across scan cycles: an overextended idea is not discarded,
and a falling pullback is not mistaken for the retest having finished.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from analysis.confluence import TradeIdea
from analysis.entry_quality import EntryTimingAssessment, EntryTimingDecision
from infra.atomic import write_json_atomic


class SetupState(StrEnum):
    DETECTED = "DETECTED"
    WAIT_PULLBACK = "WAIT_PULLBACK"
    PULLBACK_RECEIVED = "PULLBACK_RECEIVED"
    WAIT_RESUMPTION = "WAIT_RESUMPTION"
    ENTER_NOW = "ENTER_NOW"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    state: SetupState
    reason: str
    tracked: bool
    age_minutes: float = 0.0
    pullback_atr: float | None = None
    resumption_atr: float | None = None

    @property
    def may_enter(self) -> bool:
        return self.state is SetupState.ENTER_NOW

    def safe_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


@dataclass(slots=True)
class _TrackedSetup:
    symbol: str
    direction: str
    setup_family: str
    horizon: str
    state: SetupState
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    #: THE LEVEL THE BREAK CLEARED, and it never moves.
    #:
    #: Taken from the detector's own `key_levels` where it publishes one (see
    #: `_level_for`), falling back to the price the setup was first seen at.
    #: For a breakout that is the resistance that just broke and is supposed
    #: to become support. `anchor_price` below follows
    #: the extreme instead, and for two months this class had only that one:
    #: the "retest" waited for a pullback of 0.4 ATR from wherever the high
    #: happened to be, which on a move that ran three ATR is an entry two and a
    #: half ATR above the level anyone would call the retest.
    #:
    #: The scorecard priced that: bought at 80-95% of its own range, 14 trades,
    #: -1.65R; at 95-100%, 8 trades, -0.99R; at 60-80%, 21 trades, +0.66R --
    #: the only positive row on the table.
    level_price: float = 0.0
    anchor_price: float = 0.0
    retest_price: float | None = None
    retest_bar_time: datetime | None = None
    observations: int = 1
    reason: str = ""

    @property
    def key(self) -> str:
        return "|".join((self.symbol, self.direction, self.setup_family, self.horizon))

    def safe_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "setup_family": self.setup_family,
            "horizon": self.horizon,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "level_price": self.level_price,
            "anchor_price": self.anchor_price,
            "retest_price": self.retest_price,
            "retest_bar_time": (
                self.retest_bar_time.isoformat() if self.retest_bar_time is not None else None
            ),
            "observations": self.observations,
            "reason": self.reason,
        }


class SetupLifecycleBook:
    """Durable across cycles and process restarts, bounded by setup expiry."""

    def __init__(
        self,
        path: Path,
        *,
        pullback_atr: dict[str, float],
        resumption_atr: dict[str, float],
        expiry_minutes: dict[str, float],
        level_retest_modules: tuple[str, ...] = (),
        retest_level_atr: float = 0.35,
    ) -> None:
        self.path = path
        self.pullback_atr = pullback_atr
        self.resumption_atr = resumption_atr
        self.expiry_minutes = expiry_minutes
        self.level_retest_modules = tuple(level_retest_modules)
        self.retest_level_atr = retest_level_atr
        self._tracked: dict[str, _TrackedSetup] = {}
        self._events: list[dict[str, object]] = []
        self._load()

    def _retests_a_level(self, setup_family: str) -> bool:
        """Does this family have a broken level to come back to?

        A breakout does: the resistance it just cleared, which is supposed to
        become support. A trend-continuation reader does not -- there is no
        level, only a move -- and for those the pullback-from-extreme is the
        right measure and is left exactly as it was.

        Matched on substring because `setup_family` is built as
        `{module}_{timeframe}` by `_classify_horizon`, and hard-coding the
        assembled strings would silently stop matching the first time a
        timeframe changes.
        """
        return any(module in setup_family for module in self.level_retest_modules)

    def _must_still_retest(
        self,
        track: _TrackedSetup,
        idea: TradeIdea,
        timing: EntryTimingAssessment,
        executable_price: float,
    ) -> str | None:
        """Why this timely-looking break may not be entered yet, or None.

        Only ever holds a setup that is ALREADY WAITING because timing refused
        it as extended. A break that is timely the first time it is seen is
        entered as before -- that is an entry at the level, not above it, and
        it is the one bucket on the scorecard that made money (60-80% of range,
        +0.66R over 21 trades).

        Unmeasurable is held, not waved through. The house rule on this account
        is that missing data is a reason not to trade, and without an ATR there
        is no statement to make about how far above the level this is.
        """
        if track.state is not SetupState.WAIT_PULLBACK:
            return None
        if not self._retests_a_level(track.setup_family):
            return None
        if idea.direction is None:
            return None
        sign = 1 if idea.direction.name == "LONG" else -1
        reference = timing.reference_atr or 0.0
        if reference <= 0.0:
            return "entry ATR unavailable, so the distance to the broken level is unmeasured"
        above = sign * (executable_price - track.level_price) / reference
        if above <= self.retest_level_atr:
            return None
        return (
            f"timing is clear but this is still {above:.2f} ATR above the level it broke; "
            f"the retest wants {self.retest_level_atr:.2f}"
        )

    def _level_for(self, idea: TradeIdea, price: float, sign: int) -> float:
        """The price the break actually cleared, from the detector that saw it.

        THE DETECTION PRICE IS NOT THE LEVEL, and using it would leave half the
        defect in place. A setup only enters this book when entry timing has
        already REFUSED it as extended -- that is what "wait for a retest"
        means -- so by construction the price recorded here is a break that has
        already run. Freezing that price stops the level drifting with the high
        but still measures the retest from somewhere above the level.

        The detectors publish the real one. `market_structure` puts the broken
        swing level first in `key_levels`; `session_breakout`,
        `volatility_squeeze` and `range_break` publish the two edges of the
        range they broke.

        Which of those edges is the level is decided by the trade, not by the
        order they were listed: a long broke through something BELOW where it
        now trades, so of the levels on that side the nearest one is the edge
        it cleared. A short is the mirror. Anything on the wrong side is a
        level the price has not broken and cannot retest.

        With no usable level the detection price stands. That is the old
        behaviour minus the drift, which is worse than a real level and better
        than a moving one.
        """
        best: float | None = None
        for signal in idea.signals:
            if not self._retests_a_level(signal.module):
                continue
            for level in signal.key_levels:
                if not math.isfinite(level) or level <= 0.0:
                    continue
                if sign * (price - level) < 0.0:
                    continue  # wrong side: not broken, so not a retest
                if best is None or sign * (level - best) > 0.0:
                    best = level
        return best if best is not None else price

    @staticmethod
    def key_for(idea: TradeIdea) -> str:
        direction = idea.direction.name if idea.direction is not None else "NONE"
        return "|".join((idea.symbol, direction, idea.setup_family, idea.horizon))

    def observe(
        self,
        idea: TradeIdea,
        timing: EntryTimingAssessment,
        *,
        executable_price: float,
        now: datetime,
        bar_time: datetime | None = None,
    ) -> LifecycleDecision:
        """Advance one setup using only data available at ``now``."""

        if idea.direction is None:
            return LifecycleDecision(SetupState.EXPIRED, "direction missing", False)
        key = self.key_for(idea)
        track = self._tracked.get(key)
        if track is not None and now >= track.expires_at:
            age = (now - track.created_at).total_seconds() / 60.0
            self._transition(track, SetupState.EXPIRED, "setup horizon expired", now)
            self._tracked.pop(key, None)
            self._save()
            return LifecycleDecision(SetupState.EXPIRED, "setup horizon expired", True, age)

        if timing.decision is EntryTimingDecision.DATA_UNAVAILABLE:
            return LifecycleDecision(
                track.state if track is not None else SetupState.DETECTED,
                timing.detail,
                track is not None,
                self._age(track, now),
            )

        if track is None:
            if timing.passed:
                self._record_event(idea, SetupState.ENTER_NOW, timing.detail, now)
                self._save()
                return LifecycleDecision(SetupState.ENTER_NOW, timing.detail, False)
            initial = (
                SetupState.WAIT_RESUMPTION
                if timing.reason_code == "PULLBACK_STILL_ACTIVE"
                else SetupState.WAIT_PULLBACK
            )
            track = self._new_track(
                idea, initial, executable_price, timing.detail, now, bar_time=bar_time
            )
            self._tracked[key] = track
            self._record_track(track)
            self._save()
            return LifecycleDecision(initial, timing.detail, True)

        track.observations += 1
        track.updated_at = now

        # The lifecycle is memory, not a second entry-quality gate. Once the
        # native timing assessment says the executable price is no longer
        # extended and the closed entry bar is no longer adverse, the reason
        # for waiting has cleared. Requiring another fixed ATR pullback and a
        # separate resumption threshold asked the recovered setup to prove the
        # same thing twice and discarded otherwise timely entries.
        #
        # WITH ONE EXCEPTION, AND WITHOUT IT THIS WHOLE FILE IS DECORATION.
        #
        # This short circuit runs before every state below it, so for a tracked
        # setup the answer was ALWAYS "enter exactly when `entry_quality` says
        # the price is not extended" -- WAIT_PULLBACK, PULLBACK_RECEIVED and
        # WAIT_RESUMPTION could not admit an entry that timing refused, and
        # could not refuse one that timing admitted. The retest measured
        # nothing. It filled the journal with states, and the states never
        # touched a decision. Fixing where the pullback was measured FROM,
        # without this, would have changed the wording of a log line and
        # nothing else.
        #
        # `_must_still_retest` is the one case where the lifecycle overrules
        # timing: a break that was refused as extended, and has since eased
        # off enough to look timely, but has NOT come back to the level it
        # broke. Timing reads distance-from-extreme, so it clears there --
        # which is precisely the 80-95%-of-range entry that lost 1.65R a trade.
        held = self._must_still_retest(track, idea, timing, executable_price)
        if timing.passed and held is None:
            previous = track.state
            reason = (
                f"{previous.value} cleared: current {timing.timeframe} entry timing "
                "is executable again"
            )
            self._transition(track, SetupState.ENTER_NOW, reason, now)
            self._tracked.pop(key, None)
            self._save()
            return LifecycleDecision(
                SetupState.ENTER_NOW,
                reason,
                True,
                self._age(track, now),
                resumption_atr=(
                    round(timing.last_bar_directional_atr, 3)
                    if timing.last_bar_directional_atr is not None
                    else None
                ),
            )
        if timing.passed and held is not None:
            self._save()
            return LifecycleDecision(SetupState.WAIT_PULLBACK, held, True, self._age(track, now))

        sign = 1 if idea.direction.name == "LONG" else -1
        reference = timing.reference_atr or 0.0
        if reference <= 0.0:
            self._save()
            return LifecycleDecision(
                track.state, "entry ATR unavailable", True, self._age(track, now)
            )

        if track.state is SetupState.WAIT_PULLBACK:
            # Follow a spike to its final extreme. The retest is measured from
            # the best quote actually seen, not from an earlier arbitrary scan.
            if sign * (executable_price - track.anchor_price) > 0.0:
                track.anchor_price = executable_price

            # A BREAKOUT RETESTS ITS LEVEL. EVERYTHING ELSE RETESTS ITS EXTREME.
            #
            # For two months this class had one rule for both: wait for a
            # pullback of `pullback_atr` from the running high. On a break that
            # ran three ATR that is an entry two and a half ATR above the level
            # anyone would call the retest -- the trade the textbook describes
            # enters AT the broken resistance with the stop just under it, and
            # this entered near the top with the stop an ATR and a half below.
            #
            # Those are different trades and the scorecard says which one this
            # was: bought at 80-95% of its own range, -1.65R over 14 trades; at
            # 95-100%, -0.99R over 8; at 60-80%, +0.66R over 21 and the only
            # positive row on the table. It bought extended, and buying
            # extended is what a retest without a level produces.
            #
            # A trend-continuation reader has no broken level to come back to,
            # so for those the pullback-from-extreme IS the right measure and
            # is left exactly as it was.
            if self._retests_a_level(track.setup_family):
                # How far above the level we still are, in ATR. Zero means
                # price has come all the way back to it.
                above = sign * (executable_price - track.level_price) / reference
                needed_level = self.retest_level_atr
                if above <= needed_level:
                    track.retest_price = executable_price
                    track.retest_bar_time = bar_time
                    self._transition(
                        track,
                        SetupState.PULLBACK_RECEIVED,
                        f"price returned to within {above:.2f} ATR of the level it broke "
                        f"(needs {needed_level:.2f})",
                        now,
                    )
                    self._save()
                    return LifecycleDecision(
                        SetupState.PULLBACK_RECEIVED,
                        track.reason,
                        True,
                        self._age(track, now),
                        round(above, 3),
                    )
                self._save()
                return LifecycleDecision(
                    SetupState.WAIT_PULLBACK,
                    f"{above:.2f} ATR above the level it broke; the retest wants "
                    f"{needed_level:.2f}",
                    True,
                    self._age(track, now),
                    round(above, 3),
                )

            pullback = sign * (track.anchor_price - executable_price) / reference
            needed = self.pullback_atr[idea.horizon]
            if pullback >= needed:
                track.retest_price = executable_price
                track.retest_bar_time = bar_time
                self._transition(
                    track,
                    SetupState.PULLBACK_RECEIVED,
                    f"pullback reached {pullback:.2f} ATR (need {needed:.2f})",
                    now,
                )
                self._save()
                return LifecycleDecision(
                    SetupState.PULLBACK_RECEIVED,
                    track.reason,
                    True,
                    self._age(track, now),
                    round(pullback, 3),
                )
            self._save()
            return LifecycleDecision(
                SetupState.WAIT_PULLBACK,
                f"pullback is {pullback:.2f} ATR; waiting for {needed:.2f}",
                True,
                self._age(track, now),
                round(pullback, 3),
            )

        if track.state is SetupState.PULLBACK_RECEIVED:
            self._transition(track, SetupState.WAIT_RESUMPTION, "retest received", now)
            self._save()
            return LifecycleDecision(
                SetupState.WAIT_RESUMPTION,
                "retest received; waiting for a newly closed directional bar",
                True,
                self._age(track, now),
            )

        if track.state is SetupState.WAIT_RESUMPTION:
            if timing.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED":
                track.anchor_price = executable_price
                track.retest_price = None
                self._transition(track, SetupState.WAIT_PULLBACK, timing.detail, now)
                self._save()
                return LifecycleDecision(
                    SetupState.WAIT_PULLBACK, timing.detail, True, self._age(track, now)
                )
            resumed = timing.last_bar_directional_atr or 0.0
            needed = self.resumption_atr[idea.horizon]
            fresh_bar = track.retest_bar_time is None or (
                bar_time is not None and bar_time > track.retest_bar_time
            )
            if timing.passed and resumed >= needed and fresh_bar:
                reason = (
                    f"retest completed and latest closed {timing.timeframe} bar resumed "
                    f"{resumed:.2f} ATR (need {needed:.2f})"
                )
                self._transition(track, SetupState.ENTER_NOW, reason, now)
                self._tracked.pop(key, None)
                self._save()
                return LifecycleDecision(
                    SetupState.ENTER_NOW,
                    reason,
                    True,
                    self._age(track, now),
                    resumption_atr=round(resumed, 3),
                )
            if timing.passed and resumed >= needed and not fresh_bar:
                reason = "retest received; waiting for a newly closed directional bar"
            else:
                reason = (
                    timing.detail
                    if not timing.passed
                    else f"closed-bar resumption is {resumed:.2f} ATR; waiting for {needed:.2f}"
                )
            self._save()
            return LifecycleDecision(
                SetupState.WAIT_RESUMPTION,
                reason,
                True,
                self._age(track, now),
                resumption_atr=round(resumed, 3),
            )

        return LifecycleDecision(track.state, track.reason, True, self._age(track, now))

    def snapshot(self) -> dict[str, object]:
        return {
            "tracked": [row.safe_dict() for row in self._tracked.values()],
            "recent_events": self._events[-200:],
        }

    def _new_track(
        self,
        idea: TradeIdea,
        state: SetupState,
        price: float,
        reason: str,
        now: datetime,
        *,
        bar_time: datetime | None = None,
    ) -> _TrackedSetup:
        return _TrackedSetup(
            symbol=idea.symbol,
            direction=idea.direction.name if idea.direction else "NONE",
            setup_family=idea.setup_family,
            horizon=idea.horizon,
            state=state,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(minutes=self.expiry_minutes[idea.horizon]),
            level_price=self._level_for(
                idea, price, 1 if idea.direction and idea.direction.name == "LONG" else -1
            ),
            anchor_price=price,
            retest_bar_time=(bar_time if state is SetupState.WAIT_RESUMPTION else None),
            reason=reason,
        )

    def _transition(
        self, track: _TrackedSetup, state: SetupState, reason: str, now: datetime
    ) -> None:
        track.state = state
        track.reason = reason
        track.updated_at = now
        self._record_track(track)

    def _record_track(self, track: _TrackedSetup) -> None:
        self._events.append(
            {
                "timestamp": track.updated_at.isoformat(),
                "key": track.key,
                "state": track.state.value,
                "reason": track.reason,
            }
        )
        self._events = self._events[-200:]

    def _record_event(self, idea: TradeIdea, state: SetupState, reason: str, now: datetime) -> None:
        self._events.append(
            {
                "timestamp": now.isoformat(),
                "key": self.key_for(idea),
                "state": state.value,
                "reason": reason,
            }
        )
        self._events = self._events[-200:]

    @staticmethod
    def _age(track: _TrackedSetup | None, now: datetime) -> float:
        return 0.0 if track is None else max(0.0, (now - track.created_at).total_seconds() / 60.0)

    def _save(self) -> None:
        write_json_atomic(self.path, self.snapshot())

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._events = list(payload.get("recent_events", []))[-200:]
            for item in payload.get("tracked", []):
                track = _TrackedSetup(
                    symbol=str(item["symbol"]),
                    direction=str(item["direction"]),
                    setup_family=str(item["setup_family"]),
                    horizon=str(item["horizon"]),
                    state=SetupState(str(item["state"])),
                    created_at=datetime.fromisoformat(str(item["created_at"])),
                    updated_at=datetime.fromisoformat(str(item["updated_at"])),
                    expires_at=datetime.fromisoformat(str(item["expires_at"])),
                    # Older files predate the level, and the setup they describe
                    # was tracked without one. Falling back to the anchor keeps
                    # them loadable and is the honest value: it is what that
                    # setup was actually being measured against.
                    level_price=float(item.get("level_price", item["anchor_price"])),
                    anchor_price=float(item["anchor_price"]),
                    retest_price=(
                        float(item["retest_price"])
                        if item.get("retest_price") is not None
                        else None
                    ),
                    retest_bar_time=(
                        datetime.fromisoformat(str(item["retest_bar_time"]))
                        if item.get("retest_bar_time") is not None
                        else None
                    ),
                    observations=int(item.get("observations", 1)),
                    reason=str(item.get("reason", "")),
                )
                self._tracked[track.key] = track
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._tracked = {}
            self._events = []
