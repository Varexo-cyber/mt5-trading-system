"""Keep a valid directional thesis alive until its entry is actually timely.

The confluence engine answers *which way*. Entry quality answers whether the
current quote is late or whether a pullback is still moving. This module joins
those observations across scan cycles: an overextended idea is not discarded,
and a falling pullback is not mistaken for the retest having finished.
"""

from __future__ import annotations

import json
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
    anchor_price: float
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
    ) -> None:
        self.path = path
        self.pullback_atr = pullback_atr
        self.resumption_atr = resumption_atr
        self.expiry_minutes = expiry_minutes
        self._tracked: dict[str, _TrackedSetup] = {}
        self._events: list[dict[str, object]] = []
        self._load()

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
