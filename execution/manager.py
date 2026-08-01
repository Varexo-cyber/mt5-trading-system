"""Open-position management and fail-closed restart reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from config.schema import Settings
from core.broker import Broker
from core.types import Direction, Position, Timeframe
from journal.database import Journal


@dataclass(frozen=True, slots=True)
class ManagementEvent:
    ticket: int
    action: str
    detail: str
    exit_price: float | None = None
    pnl_money: float | None = None


class PositionManager:
    def __init__(self, broker: Broker, journal: Journal, settings: Settings) -> None:
        self.broker = broker
        self.journal = journal
        self.settings = settings

    def reconcile(self, positions: list[Position]) -> list[ManagementEvent]:
        events: list[ManagementEvent] = []
        broker_tickets = {position.ticket for position in positions}
        for position in positions:
            if not position.has_stop:
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "EMERGENCY_CLOSE",
                        "position had no stop",
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            if self.journal.open_trade_by_ticket(position.ticket) is None:
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "ORPHAN_CLOSE",
                        "strategy-magic position was absent from journal",
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
        for row in self.journal.open_trades():
            ticket = int(row["ticket"] or 0)
            if ticket and ticket not in broker_tickets:
                events.append(
                    ManagementEvent(
                        ticket,
                        "BROKER_CLOSED_PENDING_HISTORY",
                        "journal trade absent at broker; block new risk until "
                        "closure is reconciled",
                    )
                )
        return events

    def manage(self, positions: list[Position], now: datetime) -> list[ManagementEvent]:
        events: list[ManagementEvent] = []
        config = self.settings.trade_management
        for position in positions:
            row = self.journal.open_trade_by_ticket(position.ticket)
            if row is None:
                continue
            original_sl = float(row["sl"])
            risk = abs(position.price_open - original_sl)
            if risk <= 0:
                continue
            tick = self.broker.tick(position.symbol)
            price = tick.bid if position.direction is Direction.LONG else tick.ask
            r_now = (price - position.price_open) * int(position.direction) / risk
            age_hours = (now - position.opened_at).total_seconds() / 3600.0
            if (
                config.time_exit_hours is not None
                and age_hours >= config.time_exit_hours
                and abs(r_now) < config.time_exit_min_abs_r
            ):
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "TIME_EXIT",
                        f"{age_hours:.1f}h, {r_now:.2f}R",
                        result.filled_price,
                        position.profit + position.swap,
                    )
                )
                continue
            atr = self._atr(position.symbol)
            break_even = position.price_open + atr * config.break_even_offset_atr * int(
                position.direction
            )
            stop_behind_break_even = (
                position.direction is Direction.LONG and position.sl < break_even
            ) or (position.direction is Direction.SHORT and position.sl > break_even)
            if r_now >= config.break_even_at_r and stop_behind_break_even:
                result = self.broker.modify_stops(
                    position,
                    sl=self.broker.spec(position.symbol).normalize_price(break_even),
                    tp=position.tp,
                )
                if result.ok:
                    events.append(
                        ManagementEvent(
                            position.ticket, "BREAK_EVEN", f"stop protected at {r_now:.2f}R"
                        )
                    )
                    continue
            if config.trailing_mode == "atr" and r_now >= config.partial_close_at_r:
                trailing = price - atr * config.trailing_atr_multiple * int(position.direction)
                improves = (position.direction is Direction.LONG and trailing > position.sl) or (
                    position.direction is Direction.SHORT and trailing < position.sl
                )
                if improves:
                    result = self.broker.modify_stops(
                        position,
                        sl=self.broker.spec(position.symbol).normalize_price(trailing),
                        tp=position.tp,
                    )
                    if result.ok:
                        events.append(
                            ManagementEvent(
                                position.ticket, "ATR_TRAIL", f"stop trailed at {r_now:.2f}R"
                            )
                        )
        return events

    def _atr(self, symbol: str, period: int = 14) -> float:
        raw = self.broker.copy_rates(symbol, Timeframe.H1.mt5_value, period + 2)
        frame = pd.DataFrame(raw)
        previous = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.tail(period).mean())
