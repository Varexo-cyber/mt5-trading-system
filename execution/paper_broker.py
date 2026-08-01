"""Persistent paper execution using live broker quotes but never live orders."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.broker import Broker
from core.instrument import InstrumentSpec
from core.types import (
    AccountSnapshot,
    ClosedPosition,
    Direction,
    OrderRequest,
    OrderResult,
    Position,
    SymbolDescriptor,
    Tick,
)


class PaperBroker:
    """Broker-compatible simulator whose positions survive process restarts."""

    def __init__(
        self,
        market: Broker,
        state_path: Path | str,
        *,
        starting_balance: float = 100.0,
        currency: str = "EUR",
    ) -> None:
        self.market = market
        self.state_path = Path(state_path)
        self.starting_balance = starting_balance
        self.currency = currency
        self._balance = starting_balance
        self._positions: list[Position] = []
        self._closed: dict[int, ClosedPosition] = {}
        self._next_ticket = 1_000_000
        self._connected = False
        self._load()

    @property
    def is_connected(self) -> bool:
        return self._connected and self.market.is_connected

    @property
    def server_offset(self) -> timedelta:
        return self.market.server_offset

    def connect(self) -> AccountSnapshot:
        self.market.connect()
        self._connected = True
        self.mark_to_market()
        return self.account()

    def ensure_connected(self) -> None:
        self.market.ensure_connected()
        self._connected = True

    def shutdown(self) -> None:
        self._save()
        self.market.shutdown()
        self._connected = False

    def account(self) -> AccountSnapshot:
        floating = sum(position.profit + position.swap for position in self._positions)
        equity = self._balance + floating
        live = self.market.account()
        return AccountSnapshot(
            login=0,
            server="PAPER/" + live.server,
            currency=self.currency,
            balance=self._balance,
            equity=equity,
            margin=0.0,
            margin_free=equity,
            margin_level=0.0,
            leverage=live.leverage,
            is_demo=True,
            taken_at=datetime.now(UTC),
        )

    def positions(self, symbol: str | None = None, magic: int | None = None) -> list[Position]:
        values = self._positions
        if symbol is not None:
            values = [position for position in values if position.symbol == symbol]
        if magic is not None:
            values = [position for position in values if position.magic == magic]
        return list(values)

    def symbols(self) -> list[SymbolDescriptor]:
        return self.market.symbols()

    def select(self, symbol: str) -> None:
        self.market.select(symbol)

    def spec(self, symbol: str, *, refresh: bool = False) -> InstrumentSpec:
        return self.market.spec(symbol, refresh=refresh)

    def tick(self, symbol: str) -> Tick:
        return self.market.tick(symbol)

    def copy_rates(self, symbol: str, timeframe: int, count: int, start_pos: int = 0) -> Any:
        return self.market.copy_rates(symbol, timeframe, count, start_pos)

    def copy_rates_range(self, symbol: str, timeframe: int, start: datetime, end: datetime) -> Any:
        return self.market.copy_rates_range(symbol, timeframe, start, end)

    def order_send(self, request: OrderRequest, _spec: InstrumentSpec) -> OrderResult:
        tick = self.tick(request.symbol)
        fill = tick.ask if request.direction is Direction.LONG else tick.bid
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(
            Position(
                ticket=ticket,
                symbol=request.symbol,
                direction=request.direction,
                volume=request.volume,
                price_open=fill,
                sl=request.sl,
                tp=request.tp,
                profit=0.0,
                swap=0.0,
                opened_at=datetime.now(UTC),
                magic=request.magic,
                comment=request.comment or "jarvis-paper",
            )
        )
        self._save()
        return self._result(request, fill, ticket, "PAPER_FILLED", tick.spread)

    def modify_stops(self, position: Position, *, sl: float, tp: float) -> OrderResult:
        updated = Position(**{**asdict(position), "sl": sl, "tp": tp})
        self._positions = [
            updated if item.ticket == position.ticket else item for item in self._positions
        ]
        self._save()
        return OrderResult(
            ok=True,
            retcode=0,
            retcode_name="PAPER_MODIFIED",
            comment="PAPER_MODIFIED",
            order_ticket=position.ticket,
            deal_ticket=None,
            position_ticket=position.ticket,
            requested_volume=position.volume,
            filled_volume=position.volume,
            requested_price=position.price_open,
            filled_price=position.price_open,
            slippage_pips=0.0,
            latency_ms=0.0,
            spread_at_send=0.0,
            attempts=1,
            sent_at=datetime.now(UTC),
        )

    def close_position(self, position: Position, volume: float | None = None) -> OrderResult:
        tick = self.tick(position.symbol)
        fill = tick.bid if position.direction is Direction.LONG else tick.ask
        close_volume = volume if volume is not None else position.volume
        if close_volume <= 0 or close_volume > position.volume:
            raise ValueError(f"invalid paper close volume {close_volume:g}")
        pnl = self._pnl(position, fill) * (close_volume / position.volume)
        self._balance += pnl
        remaining = position.volume - close_volume
        if remaining > 0:
            reduced = Position(
                **{
                    **asdict(position),
                    "volume": remaining,
                    "profit": position.profit - pnl,
                }
            )
            self._positions = [
                reduced if item.ticket == position.ticket else item for item in self._positions
            ]
            self._remember_closed(position, fill, pnl, "PARTIAL", volume=close_volume)
        else:
            self._positions = [item for item in self._positions if item.ticket != position.ticket]
            self._remember_closed(position, fill, pnl, "MANUAL")
        self._save()
        return OrderResult(
            ok=True,
            retcode=0,
            retcode_name="PAPER_CLOSED",
            comment="PAPER_CLOSED",
            order_ticket=position.ticket,
            deal_ticket=position.ticket,
            position_ticket=position.ticket,
            requested_volume=close_volume,
            filled_volume=close_volume,
            requested_price=fill,
            filled_price=fill,
            slippage_pips=0.0,
            latency_ms=0.0,
            spread_at_send=tick.spread,
            attempts=1,
            sent_at=datetime.now(UTC),
        )

    def mark_to_market(self) -> list[tuple[Position, str]]:
        """Update floating PnL and simulate broker-side stop/target fills."""
        events: list[tuple[Position, str]] = []
        updated: list[Position] = []
        for position in self._positions:
            try:
                tick = self.tick(position.symbol)
                exit_price = tick.bid if position.direction is Direction.LONG else tick.ask
                stop_hit = (position.direction is Direction.LONG and exit_price <= position.sl) or (
                    position.direction is Direction.SHORT and exit_price >= position.sl
                )
                target_hit = bool(position.tp) and (
                    (position.direction is Direction.LONG and exit_price >= position.tp)
                    or (position.direction is Direction.SHORT and exit_price <= position.tp)
                )
                marked = Position(**{**asdict(position), "profit": self._pnl(position, exit_price)})
                if stop_hit or target_hit:
                    self._balance += marked.profit
                    self._remember_closed(
                        marked,
                        exit_price,
                        marked.profit,
                        "SL" if stop_hit else "TP",
                    )
                    events.append((marked, "SL" if stop_hit else "TP"))
                else:
                    updated.append(marked)
            except Exception:  # noqa: BLE001 - an unavailable quote cannot erase a position
                updated.append(position)
        self._positions = updated
        self._save()
        return events

    def closed_position(self, position_ticket: int) -> ClosedPosition | None:
        return self._closed.get(position_ticket)

    def estimate_margin(
        self,
        symbol: str,
        direction: Direction,
        volume: float,
        price: float,
    ) -> float:
        return self.market.estimate_margin(symbol, direction, volume, price)

    def _remember_closed(
        self,
        position: Position,
        exit_price: float,
        pnl_money: float,
        reason: str,
        *,
        volume: float | None = None,
    ) -> None:
        closed_volume = position.volume if volume is None else volume
        previous = self._closed.get(position.ticket)
        total_volume = closed_volume + (previous.volume if previous is not None else 0.0)
        weighted_price = (
            exit_price * closed_volume
            + (previous.exit_price * previous.volume if previous is not None else 0.0)
        ) / total_volume
        self._closed[position.ticket] = ClosedPosition(
            position_ticket=position.ticket,
            symbol=position.symbol,
            closed_at=datetime.now(UTC),
            exit_price=weighted_price,
            volume=total_volume,
            pnl_money=pnl_money + (previous.pnl_money if previous is not None else 0.0),
            reason=reason,
            deal_tickets=(
                *(previous.deal_tickets if previous is not None else ()),
                position.ticket,
            ),
        )
        if len(self._closed) > 1_000:
            oldest = min(self._closed.values(), key=lambda item: item.closed_at)
            self._closed.pop(oldest.position_ticket, None)

    def _pnl(self, position: Position, exit_price: float) -> float:
        spec = self.spec(position.symbol)
        value = spec.money_per_lot(abs(exit_price - position.price_open)) * position.volume
        return (
            value if (exit_price - position.price_open) * int(position.direction) >= 0 else -value
        )

    @staticmethod
    def _result(
        request: OrderRequest, fill: float, ticket: int, comment: str, spread: float
    ) -> OrderResult:
        return OrderResult(
            ok=True,
            retcode=0,
            retcode_name=comment,
            comment=comment,
            order_ticket=ticket,
            deal_ticket=ticket,
            position_ticket=ticket,
            requested_volume=request.volume,
            filled_volume=request.volume,
            requested_price=request.reference_price,
            filled_price=fill,
            slippage_pips=0.0,
            latency_ms=0.0,
            spread_at_send=spread,
            attempts=1,
            sent_at=datetime.now(UTC),
        )

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._balance = float(payload["balance"])
        self.currency = str(payload.get("currency", self.currency))
        self._next_ticket = int(payload["next_ticket"])
        self._positions = [
            Position(
                **{
                    **row,
                    "direction": Direction[row["direction"]],
                    "opened_at": datetime.fromisoformat(row["opened_at"]),
                }
            )
            for row in payload.get("positions", [])
        ]
        self._closed = {
            int(row["position_ticket"]): ClosedPosition(
                **{
                    **row,
                    "closed_at": datetime.fromisoformat(row["closed_at"]),
                    "deal_tickets": tuple(row.get("deal_tickets", [])),
                }
            )
            for row in payload.get("closed_positions", [])
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "balance": self._balance,
            "currency": self.currency,
            "next_ticket": self._next_ticket,
            "positions": [
                {
                    **asdict(position),
                    "direction": position.direction.name,
                    "opened_at": position.opened_at.isoformat(),
                }
                for position in self._positions
            ],
            "closed_positions": [
                {
                    **asdict(position),
                    "closed_at": position.closed_at.isoformat(),
                }
                for position in self._closed.values()
            ],
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)
