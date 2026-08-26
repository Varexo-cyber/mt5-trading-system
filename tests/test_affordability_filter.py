"""A market this account can never hold should not be analysed every cycle.

WHAT THIS COST, and it was visible on the deck rather than hidden. The AI
exchange panel was a wall of INSUFFICIENT_MARGIN on single-name stocks:

    ADS   0.23 lots needs 718.01 EUR margin
    NXTL  0.20 lots needs 727.66 EUR margin
    ZAL   0.32 lots needs 148.80 EUR margin

against an account holding 176 EUR. Those are not near misses that a smaller
stop would fix. The smallest position the contract allows is already out of
reach, so no price, no signal and no adviser opinion could ever turn one into
a trade.

Every one of them was still costing a full multi-timeframe analysis on every
cycle, on a one-vCPU box, which is time the position guard and the two
sections that CAN trade were not getting.

The refusal is deliberately narrow: the MINIMUM lot, not the sized one. A
sized position being too large is a fact about today's stop and the sizer
already refuses it per trade. This asks whether the account can hold any
position in this instrument at all, and only removes the ones where the
answer is no on every possible setup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from core.clock import SimulatedClock
from core.instrument import AssetClass
from scanner.universe import UniverseScanner

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


class _Broker:
    """Enough broker to reach the affordability gate and no further."""

    def __init__(self, *, equity: float, margin: float | None, volume_min: float = 0.01) -> None:
        self.equity = equity
        self.margin = margin
        self.volume_min = volume_min
        self.rate_calls = 0

    def account(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(equity=self.equity)

    def spec(self, symbol: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            symbol=symbol,
            is_tradable=True,
            trade_mode=4,
            asset_class=AssetClass.STOCK,
            volume_min=self.volume_min,
        )

    def tick(self, symbol: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(bid=99.99, ask=100.01, mid=100.0, spread=0.02, time=NOW)

    def estimate_margin(self, symbol, direction, volume, price):  # type: ignore[no-untyped-def]
        if self.margin is None:
            raise RuntimeError("the broker cannot price this contract")
        return self.margin

    def copy_rates(self, symbol, timeframe, count):  # type: ignore[no-untyped-def]
        """Reaching here means the affordability gate let the symbol through.

        Counted rather than merely returned, because "was this symbol analysed"
        is the whole question and the H1 fetch is where the cost starts.
        """
        self.rate_calls += 1
        base = pd.Timestamp("2026-08-20", tz=UTC)
        return [
            {
                "time": int((base + pd.Timedelta(hours=index)).timestamp()),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "tick_volume": 100,
                "spread": 2,
                "real_volume": 0,
            }
            for index in range(90)
        ]


def _scanner(broker: _Broker) -> UniverseScanner:
    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    # A simulated clock pinned to the fixture's quote time. With the real one
    # the tick ages as the test suite runs and the symbol is refused for a
    # stale quote before it ever reaches the gate under test.
    return UniverseScanner(broker, settings, SimulatedClock(NOW))  # type: ignore[arg-type]


def _verdict(broker: _Broker):  # type: ignore[no-untyped-def]
    scanner = _scanner(broker)
    spec = broker.spec("ADS")
    return scanner._unaffordable("ADS", spec, broker.tick("ADS"))


def test_a_stock_whose_minimum_lot_costs_four_accounts_is_refused() -> None:
    """The live case: 718 EUR of margin against 176 EUR of equity."""
    verdict = _verdict(_Broker(equity=176.0, margin=718.01))

    assert verdict is not None
    assert "718.01" in verdict and "176.00" in verdict


def test_an_affordable_market_is_left_alone() -> None:
    """The gate must not be a way of refusing the whole catalogue."""
    assert _verdict(_Broker(equity=176.0, margin=8.0)) is None


def test_the_buffer_is_the_one_the_margin_check_already_uses() -> None:
    """Refused on the buffered figure, not the raw one, so this gate and the
    per-trade margin check cannot disagree about the same instrument."""
    scanner = _scanner(_Broker(equity=176.0, margin=1.0))
    factor = scanner.settings.risk.margin_safety_factor
    # Just inside the buffer, and just outside it.
    assert _verdict(_Broker(equity=176.0, margin=176.0 / factor - 1)) is None
    assert _verdict(_Broker(equity=176.0, margin=176.0 / factor + 1)) is not None


def test_a_broker_that_cannot_price_the_margin_refuses_nothing() -> None:
    """Unknown is not a refusal. A contract the broker will not quote stays in
    the catalogue and the per-trade margin check has the last word."""
    assert _verdict(_Broker(equity=176.0, margin=None)) is None


def test_unknown_equity_refuses_nothing() -> None:
    assert _verdict(_Broker(equity=0.0, margin=718.01)) is None


def test_the_refusal_happens_before_the_expensive_part() -> None:
    """The point of the gate. An unaffordable symbol must not reach the H1
    history fetch, because that is where a cycle's time actually goes."""
    broker = _Broker(equity=176.0, margin=718.01)
    scanner = _scanner(broker)

    candidate, inspection = scanner._inspect(
        SimpleNamespace(name="ADS", path="CFD\\Equity\\ADS")  # type: ignore[arg-type]
    )

    assert candidate is None
    assert inspection.stage == "affordability"
    assert broker.rate_calls == 0, "the symbol was analysed anyway"
