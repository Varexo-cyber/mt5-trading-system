"""Instrument specifications: pip maths, volume normalisation, stop distances.

This is the module where a quiet bug costs real money. Every position size and
every stop distance in the system flows through the maths here, so it is fully
unit-tested and deliberately free of any dependency on the terminal.

Terminology used consistently across the codebase:

    point   the smallest price increment the broker quotes (10**-digits)
    tick    the smallest tradable price increment (`trade_tick_size`;
            equals `point` for every FX symbol seen so far, but not guaranteed)
    pip     the conventional unit traders talk in: 0.0001 for most FX pairs,
            0.01 for JPY crosses, and 1 point for metals/indices/crypto

MT5 reports `trade_tick_value` in the ACCOUNT currency. That is what makes the
pip value computation here correct for a EUR-denominated account trading
USD-quoted pairs without us running a currency conversion of our own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.mt5_codes import (
    SYMBOL_FILLING_FOK,
    SYMBOL_FILLING_IOC,
    SYMBOL_TRADE_MODE_FULL,
    OrderFilling,
)

#: Symbols whose "pip" is one point rather than ten (metals, indices, crypto,
#: and anything else quoted with few decimals). Matched on the resolved digits,
#: not the name, so broker suffixes like `XAUUSD.pro` need no special-casing.
_FX_PIP_DIGITS = (3, 5)  # fractional-pip FX quoting
_FX_WHOLE_PIP_DIGITS = (2, 4)  # legacy whole-pip FX quoting

#: ISO 4217 assigns XAU/XAG/XPT/XPD to metals, so they pass a naive "is it a
#: three-letter currency code" test while behaving nothing like FX. Crypto
#: tickers are listed for the same reason.
_NON_FX_BASE_CODES = frozenset({"XAU", "XAG", "XPT", "XPD", "BTC", "ETH", "LTC", "XRP"})

#: FX contracts are 100 000 units (1 000 on cent accounts). Gold is 100 ounces,
#: indices are 1 or 10 per point. The size is the cleanest structural signal.
_MIN_FX_CONTRACT_SIZE = 1_000.0


class AssetClass(StrEnum):
    """Broker-neutral market families with materially different microstructure."""

    FOREX = "forex"
    CRYPTO = "crypto"
    STOCK = "stock"
    INDEX = "index"
    METAL = "metal"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Immutable snapshot of a broker's contract specification for one symbol.

    Built from `mt5.symbol_info()`. Snapshotted rather than queried on demand so
    that a broker changing specs mid-session cannot silently change the size of
    an order we already computed.
    """

    symbol: str
    digits: int
    point: float
    tick_size: float
    #: Profit/loss of a one-tick move on 1.00 lot, in ACCOUNT currency.
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    #: Minimum distance in POINTS between price and SL/TP (broker stop level).
    stops_level: int
    #: Distance in POINTS within which modification is blocked (freeze level).
    freeze_level: int
    currency_base: str
    currency_profit: str
    currency_margin: str
    filling_mode_mask: int
    trade_mode: int
    #: Whether the symbol is FX-like (pip = 10 points) or point-quoted.
    is_forex: bool
    #: Broker catalogue metadata. `path` is especially useful because MT5
    #: symbols such as BTCUSD do not reliably expose a crypto base currency.
    path: str
    description: str
    asset_class: AssetClass

    # -- construction -----------------------------------------------------

    @classmethod
    def from_mt5(cls, info: Any) -> InstrumentSpec:
        """Build from an `mt5.symbol_info()` named tuple.

        Raises ValueError on values that would make sizing maths nonsensical,
        rather than letting a zero propagate into a division.
        """
        digits = int(info.digits)
        point = float(info.point)
        tick_size = float(info.trade_tick_size) or point
        tick_value = float(info.trade_tick_value)
        volume_step = float(info.volume_step)

        if point <= 0 or tick_size <= 0:
            raise ValueError(f"{info.name}: non-positive point/tick_size from broker")
        if tick_value <= 0:
            raise ValueError(
                f"{info.name}: broker reports trade_tick_value={tick_value}; "
                "cannot size positions without it (symbol may not be selected "
                "in Market Watch, or the market is closed)"
            )
        if volume_step <= 0:
            raise ValueError(f"{info.name}: non-positive volume_step from broker")

        currency_profit = str(info.currency_profit)
        currency_base = str(info.currency_base)
        # An FX symbol needs all four to hold: two real currency codes, neither
        # of them a metal or crypto ticker, FX-typical quote precision, and an
        # FX-sized contract. Gold satisfies the first and third tests (XAU/USD,
        # 2 digits) and fails the other two, which is what keeps its pip at one
        # point instead of ten.
        is_forex = (
            len(currency_base) == 3
            and len(currency_profit) == 3
            and currency_base.isalpha()
            and currency_profit.isalpha()
            and currency_base.upper() not in _NON_FX_BASE_CODES
            and currency_profit.upper() not in _NON_FX_BASE_CODES
            and digits in (*_FX_PIP_DIGITS, *_FX_WHOLE_PIP_DIGITS)
            and float(info.trade_contract_size) >= _MIN_FX_CONTRACT_SIZE
        )
        path = str(getattr(info, "path", ""))
        description = str(getattr(info, "description", ""))
        asset_class = _classify_asset(path, currency_base, is_forex)

        return cls(
            symbol=str(info.name),
            digits=digits,
            point=point,
            tick_size=tick_size,
            tick_value=tick_value,
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=volume_step,
            stops_level=int(getattr(info, "trade_stops_level", 0)),
            freeze_level=int(getattr(info, "trade_freeze_level", 0)),
            currency_base=currency_base,
            currency_profit=currency_profit,
            currency_margin=str(info.currency_margin),
            filling_mode_mask=int(getattr(info, "filling_mode", 0)),
            trade_mode=int(getattr(info, "trade_mode", SYMBOL_TRADE_MODE_FULL)),
            is_forex=is_forex,
            path=path,
            description=description,
            asset_class=asset_class,
        )

    # -- pip maths --------------------------------------------------------

    @property
    def pip_size(self) -> float:
        """Price distance of one pip.

        FX quoted with 3 or 5 digits: pip = 10 points (fractional-pip quoting).
        FX quoted with 2 or 4 digits: pip = 1 point (legacy quoting).
        Everything else (gold, indices, crypto): pip = 1 point. We do not try to
        be clever about "gold pips" — there is no market-wide convention and an
        ambiguous unit is worse than an explicit one.
        """
        if self.is_forex and self.digits in _FX_PIP_DIGITS:
            return self.point * 10.0
        return self.point

    @property
    def points_per_pip(self) -> float:
        return self.pip_size / self.point

    def pips_to_price(self, pips: float) -> float:
        """Convert a pip distance to a price distance."""
        return pips * self.pip_size

    def price_to_pips(self, price_distance: float) -> float:
        """Convert a price distance to pips. Sign is preserved."""
        return price_distance / self.pip_size

    def pip_value_per_lot(self) -> float:
        """Money value, in ACCOUNT currency, of a one-pip move on 1.00 lot.

        Derived from the broker's own `trade_tick_value` so that cross-currency
        conversion is the broker's problem, not ours. For a EUR account trading
        EURUSD this returns roughly 9.2 EUR, not 10 USD — which is exactly the
        number the position sizer needs.
        """
        return self.tick_value * (self.pip_size / self.tick_size)

    def money_per_lot(self, price_distance: float) -> float:
        """Account-currency P/L of `price_distance` on 1.00 lot."""
        return self.tick_value * (abs(price_distance) / self.tick_size)

    # -- volume normalisation ---------------------------------------------

    def round_volume_down(self, volume: float) -> float:
        """Snap a volume DOWN to the broker's lot step.

        Always down, never to nearest: rounding up increases risk beyond what
        the sizer computed, which would make the configured risk-per-trade a
        lie. The residual is dropped, not accumulated.
        """
        if volume <= 0:
            return 0.0
        steps = math.floor(round(volume / self.volume_step, 9))
        snapped = steps * self.volume_step
        # volume_step is often 0.01; float noise here shows up as 0.30000000000004
        return round(snapped, self._volume_decimals)

    @property
    def _volume_decimals(self) -> int:
        """Decimal places implied by the lot step (0.01 -> 2, 0.001 -> 3)."""
        step_text = f"{self.volume_step:.8f}".rstrip("0")
        if "." not in step_text:
            return 0
        return len(step_text.split(".", 1)[1])

    def is_volume_tradable(self, volume: float) -> bool:
        """True if `volume` is a volume this broker will actually accept."""
        if volume < self.volume_min or volume > self.volume_max:
            return False
        steps = volume / self.volume_step
        return math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-6)

    def clamp_volume_to_max(self, volume: float) -> float:
        return min(volume, self.volume_max)

    # -- risk feasibility -------------------------------------------------

    def min_risk_money(self, sl_distance_price: float) -> float:
        """Smallest risk, in account currency, this instrument can express.

        That is: the loss taken if the minimum lot is stopped out at
        `sl_distance_price`. Below this there is no valid trade — the correct
        response is to skip, never to shrink the stop or round the lot up.
        """
        return self.money_per_lot(sl_distance_price) * self.volume_min

    def min_risk_pct(self, sl_distance_price: float, equity: float) -> float:
        """`min_risk_money` as a percentage of equity."""
        if equity <= 0:
            return math.inf
        return 100.0 * self.min_risk_money(sl_distance_price) / equity

    def max_sl_pips_for_risk(self, risk_money: float) -> float:
        """Widest stop, in pips, affordable at the minimum lot for `risk_money`.

        The startup guard uses this to tell you, in plain numbers, what this
        account can and cannot trade. On a EUR 100 account risking 1% (EUR 1) on
        EURUSD at 0.01 lots this comes out near 11 pips — which is the honest
        answer to "can I trade this account", and the answer is mostly "no".
        """
        per_pip = self.pip_value_per_lot() * self.volume_min
        if per_pip <= 0:
            return 0.0
        return risk_money / per_pip

    # -- broker constraints ------------------------------------------------

    @property
    def min_stop_distance_price(self) -> float:
        """Broker's minimum SL/TP distance from price, as a price distance.

        A `stops_level` of 0 means "no fixed limit" (dynamic, spread-based).
        We still keep our own ATR buffer well above zero regardless.
        """
        return self.stops_level * self.point

    def violates_stop_level(self, entry: float, stop: float) -> bool:
        """True if the broker would reject this stop for being too close."""
        if self.stops_level <= 0:
            return False
        return abs(entry - stop) < self.min_stop_distance_price - 1e-12

    @property
    def is_tradable(self) -> bool:
        return self.trade_mode == SYMBOL_TRADE_MODE_FULL

    def preferred_filling(self) -> OrderFilling:
        """Pick a filling mode the broker actually supports.

        Sending an unsupported filling mode is a common source of retcode 10030
        (INVALID_FILL) that looks like a mysterious rejection. IOC is preferred
        over FOK so that a partial fill is possible rather than a hard reject;
        RETURN is the fallback for symbols advertising neither bit.
        """
        if self.filling_mode_mask & SYMBOL_FILLING_IOC:
            return OrderFilling.IOC
        if self.filling_mode_mask & SYMBOL_FILLING_FOK:
            return OrderFilling.FOK
        return OrderFilling.RETURN

    # -- price normalisation ----------------------------------------------

    def normalize_price(self, price: float) -> float:
        """Round a price to the broker's tick grid and digit count."""
        if self.tick_size > 0:
            price = round(price / self.tick_size) * self.tick_size
        return round(price, self.digits)

    def describe(self) -> str:
        """One-line summary for startup logs and the execution report."""
        return (
            f"{self.symbol}: class={self.asset_class.value} digits={self.digits} "
            f"pip={self.pip_size:g} "
            f"pip_value/lot={self.pip_value_per_lot():.4f} (account ccy) "
            f"lots[{self.volume_min:g}..{self.volume_max:g} step {self.volume_step:g}] "
            f"stops_level={self.stops_level}pt filling={self.preferred_filling().name}"
        )


def _classify_asset(path: str, currency_base: str, is_forex: bool) -> AssetClass:
    """Classify from MT5's catalogue path, with conservative fallbacks."""
    if is_forex:
        return AssetClass.FOREX
    root = path.split("\\", 1)[0].strip().lower()
    if root in {"crypto", "cryptos"}:
        return AssetClass.CRYPTO
    if root in {"stock", "stocks", "shares"}:
        return AssetClass.STOCK
    if root in {"index", "indices"}:
        return AssetClass.INDEX
    if root in {"commodity", "commodities"}:
        return AssetClass.METAL if "metal" in path.lower() else AssetClass.COMMODITY
    if currency_base.upper() in {"XAU", "XAG", "XPT", "XPD"}:
        return AssetClass.METAL
    if currency_base.upper() in {"BTC", "ETH", "LTC", "XRP"}:
        return AssetClass.CRYPTO
    return AssetClass.UNKNOWN
