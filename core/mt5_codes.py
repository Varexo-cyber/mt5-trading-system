"""MetaTrader 5 constants, mirrored so the codebase imports cleanly on Linux.

The `MetaTrader5` package is Windows-only. Every numeric constant we depend on
is redefined here with its documented value, so that analysis, backtesting and
unit tests run anywhere. `MT5Connector` asserts at connect time that the values
below match the ones exposed by the real package, so a silent divergence after
a terminal update turns into a loud startup failure instead of a wrong order.
"""

from __future__ import annotations

from enum import IntEnum

# --- timeframes ----------------------------------------------------------

TIMEFRAME_VALUES: dict[str, int] = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 16385,
    "H2": 16386,
    "H3": 16387,
    "H4": 16388,
    "H6": 16390,
    "H8": 16392,
    "H12": 16396,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}


# --- order / trade enums -------------------------------------------------


class OrderType(IntEnum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


class TradeAction(IntEnum):
    DEAL = 1
    PENDING = 5
    SLTP = 6
    MODIFY = 7
    REMOVE = 8
    CLOSE_BY = 10


class OrderFilling(IntEnum):
    FOK = 0
    IOC = 1
    RETURN = 2


class OrderTime(IntEnum):
    GTC = 0
    DAY = 1
    SPECIFIED = 2
    SPECIFIED_DAY = 3


class PositionType(IntEnum):
    BUY = 0
    SELL = 1


# Bitmask exposed by `symbol_info(...).filling_mode`.
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2

# `symbol_info(...).trade_mode`
SYMBOL_TRADE_MODE_DISABLED = 0
SYMBOL_TRADE_MODE_LONGONLY = 1
SYMBOL_TRADE_MODE_SHORTONLY = 2
SYMBOL_TRADE_MODE_CLOSEONLY = 3
SYMBOL_TRADE_MODE_FULL = 4


# --- trade server return codes ------------------------------------------


class Retcode(IntEnum):
    REQUOTE = 10004
    REJECT = 10006
    CANCEL = 10007
    PLACED = 10008
    DONE = 10009
    DONE_PARTIAL = 10010
    ERROR = 10011
    TIMEOUT = 10012
    INVALID = 10013
    INVALID_VOLUME = 10014
    INVALID_PRICE = 10015
    INVALID_STOPS = 10016
    TRADE_DISABLED = 10017
    MARKET_CLOSED = 10018
    NO_MONEY = 10019
    PRICE_CHANGED = 10020
    PRICE_OFF = 10021
    INVALID_EXPIRATION = 10022
    ORDER_CHANGED = 10023
    TOO_MANY_REQUESTS = 10024
    NO_CHANGES = 10025
    SERVER_DISABLES_AT = 10026
    CLIENT_DISABLES_AT = 10027
    LOCKED = 10028
    FROZEN = 10029
    INVALID_FILL = 10030
    CONNECTION = 10031
    ONLY_REAL = 10032
    LIMIT_ORDERS = 10033
    LIMIT_VOLUME = 10034
    INVALID_ORDER = 10035
    POSITION_CLOSED = 10036
    INVALID_CLOSE_VOLUME = 10038
    CLOSE_ORDER_EXIST = 10039
    LIMIT_POSITIONS = 10040
    REJECT_CANCEL = 10041
    LONG_ONLY = 10042
    SHORT_ONLY = 10043
    CLOSE_ONLY = 10044
    FIFO_CLOSE = 10045
    HEDGE_PROHIBITED = 10046


SUCCESS_RETCODES: frozenset[int] = frozenset({Retcode.PLACED, Retcode.DONE, Retcode.DONE_PARTIAL})

#: Transient conditions. Re-sending with a refreshed price is legitimate.
RETRYABLE_RETCODES: frozenset[int] = frozenset(
    {
        Retcode.REQUOTE,
        Retcode.TIMEOUT,
        Retcode.PRICE_CHANGED,
        Retcode.PRICE_OFF,
        Retcode.TOO_MANY_REQUESTS,
        Retcode.CONNECTION,
        Retcode.FROZEN,
    }
)

#: Permanent for this attempt. Retrying either cannot help or is dangerous
#: (NO_MONEY especially — a retry loop there is how accounts get margin-called).
FATAL_RETCODES: frozenset[int] = frozenset(
    {
        Retcode.REJECT,
        Retcode.INVALID,
        Retcode.INVALID_VOLUME,
        Retcode.INVALID_PRICE,
        Retcode.INVALID_STOPS,
        Retcode.TRADE_DISABLED,
        Retcode.MARKET_CLOSED,
        Retcode.NO_MONEY,
        Retcode.INVALID_EXPIRATION,
        Retcode.SERVER_DISABLES_AT,
        Retcode.CLIENT_DISABLES_AT,
        Retcode.LOCKED,
        Retcode.INVALID_FILL,
        Retcode.ONLY_REAL,
        Retcode.LIMIT_ORDERS,
        Retcode.LIMIT_VOLUME,
        Retcode.INVALID_ORDER,
        Retcode.POSITION_CLOSED,
        Retcode.INVALID_CLOSE_VOLUME,
        Retcode.LIMIT_POSITIONS,
        Retcode.LONG_ONLY,
        Retcode.SHORT_ONLY,
        Retcode.CLOSE_ONLY,
        Retcode.FIFO_CLOSE,
        Retcode.HEDGE_PROHIBITED,
    }
)


def describe_retcode(retcode: int | None) -> str:
    """Human-readable name for a trade server return code."""
    if retcode is None:
        return "NO_RESULT"
    try:
        return Retcode(retcode).name
    except ValueError:
        return f"UNKNOWN_{retcode}"
