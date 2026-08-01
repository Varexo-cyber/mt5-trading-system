"""Exception hierarchy.

Design rule: a failure that could cause the system to trade on wrong
assumptions must raise, never return a default. Silent degradation is how
trading systems lose money without anyone noticing.
"""

from __future__ import annotations


class TradingSystemError(Exception):
    """Base class for every error this system raises deliberately."""


# --- configuration -------------------------------------------------------


class ConfigError(TradingSystemError):
    """Configuration is missing, malformed, or internally inconsistent."""


class StartupGuardError(ConfigError):
    """Startup consistency check failed; the system must not start."""


# --- connectivity / platform --------------------------------------------


class MT5Error(TradingSystemError):
    """Base class for MetaTrader 5 platform errors."""


class MT5NotAvailableError(MT5Error):
    """The MetaTrader5 package cannot be imported (non-Windows host)."""


class MT5ConnectionError(MT5Error):
    """Could not initialise or log in to the terminal, or the link dropped."""


class SymbolNotAvailableError(MT5Error):
    """Symbol is unknown to the broker or cannot be selected in Market Watch."""


class OrderFailedError(MT5Error):
    """An order was rejected. Carries the raw MT5 return code."""

    def __init__(self, message: str, retcode: int | None = None, comment: str = "") -> None:
        super().__init__(message)
        self.retcode = retcode
        self.comment = comment


# --- data ----------------------------------------------------------------


class DataError(TradingSystemError):
    """Base class for market-data problems."""


class InsufficientDataError(DataError):
    """Fewer bars available than the caller requires to compute safely."""


class StaleDataError(DataError):
    """The newest closed bar is older than the freshness budget allows."""


class DataIntegrityError(DataError):
    """Bars are non-monotonic, duplicated, or contain impossible values."""


# --- risk ----------------------------------------------------------------


class RiskViolationError(TradingSystemError):
    """A hard risk rule was violated. Never catch this to continue trading."""


class ForbiddenStrategyError(RiskViolationError):
    """Martingale / grid / averaging-down / no-stop behaviour was attempted."""


# --- control -------------------------------------------------------------


class KillSwitchEngaged(TradingSystemError):
    """The STOP file exists: flatten and halt."""
