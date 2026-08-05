"""Account-bound contract for the explicitly accepted micro-live experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from config.schema import Settings
from core.types import AccountSnapshot, TradingMode
from infra.atomic import write_json_atomic

EXPERIMENTAL_LIVE_FILENAME = "EXPERIMENTAL_LIVE.json"
EXPERIMENTAL_LIVE_PHRASE = "BEVESTIG EXPERIMENTEEL LIVE"
# 2%, not 1%, and the reason is arithmetic rather than appetite.
#
# At EUR 100 the sizer reported: "1.00% of 100.00 is 1.00, which buys 0.0077
# lots" against a broker minimum of 0.01. Every setup was refused as
# TRADE_SKIPPED_UNDERCAPITALIZED — correctly, because rounding up to the minimum
# lot would have risked more than the rule allows. One percent of one hundred
# euro cannot express a trade on this broker at all, so the experiment could
# never produce a single observation.
#
# Two percent doubles that to 0.0154 lots, which rounds *down* to 0.01 and fits.
# It is the smallest number that makes the experiment possible, and it stays
# inside the 2% ceiling `micro_live` already declared, so no limit is being
# raised to accommodate it.
#
# This note used to add "and the 15% drawdown floor still ends the experiment
# after seven or eight losers". That is no longer true — the breaker is off (see
# below) and the fixed capital floor is what ends it now. At 2% of an 88 EUR
# account, roughly 1.77 EUR a trade, the distance from here to that floor is a
# great many losing trades rather than a handful.
EXPERIMENTAL_RISK_PER_TRADE_PCT = 2.0
# Zero: the automatic peak-to-current halt is off for this experiment.
#
# The operator turned it off after seeing what it left. At 88.28 EUR against a
# 96.17 peak the breaker had 6.54 EUR of room — three losing trades — while the
# posture throttle was refusing fifteen of every sixteen setups over the same
# drawdown. The account was too far down to trade and not far enough down to
# stop, which is the one state a risk system should never produce.
#
# The fixed capital floor below is now the only unconditional stop. That is a
# real reduction in protection and it is the choice that was made, with the
# numbers on the table.
EXPERIMENTAL_MAX_DRAWDOWN_PCT = 0.0

# The absolute capital stop, in account currency. A fixed number, chosen by the
# operator, and deliberately not derived from anything.
#
# It used to be `equity_at_arming * 0.85`, which meant the protection weakened
# every time it was re-armed after a loss. Armed at 100 the floor was 85; armed
# again at 88.28 it became 75.04; a run down to 75 and another re-arm would have
# put it at 63.75. The capital stop eroded fastest in exactly the situation it
# exists for, and it did so silently — the operator saw a new floor printed and
# had no way to know it used to be ten euro higher.
#
# A constant cannot do that. Re-arming does not move it, a losing week does not
# move it, and changing it is an edit to this line that shows up in a diff.
#
# With the peak-to-current breaker off, this is not the deeper of two backstops
# any more — it is the only one that cannot be reset, waited out, or recovered
# by a new day. Nothing automatic stands between the current equity and this
# number. Reaching it flattens everything and halts until a human restarts.
EXPERIMENTAL_EQUITY_FLOOR = 50.0


@dataclass(frozen=True, slots=True)
class ExperimentalLiveContract:
    """Durable approval bound to one broker account and one capital baseline."""

    version: int
    login: int
    server: str
    currency: str
    initial_equity: float
    risk_per_trade_pct: float
    max_drawdown_pct: float
    phrase: str
    armed_at: str

    @classmethod
    def create(cls, account: AccountSnapshot) -> ExperimentalLiveContract:
        if account.is_demo:
            raise RuntimeError("EXPERIMENTAL_LIVE_REQUIRES_REAL_ACCOUNT")
        if account.equity <= 0:
            raise RuntimeError("EXPERIMENTAL_LIVE_REQUIRES_POSITIVE_EQUITY")
        return cls(
            version=1,
            login=account.login,
            server=account.server,
            currency=account.currency,
            initial_equity=account.equity,
            risk_per_trade_pct=EXPERIMENTAL_RISK_PER_TRADE_PCT,
            max_drawdown_pct=EXPERIMENTAL_MAX_DRAWDOWN_PCT,
            phrase=EXPERIMENTAL_LIVE_PHRASE,
            armed_at=datetime.now(UTC).isoformat(),
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentalLiveContract:
        if not path.exists():
            raise RuntimeError("EXPERIMENTAL_LIVE_NOT_ARMED: run scripts/arm_experimental_live.py")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("EXPERIMENTAL_LIVE_CONTRACT_INVALID") from exc

    def write(self, path: Path) -> None:
        # required: without this file experimental live refuses to start, so a
        # silently dropped write would look like the arming never happened.
        write_json_atomic(path, asdict(self), required=True)

    @property
    def equity_floor(self) -> float:
        """Absolute capital floor, in addition to peak-to-current drawdown.

        Read from the build rather than from this contract, so an old contract
        file cannot carry a stale floor and re-arming cannot move it. See
        `EXPERIMENTAL_EQUITY_FLOOR` for why it is a constant and not derived
        from the equity at arming time.
        """
        return EXPERIMENTAL_EQUITY_FLOOR

    def floor_breached(self, equity: float) -> bool:
        return equity <= self.equity_floor + 1e-9

    def assert_compatible(self, account: AccountSnapshot, settings: Settings) -> None:
        failures: list[str] = []
        if self.version != 1:
            failures.append("unsupported contract version")
        if self.phrase != EXPERIMENTAL_LIVE_PHRASE:
            failures.append("confirmation phrase mismatch")
        if self.login != account.login:
            failures.append(f"account {account.login} does not match {self.login}")
        if self.server != account.server:
            failures.append(f"server {account.server!r} does not match {self.server!r}")
        if self.currency != account.currency:
            failures.append(f"currency {account.currency!r} does not match {self.currency!r}")
        if account.is_demo:
            failures.append("connected account is demo")
        if abs(self.risk_per_trade_pct - EXPERIMENTAL_RISK_PER_TRADE_PCT) > 1e-9:
            # The numbers, not a hardcoded sentence. This said "exactly 1%"
            # after the constant moved to 2%, so the message named a figure
            # nothing in the system used any more and gave the operator no way
            # to see which of the two values was actually wrong.
            failures.append(
                f"contract was armed at {self.risk_per_trade_pct:.2f}% but this build "
                f"requires {EXPERIMENTAL_RISK_PER_TRADE_PCT:.2f}% — re-arm with "
                f"--risk-percent {EXPERIMENTAL_RISK_PER_TRADE_PCT:g}"
            )
        if abs(self.max_drawdown_pct - EXPERIMENTAL_MAX_DRAWDOWN_PCT) > 1e-9:
            failures.append(
                f"contract drawdown is {self.max_drawdown_pct:.2f}% but this build "
                f"requires {EXPERIMENTAL_MAX_DRAWDOWN_PCT:.2f}%"
            )
        if abs(settings.effective_risk_pct() - self.risk_per_trade_pct) > 1e-9:
            failures.append(
                f"effective risk resolves to {settings.effective_risk_pct():.2f}% but the "
                f"contract binds it to {self.risk_per_trade_pct:.2f}%"
            )
        if settings.effective_max_risk_pct() > self.risk_per_trade_pct + 1e-9:
            failures.append(
                f"effective risk ceiling is {settings.effective_max_risk_pct():.2f}%, above "
                f"the contract's {self.risk_per_trade_pct:.2f}%"
            )
        if settings.risk.max_drawdown_circuit_breaker_pct > self.max_drawdown_pct + 1e-9:
            failures.append(
                f"configured drawdown breaker is "
                f"{settings.risk.max_drawdown_circuit_breaker_pct:.2f}%, above the "
                f"contract's {self.max_drawdown_pct:.2f}%"
            )
        if account.equity < settings.active_limits.min_equity:
            failures.append("equity is below the micro-live minimum")
        maximum = settings.active_limits.max_equity
        if maximum is not None and account.equity > maximum:
            failures.append("equity is above the micro-live maximum")
        if self.floor_breached(account.equity):
            failures.append(
                f"equity {account.equity:.2f} reached contract floor {self.equity_floor:.2f}"
            )
        if failures:
            raise RuntimeError("EXPERIMENTAL_LIVE_BLOCKED: " + "; ".join(failures))


def contract_path(root: Path) -> Path:
    return root / "runtime" / EXPERIMENTAL_LIVE_FILENAME


def _breaker_for(settings: Settings) -> float:
    """The circuit-breaker percentage this contract binds the config to.

    Spelled out rather than `min(config, EXPERIMENTAL)`, which used to be here
    and read as "take the tighter of the two". It stopped meaning that the
    moment zero came to mean *off*: with the contract at 0 and the config at 10,
    `min` returns 0 and quietly loosens the account, which is the exact opposite
    of what the expression looks like it is doing.

    The contract wins outright. That is its job — the whole point of arming is
    that the envelope cannot drift with a config edit — and it is now explicit
    in both directions rather than true only by arithmetic accident.
    """
    if EXPERIMENTAL_MAX_DRAWDOWN_PCT == 0.0:
        return 0.0
    return min(settings.risk.max_drawdown_circuit_breaker_pct, EXPERIMENTAL_MAX_DRAWDOWN_PCT)


def apply_experimental_live_limits(settings: Settings) -> Settings:
    """Return the fixed, non-configurable risk envelope for this experiment."""
    system = settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})
    risk = settings.risk.model_copy(
        update={
            "risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
            "max_risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
            "max_drawdown_circuit_breaker_pct": _breaker_for(settings),
        }
    )
    limits = dict(settings.modes)
    limits[TradingMode.MICRO_LIVE.value] = limits[TradingMode.MICRO_LIVE.value].model_copy(
        update={"max_risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT}
    )
    # `live_enabled_modules` is deliberately NOT derived here. Auto-promoting
    # every weighted module the moment the experiment is armed would mean an
    # unvalidated module reaches real money as a side effect of arming, which
    # is exactly the silent step this contract exists to prevent. Promoting a
    # module to live is an explicit edit in config, visible in a diff.
    return settings.model_copy(update={"system": system, "risk": risk, "modes": limits})
