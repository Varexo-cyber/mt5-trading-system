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
#
# This is now the FLOOR of a ramp rather than the whole story. See
# `EXPERIMENTAL_MAX_STAKE_PCT` below: an ordinary approval is still sized at
# exactly this number, and only a reviewer confidence above 0.70 buys more.
#
# 2.0 -> 3.0, 24 August, owner's instruction: "hij speelt nu continu met
# centen". At 2% of a 174 EUR account an ordinary trade risks 3.48 EUR, and a
# 1R winner nets a little over three euro after commission. The complaint is
# accurate. This is the number that answers it — the ordinary trade, not the
# exceptional one, is what "playing with cents" describes.
#
# WHAT IT DOES NOT DO. Commission is 5.50 EUR PER LOT, so a 1.5x position pays
# 1.5x commission. Profit, loss and cost all scale by the same factor and the
# ratio between them does not move at all. This makes outcomes bigger, not
# better, and anyone reading it as a fix for the cost drag is misreading it.
#
# WHAT IT COSTS IN RUNWAY. The 50 EUR floor is the only unconditional stop
# left. At 174 EUR that is 124 EUR of room: roughly 36 ordinary losers at 2%,
# roughly 24 at 3%. A third of the distance to the floor, bought for 50% more
# on each outcome.
EXPERIMENTAL_RISK_PER_TRADE_PCT = 3.0

# The most one trade may risk, reached only at maximum reviewer confidence.
#
# Authorised by the owner in these words: minimum 2%, up to 6-8% on a highly
# convincing setup, less as the conviction falls. Six rather than eight, and
# that is a choice worth naming: eight percent of a 140 EUR account is 11.20
# EUR a trade against a 50 EUR floor, which is eight maximum-conviction losers
# from the end of the experiment. Six makes it eleven. The owner authorised the
# range; this picks the conservative end of it, and moving to eight is a
# one-line edit plus a re-arm.
#
# WHAT DECIDES WHERE ON THE RAMP A TRADE LANDS is the reviewer's confidence and
# NOT the engine's own conviction score. That distinction is the whole design.
# The engine's conviction is measured on this account twice over not to predict
# the outcome — the "20+ points over the bar" bucket was the worst of all of
# them at -4.92R over 23 trades, and across 84 paid reviews the 40-45 band
# produced nothing useful while 20-25 produced 33%. Staking on that number
# would put the most money on the trades the record says are the worst.
#
# The reviewer's confidence has not been proven either. It is simply the one
# number in the chain that has not yet been disproven, and the ramp starts at
# 0.70 — well above the 0.55 approval bar — so that an approval which merely
# cleared the bar is sized exactly as it was before any of this existed.
#
# 6.0 -> 8.0, 24 August. The owner asked for "1.5x", which is 9. This is 8,
# and the reason is written four paragraphs above: the authorisation on record
# is "minimum 2%, up to 6-8% on a highly convincing setup", and 6 was chosen as
# the conservative end of a range the owner had already approved. Eight is the
# other end of that same range — the increment this file predicted in as many
# words ("moving to eight is a one-line edit plus a re-arm"). Nine would be a
# new authorisation rather than the use of an existing one, and the difference
# between 8 and 9 is not worth taking that step without being asked twice.
#
# At 174 EUR a maximum-conviction trade now risks 13.92 EUR against a 50 EUR
# floor. That is roughly nine such losers from the end of the experiment.
EXPERIMENTAL_MAX_STAKE_PCT = 8.0

# The most the whole book may risk at once, across every open position.
#
# Four slots at the 6% ceiling is 24% of the account on the table at one time,
# and nothing in the per-trade rules would have noticed. Six percent is a
# decision about one trade; twenty-four percent is a decision about the
# account, and no one made it. This is that decision: twelve percent, so two
# maximum-conviction trades fill the book and a third waits for one of them to
# close or to move its stop to break-even.
#
# Measured against the CURRENT stop of each open position, so banking a stop at
# break-even genuinely frees capacity rather than merely appearing to.
#
# 12.0 -> 16.0, moved WITH the ceiling rather than independently. The number
# was never a taste: it is two maximum-conviction trades, so that a third waits
# for one of them to close or to reach break-even. At a 6% ceiling that meant
# 12; at 8% it means 16. Leaving it at 12 would have quietly changed the design
# to "one big trade and a fragment", which is a different decision about the
# account and not one anybody made.
EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT = 16.0
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

# Bumped from 1 when conviction-scaled staking arrived. A contract armed before
# that change describes an envelope this build no longer runs — it pins one
# stake and knows nothing about a ceiling or an aggregate cap — so it must be
# re-armed rather than silently reinterpreted.
#
# Bumped to 3 on 24 August with the whole envelope: 2/6/12 became 3/8/16. A
# contract armed under version 2 describes a smaller account risk than this
# build applies, and the difference is the operator's to approve rather than
# this file's to assume. Live trading refuses until `rearm_experimental_live`
# has been run.
CONTRACT_VERSION = 3


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
    #: Defaulted only so that a version-1 file still *loads*, which is what
    #: lets `assert_compatible` reject it by name instead of the loader
    #: throwing `EXPERIMENTAL_LIVE_CONTRACT_INVALID` at an operator who has no
    #: way to tell a stale contract from a corrupt one.
    max_stake_pct: float = 0.0
    max_total_open_risk_pct: float = 0.0

    @classmethod
    def create(cls, account: AccountSnapshot) -> ExperimentalLiveContract:
        if account.is_demo:
            raise RuntimeError("EXPERIMENTAL_LIVE_REQUIRES_REAL_ACCOUNT")
        if account.equity <= 0:
            raise RuntimeError("EXPERIMENTAL_LIVE_REQUIRES_POSITIVE_EQUITY")
        return cls(
            version=CONTRACT_VERSION,
            login=account.login,
            server=account.server,
            currency=account.currency,
            initial_equity=account.equity,
            risk_per_trade_pct=EXPERIMENTAL_RISK_PER_TRADE_PCT,
            max_drawdown_pct=EXPERIMENTAL_MAX_DRAWDOWN_PCT,
            phrase=EXPERIMENTAL_LIVE_PHRASE,
            armed_at=datetime.now(UTC).isoformat(),
            max_stake_pct=EXPERIMENTAL_MAX_STAKE_PCT,
            max_total_open_risk_pct=EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT,
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
        if self.version != CONTRACT_VERSION:
            failures.append(
                f"contract is version {self.version} but this build writes version "
                f"{CONTRACT_VERSION} — conviction-scaled staking changed the envelope, "
                f"so re-arm with rearm_experimental_live.cmd"
            )
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
        if abs(self.max_stake_pct - EXPERIMENTAL_MAX_STAKE_PCT) > 1e-9:
            failures.append(
                f"contract caps a single trade at {self.max_stake_pct:.2f}% but this build "
                f"caps it at {EXPERIMENTAL_MAX_STAKE_PCT:.2f}% — re-arm"
            )
        if abs(self.max_total_open_risk_pct - EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT) > 1e-9:
            failures.append(
                f"contract caps total open risk at {self.max_total_open_risk_pct:.2f}% but "
                f"this build caps it at {EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT:.2f}% — re-arm"
            )
        # The per-trade ceiling, not the ordinary stake. These are two different
        # numbers now: an ordinary approval is sized at `risk_per_trade_pct` and
        # a very convincing one may reach `max_stake_pct`, so a ceiling above the
        # ordinary stake is the intended state rather than a loosened limit.
        if settings.effective_max_risk_pct() > self.max_stake_pct + 1e-9:
            failures.append(
                f"effective risk ceiling is {settings.effective_max_risk_pct():.2f}%, above "
                f"the contract's {self.max_stake_pct:.2f}%"
            )
        conviction = settings.risk.conviction_risk
        if conviction.enabled and conviction.ceiling_pct > self.max_stake_pct + 1e-9:
            failures.append(
                f"conviction staking reaches {conviction.ceiling_pct:.2f}%, above the "
                f"contract's {self.max_stake_pct:.2f}%"
            )
        if conviction.enabled and conviction.floor_pct + 1e-9 < self.risk_per_trade_pct:
            # A floor below the ordinary stake would make this change quietly
            # SHRINK unremarkable trades, which is the opposite of what was asked
            # for and would be invisible in the sizing line.
            failures.append(
                f"conviction staking floors at {conviction.floor_pct:.2f}%, below the "
                f"contract's ordinary {self.risk_per_trade_pct:.2f}%"
            )
        if settings.risk.max_total_open_risk_pct > self.max_total_open_risk_pct + 1e-9:
            failures.append(
                f"configured total open risk cap is "
                f"{settings.risk.max_total_open_risk_pct:.2f}%, above the contract's "
                f"{self.max_total_open_risk_pct:.2f}%"
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
    # The conviction ramp is forced here rather than read from the overlay, for
    # the same reason every other number in this function is: the point of
    # arming is that the envelope cannot drift with a config edit. A yaml change
    # to `risk.conviction_risk` cannot widen what this account can stake.
    risk = settings.risk.model_copy(
        update={
            "risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
            "max_risk_per_trade_pct": EXPERIMENTAL_MAX_STAKE_PCT,
            "max_total_open_risk_pct": EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT,
            "max_drawdown_circuit_breaker_pct": _breaker_for(settings),
            "conviction_risk": settings.risk.conviction_risk.model_copy(
                update={
                    "enabled": True,
                    "floor_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
                    "ceiling_pct": EXPERIMENTAL_MAX_STAKE_PCT,
                }
            ),
        }
    )
    limits = dict(settings.modes)
    limits[TradingMode.MICRO_LIVE.value] = limits[TradingMode.MICRO_LIVE.value].model_copy(
        update={"max_risk_per_trade_pct": EXPERIMENTAL_MAX_STAKE_PCT}
    )
    # `live_enabled_modules` is deliberately NOT derived here. Auto-promoting
    # every weighted module the moment the experiment is armed would mean an
    # unvalidated module reaches real money as a side effect of arming, which
    # is exactly the silent step this contract exists to prevent. Promoting a
    # module to live is an explicit edit in config, visible in a diff.
    return settings.model_copy(update={"system": system, "risk": risk, "modes": limits})
