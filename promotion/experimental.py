"""Account-bound contract for the explicitly accepted micro-live experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from config.schema import Settings
from core.types import AccountSnapshot, TradingMode

EXPERIMENTAL_LIVE_FILENAME = "EXPERIMENTAL_LIVE.json"
EXPERIMENTAL_LIVE_PHRASE = "BEVESTIG EXPERIMENTEEL LIVE"
EXPERIMENTAL_RISK_PER_TRADE_PCT = 1.0
EXPERIMENTAL_MAX_DRAWDOWN_PCT = 15.0


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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)

    @property
    def equity_floor(self) -> float:
        """Absolute capital floor in addition to peak-to-current drawdown."""
        return self.initial_equity * (1.0 - self.max_drawdown_pct / 100.0)

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
            failures.append("contract risk must be exactly 1%")
        if abs(self.max_drawdown_pct - EXPERIMENTAL_MAX_DRAWDOWN_PCT) > 1e-9:
            failures.append("contract drawdown must be exactly 15%")
        if abs(settings.effective_risk_pct() - self.risk_per_trade_pct) > 1e-9:
            failures.append("effective risk is not exactly 1%")
        if settings.effective_max_risk_pct() > self.risk_per_trade_pct + 1e-9:
            failures.append("effective risk ceiling exceeds 1%")
        if settings.risk.max_drawdown_circuit_breaker_pct > self.max_drawdown_pct + 1e-9:
            failures.append("configured drawdown breaker exceeds 15%")
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


def apply_experimental_live_limits(settings: Settings) -> Settings:
    """Return the fixed, non-configurable risk envelope for this experiment."""
    system = settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})
    risk = settings.risk.model_copy(
        update={
            "risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
            "max_risk_per_trade_pct": EXPERIMENTAL_RISK_PER_TRADE_PCT,
            "max_drawdown_circuit_breaker_pct": min(
                settings.risk.max_drawdown_circuit_breaker_pct,
                EXPERIMENTAL_MAX_DRAWDOWN_PCT,
            ),
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
