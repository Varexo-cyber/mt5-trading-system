from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_connector import MT5Connector
from core.types import AccountSnapshot
from promotion.experimental import (
    ExperimentalLiveContract,
    apply_experimental_live_limits,
    contract_path,
)
from runner.service import JarvisRunner, OperationMode
from tests.fakes.fake_mt5 import FakeMT5


def account(
    *,
    login: int = 5_049_535,
    server: str = "Eightcap-Live",
    currency: str = "EUR",
    equity: float = 100.0,
    is_demo: bool = False,
) -> AccountSnapshot:
    return AccountSnapshot(
        login=login,
        server=server,
        currency=currency,
        balance=equity,
        equity=equity,
        margin=0.0,
        margin_free=equity,
        margin_level=0.0,
        leverage=500,
        is_demo=is_demo,
        taken_at=datetime.now(UTC),
    )


def market(fake: FakeMT5) -> MT5Connector:
    return MT5Connector(MT5Config(), mt5_module=fake)


def write_contract(root: Path, snapshot: AccountSnapshot) -> ExperimentalLiveContract:
    contract = ExperimentalLiveContract.create(snapshot)
    contract.write(contract_path(root))
    return contract


def test_contract_is_account_bound_and_has_absolute_floor(tmp_path: Path) -> None:
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    contract = write_contract(tmp_path, account())

    restored = ExperimentalLiveContract.load(contract_path(tmp_path))
    restored.assert_compatible(account(), settings)

    assert restored == contract
    assert restored.equity_floor == pytest.approx(85.0)
    assert not restored.floor_breached(85.01)
    assert restored.floor_breached(85.0)
    with pytest.raises(RuntimeError, match=r"account .* does not match"):
        restored.assert_compatible(account(login=123), settings)
    with pytest.raises(RuntimeError, match="connected account is demo"):
        restored.assert_compatible(account(is_demo=True), settings)


def test_experimental_settings_fix_risk_and_explicitly_enable_research_modules() -> None:
    original = load_settings(env_overrides=False)
    experimental = apply_experimental_live_limits(original)

    assert experimental.effective_risk_pct() == 1.0
    assert experimental.effective_max_risk_pct() == 1.0
    assert experimental.risk.max_drawdown_circuit_breaker_pct == 15.0
    assert experimental.analysis.confluence.live_enabled_modules == (
        "market_structure",
        "trend_momentum",
        "liquidity_sweep",
        "level_reaction",
    )
    assert original.analysis.confluence.live_enabled_modules == ()


def test_experimental_runner_refuses_missing_contract(tmp_path: Path) -> None:
    fake = FakeMT5(
        equity=100.0,
        balance=100.0,
        currency="EUR",
        login_id=5_049_535,
        server="Eightcap-Live",
        is_demo=False,
    )
    runner = JarvisRunner(
        market(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.EXPERIMENTAL_LIVE,
    )

    with pytest.raises(RuntimeError, match="EXPERIMENTAL_LIVE_NOT_ARMED"):
        runner.connect()

    assert not fake.orders_sent
    assert not runner.broker.is_connected


def test_experimental_runner_connects_without_sending_an_order(tmp_path: Path) -> None:
    snapshot = account()
    write_contract(tmp_path, snapshot)
    fake = FakeMT5(
        equity=snapshot.equity,
        balance=snapshot.balance,
        currency=snapshot.currency,
        login_id=snapshot.login,
        server=snapshot.server,
        is_demo=False,
    )
    runner = JarvisRunner(
        market(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.EXPERIMENTAL_LIVE,
    )

    runner.connect()
    try:
        assert runner.experimental_contract is not None
        assert runner.settings.effective_risk_pct() == 1.0
        assert not fake.orders_sent
    finally:
        runner.close()


def test_experimental_capital_floor_engages_persistent_stop(tmp_path: Path) -> None:
    snapshot = account()
    write_contract(tmp_path, snapshot)
    fake = FakeMT5(
        equity=snapshot.equity,
        balance=snapshot.balance,
        currency=snapshot.currency,
        login_id=snapshot.login,
        server=snapshot.server,
        is_demo=False,
    )
    runner = JarvisRunner(
        market(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.EXPERIMENTAL_LIVE,
    )
    runner.connect()
    try:
        fake.equity = 85.0
        runner.run_once()

        assert runner.kill_switch.is_engaged()
        assert "EXPERIMENTAL LIVE CAPITAL STOP" in runner.kill_switch.reason()
        assert not fake.orders_sent
    finally:
        runner.close()


def test_experimental_live_refuses_demo_even_with_contract(tmp_path: Path) -> None:
    write_contract(tmp_path, account())
    fake = FakeMT5(
        equity=100.0,
        balance=100.0,
        currency="EUR",
        login_id=5_049_535,
        server="Eightcap-Live",
        is_demo=True,
    )
    runner = JarvisRunner(
        market(fake),
        load_settings(env_overrides=False),
        tmp_path,
        OperationMode.EXPERIMENTAL_LIVE,
    )

    with pytest.raises(RuntimeError, match="LIVE_ACCOUNT_REQUIRED"):
        runner.connect()

    assert not fake.orders_sent
