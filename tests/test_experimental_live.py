from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from advisory import Advice, Reflection
from advisory.providers import Supervision
from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_connector import MT5Connector
from core.types import AccountSnapshot
from promotion.experimental import (
    CONTRACT_VERSION,
    EXPERIMENTAL_EQUITY_FLOOR,
    EXPERIMENTAL_MAX_DRAWDOWN_PCT,
    EXPERIMENTAL_MAX_STAKE_PCT,
    EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT,
    EXPERIMENTAL_RISK_PER_TRADE_PCT,
    ExperimentalLiveContract,
    apply_experimental_live_limits,
    contract_path,
)
from runner.service import JarvisRunner, OperationMode
from tests.fakes.fake_mt5 import FakeMT5


class ApprovingAdvisor:
    def review(self, _idea, _context, _proposal=None):  # type: ignore[no-untyped-def]
        return Advice(True, 0.9, "approved by test adviser", provider="test")

    def reflect(self, _outcome):  # type: ignore[no-untyped-def]
        return Reflection("test reflection", provider="test")

    def supervise(self, _state):  # type: ignore[no-untyped-def]
        return Supervision("hold", "test supervisor holds", provider="test")


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
    # A fixed number, not equity-at-arming times 0.85. The derived version
    # weakened every time it was re-armed after a loss: armed at 100 the floor
    # was 85, armed again at 88.28 it became 75.04, and a further re-arm at 75
    # would have put it at 63.75. The capital stop eroded fastest in exactly the
    # situation it exists for.
    assert restored.equity_floor == pytest.approx(EXPERIMENTAL_EQUITY_FLOOR)
    assert not restored.floor_breached(EXPERIMENTAL_EQUITY_FLOOR + 0.01)
    assert restored.floor_breached(EXPERIMENTAL_EQUITY_FLOOR)
    with pytest.raises(RuntimeError, match=r"account .* does not match"):
        restored.assert_compatible(account(login=123), settings)
    with pytest.raises(RuntimeError, match="connected account is demo"):
        restored.assert_compatible(account(is_demo=True), settings)


def test_experimental_settings_fix_risk_without_promoting_modules() -> None:
    """Arming pins the risk envelope. It must not promote a module to live.

    Auto-enabling every weighted module the moment the experiment is armed
    would put unvalidated analysis on real money as a side effect of arming.
    Promotion has to be an explicit edit in config, visible in a diff.
    """
    original = load_settings(env_overrides=False)
    experimental = apply_experimental_live_limits(original)

    # 2%, because 1% of EUR 100 buys 0.0077 lots against a 0.01 minimum and
    # therefore cannot express a trade at all. See EXPERIMENTAL_RISK_PER_TRADE_PCT.
    # This is the ORDINARY stake and remains what an unremarkable approval gets.
    assert experimental.effective_risk_pct() == EXPERIMENTAL_RISK_PER_TRADE_PCT
    # The ceiling is a different number now. Only a reviewer confidence of 0.90
    # reaches it; see EXPERIMENTAL_MAX_STAKE_PCT.
    assert experimental.effective_max_risk_pct() == EXPERIMENTAL_MAX_STAKE_PCT
    assert experimental.risk.max_total_open_risk_pct == EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT
    # Zero: the automatic peak-to-current halt is off for this experiment, by
    # explicit choice. The fixed capital floor is the only unconditional stop
    # that remains — see EXPERIMENTAL_MAX_DRAWDOWN_PCT for the reasoning.
    assert experimental.risk.max_drawdown_circuit_breaker_pct == EXPERIMENTAL_MAX_DRAWDOWN_PCT

    # Whatever the shipped config says is what live gets — no more, no less.
    assert (
        experimental.analysis.confluence.live_enabled_modules
        == original.analysis.confluence.live_enabled_modules
    )
    assert original.analysis.confluence.live_enabled_modules == ()


def test_arming_forces_the_conviction_ramp_rather_than_reading_it() -> None:
    """A config edit must not be able to widen what the live account may stake.

    That is the whole reason arming exists, and the conviction ramp is the
    newest way it could have been got round: leaving `conviction_risk` to the
    overlay would have made the per-trade ceiling a yaml value again.
    """
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    ramp = settings.risk.conviction_risk

    assert ramp.enabled
    assert ramp.floor_pct == EXPERIMENTAL_RISK_PER_TRADE_PCT
    assert ramp.ceiling_pct == EXPERIMENTAL_MAX_STAKE_PCT
    # The floor equals the old fixed stake, so nothing an ordinary approval gets
    # has changed. Only setups above the confidence floor are sized differently.
    assert ramp.stake_for(0.60) == EXPERIMENTAL_RISK_PER_TRADE_PCT
    assert ramp.stake_for(1.00) == EXPERIMENTAL_MAX_STAKE_PCT


def test_a_contract_armed_before_conviction_staking_must_be_re_armed(
    tmp_path: Path,
) -> None:
    """The operational point of the version bump.

    A contract armed under the old build describes an envelope this build no
    longer runs — one stake, no ceiling, no aggregate cap. Silently
    reinterpreting it would mean the account trades a 6% ceiling that nobody
    ever confirmed, which is exactly the drift arming exists to prevent.
    """
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    stale = ExperimentalLiveContract(
        version=1,
        login=5_049_535,
        server="Eightcap-Live",
        currency="EUR",
        initial_equity=140.0,
        risk_per_trade_pct=EXPERIMENTAL_RISK_PER_TRADE_PCT,
        max_drawdown_pct=EXPERIMENTAL_MAX_DRAWDOWN_PCT,
        phrase="BEVESTIG EXPERIMENTEEL LIVE",
        armed_at=datetime.now(UTC).isoformat(),
    )

    with pytest.raises(RuntimeError, match="re-arm"):
        stale.assert_compatible(account(equity=140.0), settings)


def test_a_fresh_contract_records_the_whole_envelope(tmp_path: Path) -> None:
    contract = write_contract(tmp_path, account(equity=140.0))

    assert contract.version == CONTRACT_VERSION
    assert contract.max_stake_pct == EXPERIMENTAL_MAX_STAKE_PCT
    assert contract.max_total_open_risk_pct == EXPERIMENTAL_MAX_TOTAL_OPEN_RISK_PCT
    # It survives the round trip through disk, which is the only reason it is
    # written down at all.
    assert ExperimentalLiveContract.load(contract_path(tmp_path)) == contract


def test_a_config_reaching_past_the_contract_ceiling_is_refused(tmp_path: Path) -> None:
    """The check that makes the contract binding rather than decorative."""
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    over_reaching = settings.model_copy(
        update={
            "risk": settings.risk.model_copy(
                update={
                    "conviction_risk": settings.risk.conviction_risk.model_copy(
                        update={"ceiling_pct": EXPERIMENTAL_MAX_STAKE_PCT + 2.0}
                    )
                }
            )
        }
    )
    contract = write_contract(tmp_path, account(equity=140.0))

    with pytest.raises(RuntimeError, match="conviction staking reaches"):
        contract.assert_compatible(account(equity=140.0), over_reaching)


def test_a_conviction_floor_below_the_ordinary_stake_is_refused(tmp_path: Path) -> None:
    """A floor under 2% would quietly SHRINK unremarkable trades — the opposite
    of what was authorised, and invisible in the sizing line."""
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    shrunken = settings.model_copy(
        update={
            "risk": settings.risk.model_copy(
                update={
                    "conviction_risk": settings.risk.conviction_risk.model_copy(
                        update={"floor_pct": 1.0}
                    )
                }
            )
        }
    )
    contract = write_contract(tmp_path, account(equity=140.0))

    with pytest.raises(RuntimeError, match="conviction staking floors at"):
        contract.assert_compatible(account(equity=140.0), shrunken)


def test_shipped_config_cannot_trade_live_without_an_explicit_promotion() -> None:
    """With no module promoted, the confluence engine refuses every live entry."""
    settings = apply_experimental_live_limits(load_settings(env_overrides=False))
    assert settings.mode.is_live
    assert settings.analysis.confluence.live_enabled_modules == ()


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
        advisor=ApprovingAdvisor(),
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
        advisor=ApprovingAdvisor(),
    )

    runner.connect()
    try:
        assert runner.experimental_contract is not None
        assert runner.settings.effective_risk_pct() == EXPERIMENTAL_RISK_PER_TRADE_PCT
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
        advisor=ApprovingAdvisor(),
    )
    runner.connect()
    try:
        fake.equity = EXPERIMENTAL_EQUITY_FLOOR
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


def test_experimental_live_refuses_disabled_ai_gate(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="EXPERIMENTAL_LIVE_REQUIRES_AI"):
        runner.connect()

    assert not fake.orders_sent


def test_re_arming_after_a_loss_does_not_lower_the_floor(tmp_path: Path) -> None:
    """The bug this constant replaced, stated as a test.

    The floor used to be equity-at-arming times 0.85. Armed at 100 it was 85;
    re-armed at 88.28 after a losing day it silently became 75.04, and another
    re-arm at 75 would have made it 63.75. Ten euro of protection vanished with
    nothing on screen to say it had — the operator saw a new floor printed and
    no way to know it used to be higher.
    """
    rich = write_contract(tmp_path, account(equity=100.0))
    poor = write_contract(tmp_path, account(equity=88.28))
    broke = write_contract(tmp_path, account(equity=60.0))

    assert rich.equity_floor == poor.equity_floor == broke.equity_floor


def test_the_floor_survives_a_contract_written_by_an_older_build(tmp_path: Path) -> None:
    """Old contract files carry `initial_equity` and the old percentage. The
    floor must come from this build regardless, or a file written last week
    would still be enforcing last week's weaker number."""
    stale = write_contract(tmp_path, account(equity=1_000.0))
    assert stale.initial_equity == pytest.approx(1_000.0)
    assert stale.equity_floor == pytest.approx(EXPERIMENTAL_EQUITY_FLOOR)


def test_the_floor_is_below_the_drawdown_breaker_not_instead_of_it(tmp_path: Path) -> None:
    """Two backstops, and this is the deeper one.

    The 15% peak-to-current circuit breaker measures from the all-time equity
    high and normally halts trading well before the absolute floor is reached.
    Reading the floor as "the point at which I stop losing money" would be
    wrong by a wide margin, in the safe direction.
    """
    contract = write_contract(tmp_path, account(equity=100.0))
    peak_breaker_trips_at = 100.0 * (1.0 - EXPERIMENTAL_MAX_DRAWDOWN_PCT / 100.0)
    assert contract.equity_floor < peak_breaker_trips_at
